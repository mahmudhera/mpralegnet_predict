#!/usr/bin/env python
"""Fine-tune a stored MPRA-LegNet (LegNet) model on our data.

This script:
- loads a stored LegNet checkpoint
- reads a table with DNA sequences + continuous targets
- creates train/val/test splits
- fine-tunes with our choice of optimizer: SGD, Adam, or AdamW
- uses validation set for model selection (best epoch)
- evaluates the best model on the test set

Only requires PyTorch (and CUDA if available). No pandas, no Lightning.

Input data format:
- TSV/CSV table with at least:
    * a sequence column (default: sequence)
    * a target column (default: target)

Example:
    python scripts/finetune.py \
        --model_dir /path/to/pretrained_model_dir \
        --data my_train_data.tsv --seq_col seq --target_col y \
        --out_dir out_finetune \
        --device cuda:0 \
        --optimizer adamw --lr 1e-4 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import json
from pathlib import Path
# Ensure repo root on sys.path so `import legnet` works without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from legnet import (
    IndexedSequenceRegressionDataset,
    LegNetConfig,
    load_model,
    save_checkpoint,
    split_indices,
)
from legnet.train_utils import eval_regression, run_epoch_train, set_seed

import pandas as pd


def build_optimizer(
    name: str,
    params,
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()

    src = parser.add_argument_group("model")
    src.add_argument("--model_dir", type=str, default=None, help="Directory with config.json + checkpoint(s)")
    src.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.ckpt/.pt/.pth)")
    src.add_argument("--config", type=str, default=None, help="Path to config.json (required for .ckpt)")

    data = parser.add_argument_group("data")
    data.add_argument("--data", type=str, required=True, help="TSV/CSV path with sequences + targets")
    data.add_argument("--ref_seq_col", type=str, default="reference sequence")
    data.add_argument("--ref_activity_col", type=str, default="reference activity")
    data.add_argument("--alt_seq_col", type=str, default="alternate sequence")
    data.add_argument("--alt_activity_col", type=str, default="alternate sequence activity")
    data.add_argument("--sep", type=str, default=None, help="Delimiter (default: inferred from extension)")
    data.add_argument("--seq_len", type=int, default=None, help="Pad/truncate sequences to this length")

    split = parser.add_argument_group("split")
    split.add_argument("--train_frac", type=float, default=0.8)
    split.add_argument("--val_frac", type=float, default=0.1)
    split.add_argument("--test_frac", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=777)

    train = parser.add_argument_group("train")
    train.add_argument("--out_dir", type=str, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=0.0)
    train.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    train.add_argument("--grad_clip", type=float, default=1.0)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    aug = parser.add_argument_group("augmentation")
    aug.add_argument(
        "--rc_augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Random reverse-complement augmentation during training",
    )
    aug.add_argument(
        "--rc_average",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Average forward+reverse predictions for val/test",
    )

    sel = parser.add_argument_group("model selection")
    sel.add_argument(
        "--select_metric",
        type=str,
        default="pearson",
        choices=["pearson", "mse"],
        help="Validation metric used to pick best epoch",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint + config
    if args.model_dir:
        model_dir = Path(args.model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"Could not find config.json in: {model_dir}")
        ckpt_path = None
        # Prefer a 'pearson*' ckpt if present, else newest.
        ckpts = list(model_dir.rglob("*.ckpt")) + list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth"))
        if not ckpts:
            raise SystemExit(f"No checkpoints found under: {model_dir}")
        pearson = [p for p in ckpts if p.name.startswith("pearson")]
        if pearson:
            pearson.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_path = pearson[0]
        else:
            ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_path = ckpts[0]
    else:
        if args.checkpoint is None:
            raise SystemExit("Provide either --model_dir or --checkpoint")
        ckpt_path = Path(args.checkpoint)
        config_path = Path(args.config) if args.config else None

    config: Optional[LegNetConfig] = None
    if config_path is not None:
        config = LegNetConfig.from_json(config_path)

    # If config isn't provided, try default (works for repo-native checkpoints)
    if config is None:
        config = LegNetConfig()

    # Default augmentation to upstream settings
    if args.rc_augment is None:
        args.rc_augment = bool(config.reverse_augment)
    if args.rc_average is None:
        args.rc_average = bool(config.reverse_augment)

    # Load data
    data_path = Path(args.data)
    sep = args.sep
    if sep is None:
        sep = "," if data_path.suffix.lower() == ".csv" else "\t"

    df = pd.read_csv(data_path, sep=sep)
    ref_seqs = df[args.ref_seq_col].str.upper().tolist()
    ref_targets = df[args.ref_activity_col].astype(float).tolist()
    alt_seqs = df[args.alt_seq_col].str.upper().tolist()
    alt_targets = df[args.alt_activity_col].astype(float).tolist()
    if len(ref_seqs) != len(ref_targets):
        raise SystemExit("Number of reference sequences and reference targets do not match")
    if len(alt_seqs) != len(alt_targets):
        raise SystemExit("Number of alternate sequences and alternate targets do not match")

    seqs = ref_seqs + alt_seqs
    targets = ref_targets + alt_targets

    # Decide seq_len
    seq_len = args.seq_len
    if seq_len is None:
        # Assume fixed-length sequences; keep as-is.
        seq_len = len(seqs[0])

    # Split
    set_seed(args.seed)
    train_idx, val_idx, test_idx = split_indices(
        len(seqs), train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )

    add_reverse_channel = bool(config.use_reverse_channel)

    train_ds = IndexedSequenceRegressionDataset(
        seqs,
        targets,
        train_idx,
        seq_len=seq_len,
        rc_augment=bool(args.rc_augment),
        add_reverse_channel=add_reverse_channel,
    )
    val_ds = IndexedSequenceRegressionDataset(
        seqs,
        targets,
        val_idx,
        seq_len=seq_len,
        rc_augment=False,
        add_reverse_channel=add_reverse_channel,
    )
    test_ds = IndexedSequenceRegressionDataset(
        seqs,
        targets,
        test_idx,
        seq_len=seq_len,
        rc_augment=False,
        add_reverse_channel=add_reverse_channel,
    )

    device = torch.device(args.device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model, meta = load_model(ckpt_path, config, map_location="cpu", device=device, strict=False)

    # Optional: update config.seq_len for self-contained saving
    config.seq_len = seq_len

    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    best_state = None
    best_metric: Optional[float] = None

    print("Loaded checkpoint:", ckpt_path)
    if config_path is not None:
        print("Loaded config:", config_path)
    print("Checkpoint key prefix used:", meta.get("used_prefix"))
    print(f"Data: n={len(seqs)} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print("Training with:", args.optimizer, "lr=", args.lr, "wd=", args.weight_decay)
    print("Augmentation: rc_augment=", bool(args.rc_augment), "| rc_average(eval)=", bool(args.rc_average))

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch_train(
            model,
            train_loader,
            optimizer,
            device,
            amp=args.amp,
            grad_clip=args.grad_clip,
        )

        # Evaluate (forward only)
        va = eval_regression(model, val_loader, device, amp=args.amp)

        # If requested, evaluate as strand-averaged by also flipping inputs.
        if args.rc_average:
            # Build a reverse val loader on the fly.
            from legnet.encoding import reverse_complement, encode_seq

            # Instead of re-reading sequences, we can just run a second pass with reverse=True
            # by wrapping the dataset (cheap).
            class _ReverseWrapper(torch.utils.data.Dataset):
                def __init__(self, base: IndexedSequenceRegressionDataset):
                    self.base = base

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, i):
                    seq, y = None, None
                    # we need the underlying sequence string, but IndexedSequenceRegressionDataset only stores seqs.
                    idx = self.base.indices[i]
                    s = self.base.seqs[idx]
                    y = self.base.targets[idx]
                    x = encode_seq(s, reverse=True, add_reverse_channel=add_reverse_channel, seq_len=seq_len)
                    return x, y

            rev_val_ds = _ReverseWrapper(val_ds)
            rev_val_loader = DataLoader(
                rev_val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            # average preds from forward+reverse
            from legnet.train_utils import predict_loader

            p_forw, y_true = predict_loader(model, val_loader, device, amp=args.amp)
            p_rev, _ = predict_loader(model, rev_val_loader, device, amp=args.amp)
            p_avg = (p_forw + p_rev) / 2.0
            va_loss = torch.nn.functional.mse_loss(p_avg, y_true).item()
            from legnet.metrics import pearsonr

            va_pearson = pearsonr(p_avg, y_true).item()
            va = type(va)(loss=float(va_loss), pearson=float(va_pearson), n=va.n)

        # Decide if best
        cur = va.pearson if args.select_metric == "pearson" else -va.loss
        if best_metric is None or cur > best_metric:
            best_metric = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_mse={tr.loss:.6f} | val_mse={va.loss:.6f} | val_pearson={va.pearson:.4f}"
        )

    if best_state is None:
        raise RuntimeError("No best state captured")

    model.load_state_dict(best_state)

    # Test evaluation (optionally strand-averaged)
    te = eval_regression(model, test_loader, device, amp=args.amp)

    if args.rc_average:
        class _ReverseWrapper(torch.utils.data.Dataset):
            def __init__(self, base: IndexedSequenceRegressionDataset):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                idx = self.base.indices[i]
                s = self.base.seqs[idx]
                y = self.base.targets[idx]
                from legnet.encoding import encode_seq

                x = encode_seq(s, reverse=True, add_reverse_channel=add_reverse_channel, seq_len=seq_len)
                return x, y

        rev_test_ds = _ReverseWrapper(test_ds)
        rev_test_loader = DataLoader(
            rev_test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        from legnet.train_utils import predict_loader

        p_forw, y_true = predict_loader(model, test_loader, device, amp=args.amp)
        p_rev, _ = predict_loader(model, rev_test_loader, device, amp=args.amp)
        p_avg = (p_forw + p_rev) / 2.0
        te_loss = torch.nn.functional.mse_loss(p_avg, y_true).item()
        from legnet.metrics import pearsonr

        te_pearson = pearsonr(p_avg, y_true).item()
        te = type(te)(loss=float(te_loss), pearson=float(te_pearson), n=te.n)

    print("\nBest validation metric:", args.select_metric, "=", best_metric)
    print(f"Test: mse={te.loss:.6f} | pearson={te.pearson:.4f} | n={te.n}")

    # Save best model (repo-native checkpoint)
    best_path = out_dir / "best_legnet.pt"
    extra = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "seed": args.seed,
        "rc_augment": bool(args.rc_augment),
        "rc_average": bool(args.rc_average),
        "select_metric": args.select_metric,
        "test_mse": te.loss,
        "test_pearson": te.pearson,
    }
    save_checkpoint(best_path, model, config, extra=extra)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump({"val_best_metric": best_metric, "test": {"mse": te.loss, "pearson": te.pearson}}, f, indent=2)

    print("Saved best checkpoint:", best_path)
    print("Saved metrics:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
