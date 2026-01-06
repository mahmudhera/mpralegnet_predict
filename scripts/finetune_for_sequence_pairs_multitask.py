from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure repo root on sys.path so `import legnet` works without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from legnet import LegNetConfig, load_model, split_indices
from legnet.train_utils import set_seed
from legnet.encoding import encode_seq

import pandas as pd


# -------------------------
# Utilities / metrics
# -------------------------

@torch.no_grad()
def pearsonr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = torch.sqrt((x * x).sum() + eps) * torch.sqrt((y * y).sum() + eps)
    return float((num / (den + eps)).clamp(-1.0, 1.0).item())


@torch.no_grad()
def mse_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x.float().flatten(), y.float().flatten()).item())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def seed_worker(worker_id: int) -> None:
    # Make python/random deterministic-ish per worker
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


# -------------------------
# Data loading
# -------------------------

# -------------------------
# Pair datasets
# -------------------------

class PairDeltaDataset(Dataset):
    """
    Returns: (x_ref, x_alt, delta)
    - delta = y_alt - y_ref
    Augmentations (train):
      - flip_pairs: swap (ref,alt) with probability 0.5 and negate delta
      - rc_pair_augment: reverse-complement BOTH sequences with probability 0.5
    """
    def __init__(
        self,
        ref_seqs: Sequence[str],
        ref_y: Sequence[float],
        alt_seqs: Sequence[str],
        alt_y: Sequence[float],
        indices: Sequence[int],
        *,
        seq_len: int,
        add_reverse_channel: bool,
        flip_pairs: bool = False,
        rc_pair_augment: bool = False,
        deterministic: bool = False,
        normalize_delta: bool = False,
        normalize_mean: Optional[float] = None,
        normalize_std: Optional[float] = None,
    ) -> None:
        self.ref_seqs = ref_seqs
        self.ref_y = ref_y
        self.alt_seqs = alt_seqs
        self.alt_y = alt_y
        self.indices = list(indices)

        self.seq_len = int(seq_len)
        self.add_reverse_channel = bool(add_reverse_channel)

        self.flip_pairs = bool(flip_pairs)
        self.rc_pair_augment = bool(rc_pair_augment)

        # if deterministic=True, no randomness even if flags enabled
        self.deterministic = bool(deterministic)

        self.normalize_delta = bool(normalize_delta)
        if normalize_mean is not None and normalize_std is not None:
            self.delta_mean = float(normalize_mean)
            self.delta_std = float(normalize_std)
        else:
            deltas = [self.alt_y[i] - self.ref_y[i] for i in self.indices]
            self.delta_mean = float(torch.tensor(deltas, dtype=torch.float32).mean().item())
            self.delta_std = float(torch.tensor(deltas, dtype=torch.float32).std().item())
            if self.delta_std < 1e-6:
                raise RuntimeError("Delta standard deviation is too small; cannot normalize.")

    def __len__(self) -> int:
        return len(self.indices)

    def get_mean_std_of_deltas(self) -> Tuple[float, float]:
        return self.delta_mean, self.delta_std

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rs = self.ref_seqs[idx]
        asq = self.alt_seqs[idx]
        delta = float(self.alt_y[idx] - self.ref_y[idx])
        if self.normalize_delta:
            delta = (delta - self.delta_mean) / self.delta_std

        do_flip = False
        do_rc = False
        if not self.deterministic:
            if self.flip_pairs and random.random() < 0.5:
                do_flip = True
            if self.rc_pair_augment and random.random() < 0.5:
                do_rc = True

        if do_flip:
            rs, asq = asq, rs
            delta = -delta

        x_ref = encode_seq(rs, reverse=do_rc, add_reverse_channel=self.add_reverse_channel, seq_len=self.seq_len)
        x_alt = encode_seq(asq, reverse=do_rc, add_reverse_channel=self.add_reverse_channel, seq_len=self.seq_len)
        y = torch.tensor(delta, dtype=torch.float32)
        return x_ref, x_alt, y



class ReversePairWrapper(Dataset):
    """
    Deterministic wrapper that returns reverse-complement encodings for BOTH sequences.
    Use for rc_average at eval-time.
    """
    def __init__(self, base: PairDeltaDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        idx = self.base.indices[i]
        rs = self.base.ref_seqs[idx]
        asq = self.base.alt_seqs[idx]
        delta = float(self.base.alt_y[idx] - self.base.ref_y[idx])

        x_ref = encode_seq(rs, reverse=True, add_reverse_channel=self.base.add_reverse_channel, seq_len=self.base.seq_len)
        x_alt = encode_seq(asq, reverse=True, add_reverse_channel=self.base.add_reverse_channel, seq_len=self.base.seq_len)
        y = torch.tensor(delta, dtype=torch.float32)
        return x_ref, x_alt, y

# -------------------------
# Encoder: reuse LegNet as embedding extractor
# -------------------------

def legnet_embedding_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Returns the penultimate embedding used by the LegNet head:
      stem -> main -> mapper -> adaptive_avg_pool -> squeeze
    This matches upstream LegNet forward structure.

    If the checkpoint uses a different module layout, we can modify here.
    """
    # Common in human_legnet: model.stem, model.main, model.mapper, model.head
    if hasattr(model, "stem") and hasattr(model, "main") and hasattr(model, "mapper"):
        x = model.stem(x)
        x = model.main(x)
        x = model.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

    # Fallback: hook the input to the first Linear in model.head
    if hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        first_linear = None
        for m in model.head:
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            raise RuntimeError("Could not find a Linear layer in model.head to hook for embeddings.")

        emb_holder: Dict[str, torch.Tensor] = {}

        def _hook(_module, inputs, _output):
            emb_holder["emb"] = inputs[0]

        handle = first_linear.register_forward_hook(_hook)
        _ = model(x)
        handle.remove()
        if "emb" not in emb_holder:
            raise RuntimeError("Embedding hook did not fire; cannot extract embedding.")
        return emb_holder["emb"]

    raise RuntimeError("Unsupported model layout: cannot extract embeddings.")


class LegNetEncoder(nn.Module):
    def __init__(self, legnet: nn.Module):
        super().__init__()
        self.legnet = legnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return legnet_embedding_forward(self.legnet, x)



# -------------------------
# Method B: Siamese Δ head
# -------------------------

class SiameseDeltaHead(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        print(f"Created SiameseDeltaHead: embed_dim={embed_dim}, hidden_dim={hidden_dim}, dropout={dropout}")

    def forward(self, x_ref: torch.Tensor, x_alt: torch.Tensor) -> torch.Tensor:
        h_ref = self.encoder(x_ref)
        h_alt = self.encoder(x_alt)
        d = h_alt - h_ref
        return self.head(d).squeeze(-1)


@torch.no_grad()
def predict_delta_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for x_ref, x_alt, y in loader:
        x_ref = x_ref.to(device, non_blocking=True)
        x_alt = x_alt.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            p = model(x_ref, x_alt)
        preds.append(p.detach().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(preds, dim=0).view(-1), torch.cat(ys, dim=0).view(-1)


@torch.no_grad()
def eval_delta_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
) -> Dict[str, float]:
    if not rc_average:
        p, y = predict_delta_loader(model, loader, device, amp=amp)
        return {"mse": mse_torch(p, y), "pearson": pearsonr_torch(p, y), "n": int(len(y))}

    # rc_average: average forward+reverse predictions
    if not isinstance(loader.dataset, PairDeltaDataset):
        raise RuntimeError("rc_average expects loader.dataset to be PairDeltaDataset")
    rev_ds = ReversePairWrapper(loader.dataset)
    rev_loader = DataLoader(
        rev_ds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    p_fwd, y = predict_delta_loader(model, loader, device, amp=amp)
    p_rev, _ = predict_delta_loader(model, rev_loader, device, amp=amp)
    p = (p_fwd + p_rev) / 2.0
    return {"mse": mse_torch(p, y), "pearson": pearsonr_torch(p, y), "n": int(len(y))}


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


def run_siamese(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
    epochs: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    grad_clip: float,
    hidden_dim: int,
    dropout: float,
    loss_name: str,
    select_metric: str,
    out_dir: Path,
) -> Dict:
    # infer embed dim
    encoder.eval()
    x_ref0, _, _ = next(iter(train_loader))
    x_ref0 = x_ref0.to(device)
    with torch.no_grad():
        h0 = encoder(x_ref0)
    embed_dim = int(h0.shape[-1])

    model = SiameseDeltaHead(encoder=encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)

    if loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    else:
        raise ValueError("loss_name must be 'mse' or 'huber'")

    opt = build_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_state = None
    best_score = None

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for x_ref, x_alt, y in train_loader:
            x_ref = x_ref.to(device, non_blocking=True)
            x_alt = x_alt.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                pred = model(x_ref, x_alt)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])

        train_loss = total / max(1, n)
        va = eval_delta_model(model, val_loader, device, amp=amp, rc_average=rc_average)

        score = va["pearson"] if select_metric == "pearson" else -va["mse"]
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[siamese] epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} "
            f"| val_mse={va['mse']:.6f} val_pearson={va['pearson']:.4f}"
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    te = eval_delta_model(model, test_loader, device, amp=amp, rc_average=rc_average)

    out_path = out_dir / "siamese_delta_head.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "rc_average": bool(rc_average),
            "select_metric": select_metric,
            "test": te,
        },
        out_path,
    )

    metrics = {
        "method": "siamese",
        "val_best_metric": float(best_score) if best_score is not None else None,
        "test": te,
        "rc_average": bool(rc_average),
        "select_metric": select_metric,
        "saved_model": str(out_path),
    }
    with (out_dir / "metrics_siamese.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics



# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    src = parser.add_argument_group("model")
    src.add_argument("--model_dir", type=str, default=None, help="Directory with config.json + checkpoint(s)")
    src.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (.ckpt/.pt/.pth)")
    src.add_argument("--config", type=str, default=None, help="Path to config.json (required for .ckpt)")
    src.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False, help="strict load_state_dict")

    data = parser.add_argument_group("data")
    data.add_argument("--data", type=str, required=True, help="TSV/CSV path with paired sequences + activities")
    data.add_argument("--ref_seq_col", type=str, default="reference sequence")
    data.add_argument("--ref_activity_col", type=str, default="reference activity")
    data.add_argument("--alt_seq_col", type=str, default="alternate sequence")
    data.add_argument("--alt_activity_col", type=str, default="alternate sequence activity")
    data.add_argument("--sep", type=str, default=None, help="Delimiter (default: inferred from extension)")
    data.add_argument("--seq_len", type=int, default=200, help="Pad/truncate sequences to this length (default: 200)")
    data.add_argument("--normalize_delta", action=argparse.BooleanOptionalAction, default=False,
                      help="Normalize delta values to zero mean/unit variance")

    split = parser.add_argument_group("split")
    split.add_argument("--train_frac", type=float, default=0.8)
    split.add_argument("--val_frac", type=float, default=0.1)
    split.add_argument("--test_frac", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=777)

    run = parser.add_argument_group("run")
    run.add_argument("--out_dir", type=str, required=True)
    run.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--select_metric", type=str, default="pearson", choices=["pearson", "mse"])
    run.add_argument("--rc_pair_augment", action=argparse.BooleanOptionalAction, default=True,
                     help="Train-time pairwise reverse-complement augmentation (same orientation for ref+alt)")
    run.add_argument("--rc_average", action=argparse.BooleanOptionalAction, default=True,
                     help="Eval-time average forward+reverse predictions/embeddings")
    run.add_argument("--flip_pairs", action=argparse.BooleanOptionalAction, default=True,
                     help="Train-time (siamese/softcls) random swap(ref,alt) with label sign flip")

    train = parser.add_argument_group("training hyperparameters")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--num_workers", type=int, default=1)
    train.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=1e-3)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--grad_clip", type=float, default=1.0)
    train.add_argument("--hidden_dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])

    enc = parser.add_argument_group("encoder freezing")
    enc.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=True,
                     help="Freeze LegNet encoder (recommended for 5-10k pairs).")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Resolve checkpoint + config (same style as your finetune.py)
    if args.model_dir:
        model_dir = Path(args.model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"Could not find config.json in: {model_dir}")

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
    if config is None:
        config = LegNetConfig()

    add_reverse_channel = bool(getattr(config, "use_reverse_channel", False))

    # Reproducibility
    set_seed(args.seed)

    # Load data
    data_path = Path(args.data)
    if args.sep is not None:
        sep = args.sep
    else:
        if data_path.suffix in {".tsv", ".txt"}:
            sep = "\t"
        else:
            sep = ","

    df = pd.read_csv(data_path, sep=sep)
    ref_seqs = df[args.ref_seq_col].astype(str).tolist()
    ref_y = df[args.ref_activity_col].astype(float).tolist()
    alt_seqs = df[args.alt_seq_col].astype(str).tolist()
    alt_y = df[args.alt_activity_col].astype(float).tolist()

    if not (len(ref_seqs) == len(ref_y) == len(alt_seqs) == len(alt_y)):
        raise SystemExit("Mismatched number of sequences/activities in the provided data.")

    n_pairs = len(ref_seqs)
    seq_len = int(args.seq_len)

    # Pair-level split
    train_idx, val_idx, test_idx = split_indices(
        n_pairs,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    # create device 
    device = torch.device(args.device)

    # Create datasets + loaders
    train_ds = PairDeltaDataset(
        ref_seqs,
        ref_y,
        alt_seqs,
        alt_y,
        train_idx,
        seq_len=seq_len,
        add_reverse_channel=add_reverse_channel,
        flip_pairs=args.flip_pairs,
        rc_pair_augment=args.rc_pair_augment,
        deterministic=False,
        normalize_delta=args.normalize_delta,
    )
    val_ds = PairDeltaDataset(
        ref_seqs,
        ref_y,
        alt_seqs,
        alt_y,
        val_idx,
        seq_len=seq_len,
        add_reverse_channel=add_reverse_channel,
        flip_pairs=False,
        rc_pair_augment=False,
        deterministic=True,
        normalize_delta=args.normalize_delta,
        normalize_mean=train_ds.delta_mean if args.normalize_delta else None,
        normalize_std=train_ds.delta_std if args.normalize_delta else None,
    )
    test_ds = PairDeltaDataset(
        ref_seqs,
        ref_y,
        alt_seqs,
        alt_y,
        test_idx,
        seq_len=seq_len,
        add_reverse_channel=add_reverse_channel,
        flip_pairs=False,
        rc_pair_augment=False,
        deterministic=True,
        normalize_delta=args.normalize_delta,
        normalize_mean=train_ds.delta_mean if args.normalize_delta else None,
        normalize_std=train_ds.delta_std if args.normalize_delta else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
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

    # Load LegNet encoder
    model, meta = load_model(ckpt_path, config, map_location="cpu", device=device, strict=bool(args.strict))
    print("Loaded checkpoint:", ckpt_path)
    if config_path is not None:
        print("Loaded config:", config_path)
    print("Checkpoint key prefix used:", meta.get("used_prefix"))
    print(f"Pairs: n={n_pairs} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("seq_len:", seq_len, "| add_reverse_channel:", add_reverse_channel)
    print("methods:", args.methods)
    print("rc_pair_augment(train):", bool(args.rc_pair_augment), "| flip_pairs(train):", bool(args.flip_pairs))
    print("rc_average(eval):", bool(args.rc_average))

    # Build encoder wrapper
    encoder = LegNetEncoder(model).to(device)

    # Freeze encoder if requested
    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()  # freeze BN stats
        print("Encoder frozen: True (encoder.eval() to freeze BatchNorm stats)")
    else:
        encoder.train()
        print("Encoder frozen: False")

    # Save run config
    with (out_dir / "run_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    all_metrics: Dict[str, Dict] = {}

    siamese_dir = out_dir / "siamese"
    ensure_dir(siamese_dir)
    m = run_siamese(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        amp=bool(args.amp),
        rc_average=bool(args.rc_average),
        epochs=int(args.epochs),
        optimizer_name=args.optimizer,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        momentum=float(args.momentum),
        grad_clip=float(args.grad_clip),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        loss_name=args.loss,
        select_metric=args.select_metric,
        out_dir=siamese_dir,
    )
    all_metrics["siamese"] = m
    print("[siamese] test:", m["test"])

    print("\nDone. Wrote outputs to:", out_dir)

if __name__ == "__main__":
    main()