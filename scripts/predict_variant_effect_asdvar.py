#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure repo root on sys.path so `import legnet` works without installation.
# Ensure repo root on sys.path so `import legnet` works without installation.
_HERE = Path(__file__).resolve()
_REPO_ROOT = None
for _p in [_HERE.parent, *_HERE.parents]:
    if (_p / 'legnet').is_dir():
        _REPO_ROOT = _p
        break
if _REPO_ROOT is None:
    _REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from legnet import (
    LegNetConfig,
    SequenceOnlyDataset,
    load_model,
    resolve_checkpoint_path,
    resolve_config_path,
)

import pandas as pd
import numpy as np
from tqdm import tqdm

############### Imports complete ###############



def _predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    """Predict over an existing DataLoader (used for ensembling)."""
    model.eval()
    out_ids: List[str] = []
    preds: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch_ids, x = batch
            out_ids.extend(list(batch_ids))
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                p = model(x)
            preds.append(p.detach().float().cpu())

    return out_ids, torch.cat(preds, dim=0)



def main() -> None:
    parser = argparse.ArgumentParser()

    src = parser.add_argument_group("model")
    src.add_argument("--checkpoint_dir", type=str, default=None, help="Directory with config.json + checkpoint(s).")
    
    inp = parser.add_argument_group("input")
    inp.add_argument("--input", type=str, required=True, help="tsv with sequences where columns are pair_name,logFC,ref_sequence,alt_sequence,ref_alpha,alt_alpha")
    
    out = parser.add_argument_group("output")
    out.add_argument("--output", type=str, required=True, help="Output TSV path")
    
    run = parser.add_argument_group("runtime")
    run.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument("--batch_size", type=int, default=512)
    run.add_argument("--num_workers", type=int, default=4)
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--rc_average",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If true, average predictions for forward and reverse-complement strands",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)    # Resolve ensemble checkpoint directory + config (config.json must be inside)
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        raise SystemExit(f"--checkpoint_dir is not a directory: {ckpt_dir}")

    ckpt_paths: List[Path] = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpt_paths:
        raise SystemExit(f"No .ckpt files found in: {ckpt_dir}")

    # config path is the .json inside the checkpoint directory
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Could not find config.json in checkpoint directory: {ckpt_dir}")

    try:
        config = LegNetConfig.from_json(config_path)
    except Exception as e:
        raise SystemExit(f"Error loading config from {config_path}: {e}")

    # Default rc_average to upstream reverse_augment convention.
    if args.rc_average is None:
        args.rc_average = bool(config.reverse_augment)

    # Decide sequence length
    seq_len = config.seq_len
    device = torch.device(args.device)
    add_reverse_channel = args.rc_average or config.use_rev_channel

    # parse input TSV
    df = pd.read_csv(input_path, sep="\t")
    pair_names = df['pair_name'].tolist()
    ref_seqs = df['ref_sequence'].tolist()
    alt_seqs = df['alt_sequence'].tolist()

    # Create datasets
    ref_dataset_fwd = SequenceOnlyDataset(pair_names, ref_seqs, seq_len=seq_len, reverse=False, add_reverse_channel=add_reverse_channel)
    alt_dataset_fwd = SequenceOnlyDataset(pair_names, alt_seqs, seq_len=seq_len, reverse=False, add_reverse_channel=add_reverse_channel)
    ref_loader_fwd = DataLoader(ref_dataset_fwd, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=(device.type == "cuda"))
    alt_loader_fwd = DataLoader(alt_dataset_fwd, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=(device.type == "cuda"))

    if args.rc_average:
        ref_dataset_rev = SequenceOnlyDataset(pair_names, ref_seqs, seq_len=seq_len, reverse=True, add_reverse_channel=add_reverse_channel)
        alt_dataset_rev = SequenceOnlyDataset(pair_names, alt_seqs, seq_len=seq_len, reverse=True, add_reverse_channel=add_reverse_channel)
        ref_loader_rev = DataLoader(ref_dataset_rev, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=(device.type == "cuda"))
        alt_loader_rev = DataLoader(alt_dataset_rev, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=(device.type == "cuda"))
    else:
        ref_loader_rev = None
        alt_loader_rev = None

    # load all models
    models, meta_list = [], []
    for ckpt_path in ckpt_paths:
        model, meta = load_model(ckpt_path, config=config, device=device)
        models.append(model)
        meta_list.append(meta)

    # Predict with each model and aggregate
    print(f"Predicting variant effects using {len(models)} models...")

    all_ref_preds = []
    all_alt_preds = []

    for model in tqdm(models, desc="Models"):
        # forward predictions
        ref_ids_fwd, ref_preds_fwd = _predict_loader(model, ref_loader_fwd, device=device, amp=args.amp)
        alt_ids_fwd, alt_preds_fwd = _predict_loader(model, alt_loader_fwd, device=device, amp=args.amp)

        if args.rc_average:
            # reverse predictions if rc_average is enabled
            ref_ids_rev, ref_preds_rev = _predict_loader(model, ref_loader_rev, device=device, amp=args.amp)
            alt_ids_rev, alt_preds_rev = _predict_loader(model, alt_loader_rev, device=device, amp=args.amp)

            # sanity check on IDs
            assert ref_ids_fwd == ref_ids_rev, "Mismatch in reference sequence IDs between forward and reverse loaders"
            assert alt_ids_fwd == alt_ids_rev, "Mismatch in alternate sequence IDs between forward and reverse loaders"

            # average forward and reverse predictions
            ref_preds = (ref_preds_fwd + ref_preds_rev) / 2.0
            alt_preds = (alt_preds_fwd + alt_preds_rev) / 2.0

        else:
            ref_preds = ref_preds_fwd
            alt_preds = alt_preds_fwd

        all_ref_preds.append(ref_preds)
        all_alt_preds.append(alt_preds)

    # Aggregate predictions across models (mean)
    mean_ref_preds = torch.mean(torch.stack(all_ref_preds, dim=0), dim=0)
    mean_alt_preds = torch.mean(torch.stack(all_alt_preds, dim=0), dim=0)

    df['pred_ref'] = mean_ref_preds.numpy()
    df['pred_alt'] = mean_alt_preds.numpy()
    df['pred_effect'] = df['pred_alt'] - df['pred_ref']

    # Write output TSV
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote predictions to: {out_path}")



if __name__ == "__main__":
    main()