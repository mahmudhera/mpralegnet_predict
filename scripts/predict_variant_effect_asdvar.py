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

############### Imports complete ###############


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

    print('Created DataLoaders. Beginning inference...')



if __name__ == "__main__":
    main()