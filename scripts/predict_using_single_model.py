#!/usr/bin/env python
"""Make predictions with a stored MPRA-LegNet (LegNet) model.

Supports:
- Upstream Lightning checkpoints (.ckpt) + upstream config.json
- Repo-native checkpoints (.pt/.pth) saved by scripts/finetune.py

Input formats:
- FASTA (.fa/.fasta/.fna)
- TSV/CSV with a sequence column
- plain text (one sequence per line)

Example (upstream human_legnet layout):
    python scripts/predict.py \
        --model_dir /path/to/model_dir_or_fold_dir \
        --input my_seqs.fasta \
        --output preds.tsv \
        --device cuda:0

Example (explicit):
    python scripts/predict.py \
        --checkpoint /path/to/model.ckpt \
        --config /path/to/config.json \
        --input my_seqs.tsv --format table --seq_col seq \
        --output preds.tsv --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
# Ensure repo root on sys.path so `import legnet` works without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _read_fasta(path: Path) -> Tuple[List[str], List[str]]:
    """Read FASTA into (ids, seqs). Tries Biopython first."""
    try:
        from Bio import SeqIO  # type: ignore

        recs = list(SeqIO.parse(str(path), format="fasta"))
        ids = [r.id for r in recs]
        seqs = [str(r.seq).upper() for r in recs]
        return ids, seqs
    except Exception:
        # Minimal fallback FASTA parser.
        ids: List[str] = []
        seqs: List[str] = []
        cur_id: Optional[str] = None
        cur_chunks: List[str] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur_id is not None:
                        ids.append(cur_id)
                        seqs.append("".join(cur_chunks).upper())
                    cur_id = line[1:].split()[0]
                    cur_chunks = []
                else:
                    cur_chunks.append(line)
            if cur_id is not None:
                ids.append(cur_id)
                seqs.append("".join(cur_chunks).upper())
        return ids, seqs


def _read_txt(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []
    with path.open("r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            ids.append(str(i))
            seqs.append(line.upper())
    return ids, seqs


def _read_table(
    path: Path,
    *,
    seq_col: str,
    id_col: Optional[str],
    sep: str,
    has_header: bool,
) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    seqs: List[str] = []

    with path.open("r", newline="") as f:
        if has_header:
            reader = csv.DictReader(f, delimiter=sep)
            if reader.fieldnames is None or seq_col not in reader.fieldnames:
                raise ValueError(
                    f"Could not find sequence column '{seq_col}' in header: {reader.fieldnames}"
                )
            for i, row in enumerate(reader, start=1):
                seq = row[seq_col]
                _id = row[id_col] if (id_col and id_col in row) else str(i)
                ids.append(_id)
                seqs.append(seq.upper())
        else:
            # No header: assume either 1 column (seq) or 2 columns (id, seq)
            reader = csv.reader(f, delimiter=sep)
            for i, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) == 1:
                    ids.append(str(i))
                    seqs.append(row[0].upper())
                else:
                    ids.append(row[0])
                    seqs.append(row[1].upper())

    return ids, seqs


def _infer_format(path: Path, user_format: Optional[str]) -> str:
    if user_format:
        return user_format
    suf = path.suffix.lower()
    if suf in {".fa", ".fasta", ".fna"}:
        return "fasta"
    if suf in {".csv", ".tsv"}:
        return "table"
    return "txt"


def _predict_dataset(
    model: torch.nn.Module,
    ds: SequenceOnlyDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

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
    src.add_argument("--model_dir", type=str, default=None, help="Directory with config.json + checkpoint(s).")
    src.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint (.ckpt/.pt/.pth).")
    src.add_argument("--config", type=str, default=None, help="Path to config.json (required for .ckpt).")

    inp = parser.add_argument_group("input")
    inp.add_argument("--input", type=str, required=True, help="FASTA / table / txt with sequences")
    inp.add_argument("--format", type=str, default=None, choices=["fasta", "table", "txt"], help="Override input format")
    inp.add_argument("--seq_col", type=str, default="sequence", help="Sequence column name for table input")
    inp.add_argument("--id_col", type=str, default=None, help="ID column name for table input")
    inp.add_argument("--sep", type=str, default=None, help="Delimiter for table input (default: inferred from extension)")
    inp.add_argument("--no_header", action="store_true", help="Table input has no header")
    inp.add_argument("--seq_len", type=int, default=None, help="Pad/truncate sequences to this length")

    out = parser.add_argument_group("output")
    out.add_argument("--output", type=str, required=True, help="Output TSV path")
    out.add_argument("--write_seq", action="store_true", help="Also write the sequence column to output")

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
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve config + checkpoint
    if args.model_dir:
        model_dir = Path(args.model_dir)
        config_path = resolve_config_path(model_dir)
        ckpt_path = resolve_checkpoint_path(model_dir)
    else:
        if args.checkpoint is None:
            raise SystemExit("Provide either --model_dir or --checkpoint")
        ckpt_path = Path(args.checkpoint)
        config_path = Path(args.config) if args.config else None

    config: Optional[LegNetConfig] = None
    if config_path is not None:
        config = LegNetConfig.from_json(config_path)

    # If user didn't provide config (possible for repo-native .pt), try to load from checkpoint.
    if config is None:
        config = LegNetConfig()

    # Default rc_average to upstream reverse_augment convention.
    if args.rc_average is None:
        args.rc_average = bool(config.reverse_augment)

    # Decide sequence length
    seq_len = args.seq_len if args.seq_len is not None else config.seq_len

    device = torch.device(args.device)

    model, meta = load_model(ckpt_path, config, map_location="cpu", device=device, strict=False)

    # Input reading
    fmt = _infer_format(input_path, args.format)
    if fmt == "fasta":
        ids, seqs = _read_fasta(input_path)
    elif fmt == "table":
        sep = args.sep
        if sep is None:
            sep = "," if input_path.suffix.lower() == ".csv" else "\t"
        ids, seqs = _read_table(
            input_path,
            seq_col=args.seq_col,
            id_col=args.id_col,
            sep=sep,
            has_header=(not args.no_header),
        )
    else:
        ids, seqs = _read_txt(input_path)

    add_reverse_channel = bool(config.use_reverse_channel)

    ds_forw = SequenceOnlyDataset(ids, seqs, seq_len=seq_len, reverse=False, add_reverse_channel=add_reverse_channel)
    out_ids, p_forw = _predict_dataset(
        model,
        ds_forw,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    if args.rc_average:
        ds_rev = SequenceOnlyDataset(ids, seqs, seq_len=seq_len, reverse=True, add_reverse_channel=add_reverse_channel)
        out_ids2, p_rev = _predict_dataset(
            model,
            ds_rev,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
        )
        if out_ids2 != out_ids:
            raise RuntimeError("Internal error: ID order mismatch between forward and reverse loaders")
        preds = (p_forw + p_rev) / 2.0
    else:
        preds = p_forw

    # Write output
    with out_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = ["id", "prediction"]
        if args.write_seq:
            header = ["id", "sequence", "prediction"]
        w.writerow(header)
        if args.write_seq:
            for _id, seq, pred in zip(out_ids, seqs, preds.tolist()):
                w.writerow([_id, seq, pred])
        else:
            for _id, pred in zip(out_ids, preds.tolist()):
                w.writerow([_id, pred])

    print("Saved:", out_path)
    print("Loaded checkpoint:", ckpt_path)
    if config_path is not None:
        print("Loaded config:", config_path)
    print("Checkpoint key prefix used:", meta.get("used_prefix"))
    if meta.get("missing_keys"):
        print("Warning: missing keys:")
        for k in meta["missing_keys"][:20]:
            print("  ", k)
        if len(meta["missing_keys"]) > 20:
            print(f"  ... ({len(meta['missing_keys'])} total)")
    if meta.get("unexpected_keys"):
        print("Warning: unexpected keys:")
        for k in meta["unexpected_keys"][:20]:
            print("  ", k)
        if len(meta["unexpected_keys"]) > 20:
            print(f"  ... ({len(meta['unexpected_keys'])} total)")


if __name__ == "__main__":
    main()
