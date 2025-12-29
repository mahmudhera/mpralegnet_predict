"""Sequence encoding utilities.

The upstream MPRA-LegNet implementation encodes DNA as A/C/G/T one-hot and uses
0.25 for each base at ambiguous 'N' positions. It also optionally adds a
"reverse" channel (all zeros for forward sequences and all ones for reverse-
complement sequences) when training with reverse-complement augmentation.

This file keeps those behaviors so we can load checkpoints trained in that
style.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


_BASE_TO_ID: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


def n2id(base: str) -> int:
    """Map a nucleotide to an integer id in {0..4}.

    Any non-ACGT character is treated as N.
    """
    if not base:
        return 4
    b = base.upper()
    return _BASE_TO_ID.get(b, 4)


def reverse_complement(seq: str) -> str:
    """Reverse-complement of a DNA string (A/C/G/T/N).

    Non-ACGT characters are mapped to 'N'.
    """
    mapping = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "N": "N",
    }
    seq = seq.upper()
    return "".join(mapping.get(b, "N") for b in reversed(seq))


def encode_seq(
    seq: str,
    *,
    reverse: bool = False,
    add_reverse_channel: bool = False,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Encode one sequence as float tensor [C, L].

    - A/C/G/T -> one-hot
    - N/other -> 0.25 in each A/C/G/T channel
    - Optionally appends a "reverse" channel as in MPRA-LegNet.

    Parameters
    ----------
    seq:
        Input DNA sequence.
    reverse:
        If True, use reverse-complement of the input.
    add_reverse_channel:
        If True, append a 5th channel of all 0 (forward) or all 1 (reverse).
    seq_len:
        If provided, sequences are either truncated (if longer) or padded with
        'N' (if shorter) to this exact length.

    Returns
    -------
    torch.Tensor
        Shape [4, L] if add_reverse_channel=False, else [5, L].
    """
    if reverse:
        seq = reverse_complement(seq)

    seq = seq.strip().upper()

    if seq_len is not None:
        if len(seq) > seq_len:
            seq = seq[:seq_len]
        elif len(seq) < seq_len:
            seq = seq + ("N" * (seq_len - len(seq)))

    ids = [n2id(ch) for ch in seq]
    code = torch.tensor(ids, dtype=torch.long)

    one_hot = F.one_hot(code, num_classes=5).float()  # [L, 5]

    # For N positions (5th class), assign 0.25 to all classes, then we'll drop the 5th.
    one_hot[one_hot[:, 4] == 1] = 0.25

    x = one_hot[:, :4].transpose(0, 1).contiguous()  # [4, L]

    if add_reverse_channel:
        rev_val = 1.0 if reverse else 0.0
        rev = torch.full((1, x.shape[1]), rev_val, dtype=torch.float32)
        x = torch.cat([x, rev], dim=0)

    return x


def encode_many(
    seqs: Sequence[str],
    *,
    reverse: bool = False,
    add_reverse_channel: bool = False,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Encode a batch of sequences into a tensor [B, C, L]."""
    xs = [encode_seq(s, reverse=reverse, add_reverse_channel=add_reverse_channel, seq_len=seq_len) for s in seqs]
    return torch.stack(xs, dim=0)
