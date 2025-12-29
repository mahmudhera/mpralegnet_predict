"""Datasets and dataloader helpers."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .encoding import encode_seq


class SequenceRegressionDataset(Dataset):
    """A minimal dataset for (sequence -> continuous target) regression.

    Sequences are encoded on-the-fly to a float tensor of shape [C, L].

    Parameters
    ----------
    seqs:
        List of DNA sequences.
    targets:
        List of floats (same length as seqs).
    seq_len:
        Optional: enforce a fixed length (pad/truncate with Ns).
    rc_augment:
        If True, randomly reverse-complement sequences during training.
    add_reverse_channel:
        If True, append the reverse-channel used by MPRA-LegNet.
        When rc_augment=True, the reverse-channel will be 1.0 for reversed
        samples and 0.0 otherwise.
    """

    def __init__(
        self,
        seqs: Sequence[str],
        targets: Sequence[float],
        *,
        seq_len: Optional[int] = None,
        rc_augment: bool = False,
        add_reverse_channel: bool = False,
    ):
        if len(seqs) != len(targets):
            raise ValueError(f"seqs and targets length mismatch: {len(seqs)} vs {len(targets)}")
        self.seqs = list(seqs)
        self.targets = torch.tensor(list(targets), dtype=torch.float32)
        self.seq_len = seq_len
        self.rc_augment = rc_augment
        self.add_reverse_channel = add_reverse_channel

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.seqs[idx]
        y = self.targets[idx]

        reverse = False
        if self.rc_augment:
            reverse = bool(random.getrandbits(1))

        x = encode_seq(
            seq,
            reverse=reverse,
            add_reverse_channel=self.add_reverse_channel,
            seq_len=self.seq_len,
        )
        return x, y


class IndexedSequenceRegressionDataset(Dataset):
    """Regression dataset that uses an explicit index list.

    This avoids copying sequences/targets into separate lists for train/val/test.
    """

    def __init__(
        self,
        seqs: Sequence[str],
        targets: Sequence[float],
        indices: Sequence[int],
        *,
        seq_len: Optional[int] = None,
        rc_augment: bool = False,
        add_reverse_channel: bool = False,
    ):
        if len(seqs) != len(targets):
            raise ValueError(f"seqs and targets length mismatch: {len(seqs)} vs {len(targets)}")
        self.seqs = list(seqs)
        self.targets = torch.tensor(list(targets), dtype=torch.float32)
        self.indices = list(indices)
        self.seq_len = seq_len
        self.rc_augment = rc_augment
        self.add_reverse_channel = add_reverse_channel

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self.indices[i]
        seq = self.seqs[idx]
        y = self.targets[idx]

        reverse = False
        if self.rc_augment:
            reverse = bool(random.getrandbits(1))

        x = encode_seq(
            seq,
            reverse=reverse,
            add_reverse_channel=self.add_reverse_channel,
            seq_len=self.seq_len,
        )
        return x, y


class SequenceOnlyDataset(Dataset):
    """Dataset for inference."""

    def __init__(
        self,
        ids: Sequence[str],
        seqs: Sequence[str],
        *,
        seq_len: Optional[int] = None,
        reverse: bool = False,
        add_reverse_channel: bool = False,
    ):
        if len(ids) != len(seqs):
            raise ValueError("ids and seqs must have the same length")
        self.ids = list(ids)
        self.seqs = list(seqs)
        self.seq_len = seq_len
        self.reverse = reverse
        self.add_reverse_channel = add_reverse_channel

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        seq = self.seqs[idx]
        x = encode_seq(
            seq,
            reverse=self.reverse,
            add_reverse_channel=self.add_reverse_channel,
            seq_len=self.seq_len,
        )
        return self.ids[idx], x


def split_indices(
    n: int,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """Random train/val/test split.

    Fractions must sum to 1 (within a small tolerance).
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test fractions must sum to 1. Got: {total}")

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    # Ensure non-empty splits when possible.
    if n >= 3:
        if n_train == 0:
            n_train = 1
            n_test -= 1
        if n_val == 0:
            n_val = 1
            n_test -= 1
        if n_test <= 0:
            # Borrow from train split.
            n_test = 1
            n_train = max(1, n - n_val - n_test)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    return train_idx, val_idx, test_idx
