"""Small metric helpers (no external deps)."""

from __future__ import annotations

import torch


def pearsonr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation (vectorized, returns a scalar tensor).

    Parameters
    ----------
    pred, target:
        1D tensors of same length.
    """
    if pred.ndim != 1 or target.ndim != 1:
        pred = pred.reshape(-1)
        target = target.reshape(-1)

    pred = pred.float()
    target = target.float()

    pred = pred - pred.mean()
    target = target - target.mean()

    num = (pred * target).sum()
    den = torch.sqrt((pred * pred).sum() + eps) * torch.sqrt((target * target).sum() + eps)
    return num / den
