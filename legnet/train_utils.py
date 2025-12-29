"""Training utilities used by scripts/finetune.py.

Kept intentionally lightweight and pure-PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .metrics import pearsonr


@dataclass
class EpochStats:
    loss: float
    pearson: float
    n: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def run_epoch_train(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = True,
    grad_clip: Optional[float] = None,
) -> EpochStats:
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_loss = 0.0
    n = 0

    for x, y in loader:
        x, y = _move_to_device(x, y, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        bs = x.shape[0]
        total_loss += float(loss.detach().item()) * bs
        n += bs

    return EpochStats(loss=total_loss / max(1, n), pearson=float('nan'), n=n)


def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (preds, targets) as CPU tensors."""
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = _move_to_device(x, y, device)
            with torch.cuda.amp.autocast(enabled=amp):
                p = model(x)
            preds.append(p.detach().float().cpu())
            targets.append(y.detach().float().cpu())

    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def eval_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool = True,
) -> EpochStats:
    preds, targets = predict_loader(model, loader, device, amp=amp)
    loss = torch.nn.functional.mse_loss(preds, targets).item()
    p = pearsonr(preds, targets).item()
    return EpochStats(loss=float(loss), pearson=float(p), n=int(targets.numel()))
