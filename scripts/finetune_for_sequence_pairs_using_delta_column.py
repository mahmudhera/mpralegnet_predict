#!/usr/bin/env python
"""
Predict SNP differential MPRA activity (Δ) from paired sequences using a pretrained MPRA-LegNet,
when you ONLY have deltas as labels.

Implements three methods:

(A) frozen-embedding + ridge (closed-form, no sklearn):
    - freeze pretrained LegNet
    - extract per-allele embeddings (penultimate vector before head)
    - build delta features: d = h_alt - h_ref
    - fit ridge regression on train, pick alpha on val, eval on test

(B) siamese-delta-head:
    - shared LegNet encoder E(.)
    - predict Δ via small MLP head g(h_alt - h_ref)
    - supports freezing encoder (recommended for 5-10k pairs)

(C) soft-classification regression for Δ:
    - head outputs logits over Δ bins
    - train with soft labels (Gaussian around true Δ)
    - prediction = expectation over bin centers

Input data format:
- TSV/CSV with at least:
    ref sequence column
    alt sequence column
    delta column (Δ = alt - ref, in log2 units)

Example:
    python scripts/delta_train_only_deltas.py \
      --model_dir /path/to/pretrained_legnet_dir \
      --data my_pairs_with_deltas.tsv \
      --out_dir out_delta \
      --device cuda:0 \
      --methods ridge siamese softcls \
      --epochs 30 --batch_size 256

Notes:
- Splits are by PAIR (row), so there is no leakage between alleles.
- Train-time augmentations:
    * flip_pairs: swap(ref,alt) with sign flip of delta
    * rc_pair_augment: reverse-complement BOTH sequences (keeps delta the same)
- Eval-time rc_average averages forward+reverse predictions/embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


# -------------------------
# Data loading (ONLY deltas)
# -------------------------

def read_pairs_with_deltas_table(
    path: Path,
    *,
    sep: Optional[str],
    ref_seq_col: str,
    alt_seq_col: str,
    delta_col: str,
    delta_multiplier: float,
) -> Tuple[List[str], List[str], List[float]]:
    if sep is None:
        sep = "," if path.suffix.lower() == ".csv" else "\t"

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=sep)
        required = [ref_seq_col, alt_seq_col, delta_col]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns {missing}. Found: {reader.fieldnames}")

        ref_seqs: List[str] = []
        alt_seqs: List[str] = []
        deltas: List[float] = []

        for row in reader:
            rs = row[ref_seq_col].strip().upper()
            asq = row[alt_seq_col].strip().upper()
            d = float(row[delta_col]) * float(delta_multiplier)
            ref_seqs.append(rs)
            alt_seqs.append(asq)
            deltas.append(d)

    if len(ref_seqs) != len(alt_seqs) or len(ref_seqs) != len(deltas):
        raise SystemExit("Row counts mismatch (unexpected).")
    return ref_seqs, alt_seqs, deltas


# -------------------------
# Pair dataset (ONLY deltas)
# -------------------------

class PairDeltaOnlyDataset(Dataset):
    """
    Returns: (x_ref, x_alt, delta)

    Augmentations (train):
      - flip_pairs: swap (ref,alt) with probability 0.5 and negate delta
      - rc_pair_augment: reverse-complement BOTH sequences with probability 0.5 (delta unchanged)
    """
    def __init__(
        self,
        ref_seqs: Sequence[str],
        alt_seqs: Sequence[str],
        deltas: Sequence[float],
        indices: Sequence[int],
        *,
        seq_len: int,
        add_reverse_channel: bool,
        flip_pairs: bool = False,
        rc_pair_augment: bool = False,
        deterministic: bool = False,
    ) -> None:
        self.ref_seqs = list(ref_seqs)
        self.alt_seqs = list(alt_seqs)
        self.deltas = list(deltas)
        self.indices = list(indices)

        self.seq_len = int(seq_len)
        self.add_reverse_channel = bool(add_reverse_channel)

        self.flip_pairs = bool(flip_pairs)
        self.rc_pair_augment = bool(rc_pair_augment)
        self.deterministic = bool(deterministic)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rs = self.ref_seqs[idx]
        asq = self.alt_seqs[idx]
        delta = float(self.deltas[idx])

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
    Used for rc_average at eval-time.
    """
    def __init__(self, base: PairDeltaOnlyDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        idx = self.base.indices[i]
        rs = self.base.ref_seqs[idx]
        asq = self.base.alt_seqs[idx]
        delta = float(self.base.deltas[idx])

        x_ref = encode_seq(rs, reverse=True, add_reverse_channel=self.base.add_reverse_channel, seq_len=self.base.seq_len)
        x_alt = encode_seq(asq, reverse=True, add_reverse_channel=self.base.add_reverse_channel, seq_len=self.base.seq_len)
        y = torch.tensor(delta, dtype=torch.float32)
        return x_ref, x_alt, y


# -------------------------
# Encoder: reuse LegNet as embedding extractor
# -------------------------

def legnet_embedding_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Returns penultimate embedding used by LegNet head.
    Assumes model has (stem, main, mapper) as in common LegNet checkpoints.

    If your model layout differs, adjust this function accordingly.
    """
    if hasattr(model, "stem") and hasattr(model, "main") and hasattr(model, "mapper"):
        x = model.stem(x)
        x = model.main(x)
        x = model.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

    # Fallback: hook input to first Linear in model.head
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
# Method A: Ridge (closed-form)
# -------------------------

@torch.no_grad()
def collect_delta_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X: (N, D) delta-embeddings
      y: (N,) true deltas
    If rc_average=True, embeddings are averaged forward+reverse per allele before differencing.
    """
    encoder.eval()
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    rev_loader = None
    if rc_average:
        if not isinstance(loader.dataset, PairDeltaOnlyDataset):
            raise RuntimeError("rc_average expects loader.dataset to be PairDeltaOnlyDataset")
        rev_ds = ReversePairWrapper(loader.dataset)
        rev_loader = DataLoader(
            rev_ds,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    it_rev = iter(rev_loader) if rev_loader is not None else None

    for x_ref, x_alt, y in loader:
        x_ref = x_ref.to(device, non_blocking=True)
        x_alt = x_alt.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            h_ref = encoder(x_ref)
            h_alt = encoder(x_alt)

        if it_rev is not None:
            x_ref_r, x_alt_r, _ = next(it_rev)
            x_ref_r = x_ref_r.to(device, non_blocking=True)
            x_alt_r = x_alt_r.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                h_ref_r = encoder(x_ref_r)
                h_alt_r = encoder(x_alt_r)
            h_ref = (h_ref + h_ref_r) / 2.0
            h_alt = (h_alt + h_alt_r) / 2.0

        d = (h_alt - h_ref).detach().cpu()
        X_list.append(d)
        y_list.append(y.detach().cpu())

    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0).view(-1)
    return X, y


def ridge_fit_closed_form(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
) -> Tuple[torch.Tensor, float]:
    """
    Ridge regression with intercept (not penalized), closed-form:
      w = (Xc^T Xc + alpha I)^-1 Xc^T yc
      b = y_mean - X_mean @ w
    """
    X = X.double()
    y = y.double()

    X_mean = X.mean(dim=0)
    y_mean = y.mean()

    Xc = X - X_mean
    yc = y - y_mean

    D = X.shape[1]
    XtX = Xc.t().mm(Xc) + alpha * torch.eye(D, dtype=X.dtype)
    Xty = Xc.t().mv(yc)

    w = torch.linalg.solve(XtX, Xty)
    b = float((y_mean - X_mean.dot(w)).item())
    return w.float(), b


@torch.no_grad()
def ridge_predict(X: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
    return X.float().matmul(w.float()) + float(b)


def run_ridge(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
    alphas: List[float],
    select_metric: str,
    out_dir: Path,
) -> Dict:
    Xtr, ytr = collect_delta_embeddings(encoder, train_loader, device, amp=amp, rc_average=rc_average)
    Xva, yva = collect_delta_embeddings(encoder, val_loader, device, amp=amp, rc_average=rc_average)
    Xte, yte = collect_delta_embeddings(encoder, test_loader, device, amp=amp, rc_average=rc_average)

    best = None
    best_alpha = None
    best_score = None

    for a in alphas:
        w, b = ridge_fit_closed_form(Xtr, ytr, alpha=float(a))
        pva = ridge_predict(Xva, w, b)
        va_mse = mse_torch(pva, yva)
        va_p = pearsonr_torch(pva, yva)

        score = va_p if select_metric == "pearson" else -va_mse
        if best_score is None or score > best_score:
            best_score = score
            best_alpha = float(a)
            best = (w, b, va_mse, va_p)

    assert best is not None and best_alpha is not None
    w, b, va_mse, va_p = best
    pte = ridge_predict(Xte, w, b)
    te_mse = mse_torch(pte, yte)
    te_p = pearsonr_torch(pte, yte)

    ridge_path = out_dir / "ridge_delta.pt"
    torch.save({"w": w, "b": b, "best_alpha": best_alpha}, ridge_path)

    metrics = {
        "method": "ridge",
        "best_alpha": best_alpha,
        "val": {"mse": va_mse, "pearson": va_p, "n": int(len(yva))},
        "test": {"mse": te_mse, "pearson": te_p, "n": int(len(yte))},
        "rc_average": bool(rc_average),
        "select_metric": select_metric,
        "alphas": alphas,
        "saved_model": str(ridge_path),
    }
    with (out_dir / "metrics_ridge.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


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

    if not isinstance(loader.dataset, PairDeltaOnlyDataset):
        raise RuntimeError("rc_average expects loader.dataset to be PairDeltaOnlyDataset")
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
    
    # for debugging, print first 20 p and y
    print('DEBUG')
    print("p:", p[:20])
    print("y:", y[:20])

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
    freeze_encoder: bool,
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
        if freeze_encoder:
            # keep frozen encoder in eval mode (BatchNorm stats won't drift)
            model.encoder.eval()

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
# Method C: Soft classification regression for Δ
# -------------------------

class SiameseDeltaSoftCls(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int, hidden_dim: int, dropout: float, num_bins: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, x_ref: torch.Tensor, x_alt: torch.Tensor) -> torch.Tensor:
        h_ref = self.encoder(x_ref)
        h_alt = self.encoder(x_alt)
        d = h_alt - h_ref
        return self.head(d)  # logits


def make_bin_centers(delta_min: float, delta_max: float, num_bins: int) -> torch.Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    return torch.linspace(delta_min, delta_max, steps=num_bins, dtype=torch.float32)


def soft_targets_gaussian(y: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    y: (B,)
    centers: (K,)
    returns: (B,K) soft labels
    """
    y = y.view(-1, 1)
    c = centers.view(1, -1)
    z = (y - c) / max(1e-8, float(sigma))
    p = torch.exp(-0.5 * (z * z))
    p = p / (p.sum(dim=1, keepdim=True) + 1e-12)
    return p


@torch.no_grad()
def softcls_predict_expectation(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * centers.view(1, -1)).sum(dim=-1)


@torch.no_grad()
def eval_softcls_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
    centers: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    centers_dev = centers.to(device)

    def _predict(loader0: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        preds: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for x_ref, x_alt, y in loader0:
            x_ref = x_ref.to(device, non_blocking=True)
            x_alt = x_alt.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x_ref, x_alt)
                p = softcls_predict_expectation(logits, centers_dev)
            preds.append(p.detach().cpu())
            ys.append(y.detach().cpu())
        return torch.cat(preds, dim=0).view(-1), torch.cat(ys, dim=0).view(-1)

    if not rc_average:
        p, y = _predict(loader)
        return {"mse": mse_torch(p, y), "pearson": pearsonr_torch(p, y), "n": int(len(y))}

    if not isinstance(loader.dataset, PairDeltaOnlyDataset):
        raise RuntimeError("rc_average expects loader.dataset to be PairDeltaOnlyDataset")
    rev_ds = ReversePairWrapper(loader.dataset)
    rev_loader = DataLoader(
        rev_ds,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    p_fwd, y = _predict(loader)
    p_rev, _ = _predict(rev_loader)
    p = (p_fwd + p_rev) / 2.0
    return {"mse": mse_torch(p, y), "pearson": pearsonr_torch(p, y), "n": int(len(y))}


def run_softcls(
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
    num_bins: int,
    delta_min: float,
    delta_max: float,
    sigma: float,
    select_metric: str,
    freeze_encoder: bool,
    out_dir: Path,
) -> Dict:
    # infer embed dim
    encoder.eval()
    x_ref0, _, _ = next(iter(train_loader))
    x_ref0 = x_ref0.to(device)
    with torch.no_grad():
        h0 = encoder(x_ref0)
    embed_dim = int(h0.shape[-1])

    centers = make_bin_centers(delta_min, delta_max, num_bins)

    model = SiameseDeltaSoftCls(
        encoder=encoder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_bins=num_bins,
    ).to(device)

    opt = build_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_state = None
    best_score = None

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_encoder:
            model.encoder.eval()

        total = 0.0
        n = 0

        for x_ref, x_alt, y in train_loader:
            x_ref = x_ref.to(device, non_blocking=True)
            x_alt = x_alt.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # clip y into [delta_min, delta_max] for stable soft labels
            y_clip = y.clamp(min=delta_min, max=delta_max)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(x_ref, x_alt)  # (B,K)
                tgt = soft_targets_gaussian(y_clip, centers.to(device), sigma=sigma)  # (B,K)
                logp = F.log_softmax(logits, dim=-1)
                loss = -(tgt * logp).sum(dim=-1).mean()

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            total += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])

        train_loss = total / max(1, n)
        va = eval_softcls_model(model, val_loader, device, amp=amp, rc_average=rc_average, centers=centers)

        score = va["pearson"] if select_metric == "pearson" else -va["mse"]
        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[softcls] epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} "
            f"| val_mse={va['mse']:.6f} val_pearson={va['pearson']:.4f}"
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    te = eval_softcls_model(model, test_loader, device, amp=amp, rc_average=rc_average, centers=centers)

    out_path = out_dir / "softcls_delta.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "num_bins": num_bins,
            "delta_min": delta_min,
            "delta_max": delta_max,
            "sigma": sigma,
            "centers": centers,
            "rc_average": bool(rc_average),
            "select_metric": select_metric,
            "test": te,
        },
        out_path,
    )

    metrics = {
        "method": "softcls",
        "val_best_metric": float(best_score) if best_score is not None else None,
        "test": te,
        "rc_average": bool(rc_average),
        "select_metric": select_metric,
        "num_bins": num_bins,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "sigma": sigma,
        "saved_model": str(out_path),
    }
    with (out_dir / "metrics_softcls.json").open("w") as f:
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
    data.add_argument("--data", type=str, required=True, help="TSV/CSV with ref/alt sequences + delta label")
    data.add_argument("--ref_seq_col", type=str, default="reference sequence")
    data.add_argument("--alt_seq_col", type=str, default="alternate sequence")
    data.add_argument("--delta_col", type=str, default="delta", help="Column containing Δ (alt - ref)")
    data.add_argument("--delta_multiplier", type=float, default=1.0, help="Multiply delta labels by this (use -1 if needed)")
    data.add_argument("--sep", type=str, default=None, help="Delimiter (default: inferred from extension)")
    data.add_argument("--seq_len", type=int, default=200, help="Pad/truncate sequences to this length (default: 200)")
    data.add_argument("--num_chars_to_ignore", type=int, default=0, help="Ignore this many chars at start of sequences")

    split = parser.add_argument_group("split")
    split.add_argument("--train_frac", type=float, default=0.8)
    split.add_argument("--val_frac", type=float, default=0.1)
    split.add_argument("--test_frac", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=777)

    run = parser.add_argument_group("run")
    run.add_argument("--out_dir", type=str, required=True)
    run.add_argument("--methods", nargs="+", default=["ridge", "siamese", "softcls"], choices=["ridge", "siamese", "softcls"])
    run.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    run.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--select_metric", type=str, default="pearson", choices=["pearson", "mse"])
    run.add_argument("--rc_pair_augment", action=argparse.BooleanOptionalAction, default=True,
                     help="Train-time pairwise reverse-complement augmentation (same orientation for ref+alt)")
    run.add_argument("--rc_average", action=argparse.BooleanOptionalAction, default=True,
                     help="Eval-time average forward+reverse predictions/embeddings")
    run.add_argument("--flip_pairs", action=argparse.BooleanOptionalAction, default=True,
                     help="Train-time (siamese/softcls) random swap(ref,alt) with label sign flip")

    ridge = parser.add_argument_group("ridge")
    ridge.add_argument("--ridge_alphas", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])

    train = parser.add_argument_group("train (siamese/softcls)")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--num_workers", type=int, default=4)
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

    soft = parser.add_argument_group("soft classification Δ")
    soft.add_argument("--num_bins", type=int, default=161, help="Number of Δ bins")
    soft.add_argument("--delta_min", type=float, default=-4.0, help="Min Δ for bins")
    soft.add_argument("--delta_max", type=float, default=4.0, help="Max Δ for bins")
    soft.add_argument("--sigma", type=float, default=0.15, help="Soft label Gaussian sigma in Δ units")

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

    # Load data (ONLY deltas)
    data_path = Path(args.data)
    ref_seqs, alt_seqs, deltas = read_pairs_with_deltas_table(
        data_path,
        sep=args.sep,
        ref_seq_col=args.ref_seq_col,
        alt_seq_col=args.alt_seq_col,
        delta_col=args.delta_col,
        delta_multiplier=float(args.delta_multiplier),
    )
    n_pairs = len(ref_seqs)
    seq_len = int(args.seq_len)

    # Debug: show first 5 sequences and deltas
    print('DEBUG')
    for i in range(min(5, n_pairs)):
        print(f"Pair {i}:")
        print("  ref_seq:", ref_seqs[i])
        print("  alt_seq:", alt_seqs[i])
        print("  delta:", deltas[i])

    # Pair-level split
    train_idx, val_idx, test_idx = split_indices(
        n_pairs,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    device = torch.device(args.device)

    # Datasets
    # - Ridge: deterministic (no random aug) because we want stable features
    train_ds_ridge = PairDeltaOnlyDataset(
        ref_seqs, alt_seqs, deltas, train_idx,
        seq_len=seq_len, add_reverse_channel=add_reverse_channel,
        flip_pairs=False, rc_pair_augment=False, deterministic=True
    )
    val_ds = PairDeltaOnlyDataset(
        ref_seqs, alt_seqs, deltas, val_idx,
        seq_len=seq_len, add_reverse_channel=add_reverse_channel,
        flip_pairs=False, rc_pair_augment=False, deterministic=True
    )
    test_ds = PairDeltaOnlyDataset(
        ref_seqs, alt_seqs, deltas, test_idx,
        seq_len=seq_len, add_reverse_channel=add_reverse_channel,
        flip_pairs=False, rc_pair_augment=False, deterministic=True
    )

    # - Train-time augmented dataset for siamese/softcls
    train_ds_aug = PairDeltaOnlyDataset(
        ref_seqs, alt_seqs, deltas, train_idx,
        seq_len=seq_len, add_reverse_channel=add_reverse_channel,
        flip_pairs=bool(args.flip_pairs),
        rc_pair_augment=bool(args.rc_pair_augment),
        deterministic=False,
    )

    def make_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker,
            drop_last=False,
        )

    train_loader_ridge = make_loader(train_ds_ridge, shuffle=False)
    train_loader = make_loader(train_ds_aug, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    # Load pretrained LegNet
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
    print("delta_col:", args.delta_col, "| delta_multiplier:", float(args.delta_multiplier))

    # Build encoder wrapper
    encoder = LegNetEncoder(model).to(device)

    # Freeze encoder if requested
    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        encoder.eval()
        print("Encoder frozen: True (encoder.eval() to freeze BatchNorm stats)")
    else:
        encoder.train()
        print("Encoder frozen: False")

    # Save run config
    with (out_dir / "run_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    all_metrics: Dict[str, Dict] = {}

    # (A) Ridge
    if "ridge" in args.methods:
        ridge_dir = out_dir / "ridge"
        ensure_dir(ridge_dir)
        m = run_ridge(
            encoder=encoder,
            train_loader=train_loader_ridge,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            amp=bool(args.amp),
            rc_average=bool(args.rc_average),
            alphas=list(args.ridge_alphas),
            select_metric=args.select_metric,
            out_dir=ridge_dir,
        )
        all_metrics["ridge"] = m
        print("[ridge] test:", m["test"])

    # (B) Siamese Δ head
    if "siamese" in args.methods:
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
            freeze_encoder=bool(args.freeze_encoder),
            out_dir=siamese_dir,
        )
        all_metrics["siamese"] = m
        print("[siamese] test:", m["test"])

    # (C) Soft classification Δ
    if "softcls" in args.methods:
        soft_dir = out_dir / "softcls"
        ensure_dir(soft_dir)
        m = run_softcls(
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
            num_bins=int(args.num_bins),
            delta_min=float(args.delta_min),
            delta_max=float(args.delta_max),
            sigma=float(args.sigma),
            select_metric=args.select_metric,
            freeze_encoder=bool(args.freeze_encoder),
            out_dir=soft_dir,
        )
        all_metrics["softcls"] = m
        print("[softcls] test:", m["test"])

    with (out_dir / "metrics_all.json").open("w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nDone. Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
