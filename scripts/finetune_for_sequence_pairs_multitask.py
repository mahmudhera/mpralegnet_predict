from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure repo root on sys.path so `import legnet` works without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from legnet import LegNetConfig, load_model, split_indices
from legnet.encoding import encode_seq
from legnet.train_utils import set_seed


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


def make_criterion(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError("loss must be 'mse' or 'huber'")


# -------------------------
# Pair dataset (multitask)
# -------------------------

class PairMultiTaskDataset(Dataset):
    """
    Returns: (x_ref, x_alt, y_ref, y_alt, y_delta)
      - y_delta = y_alt - y_ref (optionally normalized)
    Augmentations (train):
      - flip_pairs: swap (ref,alt) with probability 0.5.
          * swaps sequences and swaps y_ref/y_alt
          * negates delta
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
        self.deterministic = bool(deterministic)

        self.normalize_delta = bool(normalize_delta)

        # Compute delta stats only if we need normalization (or if user provided them).
        if self.normalize_delta:
            if normalize_mean is not None and normalize_std is not None:
                self.delta_mean = float(normalize_mean)
                self.delta_std = float(normalize_std)
            else:
                deltas = [self.alt_y[i] - self.ref_y[i] for i in self.indices]
                t = torch.tensor(deltas, dtype=torch.float32)
                self.delta_mean = float(t.mean().item())
                self.delta_std = float(t.std().item())
            if self.delta_std < 1e-6:
                raise RuntimeError("Delta standard deviation is too small; cannot normalize.")
        else:
            # Still store something reasonable for checkpoint bookkeeping.
            self.delta_mean = 0.0
            self.delta_std = 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def get_delta_mean_std(self) -> Tuple[float, float]:
        return float(self.delta_mean), float(self.delta_std)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rs = self.ref_seqs[idx]
        asq = self.alt_seqs[idx]

        y_ref = float(self.ref_y[idx])
        y_alt = float(self.alt_y[idx])
        y_delta = y_alt - y_ref

        do_flip = False
        do_rc = False
        if not self.deterministic:
            if self.flip_pairs and random.random() < 0.5:
                do_flip = True
            if self.rc_pair_augment and random.random() < 0.5:
                do_rc = True

        if do_flip:
            rs, asq = asq, rs
            y_ref, y_alt = y_alt, y_ref
            y_delta = -y_delta

        if self.normalize_delta:
            y_delta = (y_delta - self.delta_mean) / self.delta_std

        x_ref = encode_seq(
            rs,
            reverse=do_rc,
            add_reverse_channel=self.add_reverse_channel,
            seq_len=self.seq_len,
        )
        x_alt = encode_seq(
            asq,
            reverse=do_rc,
            add_reverse_channel=self.add_reverse_channel,
            seq_len=self.seq_len,
        )

        t_ref = torch.tensor(y_ref, dtype=torch.float32)
        t_alt = torch.tensor(y_alt, dtype=torch.float32)
        t_delta = torch.tensor(y_delta, dtype=torch.float32)
        return x_ref, x_alt, t_ref, t_alt, t_delta


class ReversePairWrapperMulti(Dataset):
    """
    Deterministic wrapper that returns reverse-complement encodings for BOTH sequences.
    Use for rc_average at eval-time.
    """

    def __init__(self, base: PairMultiTaskDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        idx = self.base.indices[i]
        rs = self.base.ref_seqs[idx]
        asq = self.base.alt_seqs[idx]

        y_ref = float(self.base.ref_y[idx])
        y_alt = float(self.base.alt_y[idx])
        y_delta = y_alt - y_ref
        if self.base.normalize_delta:
            y_delta = (y_delta - self.base.delta_mean) / self.base.delta_std

        x_ref = encode_seq(
            rs,
            reverse=True,
            add_reverse_channel=self.base.add_reverse_channel,
            seq_len=self.base.seq_len,
        )
        x_alt = encode_seq(
            asq,
            reverse=True,
            add_reverse_channel=self.base.add_reverse_channel,
            seq_len=self.base.seq_len,
        )

        return (
            x_ref,
            x_alt,
            torch.tensor(y_ref, dtype=torch.float32),
            torch.tensor(y_alt, dtype=torch.float32),
            torch.tensor(y_delta, dtype=torch.float32),
        )


# -------------------------
# Encoder: reuse LegNet as embedding extractor
# -------------------------

def legnet_embedding_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Returns the penultimate embedding used by the LegNet head:
      stem -> main -> mapper -> adaptive_avg_pool -> squeeze

    If the checkpoint uses a different module layout, we fall back to hooking
    the first Linear in model.head (if present).
    """
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


def legnet_head_forward(legnet: nn.Module, emb: torch.Tensor) -> torch.Tensor:
    """
    Uses the original LegNet head on a precomputed embedding, preserving the architecture.
    """
    if not hasattr(legnet, "head"):
        raise RuntimeError("LegNet model does not have a .head; cannot run original head from embeddings.")
    y = legnet.head(emb)
    # Expect [B,1] or [B]; make it [B]
    if y.ndim > 1:
        y = y.squeeze(-1)
    return y


def freeze_legnet_trunk_params(legnet: nn.Module) -> None:
    """
    Freeze ONLY the LegNet trunk (stem/main/mapper), keeping the original head trainable.
    This matches the requirement: keep original mpralegnet head for y_ref and y_alt.
    """
    found_any = False
    for attr in ("stem", "main", "mapper"):
        if hasattr(legnet, attr):
            found_any = True
            mod = getattr(legnet, attr)
            for p in mod.parameters():
                p.requires_grad_(False)

    if not found_any:
        # Conservative fallback: freeze everything except "head.*"
        for name, p in legnet.named_parameters():
            if name.startswith("head."):
                continue
            p.requires_grad_(False)


def set_legnet_trunk_eval(legnet: nn.Module) -> None:
    """
    Put trunk modules in eval mode (useful when trunk is frozen, to freeze BN stats).
    """
    for attr in ("stem", "main", "mapper"):
        if hasattr(legnet, attr):
            getattr(legnet, attr).eval()


# -------------------------
# Multitask model: y_ref, y_alt, delta
# -------------------------

class MultiTaskPairModel(nn.Module):
    """
    - y_ref: original LegNet head(embedding(ref))
    - y_alt: original LegNet head(embedding(alt))
    - delta: MLP head on (emb_alt - emb_ref)
    """

    def __init__(
        self,
        legnet: nn.Module,
        *,
        embed_dim: int,
        delta_hidden_dim: int,
        delta_dropout: float,
    ) -> None:
        super().__init__()
        self.legnet = legnet
        self.encoder = LegNetEncoder(legnet)

        self.delta_head = nn.Sequential(
            nn.Linear(embed_dim, delta_hidden_dim),
            nn.SiLU(),
            nn.Dropout(delta_dropout),
            nn.Linear(delta_hidden_dim, delta_hidden_dim),
            nn.SiLU(),
            nn.Dropout(delta_dropout),
            nn.Linear(delta_hidden_dim, 1),
        )

        print(
            f"Created MultiTaskPairModel: embed_dim={embed_dim}, "
            f"delta_hidden_dim={delta_hidden_dim}, delta_dropout={delta_dropout}"
        )

    def forward(self, x_ref: torch.Tensor, x_alt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_ref = self.encoder(x_ref)
        h_alt = self.encoder(x_alt)

        y_ref = legnet_head_forward(self.legnet, h_ref)
        y_alt = legnet_head_forward(self.legnet, h_alt)

        d = h_alt - h_ref
        y_delta = self.delta_head(d).squeeze(-1)
        return y_ref, y_alt, y_delta


# -------------------------
# Eval helpers
# -------------------------

@torch.no_grad()
def predict_multitask_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    p_ref_list: List[torch.Tensor] = []
    p_alt_list: List[torch.Tensor] = []
    p_del_list: List[torch.Tensor] = []
    y_ref_list: List[torch.Tensor] = []
    y_alt_list: List[torch.Tensor] = []
    y_del_list: List[torch.Tensor] = []

    for x_ref, x_alt, y_ref, y_alt, y_delta in loader:
        x_ref = x_ref.to(device, non_blocking=True)
        x_alt = x_alt.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            p_ref, p_alt, p_delta = model(x_ref, x_alt)

        p_ref_list.append(p_ref.detach().cpu().view(-1))
        p_alt_list.append(p_alt.detach().cpu().view(-1))
        p_del_list.append(p_delta.detach().cpu().view(-1))

        y_ref_list.append(y_ref.detach().cpu().view(-1))
        y_alt_list.append(y_alt.detach().cpu().view(-1))
        y_del_list.append(y_delta.detach().cpu().view(-1))

    p_ref_t = torch.cat(p_ref_list, dim=0)
    p_alt_t = torch.cat(p_alt_list, dim=0)
    p_del_t = torch.cat(p_del_list, dim=0)
    y_ref_t = torch.cat(y_ref_list, dim=0)
    y_alt_t = torch.cat(y_alt_list, dim=0)
    y_del_t = torch.cat(y_del_list, dim=0)
    return p_ref_t, p_alt_t, p_del_t, y_ref_t, y_alt_t, y_del_t


@torch.no_grad()
def eval_multitask_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    rc_average: bool,
) -> Dict[str, float]:
    """
    Returns a dict with ref/alt/delta metrics.
    If delta was normalized in the dataset, we also report delta_mse_raw
    (MSE after unnormalizing with train-set mean/std).
    """
    ds = loader.dataset
    if rc_average:
        if not isinstance(ds, PairMultiTaskDataset):
            raise RuntimeError("rc_average expects loader.dataset to be PairMultiTaskDataset")
        rev_ds = ReversePairWrapperMulti(ds)
        rev_loader = DataLoader(
            rev_ds,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        p_ref_f, p_alt_f, p_del_f, y_ref, y_alt, y_del = predict_multitask_loader(model, loader, device, amp=amp)
        p_ref_r, p_alt_r, p_del_r, _, _, _ = predict_multitask_loader(model, rev_loader, device, amp=amp)

        p_ref = (p_ref_f + p_ref_r) / 2.0
        p_alt = (p_alt_f + p_alt_r) / 2.0
        p_del = (p_del_f + p_del_r) / 2.0
    else:
        p_ref, p_alt, p_del, y_ref, y_alt, y_del = predict_multitask_loader(model, loader, device, amp=amp)

    out: Dict[str, float] = {
        "n": float(len(y_ref)),
        # Ref
        "ref_mse": mse_torch(p_ref, y_ref),
        "ref_pearson": pearsonr_torch(p_ref, y_ref),
        # Alt
        "alt_mse": mse_torch(p_alt, y_alt),
        "alt_pearson": pearsonr_torch(p_alt, y_alt),
        # Delta (on the scale used for training)
        "delta_mse": mse_torch(p_del, y_del),
        "delta_pearson": pearsonr_torch(p_del, y_del),
    }

    # If delta normalization was used, report MSE in original delta units too.
    if isinstance(ds, PairMultiTaskDataset) and ds.normalize_delta:
        mean, std = ds.get_delta_mean_std()
        p_del_raw = p_del * std + mean
        y_del_raw = y_del * std + mean
        out["delta_mse_raw"] = mse_torch(p_del_raw, y_del_raw)
    return out


# -------------------------
# Training loop
# -------------------------

def run_multitask(
    legnet: nn.Module,
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
    delta_hidden_dim: int,
    delta_dropout: float,
    loss_ref: str,
    loss_alt: str,
    loss_delta: str,
    w_ref: float,
    w_alt: float,
    w_delta: float,
    select_metric: str,
    freeze_encoder: bool,
    out_dir: Path,
) -> Dict:
    # Infer embed dim
    encoder = LegNetEncoder(legnet).to(device)
    encoder.eval()
    x_ref0, *_ = next(iter(train_loader))
    x_ref0 = x_ref0.to(device)
    with torch.no_grad():
        h0 = encoder(x_ref0)
    embed_dim = int(h0.shape[-1])

    model = MultiTaskPairModel(
        legnet=legnet,
        embed_dim=embed_dim,
        delta_hidden_dim=delta_hidden_dim,
        delta_dropout=delta_dropout,
    ).to(device)

    crit_ref = make_criterion(loss_ref)
    crit_alt = make_criterion(loss_alt)
    crit_del = make_criterion(loss_delta)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(optimizer_name, params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_state = None
    best_score: Optional[float] = None

    for epoch in range(1, epochs + 1):
        model.train()
        # If trunk is frozen, keep it in eval mode to freeze BN stats.
        if freeze_encoder:
            set_legnet_trunk_eval(model.legnet)

        total_loss = 0.0
        total_ref = 0.0
        total_alt = 0.0
        total_del = 0.0
        n = 0

        for x_ref, x_alt, y_ref, y_alt, y_del in train_loader:
            x_ref = x_ref.to(device, non_blocking=True)
            x_alt = x_alt.to(device, non_blocking=True)
            y_ref = y_ref.to(device, non_blocking=True)
            y_alt = y_alt.to(device, non_blocking=True)
            y_del = y_del.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                p_ref, p_alt, p_del = model(x_ref, x_alt)
                l_ref = crit_ref(p_ref, y_ref)
                l_alt = crit_alt(p_alt, y_alt)
                l_del = crit_del(p_del, y_del)
                loss = (w_ref * l_ref) + (w_alt * l_alt) + (w_delta * l_del)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = int(y_ref.shape[0])
            n += bs
            total_loss += float(loss.item()) * bs
            total_ref += float(l_ref.item()) * bs
            total_alt += float(l_alt.item()) * bs
            total_del += float(l_del.item()) * bs

        tr = {
            "loss": total_loss / max(1, n),
            "ref_loss": total_ref / max(1, n),
            "alt_loss": total_alt / max(1, n),
            "delta_loss": total_del / max(1, n),
        }

        va = eval_multitask_model(model, val_loader, device, amp=amp, rc_average=rc_average)

        # Keep backwards-compatible selection behavior:
        # select_metric='pearson' -> maximize delta_pearson
        # select_metric='mse' -> minimize delta_mse (i.e. maximize -delta_mse)
        if select_metric == "pearson":
            score = float(va["delta_pearson"])
        else:
            score = -float(va["delta_mse"])

        if best_score is None or score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        msg = (
            f"[multitask] epoch {epoch:03d}/{epochs} | "
            f"train_loss={tr['loss']:.6f} (ref={tr['ref_loss']:.6f} alt={tr['alt_loss']:.6f} del={tr['delta_loss']:.6f}) | "
            f"val: ref_mse={va['ref_mse']:.6f} ref_r={va['ref_pearson']:.4f} | "
            f"alt_mse={va['alt_mse']:.6f} alt_r={va['alt_pearson']:.4f} | "
            f"delta_mse={va['delta_mse']:.6f} delta_r={va['delta_pearson']:.4f}"
        )
        if "delta_mse_raw" in va:
            msg += f" (delta_mse_raw={va['delta_mse_raw']:.6f})"
        print(msg)

    assert best_state is not None
    model.load_state_dict(best_state)

    te = eval_multitask_model(model, test_loader, device, amp=amp, rc_average=rc_average)

    out_path = out_dir / "multitask_legnet_plus_delta_head.pt"
    # Store delta normalization stats (if used) so we can unnormalize later.
    delta_norm = {}
    ds = train_loader.dataset
    if isinstance(ds, PairMultiTaskDataset) and ds.normalize_delta:
        mean, std = ds.get_delta_mean_std()
        delta_norm = {"normalize_delta": True, "delta_mean": mean, "delta_std": std}
    else:
        delta_norm = {"normalize_delta": False}

    torch.save(
        {
            "state_dict": model.state_dict(),
            "embed_dim": embed_dim,
            "delta_hidden_dim": delta_hidden_dim,
            "delta_dropout": delta_dropout,
            "rc_average": bool(rc_average),
            "select_metric": select_metric,
            "freeze_encoder": bool(freeze_encoder),
            "delta_norm": delta_norm,
            "test": te,
        },
        out_path,
    )

    metrics = {
        "method": "multitask",
        "val_best_metric": float(best_score) if best_score is not None else None,
        "select_metric": select_metric,
        "rc_average": bool(rc_average),
        "freeze_encoder": bool(freeze_encoder),
        "loss_weights": {"ref": float(w_ref), "alt": float(w_alt), "delta": float(w_delta)},
        "losses": {"ref": loss_ref, "alt": loss_alt, "delta": loss_delta},
        "delta_norm": delta_norm,
        "test": te,
        "saved_model": str(out_path),
    }
    with (out_dir / "metrics_multitask.json").open("w") as f:
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

    # Delta normalization only (as in the original delta-only script)
    data.add_argument(
        "--normalize_delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize delta labels to zero mean/unit variance (computed on train split).",
    )

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
    run.add_argument(
        "--rc_pair_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train-time pairwise reverse-complement augmentation (same orientation for ref+alt).",
    )
    run.add_argument(
        "--rc_average",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Eval-time average forward+reverse predictions for ALL heads (ref/alt/delta).",
    )
    run.add_argument(
        "--flip_pairs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train-time random swap(ref,alt): swaps labels and negates delta.",
    )

    train = parser.add_argument_group("training hyperparameters")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch_size", type=int, default=256)
    train.add_argument("--num_workers", type=int, default=1)
    train.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adam", "adamw"])
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=1e-3)
    train.add_argument("--momentum", type=float, default=0.9)
    train.add_argument("--grad_clip", type=float, default=1.0)

    # Delta head hyperparameters (same style as the siamese delta head)
    train.add_argument("--delta_hidden_dim", type=int, default=256)
    train.add_argument("--delta_dropout", type=float, default=0.1)

    # Losses (can be the same or different per task)
    train.add_argument("--loss_ref", type=str, default="huber", choices=["mse", "huber"])
    train.add_argument("--loss_alt", type=str, default="huber", choices=["mse", "huber"])
    train.add_argument("--loss_delta", type=str, default="huber", choices=["mse", "huber"])

    # Loss weights for multitask training
    train.add_argument("--w_ref", type=float, default=1.0)
    train.add_argument("--w_alt", type=float, default=1.0)
    train.add_argument("--w_delta", type=float, default=1.0)

    enc = parser.add_argument_group("encoder freezing")
    enc.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze LegNet trunk (stem/main/mapper) but keep original LegNet head trainable.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Resolve checkpoint + config
    if args.model_dir:
        model_dir = Path(args.model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"Could not find config.json in: {model_dir}")

        ckpts = list(model_dir.rglob("*.ckpt")) + list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth"))
        if not ckpts:
            raise SystemExit(f"No checkpoints found under: {model_dir}")
        pearson_ckpts = [p for p in ckpts if p.name.startswith("pearson")]
        if pearson_ckpts:
            pearson_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            ckpt_path = pearson_ckpts[0]
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
        sep = "\t" if data_path.suffix in {".tsv", ".txt"} else ","

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

    device = torch.device(args.device)

    # Create datasets + loaders
    train_ds = PairMultiTaskDataset(
        ref_seqs,
        ref_y,
        alt_seqs,
        alt_y,
        train_idx,
        seq_len=seq_len,
        add_reverse_channel=add_reverse_channel,
        flip_pairs=bool(args.flip_pairs),
        rc_pair_augment=bool(args.rc_pair_augment),
        deterministic=False,
        normalize_delta=bool(args.normalize_delta),
    )

    # Use train delta mean/std for val/test so the delta normalization is consistent
    mean, std = train_ds.get_delta_mean_std()

    val_ds = PairMultiTaskDataset(
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
        normalize_delta=bool(args.normalize_delta),
        normalize_mean=mean if args.normalize_delta else None,
        normalize_std=std if args.normalize_delta else None,
    )

    test_ds = PairMultiTaskDataset(
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
        normalize_delta=bool(args.normalize_delta),
        normalize_mean=mean if args.normalize_delta else None,
        normalize_std=std if args.normalize_delta else None,
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

    # Load pretrained LegNet (mpralegnet)
    legnet, meta = load_model(ckpt_path, config, map_location="cpu", device=device, strict=bool(args.strict))
    print("Loaded checkpoint:", ckpt_path)
    if config_path is not None:
        print("Loaded config:", config_path)
    print("Checkpoint key prefix used:", meta.get("used_prefix"))
    print(f"Pairs: n={n_pairs} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print("seq_len:", seq_len, "| add_reverse_channel:", add_reverse_channel)
    print("rc_pair_augment(train):", bool(args.rc_pair_augment), "| flip_pairs(train):", bool(args.flip_pairs))
    print("rc_average(eval):", bool(args.rc_average))
    print("normalize_delta:", bool(args.normalize_delta), f"| delta_mean={mean:.6f} delta_std={std:.6f}" if args.normalize_delta else "")

    # Freeze trunk if requested (keeps original head trainable)
    if args.freeze_encoder:
        freeze_legnet_trunk_params(legnet)
        set_legnet_trunk_eval(legnet)  # freeze BN stats in trunk
        print("Encoder frozen: True (trunk params frozen; trunk set to eval to freeze BatchNorm stats)")
    else:
        print("Encoder frozen: False")

    # Save run config
    with (out_dir / "run_args.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    multitask_dir = out_dir / "multitask"
    ensure_dir(multitask_dir)

    metrics = run_multitask(
        legnet=legnet,
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
        delta_hidden_dim=int(args.delta_hidden_dim),
        delta_dropout=float(args.delta_dropout),
        loss_ref=str(args.loss_ref),
        loss_alt=str(args.loss_alt),
        loss_delta=str(args.loss_delta),
        w_ref=float(args.w_ref),
        w_alt=float(args.w_alt),
        w_delta=float(args.w_delta),
        select_metric=str(args.select_metric),
        freeze_encoder=bool(args.freeze_encoder),
        out_dir=multitask_dir,
    )

    print("[multitask] test:", metrics["test"])
    print("\nDone. Wrote outputs to:", out_dir)


if __name__ == "__main__":
    main()
