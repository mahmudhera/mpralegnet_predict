"""Checkpoint I/O.

Goal: load MPRA-LegNet checkpoints without requiring PyTorch Lightning.

The upstream human_legnet repository trains with Lightning and saves .ckpt files.
Those are simple torch.load()-able dicts with a 'state_dict' field. The keys are
usually prefixed (e.g. 'model.'). Here we:
- instantiate a LegNet from a LegNetConfig
- load weights from either:
  * a Lightning .ckpt (dict with 'state_dict')
  * a plain state_dict (OrderedDict)
  * a .pt/.pth produced by this repo (dict with 'state_dict' and 'config')

We try a few common prefix-stripping strategies automatically.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .config import LegNetConfig


def _extract_state_dict(obj: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        # Common: saved as {'model': state_dict}
        if "model" in obj and isinstance(obj["model"], dict) and all(isinstance(k, str) for k in obj["model"].keys()):
            return obj["model"]
        # If it already looks like a state_dict
        if all(isinstance(k, str) for k in obj.keys()) and any(hasattr(v, "shape") for v in obj.values()):
            return obj  # type: ignore[return-value]

    raise ValueError(
        "Unrecognized checkpoint format. Expected a dict with 'state_dict' or a plain state_dict."
    )


def _strip_prefix(sd: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return dict(sd)
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _score_incompatible(missing: list[str], unexpected: list[str]) -> int:
    # Lower is better.
    return len(missing) + len(unexpected)


def load_model(
    checkpoint_path: str | Path,
    config: LegNetConfig,
    *,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    strict: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a LegNet model from a checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to .ckpt/.pt/.pth.
    config:
        Model config (can be parsed from upstream config.json).
    map_location:
        torch.load map_location.
    device:
        If provided, the returned model is moved to this device.
    strict:
        If True, enforce exact state_dict match.

    Returns
    -------
    model, meta
        meta is a dict with some loading diagnostics.
    """
    checkpoint_path = Path(checkpoint_path)
    obj = torch.load(checkpoint_path, map_location=map_location)

    # If it's a repo-produced checkpoint, allow overriding config from inside.
    if isinstance(obj, dict) and "config" in obj and isinstance(obj["config"], dict):
        try:
            config = LegNetConfig.from_dict(obj["config"])
        except Exception:
            pass

    model = config.build_model()

    sd = _extract_state_dict(obj)

    # Try a few common prefix-stripping patterns.
    candidates = [
        ("", dict(sd)),
        ("model.", _strip_prefix(sd, "model.")),
        ("module.", _strip_prefix(sd, "module.")),
        ("net.", _strip_prefix(sd, "net.")),
        ("model.module.", _strip_prefix(sd, "model.module.")),
    ]

    best = None
    best_info: Dict[str, Any] = {}

    for name, cand in candidates:
        incompatible = model.load_state_dict(cand, strict=False)
        score = _score_incompatible(incompatible.missing_keys, incompatible.unexpected_keys)
        info = {
            "candidate_prefix": name,
            "missing_keys": incompatible.missing_keys,
            "unexpected_keys": incompatible.unexpected_keys,
            "score": score,
        }
        if best is None or score < best_info["score"]:
            best = cand
            best_info = info

        # Reset model weights before next try (to avoid partial loading across tries)
        model = config.build_model()

    assert best is not None

    incompatible = model.load_state_dict(best, strict=strict)
    meta = {
        "checkpoint_path": str(checkpoint_path),
        "used_prefix": best_info["candidate_prefix"],
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }

    if device is not None:
        model = model.to(device)

    return model, meta


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: LegNetConfig,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a small, self-contained checkpoint (no Lightning dependency)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)


def resolve_config_path(model_dir: str | Path) -> Path:
    """Given a model_dir, return model_dir/config.json (upstream convention)."""
    model_dir = Path(model_dir)
    cfg = model_dir / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"Could not find config.json in: {model_dir}")
    return cfg


def resolve_checkpoint_path(model_dir: str | Path) -> Path:
    """Best-effort: pick a checkpoint inside a human_legnet training directory.

    We look for:
    - any *.ckpt under model_dir (recursively)
    - prefer files that start with 'pearson' (best model names in upstream)
    - otherwise pick the most recently modified.
    """
    model_dir = Path(model_dir)
    ckpts = list(model_dir.rglob("*.ckpt"))
    if not ckpts:
        # also allow *.pt/*.pth from our repo
        ckpts = list(model_dir.rglob("*.pt")) + list(model_dir.rglob("*.pth"))

    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {model_dir}")

    pearson = [p for p in ckpts if p.name.startswith("pearson")]
    if pearson:
        # If multiple, pick latest.
        pearson.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return pearson[0]

    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]
