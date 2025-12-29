"""Configuration helpers.

The upstream MPRA-LegNet (human_legnet) repository stores a fairly large
TrainingConfig in config.json. For inference and fine-tuning we only need the
architecture + augmentation-related bits.

This module provides a small LegNetConfig that can be instantiated either from:
- a minimal config you create yourself, or
- the upstream config.json produced by human_legnet/training_config.py.

Upstream reference (fields like stem_ch, stem_ks, ef_ks, ef_block_sizes,
resize_factor, pool_sizes, reverse_augment, use_reverse_channel, ...):
- autosome-ru/human_legnet/training_config.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .model import LegNet


@dataclass
class LegNetConfig:
    # Architecture
    stem_ch: int = 64
    stem_ks: int = 11
    ef_ks: int = 9
    ef_block_sizes: Tuple[int, ...] = (80, 96, 112, 128)
    resize_factor: int = 4
    pool_sizes: Tuple[int, ...] = (2, 2, 2, 2)

    # Augmentation / input representation
    reverse_augment: bool = False
    use_reverse_channel: bool = False

    # Optional: intended sequence length (not required by the model)
    seq_len: Optional[int] = None

    @property
    def in_ch(self) -> int:
        return 4 + (1 if self.use_reverse_channel else 0)

    def build_model(self) -> LegNet:
        return LegNet(
            in_ch=self.in_ch,
            stem_ch=self.stem_ch,
            stem_ks=self.stem_ks,
            ef_ks=self.ef_ks,
            ef_block_sizes=list(self.ef_block_sizes),
            pool_sizes=list(self.pool_sizes),
            resize_factor=self.resize_factor,
        )

    def to_dict(self) -> Dict[str, Any]:
        dt = asdict(self)
        # Convert tuples to lists for JSON.
        dt["ef_block_sizes"] = list(self.ef_block_sizes)
        dt["pool_sizes"] = list(self.pool_sizes)
        return dt

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, dt: Mapping[str, Any]) -> "LegNetConfig":
        """Create config from a dict.

        Extra keys are ignored, so you can pass the upstream TrainingConfig JSON.
        """
        # Pull only the keys we care about.
        keep: Dict[str, Any] = {}
        for k in (
            "stem_ch",
            "stem_ks",
            "ef_ks",
            "ef_block_sizes",
            "resize_factor",
            "pool_sizes",
            "reverse_augment",
            "use_reverse_channel",
            "seq_len",
        ):
            if k in dt:
                keep[k] = dt[k]

        # Normalize list->tuple
        if "ef_block_sizes" in keep and isinstance(keep["ef_block_sizes"], list):
            keep["ef_block_sizes"] = tuple(int(x) for x in keep["ef_block_sizes"])
        if "pool_sizes" in keep and isinstance(keep["pool_sizes"], list):
            keep["pool_sizes"] = tuple(int(x) for x in keep["pool_sizes"])

        # Some upstream configs use "in_ch" indirectly; we don't.
        return cls(**keep)

    @classmethod
    def from_json(cls, path: str | Path) -> "LegNetConfig":
        path = Path(path)
        with path.open("r") as f:
            dt = json.load(f)
        return cls.from_dict(dt)
