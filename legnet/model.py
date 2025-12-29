"""LegNet architecture (MPRA-LegNet / human_legnet variant).

This implementation is adapted from the public autosome-ru/human_legnet repository
(MIT License). It is a 1D EfficientNetV2-inspired CNN that maps one-hot encoded
DNA sequences (shape: [B, C, L]) to a single continuous prediction per sequence.

Only the minimal pieces needed for loading, inference, and fine-tuning are kept.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m: nn.Module) -> None:
    """Weight initialization used in the original code."""
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class SELayer(nn.Module):
    def __init__(self, inp: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp // reduction)),
            nn.SiLU(),
            nn.Linear(int(inp // reduction), inp),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        b, c, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y


class EffBlock(nn.Module):
    """EfficientNetV2-inspired inverted bottleneck + depthwise conv + SE."""

    def __init__(
        self,
        in_ch: int,
        ks: int,
        resize_factor: int,
        activation: Callable[[], nn.Module],
        out_ch: Optional[int] = None,
        se_reduction: Optional[int] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks

        self.inner_dim = self.in_ch * self.resize_factor

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.inner_dim,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.inner_dim,
                kernel_size=ks,
                groups=self.inner_dim,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.inner_dim),
            activation(),
            SELayer(self.inner_dim, reduction=self.se_reduction),
            nn.Conv1d(
                in_channels=self.inner_dim,
                out_channels=self.in_ch,
                kernel_size=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.in_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        ks: int,
        activation: Callable[[], nn.Module],
        out_ch: Optional[int] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                kernel_size=self.ks,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm1d(self.out_ch),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConcat(nn.Module):
    """Concatenate fn(x) and x along channel dimension."""

    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.concat([self.fn(x, **kwargs), x], dim=1)


class MapperBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LegNet(nn.Module):
    """LegNet for sequence-to-expression regression.

    Input:  one-hot DNA, shape [B, C, L]
    Output: scalar per sequence, shape [B]
    """

    def __init__(
        self,
        in_ch: int,
        stem_ch: int,
        stem_ks: int,
        ef_ks: int,
        ef_block_sizes: List[int],
        pool_sizes: List[int],
        resize_factor: int,
        activation: Callable[[], nn.Module] = nn.SiLU,
    ):
        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)

        self.in_ch = in_ch

        self.stem = LocalBlock(in_ch=in_ch, out_ch=stem_ch, ks=stem_ks, activation=activation)

        blocks: List[nn.Module] = []
        in_ch_ = stem_ch
        out_ch = stem_ch

        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch_,
                        out_ch=in_ch_,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation,
                    )
                ),
                LocalBlock(
                    in_ch=in_ch_ * 2,
                    out_ch=out_ch,
                    ks=ef_ks,
                    activation=activation,
                ),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity(),
            )
            in_ch_ = out_ch
            blocks.append(blc)

        self.main = nn.Sequential(*blocks)

        self.mapper = MapperBlock(in_features=out_ch, out_features=out_ch * 2)

        self.head = nn.Sequential(
            nn.Linear(out_ch * 2, out_ch * 2),
            nn.BatchNorm1d(out_ch * 2),
            activation(),
            nn.Linear(out_ch * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        x = x.squeeze(-1)
        return x
