"""Mamba-based 3D CT regression network."""

from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class ResidualMambaBlock(nn.Module):
    """3D residual block that mixes spatial tokens with Mamba."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.pre_norm = nn.BatchNorm3d(dim)
        self.pre_proj = conv1x1(dim, dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_norm = nn.BatchNorm3d(dim)
        self.post_proj = conv1x1(dim, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_proj(self.pre_norm(x))
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        tokens = x.reshape(b, c, -1).transpose(1, 2)
        tokens = self.mamba(tokens)
        x = tokens.transpose(1, 2).reshape(b, c, *spatial_shape)
        x = self.post_proj(self.post_norm(x))
        return self.act(x + residual)


class DownsampleStage(nn.Module):
    """A convolutional reduction stage followed by Mamba blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        stride: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm3d(out_channels),
                nn.GELU(),
            )
        ]
        for _ in range(depth):
            layers.append(ResidualMambaBlock(out_channels))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class MambaAngleRegressor(nn.Module):
    """3D CT regressor that predicts a single collapse angle."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=4,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.GELU(),
        )
        self.stage1 = DownsampleStage(base_channels, base_channels, depths[0], stride=1)
        self.stage2 = DownsampleStage(base_channels, base_channels * 2, depths[1], stride=2)
        self.stage3 = DownsampleStage(
            base_channels * 2, base_channels * 4, depths[2], stride=2
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        feature_dim = base_channels + base_channels * 2 + base_channels * 4
        self.head = nn.Sequential(
            nn.Linear(feature_dim, base_channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        f1 = self.pool(x1).flatten(1)
        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)
        return torch.cat([f1, f2, f3], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features).squeeze(-1)
