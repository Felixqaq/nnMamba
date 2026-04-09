"""Hybrid Mamba-attention 3D CT regression network."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_regressor import DownsampleStage, norm3d


def _resolve_attention_heads(dim: int, requested_heads: int) -> int:
    """Choose a valid attention head count for the given channel width."""
    heads = max(1, min(int(requested_heads), int(dim)))
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


class HybridAttentionBlock(nn.Module):
    """A lightweight global attention block applied after Mamba stages."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(dim, int(dim * mlp_ratio))
        self.pre_norm = norm3d(dim)
        self.pos_conv = nn.Conv3d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.token_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=_resolve_attention_heads(dim, num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = x + self.pos_conv(x)

        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        tokens = x.reshape(b, c, -1).transpose(1, 2)

        attn_input = self.token_norm(tokens)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + self.attn_dropout(attn_out)
        tokens = tokens + self.mlp(self.mlp_norm(tokens))

        return residual + tokens.transpose(1, 2).reshape(b, c, *spatial_shape)


class HybridMambaAttentionRegressor(nn.Module):
    """Hybrid 3D CT regressor with Mamba stages and a global attention bridge."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depths: tuple[int, int, int] = (1, 1, 1),
        head_hidden_dim: int = 128,
        dropout: float = 0.2,
        attn_heads: int = 8,
        attn_layers: int = 1,
        attn_mlp_ratio: float = 2.0,
        attn_dropout: float = 0.1,
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
            norm3d(base_channels),
            nn.GELU(),
        )
        self.stage1 = DownsampleStage(base_channels, base_channels, depths[0], stride=1)
        self.stage2 = DownsampleStage(base_channels, base_channels * 2, depths[1], stride=2)
        self.stage3 = DownsampleStage(
            base_channels * 2, base_channels * 4, depths[2], stride=2
        )
        self.attention_layers = nn.Sequential(
            *[
                HybridAttentionBlock(
                    dim=base_channels * 4,
                    num_heads=attn_heads,
                    mlp_ratio=attn_mlp_ratio,
                    dropout=attn_dropout,
                )
                for _ in range(max(1, int(attn_layers)))
            ]
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        feature_dim = base_channels + base_channels * 2 + base_channels * 4 * 2
        head_hidden_dim = max(int(head_hidden_dim), feature_dim // 2, base_channels * 4)
        head_mid_dim = max(head_hidden_dim // 2, base_channels * 4)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, head_mid_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_mid_dim, 1),
        )
        self._init_head()

    def _init_head(self) -> None:
        """Keep initial regression outputs close to zero in normalized space."""
        final_linear = self.head[-1]
        nn.init.normal_(final_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.attention_layers(x3)

        f1 = self.pool(x1).flatten(1)
        f2 = self.pool(x2).flatten(1)
        f3 = self.pool(x3).flatten(1)
        f4 = self.pool(x4).flatten(1)
        return torch.cat([f1, f2, f3, f4], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features).squeeze(-1)
