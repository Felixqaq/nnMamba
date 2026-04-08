"""SwinUNETR v2 encoder-based 3D CT regression network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer


def _to_3tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize scalar or sequence config values into a 3D tuple."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(item) for item in value)
        if len(values) != 3:
            raise ValueError(f"Expected 3 values, got {values}")
        return values
    scalar = int(value)
    return (scalar, scalar, scalar)


class SwinUNETRV2AngleRegressor(nn.Module):
    """3D CT regressor built on the SwinUNETR v2 encoder backbone."""

    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 24,
        head_hidden_dim: int = 192,
        dropout: float = 0.2,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int | Sequence[int] = 4,
        patch_size: int | Sequence[int] = 2,
        use_checkpoint: bool = False,
        use_v2: bool = True,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if feature_size % 12 != 0:
            raise ValueError("feature_size must be divisible by 12 for SwinUNETR.")
        if len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("depths and num_heads must each contain exactly 4 values.")

        self.patch_size = _to_3tuple(patch_size)
        self.window_size = _to_3tuple(window_size)
        self.backbone = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=tuple(int(depth) for depth in depths),
            num_heads=tuple(int(heads) for heads in num_heads),
            attn_drop_rate=float(attn_drop_rate),
            drop_path_rate=float(drop_path_rate),
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
            use_v2=bool(use_v2),
        )

        feature_dim = feature_size * 31
        head_hidden_dim = max(int(head_hidden_dim), feature_size)
        head_mid_dim = max(head_hidden_dim // 2, feature_size)
        self.pool = nn.AdaptiveAvgPool3d(1)
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

    def _pad_to_valid_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Pad volumes so the Swin encoder can safely downsample them."""
        spatial_shape = x.shape[2:]
        target_shape = []
        for size, patch in zip(spatial_shape, self.patch_size):
            multiple = patch**5
            padded = ((int(size) + multiple - 1) // multiple) * multiple
            target_shape.append(padded)

        if tuple(target_shape) == tuple(int(size) for size in spatial_shape):
            return x

        pad: list[int] = []
        for current, target in zip(reversed(spatial_shape), reversed(target_shape)):
            total = int(target) - int(current)
            left = total // 2
            right = total - left
            pad.extend([left, right])

        # Replicate padding preserves edge statistics better than zero padding for CT volumes.
        return F.pad(x, pad, mode="replicate")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad_to_valid_shape(x)
        hidden_states = self.backbone(x, normalize=True)
        pooled = [self.pool(feature).flatten(1) for feature in hidden_states]
        return torch.cat(pooled, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features).squeeze(-1)
