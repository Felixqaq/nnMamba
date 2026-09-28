"""Hybrid Mamba-attention 3D CT regression network."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_regressor import DownsampleStage, norm3d


# The spirometry values predicted alongside FEV1/FVC. Names match the columns the
# trainer reads from the clinical CSV; the order fixes the ModuleDict key order so
# a checkpoint stays loadable.
AUX_PFT_TARGETS: tuple[str, ...] = ("fev1", "fvc", "fev1_pctpred", "fvc_pctpred")


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
    """Hybrid 3D CT predictor with Mamba stages and a global attention bridge."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
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
            nn.Linear(head_mid_dim, int(num_classes)),
        )
        # Auxiliary emphysema head. Trained only when a run supplies %LAA-950
        # targets; it shares forward_features with the classifier, so the shared
        # trunk has to encode emphysema extent to satisfy it. Deployment never
        # calls it -- the classifier path is unchanged, so no segmentation and no
        # extra inference cost reach the field.
        self.aux_emphysema_head = nn.Sequential(
            nn.Linear(feature_dim, head_mid_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_mid_dim, 1),
        )

        # Auxiliary spirometry heads, one per additional PFT value. The point is
        # regularisation, not these outputs: on this cohort in-sample ratio MAE
        # runs near 1.1 against 7.1 on held-out patients, so the trunk has room to
        # memorise a single target. FEV1/FVC and FVC are almost uncorrelated
        # (r = -0.03) while FEV1/FVC and FEV1 are not (r = 0.46), so satisfying all
        # of them at once demands features that describe the lung rather than the
        # patient. Deployment reads self.head alone and never runs these.
        self.aux_pft_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(feature_dim, head_mid_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(head_mid_dim, 1),
            )
            for name in AUX_PFT_TARGETS
        })

        self._init_head()

    def _init_head(self) -> None:
        """Keep initial regression outputs close to zero in normalized space."""
        modules = [self.head, self.aux_emphysema_head, *self.aux_pft_heads.values()]
        for module in modules:
            final_linear = module[-1]
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
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output

    def forward_with_emphysema(self, x: torch.Tensor):
        """Classifier logits plus the auxiliary emphysema prediction.

        Kept separate from forward() so every existing caller, checkpoint and
        deployment path behaves exactly as before.
        """
        features = self.forward_features(x)
        output = self.head(features)
        logits = output.squeeze(-1) if output.shape[-1] == 1 else output
        return logits, self.aux_emphysema_head(features).squeeze(-1)

    def forward_multitarget(self, x: torch.Tensor):
        """Main output plus one auxiliary spirometry value per head.

        Separate from forward() for the same reason as forward_with_emphysema:
        every existing caller, checkpoint and deployment path keeps its current
        behaviour, and inference cost is unchanged because nothing in the field
        calls this.
        """
        features = self.forward_features(x)
        output = self.head(features)
        main = output.squeeze(-1) if output.shape[-1] == 1 else output
        aux = {name: head(features).squeeze(-1)
               for name, head in self.aux_pft_heads.items()}
        return main, aux
