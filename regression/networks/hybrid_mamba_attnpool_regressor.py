"""Hybrid Mamba-attention regressor with attention pooling instead of average pooling.

A subclass of HybridMambaAttentionRegressor that changes one thing: how each
stage's 3D feature map is reduced to a vector. The parent averages every
position equally -- ~26,000 at stage 1 -- so an upper-lobe emphysema filling a
few hundred of them is diluted by healthy lung and, measured on the validation
set, by the ~25% of every map that is air outside the lung. Here each position
earns a score and the map is averaged with softmax weights (gated attention,
Ilse et al. 2018).

The parent class is untouched. Backbone, heads and their sizes are inherited, so
any difference against the parent comes from the pooling alone; and because the
scorer's last layer starts at zero, a freshly built model produces exactly the
parent's output until training moves it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor


class GatedAttentionPool3d(nn.Module):
    """Softmax-weighted mean over every voxel of a feature map."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.content = nn.Linear(channels, hidden)
        self.gate = nn.Linear(channels, hidden)
        self.score = nn.Linear(hidden, 1)
        # Equal scores everywhere -> uniform weights -> exactly the plain mean.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, x: torch.Tensor):
        tokens = x.flatten(2).transpose(1, 2)                      # (B, N, C)
        logits = self.score(torch.tanh(self.content(tokens))
                            * torch.sigmoid(self.gate(tokens)))    # (B, N, 1)
        # Softmax over ~26,000 positions in fp32: bf16 cannot resolve weights of
        # order 1/26,000, and this model has failed silently under reduced
        # precision before.
        weights = torch.softmax(logits.float(), dim=1)
        pooled = (weights.to(tokens.dtype) * tokens).sum(dim=1)    # (B, C)
        return pooled, weights.view(x.shape[0], *x.shape[2:])


class HybridMambaAttnPoolRegressor(HybridMambaAttentionRegressor):
    """Same network as the parent, pooled by attention; also exposes the maps."""

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
        pool_hidden: int = 64,
    ):
        super().__init__(
            in_channels=in_channels, num_classes=num_classes,
            base_channels=base_channels, depths=depths,
            head_hidden_dim=head_hidden_dim, dropout=dropout,
            attn_heads=attn_heads, attn_layers=attn_layers,
            attn_mlp_ratio=attn_mlp_ratio, attn_dropout=attn_dropout,
        )
        # One pool per stage, matching the parent's four pooled feature maps.
        self.attn_pools = nn.ModuleList([
            GatedAttentionPool3d(c, pool_hidden) for c in
            (base_channels, base_channels * 2, base_channels * 4, base_channels * 4)])

    def _stage_maps(self, x: torch.Tensor):
        x1 = self.stage1(self.stem(x))
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.attention_layers(x3)
        return x1, x2, x3, x4

    def _attention_pool(self, x: torch.Tensor):
        pooled, weights = zip(*(pool(m) for pool, m in
                                zip(self.attn_pools, self._stage_maps(x))))
        return torch.cat(pooled, dim=1), list(weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # The parent's forward, forward_with_emphysema and forward_multitarget
        # all call this, so every entry point uses attention pooling.
        return self._attention_pool(x)[0]

    def forward_with_attention(self, x: torch.Tensor):
        """Prediction plus one attention map per stage.

        Maps are softmax weights over each stage grid -- 28x34x28 down to 7x9x7
        for a 112x136x112 input -- and each sums to 1 per patient.
        """
        features, weights = self._attention_pool(x)
        output = self.head(features)
        output = output.squeeze(-1) if output.shape[-1] == 1 else output
        return output, weights
