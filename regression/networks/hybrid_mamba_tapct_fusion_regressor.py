"""Late-fusion Hybrid Mamba + TAP-CT embedding predictor."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor


class HybridMambaTapctFusionRegressor(nn.Module):
    """Fuse learned 3D CT features with frozen TAP-CT patient embeddings.

    The Hybrid Mamba branch keeps the existing image pathway intact. A small
    projection branch normalizes and compresses the frozen TAP-CT embedding,
    then the final head predicts from the concatenated feature vector.
    """

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
        tapct_embedding_dim: int = 2304,
        fusion_projection_dim: int = 128,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.tapct_embedding_dim = int(tapct_embedding_dim)
        self.fusion_projection_dim = int(fusion_projection_dim)
        if self.tapct_embedding_dim <= 0:
            raise ValueError("tapct_embedding_dim must be positive.")
        if self.fusion_projection_dim <= 0:
            raise ValueError("fusion_projection_dim must be positive.")

        self.image_encoder = HybridMambaAttentionRegressor(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depths=depths,
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            attn_heads=attn_heads,
            attn_layers=attn_layers,
            attn_mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
        )
        self.image_encoder.head = nn.Identity()
        image_feature_dim = base_channels + base_channels * 2 + base_channels * 4 * 2

        self.embedding_branch = nn.Sequential(
            nn.LayerNorm(self.tapct_embedding_dim),
            nn.Linear(self.tapct_embedding_dim, self.fusion_projection_dim),
            nn.GELU(),
            nn.Dropout(float(fusion_dropout)),
        )

        feature_dim = image_feature_dim + self.fusion_projection_dim
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
        self._init_head()

    def _init_head(self) -> None:
        """Keep initial predictions close to zero in normalized space."""
        final_linear = self.head[-1]
        nn.init.normal_(final_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    def _unpack_inputs(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CT tensor and TAP-CT embedding tensor from supported inputs."""
        if isinstance(inputs, dict):
            ct = inputs.get("ct")
            if ct is None:
                ct = inputs.get("mri")
            embedding = inputs.get("tapct_embedding")
        elif isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            ct, embedding = inputs
        else:
            raise TypeError(
                "HybridMambaTapctFusionRegressor expects a dict with keys "
                "'ct' and 'tapct_embedding', or a (ct, embedding) tuple."
            )

        if ct is None:
            raise KeyError("Fusion input is missing the CT tensor.")
        if embedding is None:
            raise KeyError("Fusion input is missing tapct_embedding.")
        if embedding.ndim > 2:
            embedding = embedding.flatten(1)
        if embedding.shape[-1] != self.tapct_embedding_dim:
            raise ValueError(
                "Unexpected TAP-CT embedding dimension: "
                f"got {embedding.shape[-1]}, expected {self.tapct_embedding_dim}."
            )
        return ct, embedding

    def forward_features(self, inputs: Any) -> torch.Tensor:
        """Return the concatenated image and TAP-CT feature vector."""
        ct, embedding = self._unpack_inputs(inputs)
        image_features = self.image_encoder.forward_features(ct)
        embedding_features = self.embedding_branch(embedding.float())
        return torch.cat([image_features, embedding_features], dim=1)

    def forward(self, inputs: Any) -> torch.Tensor:
        features = self.forward_features(inputs)
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output
