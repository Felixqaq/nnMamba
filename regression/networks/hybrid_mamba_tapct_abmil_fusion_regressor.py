"""Hybrid Mamba + TAP-CT fusion with an ABMIL classification head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor


class HybridMambaTapctABMILFusionRegressor(nn.Module):
    """Fuse CT and frozen TAP-CT features with modality-level ABMIL attention."""

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
        tapct_attention_dim: int = 128,
        tapct_gated_attention: bool = True,
    ):
        super().__init__()
        self.tapct_embedding_dim = int(tapct_embedding_dim)
        self.fusion_projection_dim = int(fusion_projection_dim)
        self.tapct_attention_dim = int(tapct_attention_dim)
        self.tapct_gated_attention = bool(tapct_gated_attention)
        if self.tapct_embedding_dim <= 0:
            raise ValueError("tapct_embedding_dim must be positive.")
        if self.fusion_projection_dim <= 0:
            raise ValueError("fusion_projection_dim must be positive.")
        if self.tapct_attention_dim <= 0:
            raise ValueError("tapct_attention_dim must be positive.")

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

        self.image_branch = nn.Sequential(
            nn.LayerNorm(image_feature_dim),
            nn.Linear(image_feature_dim, self.fusion_projection_dim),
            nn.GELU(),
            nn.Dropout(float(fusion_dropout)),
        )
        self.embedding_branch = nn.Sequential(
            nn.LayerNorm(self.tapct_embedding_dim),
            nn.Linear(self.tapct_embedding_dim, self.fusion_projection_dim),
            nn.GELU(),
            nn.Dropout(float(fusion_dropout)),
        )

        self.attention_v = nn.Sequential(
            nn.Linear(self.fusion_projection_dim, self.tapct_attention_dim),
            nn.Tanh(),
        )
        if self.tapct_gated_attention:
            self.attention_u = nn.Sequential(
                nn.Linear(self.fusion_projection_dim, self.tapct_attention_dim),
                nn.Sigmoid(),
            )
        else:
            self.attention_u = None
        self.attention_score = nn.Linear(self.tapct_attention_dim, 1)

        head_hidden_dim = max(int(head_hidden_dim), self.fusion_projection_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(self.fusion_projection_dim),
            nn.Linear(self.fusion_projection_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden_dim, int(num_classes)),
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
                "HybridMambaTapctABMILFusionRegressor expects a dict with keys "
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

    def forward_instances(self, inputs: Any) -> torch.Tensor:
        """Return projected CT and TAP-CT modality instances."""
        ct, embedding = self._unpack_inputs(inputs)
        image_features = self.image_encoder.forward_features(ct)
        image_instance = self.image_branch(image_features)
        tapct_instance = self.embedding_branch(embedding.float())
        return torch.stack([image_instance, tapct_instance], dim=1)

    def attention_pool(
        self, instances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool modality instances with ABMIL attention."""
        attention_hidden = self.attention_v(instances)
        if self.attention_u is not None:
            attention_hidden = attention_hidden * self.attention_u(instances)
        scores = self.attention_score(attention_hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(instances * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def forward_features(self, inputs: Any) -> torch.Tensor:
        """Return ABMIL-pooled CT/TAP-CT fusion features."""
        instances = self.forward_instances(inputs)
        pooled, _ = self.attention_pool(instances)
        return pooled

    def forward_attention_weights(self, inputs: Any) -> torch.Tensor:
        """Return attention weights for [CT, TAP-CT] modality instances."""
        instances = self.forward_instances(inputs)
        _, weights = self.attention_pool(instances)
        return weights

    def forward(self, inputs: Any) -> torch.Tensor:
        features = self.forward_features(inputs)
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output
