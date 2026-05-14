"""Hybrid Mamba + TAP-CT late fusion with modality attention gates."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor


class HybridMambaTapctAttentionFusionRegressor(nn.Module):
    """Reweight CT and TAP-CT branches, then keep the original concat head."""

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
        self.image_feature_dim = (
            base_channels + base_channels * 2 + base_channels * 4 * 2
        )

        self.embedding_branch = nn.Sequential(
            nn.LayerNorm(self.tapct_embedding_dim),
            nn.Linear(self.tapct_embedding_dim, self.fusion_projection_dim),
            nn.GELU(),
            nn.Dropout(float(fusion_dropout)),
        )

        self.image_attention_v = nn.Sequential(
            nn.LayerNorm(self.image_feature_dim),
            nn.Linear(self.image_feature_dim, self.tapct_attention_dim),
            nn.Tanh(),
        )
        self.tapct_attention_v = nn.Sequential(
            nn.LayerNorm(self.fusion_projection_dim),
            nn.Linear(self.fusion_projection_dim, self.tapct_attention_dim),
            nn.Tanh(),
        )
        if self.tapct_gated_attention:
            self.image_attention_u = nn.Sequential(
                nn.LayerNorm(self.image_feature_dim),
                nn.Linear(self.image_feature_dim, self.tapct_attention_dim),
                nn.Sigmoid(),
            )
            self.tapct_attention_u = nn.Sequential(
                nn.LayerNorm(self.fusion_projection_dim),
                nn.Linear(self.fusion_projection_dim, self.tapct_attention_dim),
                nn.Sigmoid(),
            )
        else:
            self.image_attention_u = None
            self.tapct_attention_u = None
        self.image_attention_score = nn.Linear(self.tapct_attention_dim, 1)
        self.tapct_attention_score = nn.Linear(self.tapct_attention_dim, 1)

        self.feature_dim = self.image_feature_dim + self.fusion_projection_dim
        head_hidden_dim = max(
            int(head_hidden_dim),
            self.feature_dim // 2,
            base_channels * 4,
        )
        head_mid_dim = max(head_hidden_dim // 2, base_channels * 4)
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, head_hidden_dim),
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
                "HybridMambaTapctAttentionFusionRegressor expects a dict with keys "
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

    def _branch_score(
        self,
        features: torch.Tensor,
        value_layer: nn.Module,
        gate_layer: nn.Module | None,
        score_layer: nn.Module,
    ) -> torch.Tensor:
        attention_hidden = value_layer(features)
        if gate_layer is not None:
            attention_hidden = attention_hidden * gate_layer(features)
        return score_layer(attention_hidden)

    def forward_branch_features(
        self,
        inputs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw CT and projected TAP-CT branch features."""
        ct, embedding = self._unpack_inputs(inputs)
        image_features = self.image_encoder.forward_features(ct)
        embedding_features = self.embedding_branch(embedding.float())
        return image_features, embedding_features

    def _attention_weights_from_features(
        self,
        image_features: torch.Tensor,
        embedding_features: torch.Tensor,
    ) -> torch.Tensor:
        image_score = self._branch_score(
            image_features,
            self.image_attention_v,
            self.image_attention_u,
            self.image_attention_score,
        )
        tapct_score = self._branch_score(
            embedding_features,
            self.tapct_attention_v,
            self.tapct_attention_u,
            self.tapct_attention_score,
        )
        scores = torch.cat([image_score, tapct_score], dim=1)
        return torch.softmax(scores, dim=1)

    def forward_attention_weights(self, inputs: Any) -> torch.Tensor:
        """Return attention weights for [CT, TAP-CT] branches."""
        image_features, embedding_features = self.forward_branch_features(inputs)
        return self._attention_weights_from_features(image_features, embedding_features)

    def forward_features(self, inputs: Any) -> torch.Tensor:
        """Return the attention-reweighted concat feature vector."""
        image_features, embedding_features = self.forward_branch_features(inputs)
        weights = self._attention_weights_from_features(
            image_features,
            embedding_features,
        )
        image_scale = 2.0 * weights[:, 0:1]
        tapct_scale = 2.0 * weights[:, 1:2]
        return torch.cat(
            [
                image_features * image_scale,
                embedding_features * tapct_scale,
            ],
            dim=1,
        )

    def forward(self, inputs: Any) -> torch.Tensor:
        features = self.forward_features(inputs)
        output = self.head(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output
