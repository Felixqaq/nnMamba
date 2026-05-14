"""Attention-based multiple-instance classifier for frozen TAP-CT features."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TapctABMILClassifier(nn.Module):
    """Classify a CT scan from frozen TAP-CT instance embeddings."""

    def __init__(
        self,
        *,
        tapct_embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        attention_dim: int = 128,
        dropout: float = 0.2,
        gated_attention: bool = True,
    ) -> None:
        super().__init__()
        if tapct_embedding_dim <= 0:
            raise ValueError("tapct_embedding_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.tapct_embedding_dim = int(tapct_embedding_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim)
        self.gated_attention = bool(gated_attention)

        self.instance_encoder = nn.Sequential(
            nn.LayerNorm(self.tapct_embedding_dim),
            nn.Linear(self.tapct_embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(self.hidden_dim, self.attention_dim),
            nn.Tanh(),
        )
        if self.gated_attention:
            self.attention_u = nn.Sequential(
                nn.Linear(self.hidden_dim, self.attention_dim),
                nn.Sigmoid(),
            )
        else:
            self.attention_u = None
        self.attention_score = nn.Linear(self.attention_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def _unpack_inputs(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return TAP-CT instance tensor and optional validity mask."""
        mask = None
        if isinstance(inputs, dict):
            embedding = inputs.get("tapct_embedding")
            if embedding is None:
                embedding = inputs.get("tapct_instances")
            mask = inputs.get("tapct_mask")
        else:
            embedding = inputs

        if embedding is None:
            raise KeyError("TapctABMILClassifier requires tapct_embedding input.")
        if not torch.is_tensor(embedding):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3:
            raise ValueError(
                "TAP-CT ABMIL expects shape (batch, dim) or "
                f"(batch, instances, dim), got {tuple(embedding.shape)}."
            )
        if int(embedding.shape[-1]) != self.tapct_embedding_dim:
            raise ValueError(
                "Unexpected TAP-CT embedding dimension: "
                f"got {embedding.shape[-1]}, expected {self.tapct_embedding_dim}."
            )
        if mask is not None:
            mask = mask.bool()
            if mask.shape != embedding.shape[:2]:
                raise ValueError(
                    "tapct_mask must match the batch and instance dimensions: "
                    f"got {tuple(mask.shape)}, expected {tuple(embedding.shape[:2])}."
                )
        return embedding.float(), mask

    def forward_features(self, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pooled bag features and attention weights."""
        embedding, mask = self._unpack_inputs(inputs)
        encoded = self.instance_encoder(embedding)
        attention_hidden = self.attention_v(encoded)
        if self.attention_u is not None:
            attention_hidden = attention_hidden * self.attention_u(encoded)
        scores = self.attention_score(attention_hidden).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def forward(self, inputs: Any) -> torch.Tensor:
        """Return classification logits."""
        pooled, _ = self.forward_features(inputs)
        return self.classifier(pooled)
