"""Smoke tests for the hybrid Mamba-attention regressor."""

from __future__ import annotations

import torch

from core.config import ModelConfig
from models import build_model


def _build_hybrid_model():
    cfg = ModelConfig(
        name="hybrid_mamba_attention",
        in_channels=1,
        num_classes=1,
        base_channels=16,
        blocks=1,
        hidden_dim=128,
        dropout=0.1,
        attn_heads=4,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
    )
    return build_model(cfg)


def test_hybrid_regressor_builds() -> None:
    model = _build_hybrid_model()
    assert type(model).__name__ == "HybridMambaAttentionRegressor"
    assert model.attention_layers[0].attn.num_heads == 4


def test_hybrid_regressor_forward_pass() -> None:
    if not torch.cuda.is_available():
        return

    model = _build_hybrid_model().cuda()
    x = torch.randn(2, 1, 64, 64, 64)
    y = model(x.cuda())

    assert y.shape == (2,)
    assert torch.isfinite(y).all()


def test_hybrid_classifier_head_respects_num_classes() -> None:
    cfg = ModelConfig(
        name="hybrid_mamba_attention",
        in_channels=1,
        num_classes=4,
        base_channels=16,
        blocks=1,
        hidden_dim=128,
        dropout=0.1,
        attn_heads=4,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
    )
    model = build_model(cfg)
    assert model.head[-1].out_features == 4
