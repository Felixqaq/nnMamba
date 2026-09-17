"""Existing hybrid backbone with a binary head and feature distillation interface."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from regression.networks.hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor


def build() -> torch.nn.Module:
    return HybridMambaAttentionRegressor(
        in_channels=1, num_classes=1, base_channels=32, depths=(3, 3, 3),
        head_hidden_dim=256, dropout=0.3, attn_heads=8, attn_layers=1,
        attn_mlp_ratio=2.0, attn_dropout=0.1)


def forward(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.forward_features(x)
    return model.head(features).flatten(), features
