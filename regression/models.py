"""Model registry and factory for CT angle regression."""

from __future__ import annotations

import torch.nn as nn

from networks.hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor
from networks.mamba_regressor import MambaAngleRegressor
from networks.swinunetr_v2_regressor import SwinUNETRV2AngleRegressor


MODEL_REGISTRY = {
    "hybrid": HybridMambaAttentionRegressor,
    "hybrid_mamba_attention": HybridMambaAttentionRegressor,
    "hybrid_mamba_attention_regressor": HybridMambaAttentionRegressor,
    "mamba": MambaAngleRegressor,
    "mamba_hybrid": HybridMambaAttentionRegressor,
    "nnmamba": MambaAngleRegressor,
    "nnmamba_regressor": MambaAngleRegressor,
    "swinunetr": SwinUNETRV2AngleRegressor,
    "swinunetr_v2": SwinUNETRV2AngleRegressor,
    "swinunetr_v2_regressor": SwinUNETRV2AngleRegressor,
    "swinunetrv2": SwinUNETRV2AngleRegressor,
    "swinunet_v2": SwinUNETRV2AngleRegressor,
    "swinunetv2": SwinUNETRV2AngleRegressor,
}


def build_model(model_config, device=None) -> nn.Module:
    """Build a regression model by name."""
    if isinstance(model_config, str):
        key = model_config.lower()
        kwargs = {}
    else:
        key = str(model_config.name).lower()
        if key in {"mamba", "nnmamba", "nnmamba_regressor"}:
            kwargs = {
                "in_channels": int(model_config.in_channels),
                "base_channels": int(model_config.base_channels),
                "depths": tuple([int(model_config.blocks)] * 3),
                "dropout": float(model_config.dropout),
            }
        elif key in {
            "hybrid",
            "hybrid_mamba_attention",
            "hybrid_mamba_attention_regressor",
            "mamba_hybrid",
        }:
            kwargs = {
                "in_channels": int(model_config.in_channels),
                "base_channels": int(model_config.base_channels),
                "depths": tuple([int(model_config.blocks)] * 3),
                "head_hidden_dim": int(model_config.hidden_dim),
                "dropout": float(model_config.dropout),
                "attn_heads": int(model_config.attn_heads),
                "attn_layers": int(model_config.attn_layers),
                "attn_mlp_ratio": float(model_config.attn_mlp_ratio),
                "attn_dropout": float(model_config.attn_dropout),
            }
        else:
            window_size = model_config.window_size
            if isinstance(window_size, (list, tuple)):
                window_size = tuple(int(size) for size in window_size)
            patch_size = model_config.patch_size
            if isinstance(patch_size, (list, tuple)):
                patch_size = tuple(int(size) for size in patch_size)
            kwargs = {
                "in_channels": int(model_config.in_channels),
                "feature_size": int(model_config.feature_size),
                "head_hidden_dim": int(model_config.hidden_dim),
                "dropout": float(model_config.dropout),
                "depths": tuple(int(depth) for depth in model_config.depths),
                "num_heads": tuple(int(heads) for heads in model_config.num_heads),
                "window_size": window_size,
                "patch_size": patch_size,
                "use_checkpoint": bool(model_config.use_checkpoint),
                "use_v2": bool(model_config.use_v2),
            }

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {key}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[key](**kwargs)
    if device is not None:
        model.to(device)
    return model.float()
