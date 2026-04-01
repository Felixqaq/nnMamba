"""Model registry and factory for CT angle regression."""

from __future__ import annotations

import torch.nn as nn

from networks.mamba_regressor import MambaAngleRegressor


MODEL_REGISTRY = {
    "mamba": MambaAngleRegressor,
    "nnmamba": MambaAngleRegressor,
    "nnmamba_regressor": MambaAngleRegressor,
}


def build_model(model_config, device=None) -> nn.Module:
    """Build a regression model by name."""
    if isinstance(model_config, str):
        key = model_config.lower()
        kwargs = {}
    else:
        key = str(model_config.name).lower()
        depths = tuple([int(model_config.blocks)] * 3)
        kwargs = {
            "in_channels": int(model_config.in_channels),
            "base_channels": int(model_config.base_channels),
            "depths": depths,
            "dropout": float(model_config.dropout),
        }

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {key}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[key](**kwargs)
    if device is not None:
        model.to(device)
    return model.float()
