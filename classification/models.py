"""Model registry and factory for nnMamba classification."""

import torch.nn as nn

from networks.ssm_nnMamba import nnMambaEncoder
from networks.conv_Densenet121 import densenet121
from networks.tr_ViT import ViT
from networks.tr_crate import CRATE_small_3D


MODEL_REGISTRY = {
    "nnmamba": nnMambaEncoder,
    "densenet": lambda: densenet121(mode="classifier", drop_rate=0.05, num_classes=2),
    "vit": lambda: ViT(
        in_channels=1,
        img_size=(112, 136, 112),
        patch_size=(8, 8, 8),
        pos_embed="conv",
        classification=True,
    ),
    "crate": CRATE_small_3D,
}


def build_model(name: str, device=None) -> nn.Module:
    """Build model by name.

    Args:
        name: Model name (nnmamba, densenet, vit, crate)
        device: Target device

    Returns:
        Initialized model
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model = MODEL_REGISTRY[name]()

    if device:
        model.to(device)

    model.float()
    return model
