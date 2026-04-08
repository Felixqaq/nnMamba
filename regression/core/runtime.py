"""PyTorch runtime tuning helpers for regression experiments."""

from __future__ import annotations

import os

import torch


def configure_torch_runtime() -> None:
    """Enable safe CUDA-side performance features for fixed-shape training."""
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
