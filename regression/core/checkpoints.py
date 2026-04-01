"""Checkpoint save/load utilities for regression experiments."""

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def generate_uuid(model_name: str = "nnMambaReg") -> str:
    """Generate a unique run identifier with timestamp."""
    return f"{model_name}_{datetime.now():%Y-%m-%d_%H:%M:%S}"


def _checkpoint_name(fold: int, epoch: int | None = None, is_best: bool = False) -> str:
    if is_best:
        return f"fold{fold}_best_weight.pth"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"fold{fold}_epoch{epoch}_weights-{timestamp}.pth"


def save_checkpoint(
    model: nn.Module,
    path: Path,
    fold: int,
    epoch: int | None = None,
    is_best: bool = False,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a model checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    save_path = path / _checkpoint_name(fold=fold, epoch=epoch, is_best=is_best)

    payload = {"state_dict": model.state_dict()}
    if extra:
        payload.update(extra)

    torch.save(payload, save_path)
    return save_path


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    return checkpoint
