"""Checkpoint save/load utilities."""

from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn


def generate_uuid(model_name: str = "nnMamba") -> str:
    """Generate unique model identifier with timestamp."""
    return f"{model_name}_{datetime.now():%Y-%m-%d_%H:%M:%S}"


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int | None = None,
    fold: int | None = None,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint.

    Args:
        model: PyTorch model to save
        path: Base directory for checkpoints
        epoch: Current epoch (for periodic saves)
        fold: Current fold number
        is_best: If True, save as best_weight.pth

    Returns:
        Path to saved checkpoint
    """
    path.mkdir(parents=True, exist_ok=True)

    if is_best:
        save_path = path / "best_weight.pth"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_path = path / f"fold_{fold}_epoch{epoch}_weights-{timestamp}.pth"

    torch.save(model.state_dict(), save_path)
    return save_path


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> nn.Module:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model instance to load weights into
        device: Target device

    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model
