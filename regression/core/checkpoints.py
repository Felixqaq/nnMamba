"""Checkpoint save/load utilities for regression experiments."""

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn as nn


def _slugify_run_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "run"


def generate_uuid(
    model_name: str = "nnMambaReg",
    experiment_name: str | None = None,
) -> str:
    """Generate a unique run identifier with timestamp."""
    parts = [_slugify_run_part(model_name)]
    if experiment_name:
        parts.append(_slugify_run_part(experiment_name))
    return f"{'_'.join(parts)}_{datetime.now():%Y-%m-%d_%H:%M:%S}"


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
