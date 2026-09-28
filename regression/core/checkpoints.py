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


# Auxiliary heads are trained only when a run supplies their targets, and no
# inference path reads them, so a checkpoint saved before they existed is still
# a complete model. Nothing else may be missing.
_OPTIONAL_PREFIXES = ("aux_pft_heads.", "aux_emphysema_head.")


def load_model_weights(model, state_dict, source="") -> None:
    """Strict load, except for auxiliary heads the inference path never reads."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unknown = [k for k in missing if not k.startswith(_OPTIONAL_PREFIXES)]
    if unknown or unexpected:
        where = f" in {source}" if source else ""
        raise SystemExit(
            f"checkpoint{where} does not match {type(model).__name__}: "
            f"{len(unknown)} unexpectedly missing key(s) {unknown[:5]}, "
            f"{len(unexpected)} unexpected key(s) {list(unexpected)[:5]}")
    if missing:
        print(f"note: {len(missing)} auxiliary-head weight(s) absent from the "
              f"checkpoint{' ' + source if source else ''}; they are left at their "
              f"initial values and no prediction reads them", flush=True)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    """Load model weights and return checkpoint metadata."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    load_model_weights(model, state_dict, str(path))
    model.to(device)
    return checkpoint
