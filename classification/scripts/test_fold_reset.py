#!/usr/bin/env python
"""Regression check for independent classification CV folds."""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

CLASSIFICATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CLASSIFICATION_ROOT))

from core.trainer import Trainer  # noqa: E402


def main() -> None:
    config = SimpleNamespace(
        gpu_device_id="0",
        training=SimpleNamespace(seed=42),
        resume=SimpleNamespace(enabled=False, uuid=None),
    )
    model = nn.Linear(2, 1)
    trainer = Trainer(config, model, loader_helper=object())
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    with torch.no_grad():
        model.weight.add_(10.0)
        model.bias.sub_(3.0)

    trainer._reset_model_for_fold()

    for name, tensor in model.state_dict().items():
        expected = initial_state[name]
        actual = tensor.detach().cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(f"{name} was not reset before the fold.")

    print("PASS: classification Trainer resets model weights before each fold.")


if __name__ == "__main__":
    main()
