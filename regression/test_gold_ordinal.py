"""Tests for GOLD ordinal classification support."""

from pathlib import Path

import torch

from core.config import Config
from core.evaluator import decode_ordinal_logits, encode_ordinal_targets
from core.trainer import build_loss
from models import build_model


def test_gold_ordinal_config_uses_three_threshold_logits() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.gold.ordinal.balanced_sampling.augmentation36.yaml")
    )
    model = build_model(config.model, output_dim=config.model_output_dim())

    assert config.is_ordinal_classification() is True
    assert config.model.num_classes == 5
    assert config.model_output_dim() == 4
    assert config.training.classification_mode == "ordinal"
    assert model.head[-1].out_features == 4


def test_ordinal_targets_and_logits_round_trip_to_gold_classes() -> None:
    targets = encode_ordinal_targets(torch.tensor([0, 1, 2, 3, 4]), num_classes=5)
    logits = torch.tensor(
        [
            [-4.0, -5.0, -6.0, -7.0],
            [4.0, -4.0, -5.0, -6.0],
            [4.0, 3.0, -4.0, -5.0],
            [5.0, 4.0, 3.0, -4.0],
            [6.0, 5.0, 4.0, 3.0],
        ]
    )
    preds, probs = decode_ordinal_logits(logits, num_classes=5)

    assert targets.tolist() == [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert preds.tolist() == [0, 1, 2, 3, 4]
    assert probs.shape == (5, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5))


def test_auto_ordinal_loss_uses_binary_threshold_targets() -> None:
    loss = build_loss("auto", is_classification=True, classification_mode="ordinal")
    logits = torch.zeros(2, 4)
    targets = encode_ordinal_targets(torch.tensor([0, 4]), num_classes=5)

    value = loss(logits, targets)

    assert isinstance(loss, torch.nn.BCEWithLogitsLoss)
    assert torch.isfinite(value)
