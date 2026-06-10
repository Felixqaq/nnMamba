"""Tests for GOLD ordinal classification support."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from core.config import Config
from core.evaluator import (
    ClassificationMetrics,
    decode_ordinal_logits,
    encode_ordinal_targets,
    evaluate,
)
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


def test_gold_severity4_ordinal_tapct_config_uses_three_threshold_logits() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.gold.severity4.ordinal.tapct_late_fusion.augmentation13.yaml"
        )
    )

    assert config.is_ordinal_classification() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 4
    assert config.model_output_dim() == 3
    assert config.training.classification_mode == "ordinal"
    assert config.data.target_mode == "gold_severity4"
    assert config.training.k_folds == 5
    assert config.data.gold_exclude_class_indices == ()
    assert config.data.gold_remap_class_indices is False
    assert config.data.reuse_underrepresented_classes_in_folds is True
    assert config.data.augmentation.target_per_class == 13
    assert config.data.augmentation.class_indices == (0, 1, 2, 3)
    assert config.data.manifest == Path(
        "./datasets/generated/gold_severity4_2026_manifest.ordinal_tapct_aug13.json"
    )


def test_gold_severity4_ordinal_evaluation_uses_classification_metrics() -> None:
    class DummyOrdinalModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 3)

    samples = [
        {
            "ct": torch.zeros(1, 1, 1, 1),
            "label": torch.tensor(class_idx, dtype=torch.long),
            "target": torch.tensor(class_idx, dtype=torch.long),
        }
        for class_idx in [0, 1, 2, 3]
    ]
    dataloader = DataLoader(samples, batch_size=2)

    metrics = evaluate(
        model=DummyOrdinalModel(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        task_type="gold_severity4",
        num_classes=4,
        classification_mode="ordinal",
    )

    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.labels is not None
    assert metrics.preds is not None
    assert metrics.probs is not None
    assert metrics.labels.numel() == 4
    assert metrics.preds.numel() == 4
    assert metrics.probs.shape == (4, 4)


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
