"""Tests for hard majority ensemble configuration and voting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from core.config import Config
from core.ensemble import majority_vote_binary
from core.evaluator import compute_classification_metrics


def test_tapct_late_fusion_majority_ensemble_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_binary_extreme.tapct_late_fusion.majority_ensemble.yaml"
        )
    )

    assert config.ensemble.enabled is True
    assert config.ensemble.voting == "majority"
    assert config.ensemble.member_count == 7
    assert config.ensemble.vote_sizes == (3, 5, 7)
    assert config.ensemble.split_seed == 42
    assert config.ensemble.member_seeds == (42, 43, 44, 45, 46, 47, 48)
    assert config.ensemble.existing_member_uuids == (
        "hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25",
    )
    assert config.split_seed() == 42
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.data.target_mode == "angle_binary_extreme"


def test_ensemble_split_seed_stays_fixed_when_member_seed_changes() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_binary_extreme.tapct_late_fusion.majority_ensemble.yaml"
        )
    )
    changed_member_seed = replace(
        config,
        training=replace(config.training, seed=99),
    )

    assert changed_member_seed.split_seed() == 42

    disabled_ensemble = replace(
        changed_member_seed,
        ensemble=replace(changed_member_seed.ensemble, enabled=False),
    )
    assert disabled_ensemble.split_seed() == 99


def test_majority_vote_binary_uses_hard_predictions() -> None:
    member_preds = torch.tensor(
        [
            [0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ]
    )

    preds, probs = majority_vote_binary(member_preds, vote_size=3)
    metrics = compute_classification_metrics(
        labels=torch.tensor([1, 1, 0, 0, 0]),
        preds=preds,
        probs=probs,
        num_classes=2,
    )

    assert preds.tolist() == [1, 1, 0, 0, 1]
    assert torch.allclose(
        probs[:, 1],
        torch.tensor([2 / 3, 2 / 3, 1 / 3, 0.0, 1.0]),
    )
    assert metrics.confusion_matrix == [[2, 1], [0, 2]]
    assert metrics.accuracy == 0.8
