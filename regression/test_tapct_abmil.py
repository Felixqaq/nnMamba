"""Tests for TAP-CT ABMIL scan-level classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.config import Config, ModelConfig
from data.loader import RegressionLoaderHelper, _collate_angle_batch
from models import build_model


def test_tapct_abmil_config_is_embedding_only_classification() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.angle_3class.tapct_abmil.yaml"))

    assert config.is_classification_task() is True
    assert config.model.name == "tapct_abmil"
    assert config.model.tapct_embedding_dim == 1152
    assert config.model.tapct_attention_dim == 128
    assert config.data.tapct_features == Path("./embeddings/tapct_s_3d/features.npz")
    assert config.data.tapct_feature_key == "features"
    assert config.data.load_ct is False
    assert config.task == "Angle_3class_classification"


def test_tapct_abmil_loader_reads_pooled_embeddings_without_ct() -> None:
    regression_root = Path(__file__).resolve().parent
    repo_root = regression_root.parent
    config = Config.from_yaml(regression_root / "config.angle_3class.tapct_abmil.yaml")

    loader = RegressionLoaderHelper(
        repo_root / "by_angle_all",
        labels_json=repo_root / "patient_angle_classification_by_group.json",
        pft_json=repo_root / "pft.json",
        target_mode=config.data.target_mode,
        image_size=config.data.image_size,
        k_folds=2,
        seed=config.training.seed,
        batch_size=4,
        val_batch_size=4,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        tapct_features=regression_root / config.data.tapct_features,
        tapct_feature_key=config.data.tapct_feature_key,
        load_ct_data=config.data.load_ct,
    )

    sample = loader.dataset[0]
    assert loader.tapct_embedding_dim == 1152
    assert sample["ct"].shape == (1, 1, 1, 1)
    assert sample["tapct_embedding"].shape == (1152,)


def test_tapct_abmil_model_builds_and_accepts_single_or_bag_inputs() -> None:
    cfg = ModelConfig(
        name="tapct_abmil",
        num_classes=3,
        hidden_dim=32,
        dropout=0.1,
        tapct_embedding_dim=16,
        tapct_attention_dim=8,
    )
    model = build_model(cfg)

    pooled_logits = model({"tapct_embedding": torch.ones(2, 16)})
    bag_logits = model(
        {
            "tapct_embedding": torch.ones(2, 4, 16),
            "tapct_mask": torch.tensor(
                [[True, True, False, False], [True, True, True, True]]
            ),
        }
    )

    assert pooled_logits.shape == (2, 3)
    assert bag_logits.shape == (2, 3)
    assert torch.isfinite(pooled_logits).all()
    assert torch.isfinite(bag_logits).all()


def test_tapct_abmil_collate_pads_variable_instance_bags() -> None:
    samples = [
        {
            "ct": torch.zeros(1, 1, 1, 1),
            "target": torch.tensor(0),
            "label": torch.tensor(0),
            "angle": torch.tensor(120.0),
            "tapct_embedding": torch.ones(2, 4),
        },
        {
            "ct": torch.zeros(1, 1, 1, 1),
            "target": torch.tensor(1),
            "label": torch.tensor(1),
            "angle": torch.tensor(140.0),
            "tapct_embedding": np.ones((3, 4), dtype=np.float32),
        },
    ]
    samples[1]["tapct_embedding"] = torch.as_tensor(samples[1]["tapct_embedding"])

    batch = _collate_angle_batch(samples)

    assert batch["tapct_embedding"].shape == (2, 3, 4)
    assert batch["tapct_mask"].tolist() == [
        [True, True, False],
        [True, True, True],
    ]
