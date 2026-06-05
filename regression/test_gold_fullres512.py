"""Tests for additive GOLD 512x512 in-plane configs."""

from __future__ import annotations

from pathlib import Path

from core.config import Config


def test_gold_fullres512_config_is_additive() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.fullres512.yaml"))

    assert config.data.target_mode == "gold"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_attention"
    assert config.model.num_classes == 5
    assert config.data.image_size == (512, 512, 112)
    assert config.data.cache_data is False
    assert config.training.batch_size == 1
    assert config.training.swin_batch_size == 1
    assert config.data.manifest == Path(
        "./datasets/generated/gold_2026_manifest.fullres512.json"
    )
    assert config.task == "GOLD_stage_classification"


def test_gold_tapct_late_fusion_fullres512_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.gold.tapct_late_fusion.fullres512.yaml")
    )

    assert config.data.target_mode == "gold"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 5
    assert config.model.tapct_embedding_dim == 2304
    assert config.model.fusion_projection_dim == 128
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.image_size == (512, 512, 112)
    assert config.data.cache_data is False
    assert config.training.batch_size == 1
    assert config.training.swin_batch_size == 1
    assert config.data.augmentation.target_per_class == 36
    assert config.data.augmentation.class_indices == (0, 1, 2, 3, 4)
    assert config.task == "GOLD_stage_classification"
