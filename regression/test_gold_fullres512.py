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


def test_gold_severity4_tapct_late_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.gold.severity4.tapct_late_fusion.augmentation13.yaml"
        )
    )

    assert config.data.target_mode == "gold_severity4"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 4
    assert config.model_output_dim() == 4
    assert config.training.k_folds == 5
    assert config.training.batch_size == 12
    assert config.training.swin_batch_size == 5
    assert config.training.loss == "focal"
    assert config.training.focal_gamma == 1.0
    assert config.training.class_weight_mode == "none"
    assert config.data.image_size == (112, 136, 112)
    assert config.data.cache_data is True
    assert config.data.gold_exclude_class_indices == ()
    assert config.data.gold_remap_class_indices is False
    assert config.data.reuse_underrepresented_classes_in_folds is True
    assert config.data.augmentation.target_per_class == 13
    assert config.data.augmentation.class_indices == (0, 1, 2, 3)
    assert config.data.manifest == Path(
        "./datasets/generated/gold_severity4_2026_manifest.tapct_aug13.json"
    )
    assert config.task == "GOLD_severity4_classification"
