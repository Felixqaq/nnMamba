"""Tests for TAP-CT late-fusion training support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.config import Config, ModelConfig
from data.loader import RegressionLoaderHelper
from data.transforms import ToTensor
from models import build_model


def test_angle_3class_tapct_late_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.tapct_late_fusion.yaml")
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 3
    assert config.model.tapct_embedding_dim == 2304
    assert config.model.fusion_projection_dim == 128
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.augmentation.target_per_class == 300
    assert config.task == "Angle_3class_classification"


def test_angle_3class_tapct_abmil_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.tapct_abmil_fusion.augmentation100.yaml")
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_abmil_fusion"
    assert config.model.num_classes == 3
    assert config.model.tapct_embedding_dim == 2304
    assert config.model.tapct_attention_dim == 128
    assert config.model.tapct_gated_attention is True
    assert config.model.fusion_projection_dim == 128
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.augmentation.target_per_class == 100
    assert config.task == "Angle_3class_classification"


def test_angle_3class_tapct_attention_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.tapct_attention_fusion.augmentation100.yaml")
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_attention_fusion"
    assert config.model.num_classes == 3
    assert config.model.tapct_embedding_dim == 2304
    assert config.model.tapct_attention_dim == 128
    assert config.model.tapct_gated_attention is True
    assert config.model.fusion_projection_dim == 128
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.augmentation.target_per_class == 100
    assert config.task == "Angle_3class_classification"


def test_angle_binary_extreme_tapct_late_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_binary_extreme.tapct_late_fusion.yaml")
    )

    assert config.data.target_mode == "angle_binary_extreme"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 2
    assert config.model.tapct_embedding_dim == 1152
    assert config.data.tapct_features == Path("./embeddings/tapct_s_3d/features.npz")
    assert config.data.augmentation.target_per_class == 100
    assert config.task == "Angle_extreme_binary_classification"


def test_gold_tapct_late_fusion_config() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.gold.tapct_late_fusion.augmentation36.yaml")
    )

    assert config.data.target_mode == "gold"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 5
    assert config.model.tapct_embedding_dim == 2304
    assert config.data.pft_json == Path("./GOLD_2026_classification.json")
    assert config.data.manifest == Path(
        "./datasets/generated/gold_2026_manifest.tapct_aug36.json"
    )
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.augmentation.target_per_class == 36
    assert config.data.augmentation.class_indices == (0, 1, 2, 3, 4)
    assert config.task == "GOLD_stage_classification"


def test_tapct_late_fusion_loader_reads_patient_embeddings() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.tapct_late_fusion.yaml")
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=None,
        target_mode=config.data.target_mode,
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=4,
        val_batch_size=4,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        intensity_window=config.data.intensity_window,
        input_normalization=config.data.input_normalization,
        augmentation_config=config.data.augmentation,
        balanced_sampling=config.data.balanced_sampling,
        tapct_features=regression_root / config.data.tapct_features,
    )

    assert loader.tapct_embedding_dim == 2304
    assert len(loader.tapct_embeddings) == 66
    first_patient = loader.records[0].patient_id
    assert loader.dataset.tapct_embeddings[first_patient].shape == (2304,)


def test_to_tensor_preserves_tapct_embedding() -> None:
    sample = {
        "ct": np.zeros((1, 8, 8, 8), dtype=np.float32),
        "target": np.array(1, dtype=np.int64),
        "angle": np.array(152.0, dtype=np.float32),
        "label": np.array(1, dtype=np.int64),
        "tapct_embedding": np.ones(2304, dtype=np.float32),
    }

    output = ToTensor()(sample)

    assert output["tapct_embedding"].shape == (2304,)
    assert output["tapct_embedding"].dtype == torch.float32
    assert output["label"].item() == 1


def test_hybrid_mamba_tapct_fusion_model_builds() -> None:
    cfg = ModelConfig(
        name="hybrid_mamba_tapct_fusion",
        in_channels=1,
        num_classes=3,
        base_channels=16,
        blocks=1,
        hidden_dim=128,
        dropout=0.1,
        attn_heads=4,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
        tapct_embedding_dim=2304,
        fusion_projection_dim=64,
        fusion_dropout=0.1,
    )

    model = build_model(cfg)

    assert type(model).__name__ == "HybridMambaTapctFusionRegressor"
    assert model.tapct_embedding_dim == 2304
    assert model.fusion_projection_dim == 64
    assert model.head[-1].out_features == 3


def test_hybrid_mamba_tapct_abmil_fusion_model_forwards() -> None:
    cfg = ModelConfig(
        name="hybrid_mamba_tapct_abmil_fusion",
        in_channels=1,
        num_classes=3,
        base_channels=8,
        blocks=1,
        hidden_dim=32,
        dropout=0.1,
        attn_heads=2,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
        tapct_embedding_dim=16,
        tapct_attention_dim=8,
        fusion_projection_dim=12,
        fusion_dropout=0.1,
    )

    model = build_model(cfg).eval()
    image_feature_dim = int(model.image_branch[0].normalized_shape[0])
    model.image_encoder.forward_features = lambda ct: torch.zeros(
        ct.shape[0],
        image_feature_dim,
        dtype=ct.dtype,
        device=ct.device,
    )
    batch = {
        "ct": torch.zeros(2, 1, 16, 16, 16),
        "tapct_embedding": torch.ones(2, 16),
    }

    with torch.no_grad():
        logits = model(batch)
        features = model.forward_features(batch)
        weights = model.forward_attention_weights(batch)

    assert type(model).__name__ == "HybridMambaTapctABMILFusionRegressor"
    assert logits.shape == (2, 3)
    assert features.shape == (2, 12)
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(logits).all()


def test_hybrid_mamba_tapct_attention_fusion_model_forwards() -> None:
    cfg = ModelConfig(
        name="hybrid_mamba_tapct_attention_fusion",
        in_channels=1,
        num_classes=3,
        base_channels=8,
        blocks=1,
        hidden_dim=32,
        dropout=0.1,
        attn_heads=2,
        attn_layers=1,
        attn_mlp_ratio=2.0,
        attn_dropout=0.1,
        tapct_embedding_dim=16,
        tapct_attention_dim=8,
        fusion_projection_dim=12,
        fusion_dropout=0.1,
    )

    model = build_model(cfg).eval()
    model.image_encoder.forward_features = lambda ct: torch.zeros(
        ct.shape[0],
        model.image_feature_dim,
        dtype=ct.dtype,
        device=ct.device,
    )
    batch = {
        "ct": torch.zeros(2, 1, 16, 16, 16),
        "tapct_embedding": torch.ones(2, 16),
    }

    with torch.no_grad():
        logits = model(batch)
        features = model.forward_features(batch)
        weights = model.forward_attention_weights(batch)

    assert type(model).__name__ == "HybridMambaTapctAttentionFusionRegressor"
    assert logits.shape == (2, 3)
    assert features.shape == (2, model.feature_dim)
    assert weights.shape == (2, 2)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(logits).all()
