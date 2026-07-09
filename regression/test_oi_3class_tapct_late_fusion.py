"""Tests for OI three-class severity classification with TAP-CT late fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import Config
from data.loader import RegressionLoaderHelper
from data.manifest import oi_3class_class_names, oi_3class_label


CONFIG = "config.oi.3class.tapct_late_fusion.augmentationX5.yaml"
NO_NAME = "No significant emphysema (OI < 3)"
MODERATE_NAME = "Moderate emphysema (3 <= OI < 7)"
SEVERE_NAME = "Severe emphysema (OI >= 7)"


def test_oi_3class_label_uses_oi_cutpoints_only() -> None:
    assert oi_3class_class_names((3.0, 7.0)) == [NO_NAME, MODERATE_NAME, SEVERE_NAME]
    for oi, expected in [
        (1.61, 0),
        (2.99, 0),
        (3.0, 1),
        (6.99, 1),
        (7.0, 2),
        (26.89, 2),
    ]:
        index, _ = oi_3class_label(oi, (3.0, 7.0))
        assert index == expected, (oi, index, expected)


def test_oi_3class_config_is_three_class_classification() -> None:
    config = Config.from_yaml(Path(__file__).with_name(CONFIG))

    assert config.data.target_mode == "oi_3class"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 3
    assert config.model_output_dim() == 3
    assert config.data.oi_thresholds == (3.0, 7.0)
    assert config.task == "OI_3class_classification"
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.views_per_sample == 5
    assert config.data.augmentation.balance_then_augment is True


def test_oi_3class_loader_distribution_and_no_leakage() -> None:
    config = Config.from_yaml(Path(__file__).with_name(CONFIG))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        oi_thresholds=config.data.oi_thresholds,
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
        load_ct_data=False,
    )

    assert len(loader.records) == 66
    assert loader.tapct_embedding_dim == 2304
    assert loader.get_class_names() == [NO_NAME, MODERATE_NAME, SEVERE_NAME]
    assert loader.targets.dtype == np.int64
    assert set(np.unique(loader.targets).tolist()) == {0, 1, 2}
    assert int((loader.targets == 0).sum()) == 32
    assert int((loader.targets == 1).sum()) == 17
    assert int((loader.targets == 2).sum()) == 17
    assert "classification" in loader.split_strategy

    # Every fold's validation split must hold out whole patients, no leakage.
    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[i] for i in train_idx}
        val_patients = {loader.patient_ids[i] for i in val_idx}
        assert not (train_patients & val_patients)


def test_oi_3class_train_loader_expands_5x_views() -> None:
    config = Config.from_yaml(Path(__file__).with_name(CONFIG))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        oi_thresholds=config.data.oi_thresholds,
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
        load_ct_data=False,
    )

    assert loader._should_balance_then_augment() is True
    assert loader._views_per_sample() == 5

    base_train, _ = loader.fold_indices[0]
    expanded, flags = loader._build_balanced_view_train_indices(list(base_train))
    assert len(expanded) == len(base_train) * 5
    # Exactly one original + four augmented views per base sample.
    assert sum(flags) == len(base_train) * 4
