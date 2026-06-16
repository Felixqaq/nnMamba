"""Tests for OI emphysema binary classification with TAP-CT late fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import Config
from data.loader import RegressionLoaderHelper
from data.manifest import (
    build_angle_manifest,
    oi_emphysema_class_names,
    oi_emphysema_label,
)


# OI >= 3.0 is the most balanced cutpoint for this cohort.
EMPHYSEMA_NAME = "Significant emphysema (OI >= 3)"
NO_EMPHYSEMA_NAME = "No significant emphysema (OI < 3)"
GRAY_ZONE_CONFIG = "config.oi.emphysema.tapct_late_fusion.gray_zone.yaml"


def test_oi_emphysema_config_is_binary_classification() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.oi.emphysema.tapct_late_fusion.yaml")
    )

    assert config.data.target_mode == "oi_emphysema"
    assert config.is_classification_task() is True
    assert config.model.name == "hybrid_mamba_tapct_fusion"
    assert config.model.num_classes == 2
    assert config.model_output_dim() == 2
    assert config.data.oi_threshold == 3.0
    assert config.data.oi_json == Path("./oi_processed.json")
    assert config.data.tapct_features == Path("./embeddings/tapct_b_3d/features.npz")
    assert config.data.manifest == Path(
        "./datasets/generated/oi_emphysema_manifest.tapct_late_fusion.json"
    )
    assert config.task == "OI_emphysema_classification"


def test_oi_emphysema_label_lists_disease_class_first() -> None:
    names = oi_emphysema_class_names(3.0)
    assert names == [EMPHYSEMA_NAME, NO_EMPHYSEMA_NAME]

    # On/above the cutpoint -> emphysema is class 0 (disease-positive first).
    assert oi_emphysema_label(3.0, 3.0) == (0, EMPHYSEMA_NAME)
    assert oi_emphysema_label(7.05, 3.0) == (0, EMPHYSEMA_NAME)
    # Below the cutpoint -> no significant emphysema.
    assert oi_emphysema_label(2.99, 3.0) == (1, NO_EMPHYSEMA_NAME)
    assert oi_emphysema_label(1.64, 3.0) == (1, NO_EMPHYSEMA_NAME)


def test_oi_emphysema_manifest_splits_cohort_at_threshold() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    regression_root = repo_root / "regression"

    manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        pft_json=repo_root / "pft.json",
        target_mode="oi_emphysema",
        oi_json=regression_root / "oi_processed.json",
        oi_threshold=3.0,
    )

    assert manifest.class_names == [EMPHYSEMA_NAME, NO_EMPHYSEMA_NAME]
    assert manifest.counts["total"] == 66
    # OI >= 3.0 is the most balanced cutpoint: 34 emphysema vs 32 not.
    assert manifest.class_counts == {EMPHYSEMA_NAME: 34, NO_EMPHYSEMA_NAME: 32}

    for record in manifest.records:
        assert record.class_index in (0, 1)
        if record.oi >= 3.0:
            assert record.class_index == 0
        else:
            assert record.class_index == 1


def test_oi_emphysema_loader_uses_class_targets_and_embeddings() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.oi.emphysema.tapct_late_fusion.yaml")
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        oi_threshold=config.data.oi_threshold,
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
    assert loader.get_class_names() == [EMPHYSEMA_NAME, NO_EMPHYSEMA_NAME]
    assert loader.targets.dtype == np.int64
    assert set(np.unique(loader.targets).tolist()) == {0, 1}
    assert int((loader.targets == 0).sum()) == 34
    assert int((loader.targets == 1).sum()) == 32
    assert "classification" in loader.split_strategy

    # Every fold's validation split must hold out whole patients, no leakage.
    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[i] for i in train_idx}
        val_patients = {loader.patient_ids[i] for i in val_idx}
        assert not (train_patients & val_patients)

    batch = next(iter(loader.get_train_dl(0)))
    assert batch["target"].dtype.is_floating_point is False
    assert batch["tapct_embedding"].shape[-1] == 2304


def test_oi_emphysema_gray_zone_config_excludes_borderline_oi() -> None:
    config = Config.from_yaml(Path(__file__).with_name(GRAY_ZONE_CONFIG))

    assert config.data.target_mode == "oi_emphysema"
    assert config.is_classification_task() is True
    assert config.data.oi_threshold == 3.0
    assert config.data.oi_exclude_range == (2.5, 3.5)
    assert config.data.manifest == Path(
        "./datasets/generated/oi_emphysema_manifest.tapct_late_fusion.gray_zone.json"
    )


def test_oi_emphysema_gray_zone_manifest_drops_borderline_cases() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    regression_root = repo_root / "regression"

    manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        pft_json=repo_root / "pft.json",
        target_mode="oi_emphysema",
        oi_json=regression_root / "oi_processed.json",
        oi_threshold=3.0,
        oi_exclude_range=(2.5, 3.5),
    )

    assert manifest.class_names == [EMPHYSEMA_NAME, NO_EMPHYSEMA_NAME]
    assert manifest.counts["total"] == 50
    assert manifest.counts["excluded_oi_range"] == 16
    assert manifest.class_counts == {EMPHYSEMA_NAME: 29, NO_EMPHYSEMA_NAME: 21}
    assert all(not (2.5 <= float(record.oi) < 3.5) for record in manifest.records)


def test_oi_emphysema_gray_zone_loader_uses_filtered_targets() -> None:
    config = Config.from_yaml(Path(__file__).with_name(GRAY_ZONE_CONFIG))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        oi_json=regression_root / config.data.oi_json,
        oi_threshold=config.data.oi_threshold,
        oi_exclude_range=config.data.oi_exclude_range,
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

    assert len(loader.records) == 50
    assert loader.manifest.counts["excluded_oi_range"] == 16
    assert int((loader.targets == 0).sum()) == 29
    assert int((loader.targets == 1).sum()) == 21
    assert all(not (2.5 <= float(record.oi) < 3.5) for record in loader.records)
    assert "classification" in loader.split_strategy

    batch = next(iter(loader.get_train_dl(0)))
    assert batch["target"].dtype.is_floating_point is False
    assert batch["tapct_embedding"].shape[-1] == 2304
