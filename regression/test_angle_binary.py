"""Tests for gray-zone excluded two-class angle classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from core.config import Config
from data.loader import AugmentedSubset, BalancedClassSampler, RegressionLoaderHelper
from data.manifest import ANGLE_BINARY_EXTREME_NAMES, build_angle_manifest


def test_angle_binary_extreme_config_uses_gray_zone_exclusion() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_binary_extreme.yaml")
    )

    assert config.data.target_mode == "angle_binary_extreme"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 2
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_binary_extreme_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is False
    assert config.data.augmentation.enabled is False
    assert config.experiment.name == "Angle extreme binary baseline"
    assert config.task == "Angle_extreme_binary_classification"


def test_angle_binary_extreme_aug100_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_binary_extreme.balanced_sampling.augmentation100.yaml"
        )
    )

    assert config.data.target_mode == "angle_binary_extreme"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 2
    assert config.data.balanced_sampling is True
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.target_per_class == 100
    assert config.data.augmentation.class_indices == (0, 1)
    assert config.experiment.name == "Extreme binary + balanced aug100/class"
    assert config.task == "Angle_extreme_binary_classification"


def test_angle_binary_extreme_manifest_drops_gray_zone() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    regression_manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        target_mode="angle",
    )
    extreme_manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        target_mode="angle_binary_extreme",
    )

    assert extreme_manifest.class_names == ANGLE_BINARY_EXTREME_NAMES
    assert extreme_manifest.class_counts == {
        "Abnormal/emphysema-like (AC <=131 deg)": 14,
        "Normal-like (AC >=152 deg)": 47,
    }
    assert extreme_manifest.counts["total"] == 61
    assert extreme_manifest.counts["unique_patients"] == 61

    gray_zone_records = [
        record for record in regression_manifest.records if 131.0 < record.angle < 152.0
    ]
    assert len(gray_zone_records) == 5
    assert all(
        record.angle <= 131.0 or record.angle >= 152.0
        for record in extreme_manifest.records
    )
    assert all(record.class_index is not None for record in extreme_manifest.records)


def test_angle_binary_extreme_loader_stratifies_without_balanced_sampler() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_binary_extreme.yaml")
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
    )

    assert loader.get_class_names() == ANGLE_BINARY_EXTREME_NAMES
    assert np.bincount(loader.targets, minlength=2).tolist() == [14, 47]
    assert len(loader.records) == 61
    assert len(set(loader.patient_ids)) == 61
    assert loader.split_strategy == "patient_stratified_classification"

    train_dl = loader.get_train_dl(0)
    assert isinstance(train_dl.dataset, Subset)
    assert not isinstance(train_dl.sampler, BalancedClassSampler)

    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[index] for index in train_idx}
        val_patients = {loader.patient_ids[index] for index in val_idx}
        assert train_patients.isdisjoint(val_patients)
        fold_counts = np.bincount(loader.targets[val_idx], minlength=2)
        assert fold_counts.tolist() in ([2, 10], [3, 9], [3, 10])


def test_angle_binary_extreme_balanced_sampling_can_augment_to_100() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_binary_extreme.balanced_sampling.augmentation100.yaml"
        )
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=None,
        target_mode=config.data.target_mode,
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=5,
        val_batch_size=6,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        intensity_window=config.data.intensity_window,
        input_normalization=config.data.input_normalization,
        augmentation_config=config.data.augmentation,
        balanced_sampling=config.data.balanced_sampling,
    )

    train_idx, _ = loader.fold_indices[0]
    train_dl = loader.get_train_dl(0)
    val_dl = loader.get_val_dl(0)

    assert isinstance(train_dl.dataset, AugmentedSubset)
    assert isinstance(train_dl.sampler, BalancedClassSampler)
    assert train_dl.dataset.augmentation is not None
    assert train_dl.dataset.augmentation.target_class_indices == {0, 1}
    assert len(train_dl.dataset) == 200
    assert len(train_dl.sampler) == 200

    train_subset_targets = loader.targets[train_dl.dataset.indices].astype(int)
    assert np.bincount(train_subset_targets, minlength=2).tolist() == [100, 100]
    assert train_dl.dataset.augment_flags is not None
    assert sum(train_dl.dataset.augment_flags) == 200 - len(train_idx)
    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)
