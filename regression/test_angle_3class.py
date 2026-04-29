"""Tests for fixed 131/152 degree angle three-class classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from core.config import Config
from data.loader import BalancedClassSampler, RegressionLoaderHelper
from data.manifest import ANGLE_3CLASS_NAMES, build_angle_manifest


def test_angle_3class_config_is_three_class_classification() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.angle_3class.yaml"))

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path("./datasets/generated/angle_3class_manifest.json")
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.target_normalization == "none"
    assert config.data.augmentation.enabled is False
    assert config.task == "Angle_3class_classification"


def test_angle_3class_manifest_uses_fixed_131_152_thresholds() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    manifest = build_angle_manifest(
        repo_root / "by_angle_all",
        repo_root / "patient_angle_classification_by_group.json",
        target_mode="angle_3class",
    )

    assert manifest.counts["total"] == 66
    assert manifest.counts["unique_patients"] == 66
    assert manifest.missing_from_source == []
    assert manifest.extra_in_source_not_in_json == []
    assert manifest.class_names == ANGLE_3CLASS_NAMES
    assert manifest.class_counts == {
        "Emphysema/Abnormal (<=131 deg)": 14,
        "Intermediate (132-151 deg)": 5,
        "Normal (>=152 deg)": 47,
    }
    assert all(record.class_index is not None for record in manifest.records)


def test_angle_3class_loader_stratifies_and_undersamples_each_epoch() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.angle_3class.yaml"))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=None,
        target_mode="angle_3class",
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

    assert loader.get_class_names() == ANGLE_3CLASS_NAMES
    assert np.bincount(loader.targets, minlength=3).tolist() == [14, 5, 47]
    assert len(loader.records) == 66
    assert len(set(loader.patient_ids)) == 66
    assert loader.split_strategy == "patient_stratified_classification"

    train_idx, _ = loader.fold_indices[0]
    train_dl = loader.get_train_dl(0)
    assert isinstance(train_dl.dataset, Subset)
    assert isinstance(train_dl.sampler, BalancedClassSampler)
    assert len(train_dl.sampler) == 12

    first_epoch_positions = list(iter(train_dl.sampler))
    second_epoch_positions = list(iter(train_dl.sampler))
    first_epoch_indices = [train_idx[position] for position in first_epoch_positions]
    second_epoch_indices = [train_idx[position] for position in second_epoch_positions]
    assert np.bincount(loader.targets[first_epoch_indices], minlength=3).tolist() == [
        4,
        4,
        4,
    ]
    assert np.bincount(loader.targets[second_epoch_indices], minlength=3).tolist() == [
        4,
        4,
        4,
    ]
    assert {
        index
        for index in first_epoch_indices
        if int(loader.targets[index]) == 2
    } != {
        index
        for index in second_epoch_indices
        if int(loader.targets[index]) == 2
    }

    class_weights = loader.get_fold_class_weights(0)
    assert tuple(class_weights.shape) == (3,)
    assert np.isfinite(class_weights.numpy()).all()

    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[index] for index in train_idx}
        val_patients = {loader.patient_ids[index] for index in val_idx}
        assert train_patients.isdisjoint(val_patients)
        assert all(
            "_aug" not in Path(loader.records[index].path).name for index in train_idx
        )
        assert all(
            "_aug" not in Path(loader.records[index].path).name for index in val_idx
        )
        fold_counts = np.bincount(loader.targets[val_idx], minlength=3)
        assert int(fold_counts[1]) == 1


def test_angle_3class_augmented_manifest_balances_minority_classes() -> None:
    regression_root = Path(__file__).resolve().parent
    manifest_path = (
        regression_root / "datasets" / "generated" / "angle_3class_manifest.augmented.json"
    )

    manifest = build_angle_manifest(
        regression_root / "../by_angle_all_angle_3class_augmented",
        regression_root / "../patient_angle_classification_by_group.json",
        target_mode="angle_3class",
    )

    assert manifest_path.exists()
    assert manifest.counts["total"] == 141
    assert manifest.counts["unique_patients"] == 66
    assert manifest.class_counts == {
        "Emphysema/Abnormal (<=131 deg)": 47,
        "Intermediate (132-151 deg)": 47,
        "Normal (>=152 deg)": 47,
    }
