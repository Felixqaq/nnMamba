"""Tests for fixed 131/152 degree angle three-class classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from core.checkpoints import generate_uuid
from core.config import Config
from data.loader import (
    AugmentedSubset,
    BalancedClassSampler,
    BalancedClassViewSampler,
    RegressionLoaderHelper,
)
from data.manifest import ANGLE_3CLASS_NAMES, build_angle_manifest


def test_angle_3class_config_is_three_class_classification() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.angle_3class.yaml"))

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all_angle_3class_augmented")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.augmented.json"
    )
    assert config.training.class_weight_mode == "balanced"
    assert config.data.balanced_sampling is False
    assert config.data.target_normalization == "none"
    assert config.data.augmentation.enabled is False
    assert config.experiment.name == "Materialized aug47 + class weights"
    assert config.task == "Angle_3class_classification"


def test_angle_3class_balanced_sampling_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.balanced_sampling.yaml")
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.target_normalization == "none"
    assert config.data.augmentation.enabled is False
    assert config.experiment.name == "Per-epoch minority undersampling"
    assert config.task == "Angle_3class_classification"


def test_angle_3class_balanced_sampling_augmentation_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation.yaml"
        )
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.target_per_class == 20
    assert config.data.augmentation.class_indices == (0, 1)
    assert config.experiment.name == "Balanced sampling + aug20/class"
    assert config.task == "Angle_3class_classification"


def test_angle_3class_balanced_sampling_augmentation100_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation100.yaml"
        )
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.target_per_class == 100
    assert config.data.augmentation.class_indices == (0, 1, 2)
    assert config.experiment.name == "Balanced sampling + aug100/class"
    assert config.task == "Angle_3class_classification"


def test_angle_3class_balanced_sampling_augmentation300_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation300.yaml"
        )
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.target_per_class == 300
    assert config.data.augmentation.class_indices == (0, 1, 2)
    assert config.experiment.name == "Balanced sampling + aug300/class"
    assert config.task == "Angle_3class_classification"


def test_angle_3class_balanced_sampling_augmentation_x12_config_is_additive() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation_x12.yaml"
        )
    )

    assert config.data.target_mode == "angle_3class"
    assert config.is_classification_task() is True
    assert config.model.num_classes == 3
    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path(
        "./datasets/generated/angle_3class_manifest.json"
    )
    assert config.training.class_weight_mode == "none"
    assert config.data.balanced_sampling is True
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.balance_then_augment is True
    assert config.data.augmentation.views_per_sample == 12
    assert config.data.augmentation.target_per_class is None
    assert config.data.augmentation.class_indices == (0, 1, 2)
    assert config.experiment.name == "Balanced sampling + augx12/epoch"
    assert config.task == "Angle_3class_classification"


def test_generate_uuid_includes_sanitized_experiment_name() -> None:
    uuid = generate_uuid(
        "hybrid_mamba_attention",
        "Balanced sampling + aug100/class",
    )

    assert uuid.startswith("hybrid_mamba_attention_balanced_sampling_aug100_class_")
    assert " " not in uuid
    assert "+" not in uuid
    assert "/" not in uuid


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


def test_angle_3class_augmented_loader_uses_materialized_dataset() -> None:
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
    assert np.bincount(loader.targets, minlength=3).tolist() == [47, 47, 47]
    assert len(loader.records) == 141
    assert len(set(loader.patient_ids)) == 66
    assert loader.split_strategy == "patient_stratified_classification"

    train_dl = loader.get_train_dl(0)
    assert isinstance(train_dl.dataset, Subset)
    assert not isinstance(train_dl.sampler, BalancedClassSampler)

    class_weights = loader.get_fold_class_weights(0)
    assert tuple(class_weights.shape) == (3,)
    assert np.isfinite(class_weights.numpy()).all()

    for train_idx, val_idx in loader.fold_indices:
        train_patients = {loader.patient_ids[index] for index in train_idx}
        val_patients = {loader.patient_ids[index] for index in val_idx}
        assert train_patients.isdisjoint(val_patients)
        assert any(
            "_aug" in Path(loader.records[index].path).name for index in train_idx
        )
        assert all(
            "_aug" not in Path(loader.records[index].path).name for index in val_idx
        )
        fold_counts = np.bincount(loader.targets[val_idx], minlength=3)
        assert int(fold_counts[1]) == 1


def test_angle_3class_loader_stratifies_and_undersamples_each_epoch() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name("config.angle_3class.balanced_sampling.yaml")
    )
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


def test_angle_3class_balanced_sampling_can_use_train_time_augmentation() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation.yaml"
        )
    )
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

    train_dl = loader.get_train_dl(0)
    val_dl = loader.get_val_dl(0)

    assert isinstance(train_dl.dataset, AugmentedSubset)
    assert isinstance(train_dl.sampler, BalancedClassSampler)
    assert train_dl.dataset.augmentation is not None
    assert train_dl.dataset.augmentation.target_class_indices == {0, 1}
    assert len(train_dl.dataset) == 77
    assert len(train_dl.sampler) == 60

    train_subset_targets = loader.targets[train_dl.dataset.indices].astype(int)
    assert np.bincount(train_subset_targets, minlength=3).tolist() == [20, 20, 37]
    assert train_dl.dataset.augment_flags is not None
    assert sum(train_dl.dataset.augment_flags) == 25
    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)


def test_angle_3class_balanced_sampling_can_augment_all_classes_to_100() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation100.yaml"
        )
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=None,
        target_mode="angle_3class",
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

    train_dl = loader.get_train_dl(0)

    assert isinstance(train_dl.dataset, AugmentedSubset)
    assert isinstance(train_dl.sampler, BalancedClassSampler)
    assert train_dl.dataset.augmentation is not None
    assert train_dl.dataset.augmentation.target_class_indices == {0, 1, 2}
    assert len(train_dl.dataset) == 300
    assert len(train_dl.sampler) == 300

    train_subset_targets = loader.targets[train_dl.dataset.indices].astype(int)
    assert np.bincount(train_subset_targets, minlength=3).tolist() == [100, 100, 100]
    assert train_dl.dataset.augment_flags is not None
    assert sum(train_dl.dataset.augment_flags) == 248


def test_angle_3class_balanced_sampling_augments_x12_after_epoch_balance() -> None:
    config = Config.from_yaml(
        Path(__file__).with_name(
            "config.angle_3class.balanced_sampling.augmentation_x12.yaml"
        )
    )
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=None,
        target_mode="angle_3class",
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=12,
        val_batch_size=12,
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
    assert isinstance(train_dl.sampler, BalancedClassViewSampler)
    assert train_dl.dataset.augmentation is not None
    assert train_dl.dataset.augmentation.target_class_indices == {0, 1, 2}
    assert len(train_dl.dataset) == len(train_idx) * 12
    assert len(train_dl.sampler) == 144

    train_targets = loader.targets[train_idx].astype(int)
    assert np.bincount(train_targets, minlength=3).tolist() == [11, 4, 37]
    assert train_dl.dataset.augment_flags is not None
    assert sum(train_dl.dataset.augment_flags) == len(train_idx) * 11

    first_epoch_positions = list(iter(train_dl.sampler))
    second_epoch_positions = list(iter(train_dl.sampler))

    first_epoch_indices = [
        train_dl.dataset.indices[position] for position in first_epoch_positions
    ]
    second_epoch_indices = [
        train_dl.dataset.indices[position] for position in second_epoch_positions
    ]
    assert np.bincount(loader.targets[first_epoch_indices], minlength=3).tolist() == [
        48,
        48,
        48,
    ]
    assert np.bincount(loader.targets[second_epoch_indices], minlength=3).tolist() == [
        48,
        48,
        48,
    ]

    first_base_positions = {position // 12 for position in first_epoch_positions}
    second_base_positions = {position // 12 for position in second_epoch_positions}
    first_base_indices = [train_idx[position] for position in first_base_positions]
    assert np.bincount(loader.targets[first_base_indices], minlength=3).tolist() == [
        4,
        4,
        4,
    ]

    first_epoch_flags = [
        train_dl.dataset.augment_flags[position] for position in first_epoch_positions
    ]
    assert sum(first_epoch_flags) == 132

    first_majority_indices = {
        train_idx[position]
        for position in first_base_positions
        if int(loader.targets[train_idx[position]]) == 2
    }
    second_majority_indices = {
        train_idx[position]
        for position in second_base_positions
        if int(loader.targets[train_idx[position]]) == 2
    }
    assert first_majority_indices != second_majority_indices
    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)


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
