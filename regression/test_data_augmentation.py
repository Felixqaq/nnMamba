"""Tests for materialized GOLD data augmentation helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import RandomSampler, Subset

from core.config import Config
from data.loader import AugmentedSubset, RegressionLoaderHelper
from data.transforms import RandomCTAugmentation


def _sample(label: int) -> dict:
    ct = torch.ones(1, 8, 8, 8)
    return {
        "ct": ct,
        "mri": ct,
        "target": torch.tensor(label, dtype=torch.long),
        "label": torch.tensor(label, dtype=torch.long),
    }


def test_random_ct_augmentation_targets_configured_gold_stages() -> None:
    augment = RandomCTAugmentation(
        enabled=True,
        probability=1.0,
        gold_stages=(2,),
        rotation_degrees=0.0,
        translation_fraction=0.0,
        scale_range=(1.0, 1.0),
        intensity_scale_range=(1.0, 1.0),
        intensity_shift_range=(5.0, 5.0),
        noise_std=0.0,
    )

    skipped = augment(_sample(label=0))
    applied = augment(_sample(label=1))

    assert skipped["augmented"] is False
    assert torch.allclose(skipped["ct"], torch.ones(1, 8, 8, 8))
    assert applied["augmented"] is True
    assert torch.allclose(applied["ct"], torch.full((1, 8, 8, 8), 6.0))
    assert applied["mri"] is applied["ct"]


def test_random_ct_augmentation_can_target_zero_based_class_indices() -> None:
    augment = RandomCTAugmentation(
        enabled=True,
        probability=1.0,
        class_indices=(0, 2),
        rotation_degrees=0.0,
        translation_fraction=0.0,
        scale_range=(1.0, 1.0),
        intensity_scale_range=(1.0, 1.0),
        intensity_shift_range=(5.0, 5.0),
        noise_std=0.0,
    )

    skipped = augment(_sample(label=1))
    applied = augment(_sample(label=2))

    assert skipped["augmented"] is False
    assert torch.allclose(skipped["ct"], torch.ones(1, 8, 8, 8))
    assert applied["augmented"] is True
    assert torch.allclose(applied["ct"], torch.full((1, 8, 8, 8), 6.0))


def test_gold_config_uses_materialized_augmented_dataset() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.yaml"))

    assert config.data.source_dir == Path("../by_angle_all_gold_augmented")
    assert config.data.manifest == Path("./datasets/generated/gold_manifest.augmented.json")
    assert config.data.balanced_sampling is False
    assert config.data.input_normalization == "zscore"
    assert config.data.augmentation.enabled is False
    assert config.data.augmentation.balance_to_majority is False
    assert config.training.class_weight_mode == "balanced"


def test_gold_augmented_dataset_uses_patient_level_validation_split() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.yaml"))
    regression_root = Path(__file__).resolve().parent

    loader = RegressionLoaderHelper(
        data_root=regression_root / config.data.source_dir,
        labels_json=regression_root / config.data.labels_json,
        pft_json=regression_root / config.data.pft_json,
        target_mode="gold",
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

    train_idx, val_idx = loader.fold_indices[0]
    train_dl = loader.get_train_dl(0)
    val_dl = loader.get_val_dl(0)

    assert len(loader.records) == 144
    assert len(set(loader.patient_ids)) == 66
    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(train_dl.dataset, Subset)
    assert not isinstance(train_dl.dataset, AugmentedSubset)

    train_patients = {loader.patient_ids[index] for index in train_idx}
    val_patients = {loader.patient_ids[index] for index in val_idx}
    assert train_patients.isdisjoint(val_patients)
    assert any("_aug" in Path(loader.records[index].path).name for index in train_idx)
    assert all("_aug" not in Path(loader.records[index].path).name for index in val_idx)

    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)
    assert len(val_dl.dataset) == len(val_idx)
