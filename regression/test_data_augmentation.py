"""Tests for GOLD data augmentation helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Subset

from core.config import Config
from data.loader import AugmentedSubset, BalancedClassSampler, RegressionLoaderHelper
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


def test_gold_config_uses_train_fold_aug200_balancing() -> None:
    config = Config.from_yaml(Path(__file__).with_name("config.gold.yaml"))

    assert config.data.source_dir == Path("../by_angle_all")
    assert config.data.manifest == Path("./datasets/generated/gold_manifest.json")
    assert config.data.balanced_sampling is True
    assert config.data.input_normalization == "zscore"
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.balance_to_majority is False
    assert config.data.augmentation.target_per_class == 200
    assert config.data.augmentation.gold_stages == (1, 2, 3, 4)
    assert config.training.class_weight_mode == "none"


def test_gold_aug200_keeps_patient_level_validation_split() -> None:
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

    assert len(loader.records) == 66
    assert len(set(loader.patient_ids)) == 66
    assert isinstance(train_dl.sampler, BalancedClassSampler)
    assert isinstance(train_dl.dataset, AugmentedSubset)
    assert len(train_dl.dataset) == 800
    assert len(train_dl.sampler) == 800
    assert sum(train_dl.dataset.augment_flags or []) == 800 - len(train_idx)

    augmented_targets = Counter(
        int(loader.targets[index]) for index in train_dl.dataset.indices
    )
    assert augmented_targets == {0: 200, 1: 200, 2: 200, 3: 200}
    assert train_dl.dataset.augmentation.target_class_indices == {0, 1, 2, 3}

    train_patients = {loader.patient_ids[index] for index in train_idx}
    val_patients = {loader.patient_ids[index] for index in val_idx}
    assert train_patients.isdisjoint(val_patients)
    assert all("_aug" not in Path(loader.records[index].path).name for index in train_idx)
    assert all("_aug" not in Path(loader.records[index].path).name for index in val_idx)

    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)
    assert len(val_dl.dataset) == len(val_idx)
