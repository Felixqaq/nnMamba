"""Tests for train-only GOLD data augmentation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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


def test_gold_config_enables_balanced_train_augmentation() -> None:
    config_path = Path(__file__).with_name("config.gold.yaml")
    config = Config.from_yaml(config_path)

    assert config.data.balanced_sampling is False
    assert config.data.input_normalization == "zscore"
    assert config.data.augmentation.enabled is True
    assert config.data.augmentation.balance_to_majority is True
    assert config.data.augmentation.probability == 1.0
    assert config.data.augmentation.gold_stages == (2, 3, 4)
    assert config.data.augmentation.intensity_shift_range == (-0.1, 0.1)
    assert config.data.augmentation.noise_std == 0.03
    assert config.training.class_weight_mode == "none"


def test_gold_train_loader_balances_with_train_only_augmentation() -> None:
    config_path = Path(__file__).with_name("config.gold.yaml")
    config = Config.from_yaml(config_path)
    repo_root = Path(__file__).resolve().parents[1]

    loader = RegressionLoaderHelper(
        data_root=repo_root / "by_angle_all",
        labels_json=repo_root / "patient_angle_classification_by_group.json",
        pft_json=repo_root / "pft.json",
        target_mode="gold",
        k_folds=config.training.k_folds,
        seed=config.training.seed,
        batch_size=4,
        val_batch_size=4,
        num_workers=0,
        cache_data=False,
        manifest_path=None,
        augmentation_config=config.data.augmentation,
        balanced_sampling=config.data.balanced_sampling,
    )

    train_idx, _ = loader.fold_indices[0]
    fold_targets = loader.targets[train_idx].astype(int)
    class_counts = np.bincount(fold_targets, minlength=len(loader.class_names))
    nonzero_counts = class_counts[class_counts > 0]
    expected_samples = int(nonzero_counts.max() * len(nonzero_counts))
    expected_augmented = expected_samples - len(train_idx)

    train_dl = loader.get_train_dl(0)
    val_dl = loader.get_val_dl(0)

    assert len(loader.records) == 66
    assert isinstance(train_dl.sampler, RandomSampler)
    assert isinstance(train_dl.dataset, AugmentedSubset)
    assert train_dl.dataset.augmentation is loader.train_augmentation
    assert len(train_dl.dataset) == expected_samples
    assert train_dl.dataset.augment_flags is not None
    assert sum(train_dl.dataset.augment_flags) == expected_augmented
    assert train_dl.dataset.augment_flags[: len(train_idx)] == [False] * len(train_idx)
    augmented_targets = [
        int(loader.targets[index])
        for index, should_augment in zip(
            train_dl.dataset.indices,
            train_dl.dataset.augment_flags,
            strict=True,
        )
        if should_augment
    ]
    assert augmented_targets
    assert set(augmented_targets).issubset({1, 2, 3})
    assert loader.train_augmentation is not None
    assert loader.train_augmentation.probability == 1.0

    assert isinstance(val_dl.dataset, Subset)
    assert not isinstance(val_dl.dataset, AugmentedSubset)
    assert len(val_dl.dataset) == len(loader.fold_indices[0][1])
