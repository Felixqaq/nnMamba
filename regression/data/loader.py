"""DataLoader helper for regression with stratified binning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .dataset import AngleRegressionDataset, DEFAULT_IMAGE_SIZE
from .manifest import build_angle_manifest
from .transforms import ToTensor


ATTENTION_HEAVY_MODELS = {
    "hybrid",
    "hybrid_mamba_attention",
    "hybrid_mamba_attention_regressor",
    "mamba_hybrid",
    "swinunetr",
}


class RegressionLoaderHelper:
    """Manage CT regression data loading and fold splitting."""

    def __init__(
        self,
        data_root: str | Path | Any = "../by_angle_all",
        labels_json: str | Path = "../patient_angle_classification_by_group.json",
        image_size: tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
        k_folds: int = 5,
        seed: int = 42,
        n_bins: int = 5,
        batch_size: int = 2,
        val_batch_size: int = 2,
        num_workers: int = 0,
        cache_data: bool = True,
        manifest_path: str | Path | None = None,
        intensity_window: tuple[float, float] | None = None,
        input_normalization: str = "zscore",
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        if hasattr(data_root, "data") and hasattr(data_root, "training"):
            config = data_root
            data_root = config.data.source_dir
            labels_json = config.data.labels_json
            image_size = config.data.image_size
            k_folds = config.training.k_folds
            seed = config.training.seed
            n_bins = config.data.angle_bin_count
            if str(config.model.name).lower() in ATTENTION_HEAVY_MODELS:
                batch_size = config.training.swin_batch_size
                val_batch_size = config.training.swin_eval_batch_size
            else:
                batch_size = config.training.batch_size
                val_batch_size = config.training.eval_batch_size
            num_workers = config.data.num_workers
            cache_data = config.data.cache_data
            manifest_path = config.data.manifest
            intensity_window = config.data.intensity_window
            input_normalization = config.data.input_normalization
            pin_memory = config.data.pin_memory
            prefetch_factor = config.data.prefetch_factor

        self.data_root = Path(data_root)
        self.labels_json = Path(labels_json)
        self.image_size = image_size
        self.k_folds = k_folds
        self.seed = seed
        self.n_bins = n_bins
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.cache_data = cache_data
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.intensity_window = intensity_window
        self.input_normalization = input_normalization
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor

        manifest = build_angle_manifest(self.data_root, self.labels_json)
        self.manifest = manifest
        self.records = list(manifest.records)
        self.targets = np.asarray([record.angle for record in self.records], dtype=np.float32)
        self.patient_ids = [record.patient_id for record in self.records]
        self.target_mean, self.target_std = self._compute_target_stats(self.targets)

        self.train_ds = AngleRegressionDataset(
            self.data_root,
            self.labels_json,
            image_size=self.image_size,
            intensity_window=self.intensity_window,
            input_normalization=self.input_normalization,
            records=self.records,
            transform=transforms.Compose([ToTensor()]),
            cache_data=self.cache_data,
        )
        self.dataset = self.train_ds

        self.bin_edges, self.strata = self._build_strata(self.targets, self.n_bins)
        self._setup_folds()

        if self.manifest_path is not None:
            from .manifest import save_manifest

            save_manifest(self.manifest, self.manifest_path)

    def _build_strata(
        self, targets: np.ndarray, max_bins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create target bins using quantiles for regression stratification."""
        if len(targets) == 0:
            return np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([], dtype=int)

        max_bins = max(2, min(max_bins, len(targets)))
        for bin_count in range(max_bins, 1, -1):
            quantiles = np.linspace(0.0, 1.0, bin_count + 1)
            edges = np.unique(np.quantile(targets, quantiles))
            if len(edges) <= 2:
                continue
            strata = np.digitize(targets, edges[1:-1], right=False)
            if np.min(np.bincount(strata, minlength=len(edges) - 1)) >= self.k_folds:
                return edges.astype(np.float32), strata.astype(int)

        return np.asarray([targets.min(), targets.max()], dtype=np.float32), np.zeros(
            len(targets), dtype=int
        )

    def _setup_folds(self) -> None:
        """Create fold indices using stratified binning when possible."""
        indices = np.arange(len(self.records))
        if len(np.unique(self.strata)) > 1 and np.bincount(self.strata).min() >= self.k_folds:
            splitter = StratifiedKFold(
                n_splits=self.k_folds, shuffle=True, random_state=self.seed
            )
            splits = splitter.split(indices, self.strata)
            self.split_strategy = "stratified_bins"
        else:
            splitter = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            splits = splitter.split(indices)
            self.split_strategy = "kfold"

        self.fold_indices = [
            (train_idx.tolist(), val_idx.tolist())
            for train_idx, val_idx in splits
        ]

        self._print_fold_distribution()

    def _print_fold_distribution(self) -> None:
        """Print fold-level label statistics for sanity checking."""
        print(
            f"\n📊 Regression folds ({self.k_folds}) using {self.split_strategy}: "
            f"{len(self.records)} samples"
        )
        for fold_idx, (_, val_idx) in enumerate(self.fold_indices, start=1):
            fold_targets = self.targets[val_idx]
            print(
                f"  Fold {fold_idx}: "
                f"n={len(val_idx)}, "
                f"mean={fold_targets.mean():.2f}, "
                f"std={fold_targets.std():.2f}, "
                f"min={fold_targets.min():.2f}, "
                f"max={fold_targets.max():.2f}"
            )

    def get_fold_targets(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Return train/val targets for a fold."""
        train_idx, val_idx = self.fold_indices[fold]
        return self.targets[train_idx], self.targets[val_idx]

    def _compute_target_stats(self, targets: np.ndarray) -> tuple[float, float]:
        """Return mean/std of the provided target array."""
        mean = float(np.mean(targets)) if len(targets) else 0.0
        std = float(np.std(targets)) if len(targets) else 1.0
        if std < 1e-8:
            std = 1.0
        return mean, std

    def get_target_stats(self) -> tuple[float, float]:
        """Return global mean/std of all regression targets in the dataset."""
        return self.target_mean, self.target_std

    def get_fold_target_stats(self, fold: int) -> tuple[float, float]:
        """Backward-compatible alias for dataset-level target stats."""
        del fold
        return self.get_target_stats()

    def _build_loader(
        self,
        indices: list[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        subset = Subset(self.train_ds, indices)
        loader_kwargs = dict(
            dataset=subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)

    def get_train_dl(self, fold: int, shuffle: bool = True) -> DataLoader:
        """Get the training loader for a given fold."""
        train_idx = self.fold_indices[fold][0]
        return self._build_loader(
            train_idx,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def get_val_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Get the validation loader for a given fold."""
        val_idx = self.fold_indices[fold][1]
        return self._build_loader(
            val_idx,
            batch_size=self.val_batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def get_test_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Alias for validation loader to mirror the classification API."""
        return self.get_val_dl(fold, shuffle=shuffle)


LoaderHelper = RegressionLoaderHelper
