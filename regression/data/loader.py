"""DataLoader helper for regression and classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import transforms

from .dataset import AngleRegressionDataset, DEFAULT_IMAGE_SIZE
from .manifest import build_angle_manifest
from .transforms import RandomCTAugmentation, ToTensor


ATTENTION_HEAVY_MODELS = {
    "hybrid",
    "hybrid_mamba_attention",
    "hybrid_mamba_attention_regressor",
    "mamba_hybrid",
    "swinunetr",
}
CLASSIFICATION_TARGET_MODES = {"gold", "angle_3class", "angle_binary_extreme"}


class BalancedClassSampler(Sampler):
    """Randomly undersample each class to the minority count on every epoch."""

    def __init__(self, targets: np.ndarray, seed: int = 42):
        targets = np.asarray(targets, dtype=int)
        if targets.ndim != 1:
            raise ValueError("BalancedClassSampler expects a 1D target array.")
        if len(targets) == 0:
            raise ValueError("BalancedClassSampler requires at least one sample.")

        self.targets = targets
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.class_positions = {
            int(class_idx): np.flatnonzero(targets == class_idx).astype(int)
            for class_idx in np.unique(targets)
        }
        nonzero_counts = [
            len(positions)
            for positions in self.class_positions.values()
            if len(positions) > 0
        ]
        if not nonzero_counts:
            raise ValueError("BalancedClassSampler requires at least one class.")

        self.samples_per_class = int(min(nonzero_counts))
        self.num_samples = int(self.samples_per_class * len(self.class_positions))

    def __iter__(self):
        sampled_positions: list[int] = []
        for positions in self.class_positions.values():
            selected = self.rng.choice(
                positions,
                size=self.samples_per_class,
                replace=False,
            )
            sampled_positions.extend(int(position) for position in selected)
        self.rng.shuffle(sampled_positions)
        return iter(sampled_positions)

    def __len__(self) -> int:
        return self.num_samples


class BalancedClassViewSampler(Sampler):
    """Undersample classes first, then emit multiple views of each selected sample."""

    def __init__(
        self,
        targets: np.ndarray,
        views_per_sample: int,
        seed: int = 42,
    ):
        targets = np.asarray(targets, dtype=int)
        if targets.ndim != 1:
            raise ValueError("BalancedClassViewSampler expects a 1D target array.")
        if len(targets) == 0:
            raise ValueError("BalancedClassViewSampler requires at least one sample.")

        self.targets = targets
        self.views_per_sample = int(views_per_sample)
        if self.views_per_sample < 1:
            raise ValueError("views_per_sample must be at least 1.")

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.class_positions = {
            int(class_idx): np.flatnonzero(targets == class_idx).astype(int)
            for class_idx in np.unique(targets)
        }
        nonzero_counts = [
            len(positions)
            for positions in self.class_positions.values()
            if len(positions) > 0
        ]
        if not nonzero_counts:
            raise ValueError("BalancedClassViewSampler requires at least one class.")

        self.samples_per_class = int(min(nonzero_counts))
        self.num_samples = int(
            self.samples_per_class
            * len(self.class_positions)
            * self.views_per_sample
        )

    def __iter__(self):
        sampled_positions: list[int] = []
        for positions in self.class_positions.values():
            selected = self.rng.choice(
                positions,
                size=self.samples_per_class,
                replace=False,
            )
            for base_position in selected:
                start = int(base_position) * self.views_per_sample
                sampled_positions.extend(
                    range(start, start + self.views_per_sample)
                )
        self.rng.shuffle(sampled_positions)
        return iter(sampled_positions)

    def __len__(self) -> int:
        return self.num_samples


class AugmentedSubset(Dataset):
    """Subset wrapper that applies train-only transforms after base loading."""

    def __init__(
        self,
        dataset: Dataset,
        indices: list[int],
        augmentation=None,
        augment_flags: list[bool] | None = None,
    ):
        self.dataset = dataset
        self.indices = list(indices)
        self.augmentation = augmentation
        self.augment_flags = (
            list(augment_flags) if augment_flags is not None else None
        )
        if self.augment_flags is not None and len(self.augment_flags) != len(
            self.indices
        ):
            raise ValueError("augment_flags must match indices length.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[self.indices[idx]]
        if self.augmentation is None:
            return sample
        if self.augment_flags is not None and not self.augment_flags[idx]:
            output = dict(sample)
            output["augmented"] = False
            return output
        return self.augmentation(sample)


class RegressionLoaderHelper:
    """Manage CT regression data loading and fold splitting."""

    def __init__(
        self,
        data_root: str | Path | Any = "../by_angle_all",
        labels_json: str | Path = "../patient_angle_classification_by_group.json",
        pft_json: str | Path = "../pft.json",
        target_mode: str = "angle",
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
        augmentation_config: Any | None = None,
        balanced_sampling: bool = False,
    ):
        if hasattr(data_root, "data") and hasattr(data_root, "training"):
            config = data_root
            data_root = config.data.source_dir
            labels_json = config.data.labels_json
            pft_json = config.data.pft_json
            target_mode = config.data.target_mode
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
            augmentation_config = config.data.augmentation
            balanced_sampling = config.data.balanced_sampling

        self.data_root = Path(data_root)
        self.labels_json = Path(labels_json)
        self.pft_json = Path(pft_json) if pft_json is not None else None
        self.target_mode = str(target_mode)
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
        self.augmentation_config = augmentation_config
        self.balanced_sampling = bool(
            balanced_sampling and self.target_mode in CLASSIFICATION_TARGET_MODES
        )
        self.train_augmentation = self._build_train_augmentation()

        manifest = build_angle_manifest(
            self.data_root,
            self.labels_json,
            pft_json=self.pft_json,
            target_mode=self.target_mode,
        )
        self.manifest = manifest
        self.records = list(manifest.records)
        self.class_names = list(manifest.class_names)
        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            self.targets = np.asarray(
                [int(record.class_index) for record in self.records],
                dtype=np.int64,
            )
        else:
            self.targets = np.asarray(
                [record.angle for record in self.records],
                dtype=np.float32,
            )
        self.patient_ids = [record.patient_id for record in self.records]
        self.target_mean, self.target_std = self._compute_target_stats(self.targets)

        self.train_ds = AngleRegressionDataset(
            self.data_root,
            self.labels_json,
            pft_json=self.pft_json,
            target_mode=self.target_mode,
            image_size=self.image_size,
            intensity_window=self.intensity_window,
            input_normalization=self.input_normalization,
            records=self.records,
            transform=transforms.Compose([ToTensor()]),
            cache_data=self.cache_data,
        )
        self.dataset = self.train_ds

        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            self.bin_edges = None
            self.strata = self.targets.astype(int)
        else:
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
        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            self._setup_classification_patient_folds(indices)
            self._print_fold_distribution()
            return
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

    def _setup_classification_patient_folds(self, indices: np.ndarray) -> None:
        """Split classification patients once, keeping augmented copies out of validation."""
        original_indices = np.asarray(
            [
                int(index)
                for index in indices
                if not self._is_augmented_record(self.records[int(index)].path)
            ],
            dtype=int,
        )
        if len(original_indices) == 0:
            original_indices = indices.astype(int)

        original_targets = self.targets[original_indices]
        unique_original_patients = {
            self.patient_ids[int(index)] for index in original_indices
        }
        if len(unique_original_patients) != len(original_indices):
            raise ValueError(
                "Classification patient-level splitting expects one non-augmented record per "
                "patient. Check augmented filenames and source_dir."
            )

        if (
            len(np.unique(original_targets)) > 1
            and np.bincount(original_targets).min() >= self.k_folds
        ):
            splitter = StratifiedKFold(
                n_splits=self.k_folds, shuffle=True, random_state=self.seed
            )
            splits = splitter.split(original_indices, original_targets)
            self.split_strategy = "patient_stratified_classification"
        else:
            splitter = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            splits = splitter.split(original_indices)
            self.split_strategy = "patient_kfold_classification"

        self.fold_indices = []
        for train_positions, val_positions in splits:
            train_original = original_indices[train_positions]
            val_original = original_indices[val_positions]
            train_patients = {self.patient_ids[int(index)] for index in train_original}
            val_patients = {self.patient_ids[int(index)] for index in val_original}
            if train_patients & val_patients:
                raise ValueError("Patient leakage detected between train and validation.")

            train_idx = [
                int(index)
                for index in indices
                if self.patient_ids[int(index)] in train_patients
            ]
            val_idx = [int(index) for index in val_original]
            self.fold_indices.append((train_idx, val_idx))

    @staticmethod
    def _is_augmented_record(path: str | Path) -> bool:
        """Return whether a filename is an augmented materialized CT."""
        return "_aug" in Path(path).name

    def _print_fold_distribution(self) -> None:
        """Print fold-level label statistics for sanity checking."""
        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            print(
                f"\n📊 Classification folds ({self.k_folds}) using "
                f"{self.split_strategy}: {len(self.records)} samples"
            )
            for fold_idx, (_, val_idx) in enumerate(self.fold_indices, start=1):
                fold_targets = self.targets[val_idx]
                counts = np.bincount(fold_targets, minlength=max(len(self.class_names), 1))
                summary = ", ".join(
                    f"{self.class_names[i] if i < len(self.class_names) else i}={int(count)}"
                    for i, count in enumerate(counts)
                    if int(count) > 0
                )
                print(f"  Fold {fold_idx}: n={len(val_idx)}, {summary}")
            return

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

    def get_fold_class_weights(self, fold: int) -> torch.Tensor:
        """Return balanced class weights computed from the training split only."""
        if self.target_mode not in CLASSIFICATION_TARGET_MODES:
            raise ValueError("Class weights are only defined for classification tasks.")

        train_idx = self.fold_indices[fold][0]
        train_targets = self.targets[train_idx].astype(int)
        if len(train_targets) == 0:
            return torch.ones(1, dtype=torch.float32)

        num_classes = max(
            len(self.class_names),
            int(train_targets.max()) + 1,
        )
        counts = np.bincount(train_targets, minlength=max(num_classes, 1)).astype(
            np.float32
        )
        weights = np.ones_like(counts, dtype=np.float32)
        nonzero = counts > 0
        if np.any(nonzero):
            weights[nonzero] = float(len(train_targets)) / (
                float(np.count_nonzero(nonzero)) * counts[nonzero]
            )
            weights = weights / float(weights[nonzero].mean())
        return torch.tensor(weights, dtype=torch.float32)

    def _compute_target_stats(self, targets: np.ndarray) -> tuple[float, float]:
        """Return mean/std of the provided target array."""
        mean = float(np.mean(targets)) if len(targets) else 0.0
        std = float(np.std(targets)) if len(targets) else 1.0
        if std < 1e-8:
            std = 1.0
        return mean, std

    def get_target_stats(self) -> tuple[float, float]:
        """Return global mean/std of all regression targets in the dataset."""
        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            return 0.0, 1.0
        return self.target_mean, self.target_std

    def get_fold_target_stats(self, fold: int) -> tuple[float, float]:
        """Backward-compatible alias for dataset-level target stats."""
        del fold
        return self.get_target_stats()

    def get_class_names(self) -> list[str]:
        """Return classification label names when available."""
        return list(self.class_names)

    def _build_train_augmentation(self):
        """Create train-only augmentation for supported classification tasks."""
        cfg = self.augmentation_config
        if (
            self.target_mode not in CLASSIFICATION_TARGET_MODES
            or cfg is None
            or not getattr(cfg, "enabled", False)
        ):
            return None
        return RandomCTAugmentation(
            enabled=True,
            probability=getattr(cfg, "probability", 0.8),
            gold_stages=tuple(getattr(cfg, "gold_stages", (2, 3, 4))),
            class_indices=getattr(cfg, "class_indices", None),
            rotation_degrees=getattr(cfg, "rotation_degrees", 7.0),
            translation_fraction=getattr(cfg, "translation_fraction", 0.05),
            scale_range=tuple(getattr(cfg, "scale_range", (0.95, 1.05))),
            intensity_scale_range=tuple(
                getattr(cfg, "intensity_scale_range", (0.95, 1.05))
            ),
            intensity_shift_range=tuple(
                getattr(cfg, "intensity_shift_range", (-25.0, 25.0))
            ),
            noise_std=getattr(cfg, "noise_std", 8.0),
        )

    def _should_balance_with_augmentation(self) -> bool:
        """Return whether the train fold should add virtual augmented samples."""
        cfg = self.augmentation_config
        return bool(
            self.target_mode in CLASSIFICATION_TARGET_MODES
            and self.train_augmentation is not None
            and cfg is not None
            and not getattr(cfg, "balance_then_augment", False)
            and (
                getattr(cfg, "balance_to_majority", False)
                or getattr(cfg, "target_per_class", None) is not None
            )
        )

    def _views_per_sample(self) -> int:
        """Return how many train-time views to emit per balanced base sample."""
        cfg = self.augmentation_config
        return max(1, int(getattr(cfg, "views_per_sample", 1)))

    def _should_balance_then_augment(self) -> bool:
        """Return whether each epoch should undersample before view expansion."""
        cfg = self.augmentation_config
        return bool(
            self.target_mode in CLASSIFICATION_TARGET_MODES
            and self.balanced_sampling
            and self.train_augmentation is not None
            and cfg is not None
            and getattr(cfg, "balance_then_augment", False)
            and self._views_per_sample() > 1
        )

    def _build_augmented_train_indices(
        self, indices: list[int]
    ) -> tuple[list[int], list[bool] | None]:
        """Add virtual augmented copies for configured classification classes."""
        if not self._should_balance_with_augmentation():
            return list(indices), None

        fold_targets = self.targets[indices].astype(int)
        if len(fold_targets) == 0:
            return list(indices), None
        num_classes = max(len(self.class_names), int(fold_targets.max()) + 1)
        counts = np.bincount(fold_targets, minlength=num_classes)
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) <= 1:
            return list(indices), None

        cfg = self.augmentation_config
        configured_target = getattr(cfg, "target_per_class", None)
        target_count = (
            int(configured_target)
            if configured_target is not None
            else int(nonzero_counts.max())
        )
        if target_count <= 0:
            return list(indices), None

        augmented_class_indices = self.train_augmentation.target_class_indices
        output_indices = list(indices)
        augment_flags = [False] * len(output_indices)

        for class_idx, count in enumerate(counts):
            if count == 0 or class_idx not in augmented_class_indices:
                continue
            needed = target_count - int(count)
            if needed <= 0:
                continue
            class_indices = [
                index
                for index, target in zip(indices, fold_targets, strict=True)
                if int(target) == class_idx
            ]
            for offset in range(needed):
                output_indices.append(class_indices[offset % len(class_indices)])
                augment_flags.append(True)

        return output_indices, augment_flags

    def _build_balanced_view_train_indices(
        self, indices: list[int]
    ) -> tuple[list[int], list[bool]]:
        """Repeat each train sample into original+augmented views for epoch sampling."""
        views_per_sample = self._views_per_sample()
        output_indices: list[int] = []
        augment_flags: list[bool] = []
        for index in indices:
            output_indices.append(index)
            augment_flags.append(False)
            for _ in range(views_per_sample - 1):
                output_indices.append(index)
                augment_flags.append(True)
        return output_indices, augment_flags

    def _build_loader(
        self,
        indices: list[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        augmentation=None,
        augment_flags: list[bool] | None = None,
        sampler: Sampler[int] | None = None,
    ) -> DataLoader:
        subset = (
            AugmentedSubset(self.train_ds, indices, augmentation, augment_flags)
            if augmentation is not None
            else Subset(self.train_ds, indices)
        )
        loader_kwargs = dict(
            dataset=subset,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
        )
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(**loader_kwargs)

    def _build_balanced_sampler(self, indices: list[int]) -> BalancedClassSampler | None:
        """Build a fold-local undersampling sampler when classification balancing is enabled."""
        if not self.balanced_sampling:
            return None
        fold_targets = self.targets[indices].astype(int)
        if len(fold_targets) == 0:
            return None
        counts = np.bincount(fold_targets)
        nonzero_counts = counts[counts > 0]
        if len(nonzero_counts) <= 1:
            return None
        return BalancedClassSampler(fold_targets, seed=self.seed)

    def _build_balanced_view_sampler(
        self, indices: list[int]
    ) -> BalancedClassViewSampler | None:
        """Build a sampler that balances base samples before emitting their views."""
        fold_targets = self.targets[indices].astype(int)
        if len(fold_targets) == 0:
            return None
        return BalancedClassViewSampler(
            fold_targets,
            views_per_sample=self._views_per_sample(),
            seed=self.seed,
        )

    def get_train_dl(self, fold: int, shuffle: bool = True) -> DataLoader:
        """Get the training loader for a given fold."""
        train_idx = self.fold_indices[fold][0]
        if self._should_balance_then_augment():
            base_train_idx = list(train_idx)
            train_idx, augment_flags = self._build_balanced_view_train_indices(
                base_train_idx
            )
            sampler = self._build_balanced_view_sampler(base_train_idx)
        else:
            train_idx, augment_flags = self._build_augmented_train_indices(train_idx)
            sampler = self._build_balanced_sampler(train_idx)
        return self._build_loader(
            train_idx,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            augmentation=self.train_augmentation,
            augment_flags=augment_flags,
            sampler=sampler,
        )

    def get_val_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Get the validation loader for a given fold."""
        val_idx = self.fold_indices[fold][1]
        return self._build_loader(
            val_idx,
            batch_size=self.val_batch_size,
            shuffle=shuffle,
            drop_last=False,
            augmentation=None,
        )

    def get_test_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Alias for validation loader to mirror the classification API."""
        return self.get_val_dl(fold, shuffle=shuffle)


LoaderHelper = RegressionLoaderHelper
