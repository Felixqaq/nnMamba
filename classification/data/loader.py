"""DataLoader helper for stratified k-fold cross-validation."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .dataset import MRIDataset, Task
from .transforms import ToTensor


class LoaderHelper:
    """Manages dataset loading and stratified k-fold cross-validation splits."""

    TASK_LABELS = {
        Task.NC_v_AD: ["NC", "AD"],
        Task.sMCI_v_pMCI: ["sMCI", "pMCI"],
        Task.Normal_v_COPD: ["Normal", "COPD"],
        Task.Normal_v_Abnormal: ["Normal", "Abnormal"],
    }

    TASK_PATHS = {
        Task.NC_v_AD: ("./datasets/adni1/", "./datasets/adni2/"),
        Task.sMCI_v_pMCI: ("./datasets/adni1/", "./datasets/adni2/"),
        Task.Normal_v_COPD: ("./datasets/copd_train/", "./datasets/copd_test/"),
        Task.Normal_v_Abnormal: ("../", "../"),
    }

    def __init__(self, task: Task = Task.NC_v_AD, k_folds: int = 5, seed: int = 42):
        self.task = task
        self.labels = self.TASK_LABELS[task]

        train_root, _ = self.TASK_PATHS[task]
        transform = transforms.Compose([ToTensor()])

        self.train_ds = MRIDataset(
            train_root, self.labels, training=True, transform=transform
        )

        self._setup_stratified_folds(k_folds, seed)

    def _setup_stratified_folds(self, k_folds: int, seed: int) -> None:
        """Create stratified k-fold splits preserving class ratios."""
        labels = [self.train_ds[i]["label"].item() for i in range(len(self.train_ds))]
        indices = np.arange(len(self.train_ds))

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        self.fold_indices = [
            (train_idx.tolist(), test_idx.tolist())
            for train_idx, test_idx in skf.split(indices, labels)
        ]

        self._print_fold_distribution(labels)

    def _print_fold_distribution(self, all_labels: list) -> None:
        """Print class distribution for each fold."""
        print(f"\n📊 Stratified K-Fold Distribution ({len(self.fold_indices)} folds):")
        for i, (train_idx, test_idx) in enumerate(self.fold_indices):
            test_labels = [all_labels[j] for j in test_idx]
            n_class0 = test_labels.count(0)
            n_class1 = test_labels.count(1)
            print(
                f"  Fold {i + 1}: {self.labels[0]}={n_class0}, "
                f"{self.labels[1]}={n_class1} (Total={len(test_idx)})"
            )
        print()

    def get_task_string(self) -> str:
        """Get task name as string."""
        return self.task.name

    def get_train_dl(self, fold: int, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader for specified fold."""
        train_idx = self.fold_indices[fold][0]
        train_ds = Subset(self.train_ds, train_idx)

        return DataLoader(
            train_ds,
            batch_size=12,
            shuffle=shuffle,
            num_workers=0,  # Data is pre-cached, no need for workers
            drop_last=True,
            pin_memory=True,
        )

    def get_test_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Get test/validation DataLoader for specified fold."""
        test_idx = self.fold_indices[fold][1]
        test_ds = Subset(self.train_ds, test_idx)

        return DataLoader(
            test_ds,
            batch_size=4,
            shuffle=shuffle,
            num_workers=0,  # Data is pre-cached
            drop_last=False,
            pin_memory=True,
        )
