"""DataLoader helper for k-fold cross-validation."""

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from .dataset import MRIDataset, Task
from .transforms import ToTensor


class LoaderHelper:
    """Manages dataset loading and k-fold cross-validation splits."""

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

        train_root, test_root = self.TASK_PATHS[task]
        transform = transforms.Compose([ToTensor()])

        self.train_ds = MRIDataset(
            train_root, self.labels, training=True, transform=transform
        )
        self.test_ds = MRIDataset(
            test_root, self.labels, training=False, transform=transform
        )

        self._setup_folds(k_folds, seed)

    def _setup_folds(self, k_folds: int, seed: int) -> None:
        """Create k-fold cross-validation splits."""
        np.random.seed(seed)

        indices = list(range(len(self.train_ds)))
        np.random.shuffle(indices)

        fold_size = len(indices) // k_folds
        self.fold_indices = []

        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(indices)

            test_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]
            self.fold_indices.append((train_idx, test_idx))

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
            num_workers=6,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def get_test_dl(self, fold: int, shuffle: bool = False) -> DataLoader:
        """Get test/validation DataLoader for specified fold."""
        test_idx = self.fold_indices[fold][1]
        test_ds = Subset(self.train_ds, test_idx)

        return DataLoader(
            test_ds,
            batch_size=4,
            shuffle=shuffle,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
