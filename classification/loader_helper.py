"""The following module deals with creating the loader he"""

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from data_declaration import MRIDataset, Task
from data_declaration import ToTensor


class LoaderHelper:
    """An abstract class for assisting with dataset creation."""

    def __init__(self, task: Task = Task.NC_v_AD):
        self.task = task
        self.labels = []

        if task == Task.NC_v_AD:
            self.labels = ["NC", "AD"]
        elif task == Task.sMCI_v_pMCI:
            self.labels = ["sMCI", "pMCI"]
        elif task == Task.Normal_v_COPD:
            self.labels = ["Normal", "COPD"]
        elif task == Task.Normal_v_Abnormal:
            self.labels = ["Normal", "Abnormal"]

        # cat12
        # 根據任務設定資料路徑
        if task == Task.Normal_v_COPD:
            train_root = "./datasets/copd_train/"
            test_root = "./datasets/copd_test/"
        elif task == Task.Normal_v_Abnormal:
            train_root = "../"  # 使用上層目錄的 Normal 和 Abnormal
            test_root = "../"  # K-fold 交叉驗證，不需要獨立測試集
        else:
            train_root = "./datasets/adni1/"
            test_root = "./datasets/adni2/"

        self.train_ds = MRIDataset(
            root_dir=train_root,
            labels=self.labels,
            training=True,
            transform=transforms.Compose([ToTensor()]),
        )
        self.test_ds = MRIDataset(
            root_dir=test_root,
            labels=self.labels,
            training=False,
            transform=transforms.Compose([ToTensor()]),
        )

        # print(self.train_ds.len)
        self.indices = []
        self.set_indices()

    def get_task(self):
        """gets task"""
        return self.task

    def get_task_string(self):
        """Gets task string"""
        if self.task == Task.NC_v_AD:
            return "NC_v_AD"
        elif self.task == Task.sMCI_v_pMCI:
            return "sMCI_v_pMCI"
        elif self.task == Task.Normal_v_COPD:
            return "Normal_v_COPD"
        elif self.task == Task.Normal_v_Abnormal:
            return "Normal_v_Abnormal"
        else:
            return "Unknown_Task"

    def change_ds_labels(self, labels_in):
        """Function to change the labels of the dataset obj."""
        self.dataset = MRIDataset(
            root_dir="../data/",
            labels=labels_in,
            transform=transforms.Compose([ToTensor()]),
        )

    def change_task(self, task: Task):
        """Function to change task of the Datasets"""
        self.task = task

        if task == Task.NC_v_AD:
            self.labels = ["NC", "AD"]
        elif task == Task.sMCI_v_pMCI:
            self.labels = ["sMCI", "pMCI"]
        elif task == Task.Normal_v_COPD:
            self.labels = ["Normal", "COPD"]
        elif task == Task.Normal_v_Abnormal:
            self.labels = ["Normal", "Abnormal"]

        self.dataset = MRIDataset(
            root_dir="../data/",
            labels=self.labels,
            transform=transforms.Compose([ToTensor()]),
        )

        self.set_indices()

    def set_indices(self, total_folds=5):
        """實現 k-fold 交叉驗證的索引分割"""
        random_seed = 42
        np.random.seed(random_seed)

        # 合併訓練集和測試集做 k-fold
        train_dataset_size = len(self.train_ds)
        test_dataset_size = len(self.test_ds)

        # 獲取所有索引並打亂
        all_train_indices = list(range(train_dataset_size))
        all_test_indices = list(range(test_dataset_size))
        np.random.shuffle(all_train_indices)
        np.random.shuffle(all_test_indices)

        # 對訓練集進行 k-fold 分割
        fold_size = train_dataset_size // total_folds
        self.fold_indices = []

        for fold in range(total_folds):
            # 計算當前折的測試集範圍
            start_idx = fold * fold_size
            end_idx = (
                start_idx + fold_size if fold < total_folds - 1 else train_dataset_size
            )

            # 當前折的測試集
            fold_test_indices = all_train_indices[start_idx:end_idx]
            # 當前折的訓練集（排除測試集）
            fold_train_indices = (
                all_train_indices[:start_idx] + all_train_indices[end_idx:]
            )

            self.fold_indices.append((fold_train_indices, fold_test_indices))

        # 保留原有的完整分割（向後兼容）
        self.indices = [all_train_indices, all_test_indices]

    def make_loaders(self, shuffle=True):
        """Makes the loaders"""
        fold_indices = self.indices()

        for k in range(5):
            train_ds = Subset(self.dataset, fold_indices[k][0])
            test_ds = Subset(self.dataset, fold_indices[k][1])

            train_dl = DataLoader(
                train_ds, batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True
            )
            test_dl = DataLoader(
                test_ds, batch_size=2, shuffle=shuffle, num_workers=4, drop_last=True
            )

        print(len(test_ds))

        return (train_dl, test_dl)

    def get_train_dl(self, fold_ind, shuffle=True):
        """獲取指定折的訓練集 DataLoader"""
        if hasattr(self, "fold_indices") and fold_ind < len(self.fold_indices):
            # K-fold 模式：使用對應折的訓練索引
            train_indices = self.fold_indices[fold_ind][0]
        else:
            # 單折模式：使用所有訓練數據
            train_indices = self.indices[0]

        train_ds = Subset(self.train_ds, train_indices)
        train_dl = DataLoader(
            train_ds,
            batch_size=12,
            shuffle=shuffle,
            num_workers=6,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        return train_dl

    def get_test_dl(self, fold_ind, shuffle=True):
        """獲取指定折的測試集 DataLoader"""
        if hasattr(self, "fold_indices") and fold_ind < len(self.fold_indices):
            # K-fold 模式：使用對應折的測試索引
            test_indices = self.fold_indices[fold_ind][1]
        else:
            # 單折模式：使用所有測試數據
            test_indices = self.indices[1]

        test_ds = Subset(
            self.train_ds, test_indices
        )  # 注意：k-fold 時測試集來自 train_ds
        test_dl = DataLoader(
            test_ds,
            batch_size=4,  # 降低 batch_size 以適應小資料集
            shuffle=False,  # 驗證時不打亂
            num_workers=4,
            drop_last=False,  # 不丟棄最後不完整的 batch
            pin_memory=True,
        )

        return test_dl
