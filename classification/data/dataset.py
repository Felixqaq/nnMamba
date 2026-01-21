"""Dataset classes for medical image classification."""

from enum import Enum
import glob
import pathlib

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage import transform


class Task(Enum):
    """Classification task types."""

    NC_v_AD = 1
    sMCI_v_pMCI = 2
    Normal_v_COPD = 3
    Normal_v_Abnormal = 4


TARGET_SHAPE = (112, 136, 112)


def get_label(path: pathlib.Path, labels: list[str]) -> np.ndarray:
    """Extract label from file path based on parent directory name."""
    label_str = path.parent.stem
    return np.array([0 if label_str == labels[0] else 1], dtype=np.float32)


def get_mri(path: pathlib.Path, training: bool = False) -> np.ndarray:
    """Load and preprocess MRI/CT image.

    Args:
        path: Path to NIfTI file
        training: If True, apply augmentation (currently disabled)

    Returns:
        Preprocessed image array with shape (1, D, H, W)
    """
    try:
        mri = nib.load(str(path)).get_fdata()

        # Handle 4D images
        if len(mri.shape) > 3:
            mri = mri[:, :, :, 0]
        elif len(mri.shape) < 3:
            raise ValueError(f"Image dimension insufficient: {mri.shape}")

        # Resize to target shape
        if mri.shape != TARGET_SHAPE:
            mri = transform.resize(
                mri, TARGET_SHAPE, order=1, preserve_range=True, anti_aliasing=True
            )

        return np.expand_dims(mri, axis=0).astype(np.float32)

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return np.zeros((1, *TARGET_SHAPE), dtype=np.float32)


class MRIDataset(Dataset):
    """PyTorch Dataset for MRI/CT images."""

    def __init__(
        self,
        root_dir: str,
        labels: list[str],
        training: bool = True,
        transform=None,
    ):
        self.root_dir = root_dir
        self.labels = labels
        self.training = training
        self.transform = transform
        self.directories = []

        for label in labels:
            for path in glob.glob(f"{root_dir}{label}/*"):
                self.directories.append(pathlib.Path(path))

    def __len__(self) -> int:
        return len(self.directories)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path = self.directories[idx]
        mri = get_mri(path, self.training)
        label = get_label(path, self.labels)

        sample = {"mri": mri, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
