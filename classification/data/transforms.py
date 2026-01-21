"""Data transforms for MRI/CT images."""

import numpy as np
import torch


class ToTensor:
    """Convert numpy arrays in sample to PyTorch tensors."""

    def __call__(self, sample: dict) -> dict:
        image = sample["mri"]
        label = sample["label"]

        return {
            "mri": torch.from_numpy(image),
            "label": torch.from_numpy(label),
        }
