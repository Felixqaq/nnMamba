"""Data transforms for CT regression."""

from __future__ import annotations

import torch


class ToTensor:
    """Convert numpy arrays in a sample to PyTorch tensors."""

    def __call__(self, sample: dict) -> dict:
        ct = torch.from_numpy(sample["ct"]).float()
        return {
            "ct": ct,
            "mri": ct,
            "angle": torch.tensor(sample["angle"], dtype=torch.float32),
            "patient_id": sample.get("patient_id"),
            "source_group": sample.get("source_group"),
            "path": sample.get("path"),
        }
