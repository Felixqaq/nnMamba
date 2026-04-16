"""Data transforms for CT regression."""

from __future__ import annotations

import numpy as np
import torch


class ToTensor:
    """Convert numpy arrays in a sample to PyTorch tensors."""

    def __call__(self, sample: dict) -> dict:
        ct = torch.from_numpy(sample["ct"]).float()
        target_np = np.asarray(sample["target"])
        target = (
            torch.tensor(target_np, dtype=torch.long)
            if np.issubdtype(target_np.dtype, np.integer)
            else torch.tensor(target_np, dtype=torch.float32)
        )
        output = {
            "ct": ct,
            "mri": ct,
            "target": target,
            "angle": torch.tensor(sample["angle"], dtype=torch.float32),
            "patient_id": sample.get("patient_id"),
            "source_group": sample.get("source_group"),
            "gold_stage_label": sample.get("gold_stage_label"),
            "post_fev1_percent_predicted": sample.get("post_fev1_percent_predicted"),
            "path": sample.get("path"),
        }
        if sample.get("label") is not None:
            output["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return output
