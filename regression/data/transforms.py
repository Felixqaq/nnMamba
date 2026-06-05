"""Data transforms for CT regression."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class RandomCTAugmentation:
    """Apply conservative train-time augmentation to 3D CT tensors."""

    def __init__(
        self,
        enabled: bool = False,
        probability: float = 0.8,
        gold_stages: tuple[int, ...] = (2, 3, 4),
        class_indices: tuple[int, ...] | None = None,
        rotation_degrees: float = 7.0,
        translation_fraction: float = 0.05,
        scale_range: tuple[float, float] = (0.95, 1.05),
        intensity_scale_range: tuple[float, float] = (0.95, 1.05),
        intensity_shift_range: tuple[float, float] = (-25.0, 25.0),
        noise_std: float = 8.0,
    ):
        self.enabled = enabled
        self.probability = float(probability)
        if class_indices is None:
            self.target_class_indices = {int(stage) - 1 for stage in gold_stages}
        else:
            self.target_class_indices = {int(class_idx) for class_idx in class_indices}
        self.gold_stage_indices = self.target_class_indices
        self.rotation_degrees = float(rotation_degrees)
        self.translation_fraction = float(translation_fraction)
        self.scale_range = tuple(float(value) for value in scale_range)
        self.intensity_scale_range = tuple(
            float(value) for value in intensity_scale_range
        )
        self.intensity_shift_range = tuple(
            float(value) for value in intensity_shift_range
        )
        self.noise_std = float(noise_std)

    def __call__(self, sample: dict) -> dict:
        if not self._should_apply(sample):
            output = dict(sample)
            output["augmented"] = False
            return output

        output = dict(sample)
        ct = sample["ct"].clone().float()
        ct = self._random_affine(ct)
        ct = self._random_intensity(ct)
        ct = torch.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)
        output["ct"] = ct.contiguous()
        output["mri"] = output["ct"]
        output["augmented"] = True
        return output

    def _should_apply(self, sample: dict) -> bool:
        if not self.enabled:
            return False
        label = sample.get("label", sample.get("target"))
        if label is None:
            return False
        label_index = int(label.item() if torch.is_tensor(label) else label)
        if self.target_class_indices and label_index not in self.target_class_indices:
            return False
        return float(torch.rand(()).item()) < self.probability

    def _random_affine(self, ct: torch.Tensor) -> torch.Tensor:
        if (
            self.rotation_degrees <= 0
            and self.translation_fraction <= 0
            and self.scale_range[0] == 1.0
            and self.scale_range[1] == 1.0
        ):
            return ct

        _, depth, height, width = ct.shape
        angle = math.radians(
            float(
                torch.empty(()).uniform_(
                    -self.rotation_degrees,
                    self.rotation_degrees,
                )
            )
        )
        scale = float(
            torch.empty(()).uniform_(
                self.scale_range[0],
                self.scale_range[1],
            )
        )
        inv_scale = 1.0 / max(scale, 1e-6)
        cos_a = math.cos(angle) * inv_scale
        sin_a = math.sin(angle) * inv_scale
        translate = [
            float(
                torch.empty(()).uniform_(
                    -self.translation_fraction,
                    self.translation_fraction,
                )
            )
            for _ in range(3)
        ]

        theta = torch.tensor(
            [
                [cos_a, -sin_a, 0.0, translate[0]],
                [sin_a, cos_a, 0.0, translate[1]],
                [0.0, 0.0, inv_scale, translate[2]],
            ],
            dtype=ct.dtype,
            device=ct.device,
        ).unsqueeze(0)
        grid = F.affine_grid(
            theta,
            size=(1, 1, depth, height, width),
            align_corners=False,
        )
        return F.grid_sample(
            ct.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)

    def _random_intensity(self, ct: torch.Tensor) -> torch.Tensor:
        if self.intensity_scale_range[0] != 1.0 or self.intensity_scale_range[1] != 1.0:
            scale = float(
                torch.empty(()).uniform_(
                    self.intensity_scale_range[0],
                    self.intensity_scale_range[1],
                )
            )
            ct = ct * scale
        if self.intensity_shift_range[0] != 0.0 or self.intensity_shift_range[1] != 0.0:
            shift = float(
                torch.empty(()).uniform_(
                    self.intensity_shift_range[0],
                    self.intensity_shift_range[1],
                )
            )
            ct = ct + shift
        if self.noise_std > 0:
            ct = ct + torch.randn_like(ct) * self.noise_std
        return ct


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
            "class_label": sample.get("class_label"),
            "gold_stage_label": sample.get("gold_stage_label"),
            "post_fev1_percent_predicted": sample.get("post_fev1_percent_predicted"),
            "path": sample.get("path"),
        }
        if sample.get("oi") is not None:
            output["oi"] = torch.tensor(sample["oi"], dtype=torch.float32)
        for key in ("a", "fvc", "pef"):
            if sample.get(key) is not None:
                output[key] = sample.get(key)
        if sample.get("label") is not None:
            output["label"] = torch.tensor(sample["label"], dtype=torch.long)
        if sample.get("tapct_embedding") is not None:
            output["tapct_embedding"] = torch.as_tensor(
                sample["tapct_embedding"],
                dtype=torch.float32,
            )
        return output
