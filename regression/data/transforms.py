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
        defer_to_device: bool = False,
    ):
        self.enabled = enabled
        # When True, __call__ only records *whether* a sample should be
        # augmented and leaves the volume untouched; the trainer then
        # calls apply_batch() once per batch on the GPU. Same transforms,
        # same distributions -- ~19x cheaper, because a 112x136x112
        # trilinear grid_sample is ~99 ms on one CPU core and ~5 ms/view
        # batched on the card.
        self.defer_to_device = bool(defer_to_device)
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
        wanted = self._should_apply(sample)

        if self.defer_to_device:
            # Decide here (cheap), transform on the GPU later (fast). The flag
            # rides along in the batch so apply_batch knows which rows to touch.
            output = dict(sample)
            output["augment"] = torch.tensor(bool(wanted))
            output["augmented"] = bool(wanted)
            return output

        if not wanted:
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
        # Channel 0 is the normalized CT. Additional channels are [0, 1]
        # physical density/occupancy maps and must not receive HU intensity
        # scaling, shifts or noise.
        image = ct[:1]
        if self.intensity_scale_range[0] != 1.0 or self.intensity_scale_range[1] != 1.0:
            scale = float(
                torch.empty(()).uniform_(
                    self.intensity_scale_range[0],
                    self.intensity_scale_range[1],
                )
            )
            image = image * scale
        if self.intensity_shift_range[0] != 0.0 or self.intensity_shift_range[1] != 0.0:
            shift = float(
                torch.empty(()).uniform_(
                    self.intensity_shift_range[0],
                    self.intensity_shift_range[1],
                )
            )
            image = image + shift
        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
        if ct.shape[0] == 1:
            return image
        return torch.cat((image, ct[1:]), dim=0)


    # ------------------------------------------------------------------
    # Batched path: same transforms, same distributions, one GPU call.
    # ------------------------------------------------------------------

    def apply_batch(
        self,
        ct: torch.Tensor,
        flags: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Augment a whole (N, C, D, H, W) batch in place on its own device.

        `flags` is the per-row decision made by __call__ in the worker. Rows
        that were not selected are returned untouched, so probability<1 and
        class_indices behave exactly as they do on the CPU path.
        """
        if not self.enabled or ct.ndim != 5:
            return ct

        mask = None if flags is None else flags.to(ct.device).reshape(-1).bool()

        if mask is None or bool(mask.all()):
            # Common case here: probability=1.0 over both classes, so every row
            # is augmented. Taking the whole batch avoids index_select, clone
            # and index_copy_ -- three extra passes over 54 MB per batch, on a
            # step that is already bandwidth-bound.
            return self._augment_all(ct)

        index = mask.nonzero(as_tuple=True)[0]
        if index.numel() == 0:
            return ct

        selected = self._augment_all(ct.index_select(0, index))
        output = ct.clone()
        output.index_copy_(0, index, selected.to(output.dtype))
        return output.contiguous()

    def _augment_all(self, ct: torch.Tensor) -> torch.Tensor:
        ct = self._batch_affine(ct.float())
        ct = self._batch_intensity(ct)
        return torch.nan_to_num(ct, nan=0.0, posinf=0.0, neginf=0.0)

    def _batch_affine(self, ct: torch.Tensor) -> torch.Tensor:
        if (
            self.rotation_degrees <= 0
            and self.translation_fraction <= 0
            and self.scale_range[0] == 1.0
            and self.scale_range[1] == 1.0
        ):
            return ct

        count, _, depth, height, width = ct.shape
        device, dtype = ct.device, ct.dtype

        angle = torch.empty(count, device=device, dtype=dtype).uniform_(
            -self.rotation_degrees,
            self.rotation_degrees,
        ) * (math.pi / 180.0)
        scale = torch.empty(count, device=device, dtype=dtype).uniform_(
            self.scale_range[0],
            self.scale_range[1],
        )
        inv_scale = 1.0 / scale.clamp_min(1e-6)
        cos_a = torch.cos(angle) * inv_scale
        sin_a = torch.sin(angle) * inv_scale
        translate = torch.empty(count, 3, device=device, dtype=dtype).uniform_(
            -self.translation_fraction,
            self.translation_fraction,
        )

        theta = torch.zeros(count, 3, 4, device=device, dtype=dtype)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        theta[:, 2, 2] = inv_scale
        theta[:, :, 3] = translate

        grid = F.affine_grid(
            theta,
            size=(count, 1, depth, height, width),
            align_corners=False,
        )
        return F.grid_sample(
            ct,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def _batch_intensity(self, ct: torch.Tensor) -> torch.Tensor:
        count = ct.shape[0]
        shape = (count, 1, 1, 1, 1)
        device, dtype = ct.device, ct.dtype
        image = ct[:, :1]

        if self.intensity_scale_range[0] != 1.0 or self.intensity_scale_range[1] != 1.0:
            image = image * torch.empty(shape, device=device, dtype=dtype).uniform_(
                self.intensity_scale_range[0],
                self.intensity_scale_range[1],
            )
        if self.intensity_shift_range[0] != 0.0 or self.intensity_shift_range[1] != 0.0:
            image = image + torch.empty(shape, device=device, dtype=dtype).uniform_(
                self.intensity_shift_range[0],
                self.intensity_shift_range[1],
            )
        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std
        if ct.shape[1] == 1:
            return image
        return torch.cat((image, ct[:, 1:]), dim=1)


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
