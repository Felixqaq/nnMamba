"""Dataset classes for CT angle regression."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import nibabel as nib
import numpy as np
from skimage import transform
import torch
from torch.utils.data import Dataset

from .manifest import AngleRecord, build_angle_manifest


DEFAULT_IMAGE_SIZE = (112, 136, 112)
InputNormalization = Literal["zscore", "none"]


def _resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a 3D CT volume to the requested shape."""
    if volume.shape == target_shape:
        return volume
    return transform.resize(
        volume,
        target_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )


def _normalize_volume(
    volume: np.ndarray,
    intensity_window: tuple[float, float] | None = None,
    input_normalization: InputNormalization = "zscore",
) -> np.ndarray:
    """Apply optional CT preprocessing and normalization."""
    if intensity_window is not None:
        lo, hi = intensity_window
        volume = np.clip(volume, lo, hi)
    if input_normalization == "none":
        return volume
    if input_normalization != "zscore":
        raise ValueError(f"Unsupported input_normalization: {input_normalization}")
    lo, hi = np.percentile(volume, [1, 99])
    if hi > lo:
        volume = np.clip(volume, lo, hi)
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        std = 1.0
    return (volume - mean) / std


def load_ct(
    path: str | Path,
    image_size: tuple[int, int, int],
    intensity_window: tuple[float, float] | None = None,
    input_normalization: InputNormalization = "zscore",
) -> np.ndarray:
    """Load a CT volume from disk and return a channel-first array."""
    path = Path(path)
    volume = nib.load(str(path)).get_fdata().astype(np.float32)

    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape} for {path}")

    volume = _resize_volume(volume, image_size).astype(np.float32)
    volume = _normalize_volume(
        volume,
        intensity_window=intensity_window,
        input_normalization=input_normalization,
    )
    return np.expand_dims(volume, axis=0).astype(np.float32)


class AngleRegressionDataset(Dataset):
    """PyTorch dataset for collapse-angle regression from CT."""

    def __init__(
        self,
        data_root: str | Path,
        labels_json: str | Path,
        image_size: tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
        intensity_window: tuple[float, float] | None = None,
        input_normalization: InputNormalization = "zscore",
        records: Sequence[AngleRecord] | None = None,
        transform=None,
        cache_data: bool = True,
    ):
        self.data_root = Path(data_root)
        self.labels_json = Path(labels_json)
        self.image_size = image_size
        self.intensity_window = intensity_window
        self.input_normalization = input_normalization
        self.transform = transform
        self.cache_data = cache_data

        if records is None:
            manifest = build_angle_manifest(self.data_root, self.labels_json)
            self.records = list(manifest.records)
        else:
            self.records = list(records)

        self.cached_data: list[dict] = []
        if self.cache_data:
            self._preload_all()

    def _preload_all(self) -> None:
        """Preload all CTs into memory for fast k-fold iteration."""
        from tqdm import tqdm

        self.cached_data = []
        for record in tqdm(self.records, desc="Caching CTs", leave=False):
            sample = {
                "ct": load_ct(
                    record.path,
                    self.image_size,
                    intensity_window=self.intensity_window,
                    input_normalization=self.input_normalization,
                ),
                "angle": np.array(record.angle, dtype=np.float32),
                "patient_id": record.patient_id,
                "source_group": record.source_group,
                "path": record.path,
            }
            if self.transform is not None:
                sample = self.transform(sample)
            self.cached_data.append(sample)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        record = self.records[idx]
        if self.cache_data and self.cached_data:
            sample = dict(self.cached_data[idx])
        else:
            sample = {
                "ct": load_ct(
                    record.path,
                    self.image_size,
                    intensity_window=self.intensity_window,
                    input_normalization=self.input_normalization,
                ),
                "angle": np.array(record.angle, dtype=np.float32),
                "patient_id": record.patient_id,
                "source_group": record.source_group,
                "path": record.path,
            }
            if self.transform is not None:
                sample = self.transform(sample)
        return sample
