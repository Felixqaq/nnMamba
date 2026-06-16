"""Dataset classes for CT regression and classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import nibabel as nib
import numpy as np
from skimage import transform
import torch
from torch.utils.data import Dataset

from .manifest import AngleRecord, build_angle_manifest


DEFAULT_IMAGE_SIZE = (112, 136, 112)
InputNormalization = Literal["zscore", "none"]
CLASSIFICATION_TARGET_MODES = {
    "gold",
    "gold_severity4",
    "angle_3class",
    "angle_binary_extreme",
    "oi_emphysema",
}


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
    """PyTorch dataset for CT tasks using a shared volume pipeline."""

    def __init__(
        self,
        data_root: str | Path,
        labels_json: str | Path,
        pft_json: str | Path | None = None,
        oi_json: str | Path | None = None,
        oi_threshold: float = 4.38,
        oi_exclude_range: tuple[float, float] | list[float] | None = None,
        target_mode: str = "angle",
        image_size: tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
        intensity_window: tuple[float, float] | None = None,
        input_normalization: InputNormalization = "zscore",
        records: Sequence[AngleRecord] | None = None,
        tapct_embeddings: Mapping[str, np.ndarray] | None = None,
        transform=None,
        cache_data: bool = True,
        load_ct_data: bool = True,
        gold_exclude_class_indices: tuple[int, ...] | list[int] | None = None,
        gold_remap_class_indices: bool = False,
    ):
        self.data_root = Path(data_root)
        self.labels_json = Path(labels_json)
        self.pft_json = Path(pft_json) if pft_json is not None else None
        self.oi_json = Path(oi_json) if oi_json is not None else None
        self.oi_threshold = float(oi_threshold)
        self.oi_exclude_range = (
            (float(oi_exclude_range[0]), float(oi_exclude_range[1]))
            if oi_exclude_range is not None
            else None
        )
        self.target_mode = target_mode
        self.image_size = image_size
        self.intensity_window = intensity_window
        self.input_normalization = input_normalization
        self.tapct_embeddings = dict(tapct_embeddings or {})
        self.transform = transform
        self.cache_data = cache_data
        self.load_ct_data = bool(load_ct_data)
        self.gold_exclude_class_indices = tuple(gold_exclude_class_indices or ())
        self.gold_remap_class_indices = bool(gold_remap_class_indices)

        if records is None:
            manifest = build_angle_manifest(
                self.data_root,
                self.labels_json,
                pft_json=self.pft_json,
                target_mode=self.target_mode,
                oi_json=self.oi_json,
                oi_threshold=self.oi_threshold,
                oi_exclude_range=self.oi_exclude_range,
                gold_exclude_class_indices=self.gold_exclude_class_indices,
                gold_remap_class_indices=self.gold_remap_class_indices,
            )
            self.records = list(manifest.records)
        else:
            self.records = list(records)

        self.cached_data: list[dict] = []
        if self.cache_data:
            self._preload_all()

    def _build_sample(self, record: AngleRecord) -> dict:
        """Load and assemble a sample dictionary for one CT volume."""
        if self.target_mode in CLASSIFICATION_TARGET_MODES:
            if record.class_index is None:
                raise ValueError(
                    f"Missing class label for patient {record.patient_id} "
                    f"in {self.target_mode} mode."
                )
            target = np.array(record.class_index, dtype=np.int64)
        else:
            target_value = record.target if record.target is not None else record.angle
            target = np.array(target_value, dtype=np.float32)

        if self.load_ct_data:
            ct = load_ct(
                record.path,
                self.image_size,
                intensity_window=self.intensity_window,
                input_normalization=self.input_normalization,
            )
        else:
            ct = np.zeros((1, 1, 1, 1), dtype=np.float32)

        sample = {
            "ct": ct,
            "target": target,
            "angle": np.array(record.angle, dtype=np.float32),
            "oi": (
                np.array(record.oi, dtype=np.float32)
                if record.oi is not None
                else None
            ),
            "a": record.a,
            "fvc": record.fvc,
            "pef": record.pef,
            "label": (
                np.array(record.class_index, dtype=np.int64)
                if record.class_index is not None
                else None
            ),
            "patient_id": record.patient_id,
            "source_group": record.source_group,
            "class_label": record.class_label,
            "gold_stage_label": record.gold_stage_label,
            "post_fev1_percent_predicted": record.post_fev1_percent_predicted,
            "path": record.path,
        }
        if self.tapct_embeddings:
            embedding = self.tapct_embeddings.get(record.patient_id)
            if embedding is None:
                raise KeyError(
                    f"Missing TAP-CT embedding for patient {record.patient_id}."
                )
            sample["tapct_embedding"] = np.asarray(embedding, dtype=np.float32)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _preload_all(self) -> None:
        """Preload all CTs into memory for fast k-fold iteration."""
        from tqdm import tqdm

        self.cached_data = []
        for record in tqdm(self.records, desc="Caching CTs", leave=False):
            self.cached_data.append(self._build_sample(record))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        record = self.records[idx]
        if self.cache_data and self.cached_data:
            sample = dict(self.cached_data[idx])
        else:
            sample = self._build_sample(record)
        return sample
