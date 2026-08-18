"""Dataset classes for CT regression and classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage import transform
import torch
from torch.utils.data import Dataset

from .manifest import AngleRecord, build_angle_manifest


DEFAULT_IMAGE_SIZE = (112, 136, 112)
InputNormalization = Literal["zscore", "none"]
LungMaskMode = Literal["off", "zero_outside", "crop", "crop_and_zero"]
CLASSIFICATION_TARGET_MODES = {
    "gold",
    "gold_severity4",
    "angle_3class",
    "angle_binary_extreme",
    "oi_emphysema",
    "oi_3class",
    "normal_v_abnormal",
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


def _resample_to_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Resample a CT volume so one voxel spans a fixed physical size."""
    factors = [
        float(source) / float(target)
        for source, target in zip(source_spacing, target_spacing)
    ]
    shape = tuple(
        max(1, int(round(size * factor))) for size, factor in zip(volume.shape, factors)
    )
    if shape == volume.shape:
        return volume
    return transform.resize(
        volume,
        shape,
        order=1,
        preserve_range=True,
        anti_aliasing=any(factor < 1.0 for factor in factors),
    ).astype(np.float32)


def _content_center(volume: np.ndarray, threshold: float = -400.0) -> tuple[int, ...]:
    """Return the bounding-box center of the scanned body, for content-aware cropping."""
    mask = volume > threshold
    if not mask.any():
        return tuple(size // 2 for size in volume.shape)
    center = []
    for axis in range(volume.ndim):
        other = tuple(i for i in range(volume.ndim) if i != axis)
        present = np.where(mask.any(axis=other))[0]
        center.append(int((present[0] + present[-1]) // 2))
    return tuple(center)


def _crop_to_shape(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    center: tuple[int, ...],
) -> np.ndarray:
    """Crop oversized axes around the given center, leaving smaller axes untouched."""
    slices = []
    for size, target, axis_center in zip(volume.shape, target_shape, center):
        if size <= target:
            slices.append(slice(0, size))
            continue
        start = int(np.clip(axis_center - target // 2, 0, size - target))
        slices.append(slice(start, start + target))
    return volume[tuple(slices)]


def _pad_to_shape(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    pad_value: float,
) -> np.ndarray:
    """Center-pad undersized axes with a constant value."""
    pads = []
    for size, target in zip(volume.shape, target_shape):
        total = max(0, target - size)
        before = total // 2
        pads.append((before, total - before))
    if all(pad == (0, 0) for pad in pads):
        return volume
    return np.pad(volume, pads, mode="constant", constant_values=pad_value)


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


def _normalize_then_pad(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    intensity_window: tuple[float, float] | None,
    input_normalization: InputNormalization,
    pad_hu: float,
) -> np.ndarray:
    """Normalize using scanned-content statistics only, then pad with air.

    Padding after normalization keeps the intensity statistics independent of how
    much padding a patient needs, so the amount of empty space cannot leak into
    the normalized values.
    """
    pad_value = float(pad_hu)
    if intensity_window is not None:
        lo, hi = intensity_window
        volume = np.clip(volume, lo, hi)
        pad_value = float(np.clip(pad_value, lo, hi))

    if input_normalization == "none":
        return _pad_to_shape(volume, target_shape, pad_value)
    if input_normalization != "zscore":
        raise ValueError(f"Unsupported input_normalization: {input_normalization}")

    lo, hi = np.percentile(volume, [1, 99])
    if hi > lo:
        volume = np.clip(volume, lo, hi)
        pad_value = float(np.clip(pad_value, lo, hi))
    mean = float(volume.mean())
    std = float(volume.std())
    if std < 1e-6:
        std = 1.0
    volume = (volume - mean) / std
    return _pad_to_shape(volume, target_shape, (pad_value - mean) / std)


def _apply_lung_mask(
    volume: np.ndarray,
    mask_path: str | Path,
    spacing: tuple[float, float, float],
    mode: LungMaskMode,
    dilate_mm: float,
    pad_hu: float,
) -> np.ndarray:
    """Blank out and/or crop to the lung region, on the source voxel grid."""
    mask = np.asarray(nib.load(str(mask_path)).dataobj).astype(bool)
    if mask.shape != volume.shape:
        raise ValueError(
            f"Lung mask shape {mask.shape} does not match image shape "
            f"{volume.shape} for {mask_path}"
        )
    if not mask.any():
        raise ValueError(f"Lung mask is empty: {mask_path}")

    if dilate_mm > 0:
        iterations = [max(0, int(round(dilate_mm / max(size, 1e-6)))) for size in spacing]
        if max(iterations) > 0:
            mask = ndimage.binary_dilation(
                mask, ndimage.generate_binary_structure(3, 1), iterations=max(iterations)
            )

    if mode in {"crop", "crop_and_zero"}:
        bounds = []
        for axis in range(3):
            other = tuple(i for i in range(3) if i != axis)
            present = np.where(mask.any(axis=other))[0]
            bounds.append((int(present[0]), int(present[-1]) + 1))
        region = tuple(slice(start, stop) for start, stop in bounds)
        volume = volume[region]
        mask = mask[region]

    if mode in {"zero_outside", "crop_and_zero"}:
        volume = np.where(mask, volume, np.float32(pad_hu))
    return volume


def load_ct(
    path: str | Path,
    image_size: tuple[int, int, int],
    intensity_window: tuple[float, float] | None = None,
    input_normalization: InputNormalization = "zscore",
    target_spacing: tuple[float, float, float] | None = None,
    pad_hu: float = -1000.0,
    lung_mask_path: str | Path | None = None,
    lung_mask_mode: LungMaskMode = "off",
    lung_mask_dilate_mm: float = 0.0,
) -> np.ndarray:
    """Load a CT volume from disk and return a channel-first array.

    With ``target_spacing`` set, the volume is resampled so one voxel spans a fixed
    physical size and then cropped/padded to ``image_size``. Anatomy therefore keeps
    its true scale across patients. Without it, the volume is resized to
    ``image_size`` directly, which makes the scale factor patient-dependent.
    """
    path = Path(path)
    image = nib.load(str(path))
    volume = image.get_fdata().astype(np.float32)

    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape} for {path}")

    source_spacing = tuple(float(zoom) for zoom in image.header.get_zooms()[:3])
    if lung_mask_mode != "off":
        if lung_mask_path is None:
            raise ValueError(f"lung_mask_mode={lung_mask_mode} requires a mask for {path}")
        volume = _apply_lung_mask(
            volume,
            mask_path=lung_mask_path,
            spacing=source_spacing,
            mode=lung_mask_mode,
            dilate_mm=lung_mask_dilate_mm,
            pad_hu=pad_hu,
        )

    if target_spacing is None:
        volume = _resize_volume(volume, image_size).astype(np.float32)
        volume = _normalize_volume(
            volume,
            intensity_window=intensity_window,
            input_normalization=input_normalization,
        )
        return np.expand_dims(volume, axis=0).astype(np.float32)

    volume = _resample_to_spacing(volume, source_spacing, target_spacing)
    volume = _crop_to_shape(volume, image_size, _content_center(volume))
    volume = _normalize_then_pad(
        volume,
        target_shape=image_size,
        intensity_window=intensity_window,
        input_normalization=input_normalization,
        pad_hu=pad_hu,
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
        target_spacing: tuple[float, float, float] | None = None,
        pad_hu: float = -1000.0,
        lung_mask_dir: str | Path | None = None,
        lung_mask_mode: LungMaskMode = "off",
        lung_mask_dilate_mm: float = 0.0,
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
        self.target_spacing = (
            tuple(float(value) for value in target_spacing)
            if target_spacing is not None
            else None
        )
        self.pad_hu = float(pad_hu)
        self.lung_mask_dir = Path(lung_mask_dir) if lung_mask_dir else None
        self.lung_mask_mode = lung_mask_mode
        self.lung_mask_dilate_mm = float(lung_mask_dilate_mm)
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

    def _lung_mask_path(self, record: AngleRecord) -> Path | None:
        """Resolve the lung mask for one record, if masking is enabled."""
        if self.lung_mask_mode == "off" or self.lung_mask_dir is None:
            return None
        mask_path = self.lung_mask_dir / f"{record.patient_id}.nii.gz"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing lung mask for patient {record.patient_id}: {mask_path}"
            )
        return mask_path

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
                target_spacing=self.target_spacing,
                pad_hu=self.pad_hu,
                lung_mask_path=self._lung_mask_path(record),
                lung_mask_mode=self.lung_mask_mode,
                lung_mask_dilate_mm=self.lung_mask_dilate_mm,
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
