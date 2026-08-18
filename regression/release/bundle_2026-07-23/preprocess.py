"""FROZEN CT preprocessing — bit-for-bit copy of nnMamba regression/data/dataset.py.

Do not edit without re-verifying against training via package_release.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from skimage import transform

InputNormalization = Literal["zscore", "none"]


def _resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
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


def _compute_hash() -> str:
    import inspect

    src = "".join(
        inspect.getsource(fn) for fn in (_resize_volume, _normalize_volume, load_ct)
    )
    return hashlib.sha256(src.encode()).hexdigest()


PREPROCESS_HASH = _compute_hash()
