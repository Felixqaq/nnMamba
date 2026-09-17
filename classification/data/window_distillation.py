"""Paired CT windows from HU NIfTI files, with explicit patient splits."""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize
from torch.utils.data import Dataset


WINDOWS = {"lung": (-600.0, 1500.0), "mediastinal": (20.0, 350.0)}


def read_manifest(path: Path) -> list[dict]:
    """Require one original HU scan per patient and preassigned disjoint splits."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"patient_id", "path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires patient_id,path,label,split columns")
        records = list(reader)
    ids, paths = set(), set()
    for row in records:
        patient = row["patient_id"].strip()
        scan = Path(row["path"])
        scan = (path.parent / scan).resolve() if not scan.is_absolute() else scan.resolve()
        if not patient or patient in ids or scan in paths:
            raise ValueError("Duplicate/empty patient ID or repeated scan; use one scan per patient")
        if row["label"] not in {"0", "1"} or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Labels must be 0/1 and splits train/val/test")
        if not scan.is_file():
            raise FileNotFoundError(scan)
        ids.add(patient)
        paths.add(scan)
        row.update(patient_id=patient, path=str(scan), label=int(row["label"]))
    for split in ("train", "val", "test"):
        if {r["label"] for r in records if r["split"] == split} != {0, 1}:
            raise ValueError(f"Both classes are required in {split}")
    return records


def window_array(hu: np.ndarray, name: str) -> np.ndarray:
    """Map a HU volume through a fixed CT window to [0, 1]."""
    level, width = WINDOWS[name]
    return ((np.clip(hu, level - width / 2, level + width / 2)
             - (level - width / 2)) / width).astype(np.float32)


def prepare_cache(records: list[dict], directory: Path, shape: tuple[int, int, int],
                  check_deadline=lambda: None) -> dict:
    """Window before resizing, and fit global normalization on training voxels only.

    RAS orientation and whole-volume resizing are deliberate nnMamba adaptations;
    this does not reproduce the paper's 32-slice sampling. Input must retain HU.
    """
    directory.mkdir(parents=True, exist_ok=False)
    totals = {name: [0.0, 0.0, 0] for name in WINDOWS}
    for index, row in enumerate(records):
        check_deadline()
        if index % 25 == 0:
            print(f"Preparing paired HU windows: {index}/{len(records)}", flush=True)
        image = nib.as_closest_canonical(nib.load(row["path"]))
        hu = image.get_fdata(dtype=np.float32)
        if hu.ndim != 3 or not np.isfinite(hu).all():
            raise ValueError("Expected a finite 3D HU volume")
        # Reject obvious normalized/windowed inputs, never fabricate missing HU.
        if hu.min() > -500 or hu.max() < 50:
            raise ValueError("Input does not appear to retain HU; verify the source scans")
        for name in WINDOWS:
            volume = resize(window_array(hu, name), shape, order=1,
                            preserve_range=True, anti_aliasing=True).astype(np.float32)
            np.save(directory / f"{index}_{name}.npy", volume)
            if row["split"] == "train":
                values = volume.astype(np.float64)
                totals[name][0] += values.sum()
                totals[name][1] += np.square(values).sum()
                totals[name][2] += values.size
    stats = {}
    for name, (total, squares, count) in totals.items():
        mean = total / count
        stats[name] = {"mean": mean, "std": max((max(squares / count - mean**2, 0))**0.5, 1e-6)}
    return stats


class WindowDataset(Dataset):
    """Return spatially aligned views; no independent random augmentations."""

    def __init__(self, records: list[dict], directory: Path, stats: dict, split: str):
        self.rows = [(i, row) for i, row in enumerate(records) if row["split"] == split]
        self.directory, self.stats = directory, stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        cache_index, row = self.rows[index]
        sample = {"patient_id": row["patient_id"], "label": float(row["label"])}
        for name in WINDOWS:
            values = np.load(self.directory / f"{cache_index}_{name}.npy", allow_pickle=False)
            values = (values - self.stats[name]["mean"]) / self.stats[name]["std"]
            sample[name] = torch.from_numpy(values[None].astype(np.float32))
        return sample
