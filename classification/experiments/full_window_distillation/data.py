"""Original-DICOM slice sampling and on-demand five-window normalization."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


WINDOWS = {"lung": (-600., 1500.), "mediastinal": (20., 350.),
           "hrct": (-600., 2000.), "zero": (0., 1500.), "bone": (250., 1000.)}


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def transform_window(x: torch.Tensor, name: str, stats: dict) -> torch.Tensor:
    level, width = WINDOWS[name]
    x = (x.clamp(level-width/2, level+width/2) - (level-width/2)) / width
    return (x - stats[name]["mean"]) / stats[name]["std"]


def load_original(record: dict) -> tuple[np.ndarray, dict]:
    """Use the already audited series UID, retaining HU and true orientation.

    Sampling interpretation: discard outer 10%, then select 32 evenly spaced
    slices between 40% and 90% of the original apex-to-base stack. This explicit
    choice resolves the paper's ambiguous 'region starting at 40%' description.
    """
    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(record["dicom_dir"], record["series_uid"]))
    if not files:
        raise ValueError("Audited DICOM series unavailable")
    dropped = 0
    if record.get("odd_slices_dropped", 0):
        sizes = []
        for file in files:
            metadata = sitk.ImageFileReader()
            metadata.SetFileName(file)
            metadata.ReadImageInformation()
            sizes.append(tuple(metadata.GetSize()[:2]))
        common = Counter(sizes).most_common(1)[0][0]
        selected = [f for f, size in zip(files, sizes) if size == common]
        dropped = len(files) - len(selected)
        files = selected
    reader.SetFileNames(files)
    image = sitk.DICOMOrient(reader.Execute(), "LPS")
    # LPS positive z points superiorly; reverse to apex -> base.
    hu = sitk.GetArrayFromImage(image).astype(np.float32)[::-1].copy()
    if hu.ndim != 3 or not np.isfinite(hu).all() or hu.min() > -500 or hu.max() < 50:
        raise ValueError("Expected finite 3D chest CT retaining HU")
    n = len(hu)
    low, high = int(np.floor(n * .4)), int(np.ceil(n * .9))
    if high-low < 32:
        low, high = int(np.floor(n * .1)), int(np.ceil(n * .9))
    if high-low < 32:
        low, high = 0, n
    if high-low < 32:
        raise ValueError("Fewer than 32 unique axial slices; manual review required")
    indices = np.rint(np.linspace(low, high-1, 32)).astype(int)
    volume = torch.from_numpy(hu[indices].copy())[:, None]
    height, width = volume.shape[-2:]
    side = max(height, width)
    if height != width:
        dh, dw = side-height, side-width
        volume = torch.nn.functional.pad(volume, (dw//2, dw-dw//2, dh//2, dh-dh//2), value=-1000.)
    # Windowing commutes with this resize only when no clipping is involved;
    # the paper resizes HU slices first and applies windows afterwards.
    volume = torch.nn.functional.interpolate(volume, size=(512, 512), mode="bilinear",
                                             align_corners=False)[:, 0].numpy()
    metadata = {"source_shape": list(hu.shape), "indices_apex_to_base": indices.tolist(),
                "dropped_odd_slices": dropped, "square_padding_h_w": [side-height, side-width],
                "source_series_uid": record["series_uid"],
                "direction_lps": list(image.GetDirection()), "spacing": list(image.GetSpacing())}
    return volume, metadata


def prepare(records: list[dict], source_records: dict, cache: Path, fingerprint: str,
            check_deadline) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    identity = cache / "identity.json"
    if identity.exists():
        if json.loads(identity.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Cache provenance mismatch")
    else:
        atomic_json(identity, {"fingerprint": fingerprint})
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)

    def one(row: dict) -> None:
        check_deadline()
        path = cache / f"{row['patient_id']}.npy"
        if path.exists():
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if existing.shape != (32, 512, 512) or existing.dtype != np.float32:
                raise ValueError("Invalid cached shape/dtype")
            return
        volume, meta = load_original(source_records[row["patient_id"]])
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.save(stream, volume, allow_pickle=False)
        atomic_json(path.with_suffix(".json"), meta)
        temp.replace(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, _ in enumerate(pool.map(one, records), 1):
            if i % 10 == 0 or i == len(records):
                print(f"DICOM cache {i}/{len(records)}", flush=True)
    stat_path = cache / "stats.json"
    if stat_path.exists():
        return json.loads(stat_path.read_text())
    totals = {w: [0., 0., 0] for w in WINDOWS}
    for i, row in enumerate(records):
        check_deadline()
        if row["split"] != "train":
            continue
        hu = np.load(cache / f"{row['patient_id']}.npy", allow_pickle=False)
        for w, (level, width) in WINDOWS.items():
            x = (np.clip(hu, level-width/2, level+width/2)-(level-width/2))/width
            totals[w][0] += x.sum(dtype=np.float64)
            totals[w][1] += np.square(x).sum(dtype=np.float64)
            totals[w][2] += x.size
        if i % 50 == 0:
            print(f"Training-only normalization {i}/{len(records)}", flush=True)
    stats = {}
    for w, (total, squared, count) in totals.items():
        mean = float(total / count)
        stats[w] = {"mean": mean, "std": float(max(squared/count-mean**2, 1e-12)**.5)}
    atomic_json(stat_path, stats)
    return stats


class RawSlices(Dataset):
    """Single HU cache shared by all windows; optional frozen teacher features."""

    def __init__(self, records: list[dict], cache: Path, split: str, teacher: dict | None = None):
        self.rows = [r for r in records if r["split"] == split]
        self.cache, self.teacher = cache, teacher

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        hu = np.load(self.cache / f"{row['patient_id']}.npy", allow_pickle=False)
        sample = {"hu": torch.from_numpy(hu[None]), "label": float(row["label"]),
                  "patient_id": row["patient_id"]}
        if self.teacher is not None:
            sample["teacher"] = self.teacher[row["patient_id"]]
        return sample
