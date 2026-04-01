"""Shared helpers for regression data scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Iterable


@dataclass(frozen=True)
class RegressionRecord:
    """Single regression sample entry."""

    patient_id: str
    path: Path
    subset: str
    original_group: str
    angle_group: str
    angle: float
    binary_label: int


def repo_root() -> Path:
    """Return repository root by walking up from this file."""
    return Path(__file__).resolve().parents[2]


def default_source_root() -> Path:
    """Return the default CT source root."""
    return repo_root() / "by_angle_all"


def default_angle_json() -> Path:
    """Return the label source JSON path."""
    return repo_root() / "patient_angle_classification_by_group.json"


def default_output_dir() -> Path:
    """Return the default regression datasets output directory."""
    return repo_root() / "regression" / "datasets" / "generated"


def load_angle_lookup(angle_json: Path) -> dict[str, dict[str, object]]:
    """Load patient angle metadata from the source JSON."""
    with angle_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    lookup: dict[str, dict[str, object]] = {}

    for group_name, group_data in raw.items():
        if not isinstance(group_data, dict):
            continue
        by_angle = group_data.get("by_angle", {})
        for angle_group, patient_map in by_angle.items():
            for patient_id, angle in patient_map.items():
                lookup[str(patient_id)] = {
                    "angle": float(angle),
                    "group": group_name,
                    "angle_group": angle_group,
                }

    return lookup


def patient_id_from_filename(path: Path) -> str:
    """Extract the patient id prefix from a CT filename."""
    return path.name.split("_", 1)[0].strip()


def iter_ct_files(source_root: Path) -> Iterable[Path]:
    """Yield all NIfTI CT files under the angle-sorted source tree."""
    for subset_dir in sorted(source_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        for path in sorted(subset_dir.glob("*.nii.gz")):
            yield path


def timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
