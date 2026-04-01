"""Manifest helpers for CT collapse-angle regression.

The dataset is organized under ``by_angle_all/`` and labels are read from
``patient_angle_classification_by_group.json``. Each CT filename begins with the
patient identifier, which is used to match the target angle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class AngleRecord:
    """Single CT sample and its regression target."""

    patient_id: str
    path: str
    angle: float
    source_group: str


@dataclass
class AngleManifest:
    """Complete dataset manifest for regression."""

    source_json: str
    data_root: str
    records: list[AngleRecord]
    counts: dict[str, int]
    missing_from_source: list[str]
    extra_in_source_not_in_json: list[str]

    def to_dict(self) -> dict:
        return {
            "source_json": self.source_json,
            "data_root": self.data_root,
            "counts": self.counts,
            "missing_from_source": self.missing_from_source,
            "extra_in_source_not_in_json": self.extra_in_source_not_in_json,
            "records": [asdict(record) for record in self.records],
        }


def load_angle_label_map(labels_json: str | Path) -> dict[str, float]:
    """Load patient-id -> angle mapping from the provided JSON annotation file."""
    labels_json = Path(labels_json)
    with labels_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: dict[str, float] = {}
    for group in data.values():
        by_angle = group.get("by_angle", {})
        for bucket in by_angle.values():
            for patient_id, angle in bucket.items():
                label_map[str(patient_id)] = float(angle)

    return label_map


def _extract_patient_id(path: Path) -> str:
    """Get the patient identifier from a CT filename."""
    name = path.name
    if name.endswith(".nii.gz"):
        base = name[:-7]
    else:
        base = path.stem
    return base.split("_", 1)[0]


def iter_ct_files(data_root: str | Path) -> list[Path]:
    """List CT files under the angle-organized folders."""
    data_root = Path(data_root)
    return sorted(p for p in data_root.rglob("*.nii.gz") if p.is_file())


def build_angle_manifest(
    data_root: str | Path,
    labels_json: str | Path,
) -> AngleManifest:
    """Create a manifest from the on-disk CTs and the angle annotation JSON."""
    data_root = Path(data_root)
    labels_json = Path(labels_json)
    label_map = load_angle_label_map(labels_json)

    records: list[AngleRecord] = []
    source_ids: set[str] = set()
    missing_from_source: list[str] = []

    for ct_path in iter_ct_files(data_root):
        patient_id = _extract_patient_id(ct_path)
        source_ids.add(patient_id)
        angle = label_map.get(patient_id)
        if angle is None:
            missing_from_source.append(patient_id)
            continue

        source_group = ct_path.parent.name
        records.append(
            AngleRecord(
                patient_id=patient_id,
                path=str(ct_path),
                angle=float(angle),
                source_group=source_group,
            )
        )

    extra_in_source_not_in_json = sorted(
        patient_id for patient_id in label_map.keys() if patient_id not in source_ids
    )

    counts = {
        "total": len(records),
        "unique_patients": len({record.patient_id for record in records}),
        "low_angle_group": sum(1 for record in records if "low" in record.source_group),
        "high_angle_group": sum(1 for record in records if "high" in record.source_group),
    }

    return AngleManifest(
        source_json=str(labels_json),
        data_root=str(data_root),
        records=records,
        counts=counts,
        missing_from_source=sorted(set(missing_from_source)),
        extra_in_source_not_in_json=extra_in_source_not_in_json,
    )


def save_manifest(manifest: AngleManifest, path: str | Path) -> Path:
    """Persist the manifest for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    return path
