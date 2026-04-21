"""Manifest helpers for CT angle regression and GOLD-stage classification.

The dataset is organized under ``by_angle_all/`` and labels are read from
``patient_angle_classification_by_group.json`` and optionally ``pft.json``.
Each CT filename begins with the patient identifier, which is used to match the
target label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re


@dataclass(frozen=True)
class AngleRecord:
    """Single CT sample and its available targets."""

    patient_id: str
    path: str
    angle: float
    source_group: str
    gold_stage: int | None = None
    gold_stage_label: str | None = None
    post_fev1_percent_predicted: float | None = None


@dataclass
class AngleManifest:
    """Complete dataset manifest for regression/classification tasks."""

    source_json: str
    data_root: str
    records: list[AngleRecord]
    counts: dict[str, int]
    missing_from_source: list[str]
    extra_in_source_not_in_json: list[str]
    missing_gold_labels: list[str]
    gold_stage_counts: dict[str, int]
    class_names: list[str]

    def to_dict(self) -> dict:
        return {
            "source_json": self.source_json,
            "data_root": self.data_root,
            "counts": self.counts,
            "missing_from_source": self.missing_from_source,
            "extra_in_source_not_in_json": self.extra_in_source_not_in_json,
            "missing_gold_labels": self.missing_gold_labels,
            "gold_stage_counts": self.gold_stage_counts,
            "class_names": self.class_names,
            "records": [asdict(record) for record in self.records],
        }


def load_angle_label_map(labels_json: str | Path) -> dict[str, float]:
    """Load patient-id -> angle mapping from the provided JSON annotation file."""
    labels_json = Path(labels_json)
    with labels_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    label_map: dict[str, float] = {}
    for group in data.values():
        by_angle = group.get("by_angle", {})
        for bucket in by_angle.values():
            for patient_id, angle in bucket.items():
                label_map[str(patient_id)] = float(angle)

    return label_map


def _gold_stage_sort_key(stage_name: str) -> tuple[int, str]:
    """Sort GOLD labels by their numeric stage when available."""
    match = re.search(r"(\d+)", stage_name)
    if match:
        return int(match.group(1)), stage_name
    return (10_000, stage_name)


def _english_gold_label(stage_name: str) -> str:
    """Normalize GOLD class names into English-only labels."""
    match = re.search(r"(\d+)", stage_name)
    if not match:
        return stage_name

    stage = int(match.group(1))
    aliases = {
        1: "GOLD 1 (Mild)",
        2: "GOLD 2 (Moderate)",
        3: "GOLD 3 (Severe)",
        4: "GOLD 4 (Very Severe)",
    }
    return aliases.get(stage, f"GOLD {stage}")


def load_gold_label_map(
    pft_json: str | Path,
) -> tuple[dict[str, dict[str, float | int | str]], list[str]]:
    """Load patient-id -> GOLD stage mapping from the provided PFT JSON."""
    pft_json = Path(pft_json)
    with pft_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    raw_class_names = sorted((str(key) for key in data.keys()), key=_gold_stage_sort_key)
    class_names = [_english_gold_label(class_name) for class_name in raw_class_names]
    label_map: dict[str, dict[str, float | int | str]] = {}

    for class_index, raw_class_name in enumerate(raw_class_names):
        english_class_name = class_names[class_index]
        for record in data.get(raw_class_name, []):
            patient_id = str(record["patient_id"])
            fev1 = record.get("post_fev1_percent_predicted")
            label_map[patient_id] = {
                "gold_stage": class_index,
                "gold_stage_label": english_class_name,
                "post_fev1_percent_predicted": (
                    float(fev1) if fev1 is not None else None
                ),
            }

    return label_map, class_names


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
    pft_json: str | Path | None = None,
    target_mode: str = "angle",
) -> AngleManifest:
    """Create a manifest from the on-disk CTs and the angle annotation JSON."""
    data_root = Path(data_root)
    labels_json = Path(labels_json)
    label_map = load_angle_label_map(labels_json)
    gold_label_map: dict[str, dict[str, float | int | str]] = {}
    class_names: list[str] = []
    if pft_json is not None:
        gold_label_map, class_names = load_gold_label_map(pft_json)
    if target_mode == "gold" and not gold_label_map:
        raise ValueError(
            "target_mode='gold' requires a valid pft_json with GOLD stage labels."
        )

    records: list[AngleRecord] = []
    source_ids: set[str] = set()
    missing_from_source: list[str] = []
    missing_gold_labels: list[str] = []

    for ct_path in iter_ct_files(data_root):
        patient_id = _extract_patient_id(ct_path)
        source_ids.add(patient_id)
        angle = label_map.get(patient_id)
        if angle is None:
            missing_from_source.append(patient_id)
            continue

        gold_meta = gold_label_map.get(patient_id)
        if target_mode == "gold" and gold_meta is None:
            missing_gold_labels.append(patient_id)
            continue

        source_group = ct_path.parent.name
        records.append(
            AngleRecord(
                patient_id=patient_id,
                path=str(ct_path),
                angle=float(angle),
                source_group=source_group,
                gold_stage=(
                    int(gold_meta["gold_stage"]) if gold_meta is not None else None
                ),
                gold_stage_label=(
                    str(gold_meta["gold_stage_label"])
                    if gold_meta is not None
                    else None
                ),
                post_fev1_percent_predicted=(
                    float(gold_meta["post_fev1_percent_predicted"])
                    if gold_meta is not None
                    and gold_meta["post_fev1_percent_predicted"] is not None
                    else None
                ),
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
    gold_stage_counts = {
        class_name: sum(1 for record in records if record.gold_stage_label == class_name)
        for class_name in class_names
    }

    return AngleManifest(
        source_json=str(labels_json),
        data_root=str(data_root),
        records=records,
        counts=counts,
        missing_from_source=sorted(set(missing_from_source)),
        extra_in_source_not_in_json=extra_in_source_not_in_json,
        missing_gold_labels=sorted(set(missing_gold_labels)),
        gold_stage_counts=gold_stage_counts,
        class_names=class_names,
    )


def save_manifest(manifest: AngleManifest, path: str | Path) -> Path:
    """Persist the manifest for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    return path
