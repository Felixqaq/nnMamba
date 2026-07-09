"""Manifest helpers for CT regression/classification tasks.

The dataset is organized under ``by_angle_all/`` and labels are read from
``patient_angle_classification_by_group.json``, optionally ``pft.json``, and
optionally OI labels. Each CT filename begins with the patient identifier,
which is used to match the target label.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re

ANGLE_3CLASS_NAMES = [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)",
]
ANGLE_BINARY_EXTREME_LOW_MAX = 131.0
ANGLE_BINARY_EXTREME_HIGH_MIN = 152.0
ANGLE_BINARY_EXTREME_NAMES = [
    "Abnormal/emphysema-like (AC <=131 deg)",
    "Normal-like (AC >=152 deg)",
]
# Disease-positive class listed first (index 0), matching the angle/OI binary
# convention so the evaluator reports sensitivity for the abnormal class.
NORMAL_V_ABNORMAL_NAMES = ["Abnormal", "Normal"]
GOLD_2026_CLASS_NAMES = [
    "Class 0 (No COPD)",
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)",
]
GOLD_SEVERITY4_CLASS_NAMES = GOLD_2026_CLASS_NAMES[1:]


@dataclass(frozen=True)
class AngleRecord:
    """Single CT sample and its available targets."""

    patient_id: str
    path: str
    angle: float
    source_group: str
    target: float | None = None
    class_index: int | None = None
    class_label: str | None = None
    gold_stage: int | None = None
    gold_stage_label: str | None = None
    post_fev1_percent_predicted: float | None = None
    oi: float | None = None
    a: float | None = None
    fvc: float | None = None
    pef: float | None = None


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
    class_counts: dict[str, int]
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
            "class_counts": self.class_counts,
            "class_names": self.class_names,
            "records": [asdict(record) for record in self.records],
        }


def load_angle_label_map(labels_json: str | Path) -> dict[str, float]:
    """Load patient-id -> angle mapping from the provided JSON annotation file."""
    labels_json = Path(labels_json)
    with labels_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    label_map: dict[str, float] = {}
    if all(isinstance(value, (int, float)) for value in data.values()):
        return {str(patient_id): float(angle) for patient_id, angle in data.items()}

    patients = data.get("patients")
    if isinstance(patients, dict):
        for patient_id, meta in patients.items():
            if isinstance(meta, dict) and "angle" in meta:
                label_map[str(patient_id)] = float(meta["angle"])
        if label_map:
            return label_map

    for group in data.values():
        if not isinstance(group, dict):
            continue
        by_angle = group.get("by_angle", {})
        for bucket in by_angle.values():
            for patient_id, angle in bucket.items():
                label_map[str(patient_id)] = float(angle)

    return label_map


def angle_3class_label(angle: float) -> tuple[int, str]:
    """Map a PFT angle into the fixed 131/152 degree three-class target."""
    if angle <= 131.0:
        return 0, ANGLE_3CLASS_NAMES[0]
    if angle < 152.0:
        return 1, ANGLE_3CLASS_NAMES[1]
    return 2, ANGLE_3CLASS_NAMES[2]


def angle_binary_extreme_label(angle: float) -> tuple[int, str] | None:
    """Map clear AC endpoints into a two-class target and drop the gray zone."""
    angle = float(angle)
    if angle <= ANGLE_BINARY_EXTREME_LOW_MAX:
        return 0, ANGLE_BINARY_EXTREME_NAMES[0]
    if angle >= ANGLE_BINARY_EXTREME_HIGH_MIN:
        return 1, ANGLE_BINARY_EXTREME_NAMES[1]
    return None


def normal_v_abnormal_label(source_group: str) -> tuple[int, str] | None:
    """Map the parent folder name (Normal/Abnormal) into the two-class target."""
    group = str(source_group).strip().lower()
    if group.startswith("abnormal"):
        return 0, NORMAL_V_ABNORMAL_NAMES[0]
    if group.startswith("normal"):
        return 1, NORMAL_V_ABNORMAL_NAMES[1]
    return None


def oi_emphysema_class_names(threshold: float) -> list[str]:
    """Return the two class names for the OI obstruction-index cutpoint.

    The abnormal (emphysema) endpoint is listed first so that binary
    sensitivity/specificity is reported for the disease-positive class,
    matching the angle binary-extreme convention.
    """
    return [
        f"Significant emphysema (OI >= {threshold:g})",
        f"No significant emphysema (OI < {threshold:g})",
    ]


def oi_emphysema_label(oi: float, threshold: float) -> tuple[int, str]:
    """Map an OI obstruction index into the emphysema two-class target."""
    names = oi_emphysema_class_names(threshold)
    if float(oi) >= float(threshold):
        return 0, names[0]
    return 1, names[1]


OI_3CLASS_DEFAULT_THRESHOLDS = (3.0, 7.0)


def oi_3class_class_names(thresholds: tuple[float, float]) -> list[str]:
    """Return the three OI-severity class names for the given cutpoints.

    Classes are ordered by ascending emphysema severity (0 = none) using the
    OI obstruction index alone (no FEV1 stratification):
      0 -> OI < low, 1 -> low <= OI < high, 2 -> OI >= high.
    """
    low, high = float(thresholds[0]), float(thresholds[1])
    return [
        f"No significant emphysema (OI < {low:g})",
        f"Moderate emphysema ({low:g} <= OI < {high:g})",
        f"Severe emphysema (OI >= {high:g})",
    ]


def oi_3class_label(oi: float, thresholds: tuple[float, float]) -> tuple[int, str]:
    """Map an OI obstruction index into the three-class severity target."""
    low, high = float(thresholds[0]), float(thresholds[1])
    names = oi_3class_class_names(thresholds)
    oi = float(oi)
    if oi < low:
        return 0, names[0]
    if oi < high:
        return 1, names[1]
    return 2, names[2]


def _gold_stage_sort_key(stage_name: str) -> tuple[int, str]:
    """Sort GOLD labels by their numeric stage when available."""
    match = re.search(r"(\d+)", stage_name)
    if match:
        return int(match.group(1)), stage_name
    return (10_000, stage_name)


def _english_gold_label(stage_name: str) -> str:
    """Normalize GOLD class names into English-only labels."""
    if re.search(r"\bclass\s*0\b", stage_name, flags=re.IGNORECASE):
        return GOLD_2026_CLASS_NAMES[0]

    match = re.search(r"(\d+)", stage_name)
    if not match:
        return stage_name

    stage = int(match.group(1))
    aliases = {
        0: GOLD_2026_CLASS_NAMES[0],
        1: "GOLD 1 (Mild)",
        2: "GOLD 2 (Moderate)",
        3: "GOLD 3 (Severe)",
        4: "GOLD 4 (Very Severe)",
    }
    return aliases.get(stage, f"GOLD {stage}")


def _gold_label_from_class_index(class_index: int) -> str:
    """Return the canonical GOLD 2026 label for a zero-based class index."""
    if 0 <= class_index < len(GOLD_2026_CLASS_NAMES):
        return GOLD_2026_CLASS_NAMES[class_index]
    return f"Class {class_index}"


def _gold_2026_records(data: dict) -> list[dict]:
    """Return patient records from a GOLD 2026 class JSON."""
    records = data.get("records")
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]

    by_class = data.get("by_class")
    if not isinstance(by_class, dict):
        return []

    output: list[dict] = []
    for class_key, class_records in by_class.items():
        match = re.search(r"(\d+)", str(class_key))
        if match is None or not isinstance(class_records, list):
            continue
        class_index = int(match.group(1))
        for record in class_records:
            if isinstance(record, dict):
                item = dict(record)
                item.setdefault("class", class_index)
                output.append(item)
    return output


def _load_gold_2026_label_map(
    data: dict,
) -> tuple[dict[str, dict[str, float | int | str | None]], list[str]]:
    """Load patient-id -> GOLD 2026 five-class mapping."""
    records = _gold_2026_records(data)
    class_indices = {
        int(record["class"])
        for record in records
        if "class" in record and record["class"] is not None
    }
    for class_key in (data.get("counts") or {}).keys():
        match = re.fullmatch(r"class_(\d+)", str(class_key))
        if match is not None:
            class_indices.add(int(match.group(1)))
    for class_key in (data.get("by_class") or {}).keys():
        match = re.fullmatch(r"class_(\d+)", str(class_key))
        if match is not None:
            class_indices.add(int(match.group(1)))

    class_names = (
        [
            _gold_label_from_class_index(class_index)
            for class_index in range(max(class_indices) + 1)
        ]
        if class_indices
        else []
    )
    label_map: dict[str, dict[str, float | int | str | None]] = {}

    for index, record in enumerate(records):
        patient_id = str(record.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"GOLD 2026 record {index} is missing patient_id.")
        if "class" not in record or record["class"] is None:
            raise ValueError(
                f"GOLD 2026 record for patient {patient_id} is missing class."
            )

        class_index = int(record["class"])
        label_map[patient_id] = {
            "gold_stage": class_index,
            "gold_stage_label": _gold_label_from_class_index(class_index),
            "post_fev1_percent_predicted": _optional_float(
                record,
                "post_fev1_percent_predicted",
            ),
        }

    return label_map, class_names


def load_gold_label_map(
    pft_json: str | Path,
) -> tuple[dict[str, dict[str, float | int | str | None]], list[str]]:
    """Load patient-id -> GOLD stage mapping from the provided PFT JSON."""
    pft_json = Path(pft_json)
    with pft_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict) and (
        isinstance(data.get("records"), list) or isinstance(data.get("by_class"), dict)
    ):
        return _load_gold_2026_label_map(data)

    raw_class_names = sorted((str(key) for key in data.keys()), key=_gold_stage_sort_key)
    class_names = [_english_gold_label(class_name) for class_name in raw_class_names]
    label_map: dict[str, dict[str, float | int | str | None]] = {}

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


def _optional_float(record: dict, key: str) -> float | None:
    """Return a finite-ish optional float from a JSON record."""
    value = record.get(key)
    if value is None or value == "":
        return None
    return float(value)


def load_oi_label_map(oi_json: str | Path) -> dict[str, dict[str, float | None]]:
    """Load patient-id -> OI regression targets from the processed OI JSON."""
    oi_json = Path(oi_json)
    with oi_json.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("OI labels must be a JSON array of patient records.")

    label_map: dict[str, dict[str, float | None]] = {}
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"OI record {index} must be a JSON object.")
        patient_id = str(record.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"OI record {index} is missing patient_id.")
        if "oi" not in record:
            raise ValueError(f"OI record for patient {patient_id} is missing oi.")
        label_map[patient_id] = {
            "oi": float(record["oi"]),
            "a": _optional_float(record, "a"),
            "fvc": _optional_float(record, "fvc"),
            "pef": _optional_float(record, "pef"),
        }

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
    pft_json: str | Path | None = None,
    target_mode: str = "angle",
    oi_json: str | Path | None = None,
    oi_threshold: float = 4.38,
    oi_thresholds: tuple[float, float] | list[float] | None = None,
    oi_exclude_range: tuple[float, float] | list[float] | None = None,
    gold_exclude_class_indices: tuple[int, ...] | list[int] | None = None,
    gold_remap_class_indices: bool = False,
) -> AngleManifest:
    """Create a manifest from the on-disk CTs and task annotation JSON."""
    data_root = Path(data_root)
    labels_json = Path(labels_json)
    target_mode = str(target_mode)
    is_gold_target = target_mode in {"gold", "gold_severity4"}
    is_oi_target = target_mode in {"oi", "oi_emphysema", "oi_3class"}
    is_folder_label_target = target_mode == "normal_v_abnormal"
    oi_3class_thresholds = (
        (float(oi_thresholds[0]), float(oi_thresholds[1]))
        if oi_thresholds is not None
        else OI_3CLASS_DEFAULT_THRESHOLDS
    )
    if oi_3class_thresholds[0] >= oi_3class_thresholds[1]:
        raise ValueError("oi_thresholds must satisfy low < high.")
    oi_exclude_bounds = (
        (float(oi_exclude_range[0]), float(oi_exclude_range[1]))
        if oi_exclude_range is not None
        else None
    )
    if oi_exclude_bounds is not None and oi_exclude_bounds[0] >= oi_exclude_bounds[1]:
        raise ValueError("oi_exclude_range must satisfy low < high.")
    excluded_gold_classes = {
        int(class_idx) for class_idx in (gold_exclude_class_indices or [])
    }
    label_map = load_angle_label_map(labels_json) if labels_json.exists() else {}
    oi_label_map: dict[str, dict[str, float | None]] = {}
    oi_path = Path(oi_json) if oi_json is not None else None
    if is_oi_target:
        if oi_path is None:
            raise ValueError(f"target_mode={target_mode!r} requires data.oi_json.")
        oi_label_map = load_oi_label_map(oi_path)
    gold_label_map: dict[str, dict[str, float | int | str | None]] = {}
    gold_class_names: list[str] = []
    pft_path = Path(pft_json) if pft_json is not None else None
    if pft_path is not None and (is_gold_target or pft_path.exists()):
        gold_label_map, gold_class_names = load_gold_label_map(pft_path)
    if is_gold_target and not gold_label_map:
        raise ValueError(
            f"target_mode={target_mode!r} requires a valid pft_json with GOLD stage labels."
        )
    active_gold_classes = [
        (class_idx, class_name)
        for class_idx, class_name in enumerate(gold_class_names)
        if class_idx not in excluded_gold_classes
    ]
    gold_class_remap = {
        class_idx: remapped_idx
        for remapped_idx, (class_idx, _) in enumerate(active_gold_classes)
    }
    if target_mode == "gold":
        class_names = (
            [class_name for _, class_name in active_gold_classes]
            if gold_remap_class_indices
            else list(gold_class_names)
        )
    elif target_mode == "gold_severity4":
        class_names = [
            class_name
            for class_idx, class_name in enumerate(gold_class_names)
            if class_idx > 0
        ]
    elif target_mode == "angle_3class":
        class_names = list(ANGLE_3CLASS_NAMES)
    elif target_mode == "angle_binary_extreme":
        class_names = list(ANGLE_BINARY_EXTREME_NAMES)
    elif target_mode == "oi_emphysema":
        class_names = oi_emphysema_class_names(oi_threshold)
    elif target_mode == "oi_3class":
        class_names = oi_3class_class_names(oi_3class_thresholds)
    elif target_mode == "normal_v_abnormal":
        class_names = list(NORMAL_V_ABNORMAL_NAMES)
    else:
        class_names = []

    records: list[AngleRecord] = []
    source_ids: set[str] = set()
    missing_from_source: list[str] = []
    missing_gold_labels: list[str] = []
    excluded_oi_range_count = 0

    for ct_path in iter_ct_files(data_root):
        patient_id = _extract_patient_id(ct_path)
        source_ids.add(patient_id)
        oi_meta = None
        if is_oi_target:
            oi_meta = oi_label_map.get(patient_id)
            if oi_meta is None:
                missing_from_source.append(patient_id)
                continue
            angle = label_map.get(patient_id, float("nan"))
            target = float(oi_meta["oi"])
            if (
                oi_exclude_bounds is not None
                and oi_exclude_bounds[0] <= target < oi_exclude_bounds[1]
            ):
                excluded_oi_range_count += 1
                continue
        elif is_folder_label_target:
            # Label comes from the parent folder name, not the angle JSON, so do
            # not skip patients that lack an angle annotation.
            angle = label_map.get(patient_id, float("nan"))
            target = float("nan")
        else:
            angle = label_map.get(patient_id)
            if angle is None:
                missing_from_source.append(patient_id)
                continue
            target = float(angle)

        gold_meta = gold_label_map.get(patient_id)
        if is_gold_target and gold_meta is None:
            missing_gold_labels.append(patient_id)
            continue
        if is_gold_target and gold_meta is not None:
            gold_stage = int(gold_meta["gold_stage"])
            if target_mode == "gold_severity4" and gold_stage == 0:
                continue
            if target_mode == "gold" and gold_stage in excluded_gold_classes:
                continue

        class_index = None
        class_label = None
        if target_mode == "gold" and gold_meta is not None:
            gold_stage = int(gold_meta["gold_stage"])
            class_index = (
                gold_class_remap[gold_stage]
                if gold_remap_class_indices
                else gold_stage
            )
            class_label = str(gold_meta["gold_stage_label"])
        elif target_mode == "gold_severity4" and gold_meta is not None:
            gold_stage = int(gold_meta["gold_stage"])
            class_index = gold_stage - 1
            class_label = str(gold_meta["gold_stage_label"])
        elif target_mode == "angle_3class":
            class_index, class_label = angle_3class_label(float(angle))
        elif target_mode == "angle_binary_extreme":
            extreme_label = angle_binary_extreme_label(float(angle))
            if extreme_label is None:
                continue
            class_index, class_label = extreme_label
        elif target_mode == "oi_emphysema" and oi_meta is not None:
            class_index, class_label = oi_emphysema_label(
                float(oi_meta["oi"]), oi_threshold
            )
        elif target_mode == "oi_3class" and oi_meta is not None:
            class_index, class_label = oi_3class_label(
                float(oi_meta["oi"]), oi_3class_thresholds
            )
        elif target_mode == "normal_v_abnormal":
            folder_label = normal_v_abnormal_label(ct_path.parent.name)
            if folder_label is None:
                continue
            class_index, class_label = folder_label
            target = float(class_index)

        source_group = ct_path.parent.name
        records.append(
            AngleRecord(
                patient_id=patient_id,
                path=str(ct_path),
                angle=float(angle),
                source_group=source_group,
                target=target,
                class_index=class_index,
                class_label=class_label,
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
                oi=float(oi_meta["oi"]) if oi_meta is not None else None,
                a=(
                    float(oi_meta["a"])
                    if oi_meta is not None and oi_meta["a"] is not None
                    else None
                ),
                fvc=(
                    float(oi_meta["fvc"])
                    if oi_meta is not None and oi_meta["fvc"] is not None
                    else None
                ),
                pef=(
                    float(oi_meta["pef"])
                    if oi_meta is not None and oi_meta["pef"] is not None
                    else None
                ),
            )
        )

    active_label_ids = oi_label_map.keys() if is_oi_target else label_map.keys()
    extra_in_source_not_in_json = sorted(
        patient_id for patient_id in active_label_ids if patient_id not in source_ids
    )

    counts = {
        "total": len(records),
        "unique_patients": len({record.patient_id for record in records}),
        "low_angle_group": sum(1 for record in records if "low" in record.source_group),
        "high_angle_group": sum(1 for record in records if "high" in record.source_group),
    }
    if oi_exclude_bounds is not None:
        counts["excluded_oi_range"] = excluded_oi_range_count
    if target_mode == "gold_severity4":
        gold_count_names = [
            class_name
            for class_idx, class_name in enumerate(gold_class_names)
            if class_idx > 0
        ]
    else:
        gold_count_names = [
            class_name
            for class_idx, class_name in enumerate(gold_class_names)
            if class_idx not in excluded_gold_classes
        ]
    gold_stage_counts = {
        class_name: sum(1 for record in records if record.gold_stage_label == class_name)
        for class_name in gold_count_names
    }
    class_counts = {
        class_name: sum(1 for record in records if record.class_label == class_name)
        for class_name in class_names
    }

    return AngleManifest(
        source_json=str(oi_path if is_oi_target else labels_json),
        data_root=str(data_root),
        records=records,
        counts=counts,
        missing_from_source=sorted(set(missing_from_source)),
        extra_in_source_not_in_json=extra_in_source_not_in_json,
        missing_gold_labels=sorted(set(missing_gold_labels)),
        gold_stage_counts=gold_stage_counts,
        class_counts=class_counts,
        class_names=class_names,
    )


def save_manifest(manifest: AngleManifest, path: str | Path) -> Path:
    """Persist the manifest for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
    return path
