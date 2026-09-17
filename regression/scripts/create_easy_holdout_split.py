#!/usr/bin/env python3
"""Create a deliberately easy 200-patient same-hospital evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from prepare_doctor_validation import cohort_digest, read_pft_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pft-csv", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=7.0)
    parser.add_argument("--validation-abnormal", type=int, default=60)
    parser.add_argument("--validation-normal", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--exclude-imaging-id", action="append", default=[])
    parser.add_argument(
        "--exclude-imaging",
        action="append",
        default=[],
        metavar="PATIENT_ID=REASON",
        help="exclude an imaging-invalid patient and preserve the exact reason",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_joint_strata(
    rows: list[dict[str, str]],
    quota: int,
) -> dict[str, int]:
    """Largest-remainder allocation over Sex x Date cells."""
    counts = Counter(f"{row['Sex']}|{row['Date']}" for row in rows)
    if not 0 <= quota <= len(rows):
        raise SystemExit(f"invalid quota {quota} for pool of {len(rows)}")
    exact = {key: quota * count / len(rows) for key, count in counts.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = quota - sum(allocated.values())
    order = sorted(
        counts,
        key=lambda key: (exact[key] - allocated[key], counts[key], key),
        reverse=True,
    )
    for key in order[:remaining]:
        allocated[key] += 1
    if sum(allocated.values()) != quota:
        raise AssertionError("joint-stratum allocation does not sum to quota")
    if any(allocated[key] > counts[key] for key in counts):
        raise AssertionError("joint-stratum allocation exceeds available cases")
    return allocated


def select_label_cases(
    rows: list[dict[str, str]],
    quota: int,
    seed: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    allocations = allocate_joint_strata(rows, quota)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_stratum[f"{row['Sex']}|{row['Date']}"].append(row["PatientID"])
    rng = random.Random(seed)
    selected: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    for key in sorted(by_stratum):
        candidates = sorted(by_stratum[key])
        rng.shuffle(candidates)
        take = allocations[key]
        selected.extend(candidates[:take])
        audit[key] = {"available": len(candidates), "selected": take}
    if len(selected) != quota or len(set(selected)) != quota:
        raise AssertionError(f"selected {len(selected)} unique={len(set(selected))}, expected {quota}")
    return selected, audit


def describe(rows: list[dict[str, str]], margin: float) -> dict[str, Any]:
    values = sorted(float(row["FEV1FVC_pct"]) for row in rows)
    midpoint = len(values) // 2
    median = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return {
        "n": len(rows),
        "abnormal": sum(row["Fixed70_Label"] == "Abnormal" for row in rows),
        "normal": sum(row["Fixed70_Label"] == "Normal" for row in rows),
        "female": sum(row["Sex"] == "F" for row in rows),
        "male": sum(row["Sex"] == "M" for row in rows),
        "hard": sum(
            abs(float(row["FEV1FVC_pct"]) - 70.0) < margin for row in rows
        ),
        "easy": sum(
            abs(float(row["FEV1FVC_pct"]) - 70.0) >= margin for row in rows
        ),
        "fev1_fvc_min": min(values),
        "fev1_fvc_median": median,
        "fev1_fvc_max": max(values),
    }


def main() -> None:
    args = parse_args()
    if args.margin <= 0:
        raise SystemExit("margin must be positive")
    if args.validation_abnormal < 1 or args.validation_normal < 1:
        raise SystemExit("validation label quotas must be positive")

    rows, excluded = read_pft_rows(args.pft_csv)
    imaging_exclusions = {
        patient_id: "invalid imaging — available CT is non-thoracic head/neck imaging"
        for patient_id in args.exclude_imaging_id
    }
    for value in args.exclude_imaging:
        patient_id, separator, reason = value.partition("=")
        patient_id = patient_id.strip()
        reason = reason.strip()
        if not separator or not patient_id or not reason:
            raise SystemExit(
                "--exclude-imaging must use PATIENT_ID=REASON with both values present"
            )
        if patient_id in imaging_exclusions and imaging_exclusions[patient_id] != reason:
            raise SystemExit(f"conflicting imaging exclusion reasons for {patient_id}")
        imaging_exclusions[patient_id] = reason
    imaging_excluded_ids = set(imaging_exclusions)
    for patient_id, reason in sorted(imaging_exclusions.items()):
        matches = [row for row in rows if row["PatientID"] == patient_id]
        if len(matches) != 1:
            raise SystemExit(
                f"--exclude-imaging-id {patient_id}: expected one eligible PFT row, "
                f"found {len(matches)}"
            )
        row = matches[0]
        rows.remove(row)
        excluded.append(
            {
                "PatientID": patient_id,
                "Batch": row.get("Date", ""),
                "Reason": reason,
            }
        )
    by_id = {row["PatientID"]: row for row in rows}
    source_split = json.loads(args.source_split.read_text(encoding="utf-8-sig"))
    source_ids = set(source_split["training_patient_ids"]) | set(
        source_split["validation_patient_ids"]
    )
    if source_ids != set(by_id):
        raise SystemExit(
            "source split does not match the eligible imaging-QA cohort after exclusions"
        )
    source_imaging_excluded_ids = set(
        source_split.get("imaging_excluded_patient_ids", [])
    )
    if source_imaging_excluded_ids and source_imaging_excluded_ids != imaging_excluded_ids:
        raise SystemExit(
            "explicit imaging exclusions differ from the source split: "
            f"requested={sorted(imaging_excluded_ids)} "
            f"source={sorted(source_imaging_excluded_ids)}"
        )

    hard_rows = [
        row for row in rows if abs(float(row["FEV1FVC_pct"]) - 70.0) < args.margin
    ]
    easy_rows = [
        row for row in rows if abs(float(row["FEV1FVC_pct"]) - 70.0) >= args.margin
    ]
    easy_by_label = {
        label: [row for row in easy_rows if row["Fixed70_Label"] == label]
        for label in ("Abnormal", "Normal")
    }
    requested = {
        "Abnormal": args.validation_abnormal,
        "Normal": args.validation_normal,
    }
    for label, quota in requested.items():
        if len(easy_by_label[label]) < quota:
            raise SystemExit(
                f"not enough easy {label} cases: have {len(easy_by_label[label])}, need {quota}"
            )

    selected: list[str] = []
    allocation_audit: dict[str, dict[str, dict[str, int]]] = {}
    for offset, label in enumerate(("Abnormal", "Normal")):
        label_selected, label_audit = select_label_cases(
            easy_by_label[label],
            requested[label],
            args.seed + offset,
        )
        selected.extend(label_selected)
        allocation_audit[label] = label_audit

    validation_ids = sorted(selected)
    training_ids = sorted(set(by_id) - set(validation_ids))
    hard_ids = {row["PatientID"] for row in hard_rows}
    easy_ids = {row["PatientID"] for row in easy_rows}
    if hard_ids - set(training_ids):
        raise AssertionError("at least one boundary-defined hard case entered easy holdout")
    if set(validation_ids) - easy_ids:
        raise AssertionError("at least one non-easy case entered easy holdout")
    expected_validation_size = args.validation_abnormal + args.validation_normal
    expected_training_size = len(rows) - expected_validation_size
    if (
        len(validation_ids) != expected_validation_size
        or len(training_ids) != expected_training_size
    ):
        raise AssertionError(
            f"expected validation={expected_validation_size} and "
            f"training={expected_training_size}"
        )

    original_new_ids = set(source_split.get("priority_new_patient_ids", []))
    new_validation_ids = sorted(original_new_ids & set(validation_ids))
    new_training_ids = sorted(original_new_ids & set(training_ids))
    validation_rows = [by_id[pid] for pid in validation_ids]
    training_rows = [by_id[pid] for pid in training_ids]
    simple_training_ids = sorted(set(training_ids) & easy_ids)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "strategy": (
            "deliberately easy same-hospital holdout: select only cases with "
            f"|FEV1/FVC-70|>={args.margin:g}; exact label quotas; proportional "
            "largest-remainder allocation over sex-by-batch cells"
        ),
        "interpretation_warning": (
            "This deliberately easy holdout is not an unbiased external validation set and "
            "will tend to overestimate performance near the clinical decision boundary."
        ),
        "difficulty_definition": f"hard if |FEV1/FVC-70| < {args.margin:g}",
        "validation_size": len(validation_ids),
        "training_size": len(training_ids),
        "hard_case_count": len(hard_rows),
        "easy_case_count": len(easy_rows),
        "imaging_excluded_patient_ids": sorted(imaging_excluded_ids),
        "imaging_exclusion_reasons": imaging_exclusions,
        "all_hard_cases_in_training": hard_ids <= set(training_ids),
        "hard_training_patient_ids": sorted(hard_ids),
        "simple_training_patient_ids": simple_training_ids,
        "validation_patient_ids": validation_ids,
        "training_patient_ids": training_ids,
        "validation_label_targets": requested,
        "validation_joint_stratum_allocation": allocation_audit,
        "priority_new_patient_ids": new_validation_ids,
        "all_new_patient_ids": sorted(original_new_ids),
        "new_patient_ids_in_training": new_training_ids,
        "new_patient_count_in_validation": len(new_validation_ids),
        "new_patient_count_in_training": len(new_training_ids),
        "cohort_summary": describe(rows, args.margin),
        "training_summary": describe(training_rows, args.margin),
        "validation_summary": describe(validation_rows, args.margin),
        "pft_csv": str(args.pft_csv),
        "cohort_sha256": cohort_digest(rows),
        "source_split": str(args.source_split),
        "source_split_sha256": file_sha256(args.source_split),
        "excluded": excluded,
    }

    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8-sig"))
        keys = (
            "seed",
            "difficulty_definition",
            "validation_patient_ids",
            "training_patient_ids",
            "cohort_sha256",
            "source_split_sha256",
            "imaging_excluded_patient_ids",
        )
        if any(existing.get(key) != payload.get(key) for key in keys):
            raise SystemExit(f"existing split differs from requested deterministic split: {args.output}")
        print(f"reusing {args.output}")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "difficulty_definition",
        "hard_case_count",
        "easy_case_count",
        "new_patient_count_in_validation",
        "new_patient_count_in_training",
        "training_summary",
        "validation_summary",
    )}, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
