#!/usr/bin/env python3
"""Freeze a larger training cohort while preserving an existing holdout exactly."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from prepare_doctor_validation import cohort_digest, read_pft_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pft-csv", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=7.0)
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


def describe(rows: list[dict[str, str]], margin: float) -> dict[str, object]:
    ratios = [float(row["FEV1FVC_pct"]) for row in rows]
    ordered = sorted(ratios)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    labels = Counter(row["Fixed70_Label"] for row in rows)
    sexes = Counter(row["Sex"] for row in rows)
    return {
        "n": len(rows),
        "normal": labels["Normal"],
        "abnormal": labels["Abnormal"],
        "female": sexes["F"],
        "male": sexes["M"],
        "hard": sum(abs(value - 70.0) < margin for value in ratios),
        "easy": sum(abs(value - 70.0) >= margin for value in ratios),
        "fev1_fvc_min": min(ratios),
        "fev1_fvc_median": median,
        "fev1_fvc_max": max(ratios),
    }


def main() -> None:
    args = parse_args()
    if args.margin <= 0:
        raise SystemExit("margin must be positive")

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
    for patient_id in sorted(imaging_excluded_ids):
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
                "Reason": imaging_exclusions[patient_id],
            }
        )

    base_split = json.loads(args.base_split.read_text(encoding="utf-8-sig"))
    validation_ids = set(base_split["validation_patient_ids"])
    previous_training_ids = set(base_split["training_patient_ids"])
    current_ids = {row["PatientID"] for row in rows}
    base_ids = validation_ids | previous_training_ids
    missing_base_ids = sorted(base_ids - current_ids)
    # A holdout patient may never disappear -- that would silently shrink the
    # frozen validation set and make the score incomparable with earlier runs.
    # A *training* patient may, but only when this run excludes them explicitly
    # and says why; the dropped ids are recorded in the split so the training
    # cohort's provenance stays reconstructable.
    missing_validation = sorted(validation_ids - current_ids)
    if missing_validation:
        raise SystemExit(
            "frozen holdout patients are missing from the current CSV: "
            f"{missing_validation}"
        )
    dropped_training = sorted(
        pid for pid in (previous_training_ids - current_ids)
        if pid in imaging_exclusions
    )
    unexplained = sorted(set(missing_base_ids) - set(dropped_training))
    if unexplained:
        raise SystemExit(f"patients from the base split are missing: {unexplained}")
    for pid in dropped_training:
        print(f"DROPPED FROM TRAINING: {pid} — {imaging_exclusions[pid]}")
    previous_training_ids = previous_training_ids - set(dropped_training)

    training_ids = current_ids - validation_ids
    if not previous_training_ids <= training_ids:
        raise AssertionError("at least one prior training patient left the training set")
    if validation_ids & training_ids:
        raise AssertionError("training and validation overlap")

    by_id = {row["PatientID"]: row for row in rows}
    validation_rows = [by_id[pid] for pid in sorted(validation_ids)]
    training_rows = [by_id[pid] for pid in sorted(training_ids)]
    new_ids = sorted(current_ids - base_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.output_dir / "fev1_fvc_frozen.csv"
    split_path = args.output_dir / "split.json"
    if snapshot_path.exists() or split_path.exists():
        raise SystemExit(
            "refusing to replace an existing frozen snapshot or split: "
            f"{args.output_dir}"
        )

    # Preserve the complete curated input, including rows that read_pft_rows will
    # place on the Excluded worksheet. Downstream runs use this immutable copy.
    snapshot_path.write_bytes(args.pft_csv.read_bytes())

    # Needed by the strategy/warning strings below, which describe the holdout
    # that is actually present rather than naming a hard-coded experiment line.
    validation_summary = describe(validation_rows, args.margin)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": base_split.get("seed"),
        "strategy": (
            "preserve the prior easy200 holdout exactly; add every currently "
            "eligible later PFT patient to training; no validation patient enters training"
        ),
        "interpretation_warning": (
            "The fixed easy200 set is a deliberately easy same-hospital internal "
            "validation set, not an unbiased external validation set."
        ),
        "difficulty_definition": f"hard if |FEV1/FVC-70| < {args.margin:g}",
        "validation_size": len(validation_ids),
        "training_size": len(training_ids),
        "validation_patient_ids": sorted(validation_ids),
        "training_patient_ids": sorted(training_ids),
        "priority_new_patient_ids": list(
            base_split.get("priority_new_patient_ids", [])
        ),
        "previous_training_patient_ids": sorted(previous_training_ids),
        "dropped_from_previous_training": sorted(
            pid for pid in base_split["training_patient_ids"]
            if pid not in current_ids and pid in imaging_exclusions
        ),
        "new_training_patient_ids": new_ids,
        "new_training_patient_count": len(new_ids),
        "all_new_patients_in_training": set(new_ids) <= training_ids,
        "imaging_excluded_patient_ids": sorted(imaging_excluded_ids),
        "imaging_exclusion_reasons": imaging_exclusions,
        "cohort_summary": describe(rows, args.margin),
        "training_summary": describe(training_rows, args.margin),
        "validation_summary": validation_summary,
        "pft_csv": str(snapshot_path),
        "source_pft_csv": str(args.pft_csv),
        "source_pft_sha256": file_sha256(args.pft_csv),
        "snapshot_pft_sha256": file_sha256(snapshot_path),
        "cohort_sha256": cohort_digest(rows),
        "base_split": str(args.base_split),
        "base_split_sha256": file_sha256(args.base_split),
        "excluded": excluded,
    }
    split_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "cohort_summary": payload["cohort_summary"],
                "training_summary": payload["training_summary"],
                "validation_summary": payload["validation_summary"],
                "new_training_patient_count": len(new_ids),
                "split": str(split_path),
                "pft_snapshot": str(snapshot_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
