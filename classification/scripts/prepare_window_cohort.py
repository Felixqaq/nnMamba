"""Freeze the valid conversion cohort while preserving the official test set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.model_selection import train_test_split



DECISIONS = Path(__file__).resolve().parents[2] / "regression/cohort_decisions.local.json"


def _excluded_ids() -> list:
    """Cohort exclusions, from the file git does not track.

    The ID used to sit in this script. It names a patient beside the reason the
    scan was rejected, and this repository is public.
    """
    if not DECISIONS.is_file():
        raise SystemExit(f"{DECISIONS} is missing; refusing to write an audit that "
                         "claims an exclusion list it cannot read")
    return list(json.loads(DECISIONS.read_text(encoding="utf-8"))["excluded"])


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    root = Path("/home/felix/Research/nnMamba/classification/datasets/normal_v_abnormal_fev1fvc70")
    summary = json.loads((root / "build_summary.json").read_text())
    records = [r for r in summary["records"] if r["ok"]]
    assert len(records) == 777, "Cohort changed; audit again"
    split_path = project / "regression/outputs/doctor_validation_official200_20260831/split.json"
    old = json.loads(split_path.read_text())
    test = set(old["validation_patient_ids"])
    assert len(test) == 200
    ids = {r["patient_id"] for r in records}
    assert len(ids) == len(records) and test <= ids
    assert not ids.intersection(summary["excluded_patients"])
    development = sorted(ids - test)
    labels = {r["patient_id"]: int(float(r["fev1_fvc_pct"]) < 70) for r in records}
    train, val = train_test_split(development, test_size=0.2, random_state=42,
                                 stratify=[labels[p] for p in development])
    assignments = {**dict.fromkeys(train, "train"), **dict.fromkeys(val, "val"),
                   **dict.fromkeys(test, "test")}
    output = project / "weights/window_distillation/cohort777"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "patients.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "path", "label", "split"])
        for row in sorted(records, key=lambda r: r["patient_id"]):
            pid = row["patient_id"]
            label = labels[pid]
            assert row["label"] == ("Abnormal" if label else "Normal")
            matches = list(root.glob(f"*/{pid}.nii.gz"))
            if not matches:
                matches = list(root.glob(f"*/{pid}_*.nii.gz"))
            assert len(matches) == 1, "Ambiguous/missing original converted scan"
            writer.writerow([pid, str(matches[0]), label, assignments[pid]])
    audit = {"valid_patients": len(records), "excluded_stale_scan": sorted(_excluded_ids()),
             "source_summary": str(root / "build_summary.json"),
             "original_dicom_root": summary["dicom_root"],
             "frozen_test_source": str(split_path),
             "new_development_patients": len(ids - set(old["training_patient_ids"]) - test),
             "split_counts": {s: {str(y): sum(assignments[p] == s and labels[p] == y for p in ids)
                              for y in (0, 1)} for s in ("train", "val", "test")},
             "label_note": "FEV1/FVC<70; original source mixes post-BD and pre-BD when post is unavailable",
             "data_note": "Reuse original HU NIfTI conversions; do not train on z-scored derivatives"}
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
