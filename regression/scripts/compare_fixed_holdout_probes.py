#!/usr/bin/env python3
"""Compare fixed-holdout prediction files on the exact same patient IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Model label and prediction path in LABEL=PATH form; may be repeated",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    patients = payload.get("patients", payload)
    if not isinstance(patients, dict):
        raise ValueError(f"{path}: expected a patient mapping")
    return patients


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    labels = {
        str(record["patient_id"]): str(
            record.get("class_label") or record.get("source_group")
        )
        for record in manifest
    }
    holdout = [
        str(patient_id)
        for patient_id in json.loads(args.split_json.read_text(encoding="utf-8"))[
            "validation_patient_ids"
        ]
    ]
    if len(holdout) != len(set(holdout)):
        raise SystemExit("validation_patient_ids contains duplicates")

    rows: list[dict] = []
    sources: dict[str, str] = {}
    for specification in args.result:
        if "=" not in specification:
            raise SystemExit(f"--result must be LABEL=PATH, got {specification!r}")
        label, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        patients = load_predictions(path)
        missing = [patient_id for patient_id in holdout if patient_id not in patients]
        if missing:
            raise SystemExit(f"{label}: missing {len(missing)} holdout predictions")

        truth = np.asarray(
            [1 if labels[patient_id].lower() == "abnormal" else 0 for patient_id in holdout],
            dtype=int,
        )
        probability = np.asarray(
            [
                float(
                    patients[patient_id].get(
                        "mean_prob_abnormal", patients[patient_id].get("prob_abnormal")
                    )
                )
                for patient_id in holdout
            ],
            dtype=float,
        )
        prediction = np.asarray(
            [
                1 if str(patients[patient_id]["pred_label"]).lower() == "abnormal" else 0
                for patient_id in holdout
            ],
            dtype=int,
        )
        tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
        row = {
            "model": label,
            "n": len(holdout),
            "accuracy": round(float(accuracy_score(truth, prediction)), 5),
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, prediction)), 5),
            "auc": round(float(roc_auc_score(truth, probability)), 5),
            "sensitivity": round(float(tp / (tp + fn)), 5),
            "specificity": round(float(tn / (tn + fp)), 5),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        rows.append(row)
        sources[label] = str(path)

    output = {
        "cohort": "frozen official 200",
        "validation_patient_ids": holdout,
        "sources": sources,
        "models": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
