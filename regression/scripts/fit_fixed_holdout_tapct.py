#!/usr/bin/env python3
"""Fit the frozen TAP-CT probe on all training patients and score the holdout once."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-patient-id", action="append", default=[])
    return parser.parse_args()


def load_labels(features_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import csv

    data = np.load(features_path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float64)
    patient_ids = [str(value) for value in data["patient_ids"]]
    with (features_path.parent / "metadata.csv").open(encoding="utf-8-sig") as handle:
        groups = {str(row["patient_id"]): row["source_group"].strip() for row in csv.DictReader(handle)}
    labels = np.array([1 if groups[pid] == "Abnormal" else 0 for pid in patient_ids], dtype=int)
    return features, labels, patient_ids


def main() -> None:
    args = parse_args()
    features, labels, patient_ids = load_labels(args.features)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    index_by_id = {pid: index for index, pid in enumerate(patient_ids)}
    expected = set(split["training_patient_ids"]) | set(split["validation_patient_ids"])
    excluded_ids = set(args.exclude_patient_id)
    if excluded_ids & expected:
        raise SystemExit("an explicitly excluded TAP-CT patient is still present in the split")
    if set(index_by_id) - excluded_ids != expected or excluded_ids - set(index_by_id):
        raise SystemExit(
            f"feature/split mismatch after explicit exclusions: "
            f"missing={sorted(expected-(set(index_by_id)-excluded_ids))} "
            f"extra={sorted((set(index_by_id)-excluded_ids)-expected)} "
            f"unknown_exclusions={sorted(excluded_ids-set(index_by_id))}"
        )
    train_indices = np.array([index_by_id[pid] for pid in split["training_patient_ids"]])
    holdout_indices = np.array([index_by_id[pid] for pid in split["validation_patient_ids"]])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=args.c,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(features[train_indices], labels[train_indices])
    probabilities = model.predict_proba(features[holdout_indices])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    truth = labels[holdout_indices]
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    patients = {
        patient_ids[index]: {
            "true_label": "Abnormal" if labels[index] else "Normal",
            "prob_abnormal": float(probability),
            "pred_label": "Abnormal" if prediction else "Normal",
        }
        for index, probability, prediction in zip(
            holdout_indices, probabilities, predictions, strict=True
        )
    }
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "features": str(args.features),
            "split_json": str(args.split_json),
            "n_training": int(len(train_indices)),
            "n_holdout": int(len(holdout_indices)),
            "C": args.c,
            "C_basis": "mode/median of 50 inner-CV selections from the prior 383-case analysis",
            "validation_used_for_model_selection": False,
            "excluded_patient_ids": sorted(excluded_ids),
        },
        "metrics": {
            "accuracy": round(float(accuracy_score(truth, predictions)), 5),
            "balanced_accuracy": round(float(balanced_accuracy_score(truth, predictions)), 5),
            "auc": round(float(roc_auc_score(truth, probabilities)), 5),
            "sensitivity": round(float(tp / (tp + fn)), 5),
            "specificity": round(float(tn / (tn + fp)), 5),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "patients": patients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
