#!/usr/bin/env python3
"""Tune a TAP-CT probe and threshold using training-only OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_C_GRID = "0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--c-grid", default=DEFAULT_C_GRID)
    return parser.parse_args()


def load_labels(features_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(features_path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float64)
    patient_ids = [str(value) for value in data["patient_ids"]]
    with (features_path.parent / "metadata.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        groups = {
            str(row["patient_id"]): row["source_group"].strip()
            for row in csv.DictReader(handle)
        }
    labels = np.asarray(
        [1 if groups[patient_id] == "Abnormal" else 0 for patient_id in patient_ids],
        dtype=int,
    )
    return features, labels, patient_ids


def build_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def choose_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                probabilities.astype(np.float64),
            ]
        )
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        key = (
            float(balanced_accuracy_score(truth, predictions)),
            float(accuracy_score(truth, predictions)),
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "threshold": float(threshold),
        "accuracy": round(float(accuracy_score(truth, predictions)), 5),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truth, predictions)), 5
        ),
        "auc": round(float(roc_auc_score(truth, probabilities)), 5),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def parse_c_grid(text: str) -> list[float]:
    values = sorted({float(value.strip()) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise SystemExit("--c-grid must contain positive values")
    return values


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    c_grid = parse_c_grid(args.c_grid)
    features, labels, patient_ids = load_labels(args.features)
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    expected = set(split["training_patient_ids"]) | set(
        split["validation_patient_ids"]
    )
    missing = sorted(expected - set(index_by_id))
    extra = sorted(set(index_by_id) - expected)
    if missing or extra:
        raise SystemExit(f"feature/split mismatch: missing={missing} extra={extra}")

    train_indices = np.asarray(
        [index_by_id[patient_id] for patient_id in split["training_patient_ids"]],
        dtype=int,
    )
    holdout_indices = np.asarray(
        [index_by_id[patient_id] for patient_id in split["validation_patient_ids"]],
        dtype=int,
    )
    train_labels = labels[train_indices]
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_positions = list(
        splitter.split(np.arange(len(train_indices)), train_labels)
    )

    candidates: list[dict[str, object]] = []
    probabilities_by_c: dict[float, np.ndarray] = {}
    for c_value in c_grid:
        oof_probabilities = np.full(len(train_indices), np.nan, dtype=np.float64)
        fold_auc: list[float] = []
        for fold_index, (fit_positions, oof_positions) in enumerate(
            fold_positions, start=1
        ):
            model = build_model(c_value, args.seed + fold_index)
            model.fit(
                features[train_indices[fit_positions]], train_labels[fit_positions]
            )
            fold_probabilities = model.predict_proba(
                features[train_indices[oof_positions]]
            )[:, 1]
            oof_probabilities[oof_positions] = fold_probabilities
            fold_auc.append(
                float(roc_auc_score(train_labels[oof_positions], fold_probabilities))
            )
        if not np.isfinite(oof_probabilities).all():
            raise RuntimeError(f"C={c_value} did not produce complete OOF predictions")
        probabilities_by_c[c_value] = oof_probabilities
        candidates.append(
            {
                "C": c_value,
                "mean_fold_auc": float(np.mean(fold_auc)),
                "pooled_oof_auc": float(
                    roc_auc_score(train_labels, oof_probabilities)
                ),
                "fold_auc": fold_auc,
            }
        )

    # Primary selection uses mean fold AUC. Ties prefer pooled AUC and then the
    # smaller C (stronger regularization), all without consulting the holdout.
    selected = max(
        candidates,
        key=lambda row: (
            float(row["mean_fold_auc"]),
            float(row["pooled_oof_auc"]),
            -float(row["C"]),
        ),
    )
    selected_c = float(selected["C"])
    oof_probabilities = probabilities_by_c[selected_c]
    threshold = choose_threshold(train_labels, oof_probabilities)
    oof_metrics = metrics(train_labels, oof_probabilities, threshold)

    final_model = build_model(selected_c, args.seed)
    final_model.fit(features[train_indices], train_labels)
    holdout_probabilities = final_model.predict_proba(features[holdout_indices])[:, 1]
    holdout_truth = labels[holdout_indices]
    holdout_predictions = (holdout_probabilities >= threshold).astype(int)
    patients = {
        patient_ids[index]: {
            "true_label": "Abnormal" if labels[index] else "Normal",
            "prob_abnormal": float(probability),
            "pred_label": "Abnormal" if prediction else "Normal",
        }
        for index, probability, prediction in zip(
            holdout_indices,
            holdout_probabilities,
            holdout_predictions,
            strict=True,
        )
    }
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "features": str(args.features),
            "split_json": str(args.split_json),
            "n_training": int(len(train_indices)),
            "n_holdout": int(len(holdout_indices)),
            "folds": args.folds,
            "fold_split_seed": args.seed,
            "C_grid": c_grid,
            "C_candidates": candidates,
            "selected_C": selected_c,
            "selected_threshold": threshold,
            "C_selection": "maximum mean training-only fold AUC",
            "threshold_selection": "maximum pooled training-only OOF balanced accuracy",
            "class_weight": "balanced",
            "validation_used_for_model_or_threshold_selection": False,
        },
        "training_oof_metrics": oof_metrics,
        "metrics": metrics(holdout_truth, holdout_probabilities, threshold),
        "patients": patients,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"selected_C": selected_c, "threshold": threshold}, indent=2))
    print("training OOF")
    print(json.dumps(oof_metrics, indent=2))
    print("fixed holdout")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
