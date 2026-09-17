#!/usr/bin/env python3
"""Train a hard-weighted TAP-CT probe without using the formal holdout for selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_C_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--hardness-csv", type=Path, required=True)
    parser.add_argument("--mamba-selection-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260827)
    parser.add_argument("--c-grid", type=float, nargs="+", default=DEFAULT_C_GRID)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def hash_ids(patient_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(patient_ids).encode("utf-8")).hexdigest()


def hash_mapping(values: dict[str, float]) -> str:
    text = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def load_feature_store(features_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(features_path, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float64)
    patient_ids = [str(value) for value in data["patient_ids"]]
    if features.ndim != 2 or features.shape[0] != len(patient_ids):
        raise SystemExit(f"invalid TAP-CT feature shape: {features.shape}, ids={len(patient_ids)}")

    metadata_path = features_path.parent / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        groups = {
            str(row["patient_id"]): str(row["source_group"]).strip()
            for row in csv.DictReader(handle)
        }
    missing = sorted(set(patient_ids) - set(groups))
    if missing:
        raise SystemExit(f"TAP-CT metadata missing labels for {missing[:10]}")
    labels = np.array([1 if groups[pid] == "Abnormal" else 0 for pid in patient_ids], dtype=int)
    return features, labels, patient_ids


def load_hardness(
    path: Path,
    train_ids: list[str],
    holdout_ids: list[str],
    selection_summary_path: Path,
) -> tuple[dict[str, float], int, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row["Patient_ID"]): row for row in rows}
    if set(by_id) != set(train_ids):
        raise SystemExit(
            "hardness/train mismatch: "
            f"missing={sorted(set(train_ids)-set(by_id))[:10]} "
            f"extra={sorted(set(by_id)-set(train_ids))[:10]}"
        )
    leaked = sorted(set(by_id) & set(holdout_ids))
    if leaked:
        raise SystemExit(f"formal holdout leaked into TAP-CT hardness weights: {leaked[:10]}")

    weights = {pid: float(by_id[pid]["Sampling_Weight"]) for pid in train_ids}
    hard_count = sum(str(by_id[pid]["Hard_Example"]).strip().lower() == "true" for pid in train_ids)
    weights_hash = hash_mapping(weights)

    mamba_summary = json.loads(selection_summary_path.read_text(encoding="utf-8"))
    if mamba_summary["training_ids_sha256"] != hash_ids(train_ids):
        raise SystemExit("Mamba hardness summary training IDs do not match the frozen split")
    if mamba_summary["formal_holdout_ids_sha256"] != hash_ids(holdout_ids):
        raise SystemExit("Mamba hardness summary holdout IDs do not match the frozen split")
    if mamba_summary["hard_example_count"] != hard_count:
        raise SystemExit("hard-example count differs from the Mamba selection summary")
    if mamba_summary["sampling_weights_sha256"] != weights_hash:
        raise SystemExit("hardness weights differ from the Mamba selection summary")
    if mamba_summary["formal_holdout_patient_ids_used_for_selection_or_mining"]:
        raise SystemExit("Mamba hardness summary reports formal holdout use")
    return weights, hard_count, weights_hash


def build_probe(c_value: float, seed: int) -> Pipeline:
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


def calculate_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "threshold": round(float(threshold), 5),
        "accuracy": round(float(accuracy_score(truth, predictions)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, predictions)), 5),
        "auc": round(float(roc_auc_score(truth, probabilities)), 5),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if not 0.0 < minimum < maximum < 1.0:
        raise SystemExit("threshold range must be strictly within (0, 1)")
    if step <= 0:
        raise SystemExit("threshold step must be positive")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    values = [round(minimum + index * step, 10) for index in range(count + 1)]
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return values


def select_threshold(
    truth: np.ndarray,
    probabilities: np.ndarray,
    candidates: list[float],
) -> tuple[float, dict[str, Any]]:
    scored = [(threshold, calculate_metrics(truth, probabilities, threshold)) for threshold in candidates]
    return max(
        scored,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["accuracy"],
            -abs(item[0] - 0.5),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.selection_folds < 2:
        raise SystemExit("selection-folds must be at least 2")
    if not args.c_grid or any(value <= 0 for value in args.c_grid):
        raise SystemExit("all C candidates must be positive")

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["training_patient_ids"]]
    holdout_ids = [str(value) for value in split["validation_patient_ids"]]
    if set(train_ids) & set(holdout_ids):
        raise SystemExit("training and formal holdout overlap")
    if len(train_ids) != 230 or len(holdout_ids) != 200:
        raise SystemExit(f"expected train=230/formal=200, got {len(train_ids)}/{len(holdout_ids)}")

    features, labels, patient_ids = load_feature_store(args.features)
    index_by_id = {pid: index for index, pid in enumerate(patient_ids)}
    expected_ids = set(train_ids) | set(holdout_ids)
    if set(index_by_id) != expected_ids:
        raise SystemExit(
            "feature/split mismatch: "
            f"missing={sorted(expected_ids-set(index_by_id))[:10]} "
            f"extra={sorted(set(index_by_id)-expected_ids)[:10]}"
        )

    weights_by_id, hard_count, weights_hash = load_hardness(
        args.hardness_csv,
        train_ids,
        holdout_ids,
        args.mamba_selection_summary,
    )
    train_indices = np.array([index_by_id[pid] for pid in train_ids], dtype=int)
    x_train = features[train_indices].copy()
    y_train = labels[train_indices].copy()
    sample_weights = np.array([weights_by_id[pid] for pid in train_ids], dtype=np.float64)

    folds = list(
        StratifiedKFold(
            n_splits=args.selection_folds,
            shuffle=True,
            random_state=args.selection_seed,
        ).split(x_train, y_train)
    )
    thresholds = threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
    selection_curve: dict[str, dict[str, Any]] = {}
    oof_by_c: dict[float, np.ndarray] = {}
    for c_value in sorted(set(float(value) for value in args.c_grid)):
        oof_probabilities = np.full(len(train_ids), np.nan, dtype=np.float64)
        fold_audit: list[dict[str, Any]] = []
        for fold_number, (fit_rows, valid_rows) in enumerate(folds, start=1):
            model = build_probe(c_value, args.selection_seed + fold_number)
            model.fit(
                x_train[fit_rows],
                y_train[fit_rows],
                clf__sample_weight=sample_weights[fit_rows],
            )
            oof_probabilities[valid_rows] = model.predict_proba(x_train[valid_rows])[:, 1]
            fold_audit.append(
                {
                    "fold": fold_number,
                    "n_fit": int(len(fit_rows)),
                    "n_internal_validation": int(len(valid_rows)),
                    "fit_patient_ids_sha256": hash_ids([train_ids[index] for index in fit_rows]),
                    "internal_validation_patient_ids_sha256": hash_ids(
                        [train_ids[index] for index in valid_rows]
                    ),
                    "formal_holdout_patient_ids_used": [],
                }
            )
        if np.isnan(oof_probabilities).any():
            raise RuntimeError(f"missing OOF predictions for C={c_value}")
        selected_threshold, metrics = select_threshold(y_train, oof_probabilities, thresholds)
        selection_curve[f"{c_value:g}"] = {
            "C": c_value,
            "selected_threshold": selected_threshold,
            "oof_metrics": metrics,
            "folds": fold_audit,
        }
        oof_by_c[c_value] = oof_probabilities
        print(
            f"C={c_value:g} threshold={selected_threshold:.2f} "
            f"OOF_balanced_accuracy={metrics['balanced_accuracy']:.5f} "
            f"OOF_auc={metrics['auc']:.5f}",
            flush=True,
        )

    selected_c = max(
        oof_by_c,
        key=lambda value: (
            selection_curve[f"{value:g}"]["oof_metrics"]["balanced_accuracy"],
            selection_curve[f"{value:g}"]["oof_metrics"]["auc"],
            selection_curve[f"{value:g}"]["oof_metrics"]["accuracy"],
            -abs(math.log10(value) - math.log10(0.01)),
        ),
    )
    selected = selection_curve[f"{selected_c:g}"]
    selected_threshold = float(selected["selected_threshold"])

    final_model = build_probe(selected_c, args.selection_seed)
    final_model.fit(x_train, y_train, clf__sample_weight=sample_weights)
    final_model_fitted_before_holdout_access = True

    # The frozen encoder features are stored together, but formal rows are not indexed until
    # training-only selection is complete and the final supervised probe has been fitted.
    holdout_indices = np.array([index_by_id[pid] for pid in holdout_ids], dtype=int)
    x_holdout = features[holdout_indices]
    truth = labels[holdout_indices]
    probabilities = final_model.predict_proba(x_holdout)[:, 1]
    predictions = (probabilities >= selected_threshold).astype(int)
    metrics = calculate_metrics(truth, probabilities, selected_threshold)

    patients = {
        pid: {
            "true_label": "Abnormal" if int(label) else "Normal",
            "prob_abnormal": float(probability),
            "pred_label": "Abnormal" if int(prediction) else "Normal",
        }
        for pid, label, probability, prediction in zip(
            holdout_ids,
            truth,
            probabilities,
            predictions,
            strict=True,
        )
    }
    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "policy": (
            f"same {hard_count} Mamba-derived hard cases; "
            "training-only 5-fold C and threshold selection"
        ),
        "encoder": "fomofo/tap-ct-s-3d frozen embeddings",
        "supervised_component": "StandardScaler + hard-weighted balanced LogisticRegression probe",
        "n_training": len(train_ids),
        "n_formal_holdout": len(holdout_ids),
        "training_ids_sha256": hash_ids(train_ids),
        "formal_holdout_ids_sha256": hash_ids(holdout_ids),
        "hard_example_count": hard_count,
        "sampling_weights_sha256": weights_hash,
        "formal_holdout_patient_ids_used_for_selection_or_mining": [],
        "formal_holdout_feature_rows_accessed_before_final_probe_fit": False,
        "final_model_fitted_before_holdout_access": final_model_fitted_before_holdout_access,
        "selection_folds": args.selection_folds,
        "selection_seed": args.selection_seed,
        "selection_metric": "pooled training-only OOF balanced_accuracy; AUC and accuracy tie-break",
        "selection_curve": selection_curve,
        "selected_C": selected_c,
        "selected_threshold": selected_threshold,
        "selected_oof_metrics": selected["oof_metrics"],
    }
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "features": str(args.features),
            "split_json": str(args.split_json),
            "hardness_csv": str(args.hardness_csv),
            "selection_audit": str(args.output_dir / "tapct_hard_selection_summary.json"),
            "n_training": len(train_ids),
            "n_holdout": len(holdout_ids),
            "selected_C": selected_c,
            "selected_threshold": selected_threshold,
            "hard_example_count": hard_count,
            "encoder_frozen": True,
            "validation_used_for_model_selection": False,
            "validation_used_for_hard_example_mining": False,
            "holdout_evaluated_only_after_final_probe_fit": True,
            "formal_holdout_ids_sha256": hash_ids(holdout_ids),
        },
        "metrics": metrics,
        "patients": patients,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "tapct_hard_selection_summary.json"
    output_path = args.output_dir / "tapct_hard_holdout_predictions.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"selected_C={selected_c:g}; selected_threshold={selected_threshold:.2f}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote {audit_path}", flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
