#!/usr/bin/env python3
"""Backfill classification metrics in existing regression result artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill sensitivity/specificity into legacy classification "
            "results under regression/figures."
        )
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="Directory to scan for legacy results.json files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only update results.json; do not regenerate summary PNG/CSV artifacts.",
    )
    return parser.parse_args()


def sensitivity_specificity(cm: list[list[int]] | np.ndarray) -> tuple[float, float]:
    """Return binary class-0 or multiclass macro sensitivity/specificity."""
    matrix = np.asarray(cm, dtype=float)
    if matrix.size == 0:
        return 0.0, 0.0
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square confusion matrix, got {matrix.shape}.")

    total = float(matrix.sum())
    true_positives = np.diag(matrix)
    false_negatives = matrix.sum(axis=1) - true_positives
    false_positives = matrix.sum(axis=0) - true_positives
    true_negatives = total - true_positives - false_negatives - false_positives

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.divide(
            true_positives,
            true_positives + false_negatives,
            out=np.zeros_like(true_positives, dtype=float),
            where=(true_positives + false_negatives) != 0,
        )
        specificities = np.divide(
            true_negatives,
            true_negatives + false_positives,
            out=np.zeros_like(true_negatives, dtype=float),
            where=(true_negatives + false_positives) != 0,
        )

    if matrix.shape == (2, 2):
        return float(sensitivities[0]), float(specificities[0])
    return float(sensitivities.mean()), float(specificities.mean())


def round_metric(value: float) -> float:
    """Match the metric precision used by training results."""
    return round(float(value), 5)


def update_metric_entry(entry: dict[str, Any]) -> bool:
    """Backfill sensitivity/specificity from one confusion matrix."""
    confusion = entry.get("confusion_matrix")
    if confusion is None:
        return False

    sensitivity, specificity = sensitivity_specificity(confusion)
    next_values = {
        "sensitivity": round_metric(sensitivity),
        "specificity": round_metric(specificity),
    }
    changed = any(entry.get(key) != value for key, value in next_values.items())
    entry.update(next_values)
    return changed


def update_summary(summary: dict[str, Any], folds: list[dict[str, Any]]) -> bool:
    """Backfill mean/std summary values for fold-level new metrics."""
    changed = False
    for metric in ("sensitivity", "specificity"):
        values = np.asarray(
            [fold[metric] for fold in folds if fold.get(metric) is not None],
            dtype=float,
        )
        if values.size == 0:
            continue
        next_values = {
            f"mean_{metric}": round_metric(values.mean()),
            f"std_{metric}": round_metric(values.std()),
        }
        changed |= any(summary.get(key) != value for key, value in next_values.items())
        summary.update(next_values)
    return changed


def update_total_confusion_matrix(
    payload: dict[str, Any],
    folds: list[dict[str, Any]],
) -> bool:
    """Backfill the total confusion matrix when fold matrices are available."""
    matrices = [
        np.asarray(fold["confusion_matrix"], dtype=int)
        for fold in folds
        if fold.get("confusion_matrix") is not None
    ]
    if not matrices:
        return False
    if any(matrix.shape != matrices[0].shape for matrix in matrices):
        raise ValueError("Fold confusion matrices do not share the same shape.")

    total = np.sum(matrices, axis=0).astype(int).tolist()
    if payload.get("total_confusion_matrix") == total:
        return False
    payload["total_confusion_matrix"] = total
    return True


def load_prediction_sidecar(path: Path) -> dict[str, Any] | None:
    """Load one per-fold prediction payload when it exists."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def update_standard_fold(results_path: Path, fold: dict[str, Any]) -> bool:
    """Hydrate and update one trainer result fold entry."""
    changed = False
    fold_number = fold.get("fold")
    prediction_path = results_path.parent / f"fold{fold_number}_predictions.json"
    prediction_payload = load_prediction_sidecar(prediction_path)

    if prediction_payload is not None:
        confusion = prediction_payload.get("confusion_matrix")
        if confusion is not None and fold.get("confusion_matrix") != confusion:
            fold["confusion_matrix"] = confusion
            changed = True
        if update_metric_entry(prediction_payload):
            write_json(prediction_path, prediction_payload)
            changed = True

    changed |= update_metric_entry(fold)
    return changed


def update_standard_result(results_path: Path, payload: dict[str, Any]) -> bool:
    """Update trainer-generated classification results JSON."""
    folds = [fold for fold in payload.get("folds", []) if isinstance(fold, dict)]

    changed = False
    for fold in folds:
        changed |= update_standard_fold(results_path, fold)
    if not any(fold.get("confusion_matrix") is not None for fold in folds):
        return False
    changed |= update_summary(payload.setdefault("summary", {}), folds)
    changed |= update_total_confusion_matrix(payload, folds)
    return changed


def update_probe_result(payload: dict[str, Any]) -> bool:
    """Update TAP-CT probe result entries and their combined metrics."""
    changed = False
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue

        folds = [fold for fold in result.get("folds", []) if isinstance(fold, dict)]
        if any(fold.get("confusion_matrix") is not None for fold in folds):
            for fold in folds:
                changed |= update_metric_entry(fold)
            changed |= update_summary(result.setdefault("summary", {}), folds)

        combined = result.get("combined")
        if isinstance(combined, dict):
            changed |= update_metric_entry(combined)
    return changed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a results JSON file with stable formatting."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def regenerate_standard_summary(path: Path, payload: dict[str, Any]) -> None:
    """Regenerate CSV and metric plots for one trainer result file."""
    from summarize_results import (
        CLASSIFICATION_METRICS,
        plot_metric_barplot,
        plot_metric_boxplot,
        write_summary_csv,
    )

    folds = [fold for fold in payload.get("folds", []) if isinstance(fold, dict)]
    write_summary_csv(folds, path.parent, CLASSIFICATION_METRICS)
    plot_metric_boxplot(folds, path.parent, CLASSIFICATION_METRICS, "classification")
    plot_metric_barplot(folds, path.parent, CLASSIFICATION_METRICS, "classification")


def regenerate_probe_plots(path: Path) -> None:
    """Regenerate metric comparison plots for one TAP-CT probe results file."""
    from plot_embedding_probe_results import plot_results_file

    plot_results_file(path, output_dir=path.parent)


def backfill_path(path: Path, regenerate_plots: bool) -> str:
    """Update one results file and return its schema category."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload.get("folds"), list):
        if not update_standard_result(path, payload):
            return "standard_skipped"
        write_json(path, payload)
        if regenerate_plots:
            regenerate_standard_summary(path, payload)
        return "standard_updated"

    if isinstance(payload.get("results"), list):
        if not update_probe_result(payload):
            return "probe_skipped"
        write_json(path, payload)
        if regenerate_plots:
            regenerate_probe_plots(path)
        return "probe_updated"

    return "unknown_skipped"


def main() -> None:
    """Backfill every legacy results file under the requested figures root."""
    args = parse_args()
    counts = Counter(
        backfill_path(path, regenerate_plots=not args.no_plots)
        for path in sorted(args.figures_root.rglob("results.json"))
    )
    print(f"Scanned results files: {sum(counts.values())}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
