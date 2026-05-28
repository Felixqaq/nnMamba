"""Tests for regression-side classification reporting metrics."""

from __future__ import annotations

import csv

import torch

from core.evaluator import compute_classification_metrics
from core.visualizer import plot_global_summary
from scripts.summarize_results import CLASSIFICATION_METRICS, write_summary_csv


def _metrics(labels: list[int], preds: list[int], num_classes: int):
    probs = torch.nn.functional.one_hot(
        torch.tensor(preds),
        num_classes=num_classes,
    ).float()
    return compute_classification_metrics(
        labels=torch.tensor(labels),
        preds=torch.tensor(preds),
        probs=probs,
        num_classes=num_classes,
    )


def test_multiclass_metrics_report_macro_sensitivity_and_specificity() -> None:
    metrics = _metrics(
        labels=[0, 0, 1, 1, 2, 2],
        preds=[0, 1, 1, 1, 2, 0],
        num_classes=3,
    )

    assert metrics.sensitivity == 0.66667
    assert metrics.specificity == 0.83333


def test_classification_summary_csv_keeps_new_fold_std_metrics(tmp_path) -> None:
    folds = [
        {"accuracy": 0.8, "sensitivity": 0.5, "specificity": 0.9},
        {"accuracy": 1.0, "sensitivity": 1.0, "specificity": 0.7},
    ]

    summary_path = write_summary_csv(folds, tmp_path, CLASSIFICATION_METRICS)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}

    assert rows["Sensitivity"] == {
        "metric": "Sensitivity",
        "mean": "0.75000",
        "std": "0.25000",
    }
    assert rows["Specificity"] == {
        "metric": "Specificity",
        "mean": "0.80000",
        "std": "0.10000",
    }


def test_global_classification_summary_writes_metric_barplot(tmp_path) -> None:
    fold_one = _metrics(
        labels=[0, 0, 1, 1, 2, 2],
        preds=[0, 1, 1, 1, 2, 0],
        num_classes=3,
    )
    fold_two = _metrics(
        labels=[0, 0, 1, 1, 2, 2],
        preds=[0, 0, 1, 2, 2, 2],
        num_classes=3,
    )

    plot_global_summary(
        [fold_one, fold_two],
        tmp_path,
        task_type="angle_3class",
        class_names=["Abnormal", "Intermediate", "Normal"],
    )

    assert (tmp_path / "metric_boxplot.png").exists()
    assert (tmp_path / "metric_barplot.png").exists()
