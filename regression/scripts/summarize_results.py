#!/usr/bin/env python3
"""Summarize cross-validation regression/classification results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


REGRESSION_METRICS = [
    ("mae", "MAE", "#2ca02c"),
    ("rmse", "RMSE", "#d62728"),
    ("r2", "R2", "#9467bd"),
    ("pearson", "Pearson", "#ff7f0e"),
]

CLASSIFICATION_METRICS = [
    ("accuracy", "Accuracy", "#2ca02c"),
    ("macro_f1", "Macro F1", "#d62728"),
    ("balanced_accuracy", "Balanced Accuracy", "#9467bd"),
    ("macro_precision", "Macro Precision", "#ff7f0e"),
    ("macro_recall", "Macro Recall", "#17becf"),
    ("sensitivity", "Sensitivity", "#8c564b"),
    ("specificity", "Specificity", "#bcbd22"),
]


def load_results(path: Path) -> list[dict]:
    """Read fold metrics from results.json."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get("folds", []))


def detect_metrics(folds: list[dict]) -> tuple[list[tuple[str, str, str]], str]:
    """Detect whether the results contain regression or classification metrics."""
    if any("macro_f1" in fold for fold in folds):
        return CLASSIFICATION_METRICS, "classification"
    return REGRESSION_METRICS, "regression"


def write_summary_csv(
    folds: list[dict],
    output_dir: Path,
    metrics: list[tuple[str, str, str]],
) -> Path:
    """Write mean/std summary to CSV."""
    output_path = output_dir / "summary.csv"
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        for key, label, _ in metrics:
            values = np.asarray([fold[key] for fold in folds if key in fold], dtype=float)
            if values.size == 0:
                continue
            writer.writerow([label, f"{values.mean():.5f}", f"{values.std():.5f}"])
    return output_path


def plot_metric_boxplot(
    folds: list[dict],
    output_dir: Path,
    metrics: list[tuple[str, str, str]],
    task_name: str,
) -> Path:
    """Create a fold-wise metric boxplot."""
    data = []
    labels = []
    colors = []
    for key, label, color in metrics:
        values = np.asarray([fold[key] for fold in folds if key in fold], dtype=float)
        if values.size == 0:
            continue
        data.append(values)
        labels.append(label)
        colors.append(color)

    if not data:
        raise ValueError("No supported metrics found in results.json.")

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    for idx, values in enumerate(data, start=1):
        x = np.random.normal(idx, 0.04, size=len(values))
        ax.scatter(x, values, color="black", s=12, alpha=0.55, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Validation {task_name.title()} Metrics", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = output_dir / "metric_boxplot.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_metric_barplot(
    folds: list[dict],
    output_dir: Path,
    metrics: list[tuple[str, str, str]],
    task_name: str,
) -> Path:
    """Create mean ± std barplot for paper summaries."""
    labels = []
    means = []
    stds = []
    colors = []

    for key, label, color in metrics:
        values = np.asarray([fold[key] for fold in folds if key in fold], dtype=float)
        if values.size == 0:
            continue
        labels.append(label)
        means.append(values.mean())
        stds.append(values.std())
        colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.78)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(f"{task_name.title()} Summary (Mean ± Std)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    for idx, (mean, std) in enumerate(zip(means, stds)):
        ax.text(idx, mean, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    output_path = output_dir / "metric_barplot.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize regression/classification CV results"
    )
    parser.add_argument("results_json", type=Path, help="Path to results.json")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory for summary files; defaults to results.json parent",
    )
    args = parser.parse_args()

    output_dir = args.out_dir or args.results_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = load_results(args.results_json)
    if not folds:
        raise ValueError("results.json does not contain any fold entries.")

    metrics, task_name = detect_metrics(folds)
    csv_path = write_summary_csv(folds, output_dir, metrics)
    box_path = plot_metric_boxplot(folds, output_dir, metrics, task_name)
    bar_path = plot_metric_barplot(folds, output_dir, metrics, task_name)

    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved metric boxplot to: {box_path}")
    print(f"Saved metric barplot to: {bar_path}")


if __name__ == "__main__":
    main()
