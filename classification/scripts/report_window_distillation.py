"""Render paired test metrics after a completed window-distillation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text())
    if result["status"] != "complete":
        raise RuntimeError("Do not report incomplete experiments as efficacy results")
    with (args.run / "test_predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([int(row["label"]) for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for name in ("control", "distilled"):
        probabilities = np.array([float(row[name]) for row in rows])
        fpr, tpr, _ = roc_curve(labels, probabilities)
        axes[0].plot(fpr, tpr, label=f"{name}: AUC={result['test'][name]['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Same held-out patients")
    axes[0].legend(loc="lower right")
    for ax, name in zip(axes[1:], ("control", "distilled")):
        matrix = np.array(result["test"][name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=len(rows))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-COPD", "COPD"],
               yticklabels=["Non-COPD", "COPD"], xlabel="Predicted", ylabel="True", title=name)
    fig.savefig(args.run / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Cross-window distillation pilot", "",
             "Single-seed, same-hospital comparison; not a reproduction of the paper.", "",
             f"Teacher: {result['teacher_window']}; student: {result['student_window']}.", "",
             "| Model | AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name, metrics in result["test"].items():
        values = [metrics[k] for k in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity")]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend(["", f"KD minus control AUC: {result['distilled_minus_control']['auc']:.4f}.",
                  f"Patient-paired bootstrap 95% CI: {result['delta_auc_bootstrap_95ci']}.", "",
                  "Thresholds and checkpoints were selected on validation data only.",
                  "The control and distilled arms start from the same student checkpoint and have equal additional epochs.",
                  "Historical ensemble scores are not a matched comparator.", "", "![Comparison](comparison.png)"])
    (args.run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.run / "report.md")


if __name__ == "__main__":
    main()
