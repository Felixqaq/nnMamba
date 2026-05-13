"""Plot figures for frozen TAP-CT embedding probe results."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


METRICS = [
    ("accuracy", "Accuracy", "#4C78A8"),
    ("macro_f1", "Macro-F1", "#F58518"),
    ("balanced_accuracy", "Balanced Acc", "#54A24B"),
]

MODEL_LABELS = {
    "logistic": "Logistic",
    "linear_svm": "Linear SVM",
    "ridge_classifier": "Ridge",
    "ordinal_logistic": "Ordinal",
    "angle_ridge_threshold": "Angle Ridge",
}

TARGET_LABELS = {
    "angle_3class": "Angle 3-Class",
    "angle_binary_extreme": "Angle Extreme Binary",
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate PNG figures from TAP-CT embedding probe results."
    )
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def safe_name(value: str) -> str:
    """Return a safe filename stem."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def short_label(value: str, width: int = 20) -> str:
    """Wrap a plot label over multiple lines."""
    return "\n".join(textwrap.wrap(value, width=width, break_long_words=False))


def load_results(path: Path) -> dict:
    """Load a TAP-CT probe results JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_dir_for(results_json: Path, output_dir: Path | None) -> Path:
    """Resolve output directory."""
    resolved = output_dir or results_json.parent
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def class_recalls(cm: np.ndarray) -> np.ndarray:
    """Compute per-class recall from a confusion matrix."""
    totals = cm.sum(axis=1)
    return np.divide(
        np.diag(cm),
        totals,
        out=np.zeros(cm.shape[0], dtype=float),
        where=totals != 0,
    )


def plot_confusion_matrix(result: dict, output_dir: Path, dpi: int) -> Path:
    """Plot one combined confusion matrix."""
    cm = np.array(result["combined"]["confusion_matrix"], dtype=int)
    class_names = result["class_names"]
    target = TARGET_LABELS.get(result["target"], result["target"])
    model = MODEL_LABELS.get(result["model"], result["model"])

    fig_width = 8.2 if len(class_names) == 3 else 7.2
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(
        f"{target} - {model}\nCombined Confusion Matrix",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([short_label(name, 24) for name in class_names], rotation=25, ha="right")
    ax.set_yticklabels([short_label(name, 28) for name in class_names])

    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            ax.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color=color,
                fontsize=16,
            )

    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()

    path = output_dir / f"{safe_name(result['target'])}_{safe_name(result['model'])}_confusion_matrix.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_recall(result: dict, output_dir: Path, dpi: int) -> Path:
    """Plot per-class recall for one result."""
    cm = np.array(result["combined"]["confusion_matrix"], dtype=int)
    recalls = class_recalls(cm)
    class_names = result["class_names"]
    target = TARGET_LABELS.get(result["target"], result["target"])
    model = MODEL_LABELS.get(result["model"], result["model"])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    bars = ax.bar(
        np.arange(len(recalls)),
        recalls,
        color=["#4C78A8", "#F58518", "#54A24B"][: len(recalls)],
        width=0.62,
    )
    ax.set_title(
        f"{target} - {model}\nPer-Class Recall",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(len(recalls)))
    ax.set_xticklabels([short_label(name, 24) for name in class_names], rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, value in zip(bars, recalls, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.035, 1.02),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()

    path = output_dir / f"{safe_name(result['target'])}_{safe_name(result['model'])}_class_recall.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_comparison(results: list[dict], output_dir: Path, dpi: int) -> list[Path]:
    """Plot metric comparison bars for each target."""
    paths = []
    targets = []
    for result in results:
        if result["target"] not in targets:
            targets.append(result["target"])

    for target in targets:
        target_results = [result for result in results if result["target"] == target]
        x = np.arange(len(target_results))
        width = 0.24
        fig, ax = plt.subplots(figsize=(max(8.5, 1.35 * len(target_results)), 5.8))
        for metric_index, (metric_key, metric_label, color) in enumerate(METRICS):
            offsets = x + (metric_index - 1) * width
            values = [
                float(result["summary"][f"mean_{metric_key}"])
                for result in target_results
            ]
            bars = ax.bar(
                offsets,
                values,
                width=width,
                label=metric_label,
                color=color,
            )
            for bar, value in zip(bars, values, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 0.025, 1.02),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        target_label = TARGET_LABELS.get(target, target)
        ax.set_title(
            f"{target_label}\nFrozen TAP-CT Probe Comparison",
            fontsize=16,
            fontweight="bold",
            pad=14,
        )
        ax.set_ylabel("5-fold Mean Score", fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS.get(result["model"], result["model"]) for result in target_results],
            rotation=20,
            ha="right",
        )
        ax.legend(loc="upper left", ncols=3)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()

        path = output_dir / f"{safe_name(target)}_metric_comparison.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def plot_all_metric_overview(results: list[dict], output_dir: Path, dpi: int) -> Path:
    """Plot one compact overview with one point per target/model."""
    labels = [
        f"{TARGET_LABELS.get(result['target'], result['target'])}\n"
        f"{MODEL_LABELS.get(result['model'], result['model'])}"
        for result in results
    ]
    x = np.arange(len(results))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(10.5, 1.25 * len(results)), 6.2))
    for metric_index, (metric_key, metric_label, color) in enumerate(METRICS):
        values = [
            float(result["summary"][f"mean_{metric_key}"])
            for result in results
        ]
        ax.bar(
            x + (metric_index - 1) * width,
            values,
            width=width,
            label=metric_label,
            color=color,
        )
    ax.set_title(
        "Frozen TAP-CT Probe Overview",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("5-fold Mean Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    path = output_dir / "probe_metric_overview.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    """Generate plots from a results.json file."""
    args = parse_args()
    written = plot_results_file(
        args.results_json,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )

    print(f"Wrote {len(written)} figures to {args.output_dir or args.results_json.parent}")
    for path in written:
        print(path)


def plot_results_file(
    results_json: Path,
    *,
    output_dir: Path | None = None,
    dpi: int = 180,
) -> list[Path]:
    """Generate all figures for one TAP-CT probe results file."""
    payload = load_results(results_json)
    results = payload["results"]
    resolved_output_dir = output_dir_for(results_json, output_dir)

    written = []
    written.append(plot_all_metric_overview(results, resolved_output_dir, dpi))
    written.extend(plot_metric_comparison(results, resolved_output_dir, dpi))
    for result in results:
        written.append(plot_confusion_matrix(result, resolved_output_dir, dpi))
        written.append(plot_class_recall(result, resolved_output_dir, dpi))
    return written


if __name__ == "__main__":
    main()
