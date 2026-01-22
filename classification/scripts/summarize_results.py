#!/usr/bin/env python3
"""
Summarize N-fold results and generate paper-ready plots.
"""

import re
import argparse
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Use a clean, modern aesthetic for paper-ready plots using matplotlib
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def parse_log(log_path):
    """
    Parse the log file to extract fold results.
    """
    with open(log_path, "r") as f:
        content = f.read()

    # regex for the metrics line
    # Format: Epoch: 5/50, Test: Acc=0.5, Sens=0.0, Spec=1.0, AUC=0.48 | Train: Acc=0.38889, AUC=0.80844
    pattern = re.compile(
        r"Epoch:?\s*(?P<epoch>\d+)(?:/\d+)?,\s*Test:?\s*(?P<metrics>.*?)\|",
        re.IGNORECASE,
    )

    lines = content.split("\n")
    fold_data = []
    fold_results = []
    last_epoch = 999

    for line in lines:
        match = pattern.search(line)
        if match:
            epoch = int(match.group("epoch"))
            metrics_str = match.group("metrics")

            m = {}
            # Clean up the string to handle colons and commas
            cleaned_metrics = metrics_str.replace(":", "").replace("Test Acc", "acc")
            for pair in cleaned_metrics.split(","):
                if "=" in pair:
                    k, v = pair.strip().split("=")
                    m[k.strip().lower()] = float(v.strip())

            if epoch <= last_epoch and fold_data:
                # New fold detected
                best_idx = np.argmax([r.get("auc", 0) for r in fold_data])
                fold_results.append(fold_data[best_idx])
                fold_data = []

            m["epoch"] = epoch
            fold_data.append(m)
            last_epoch = epoch

    if fold_data:
        # Add last fold
        best_idx = np.argmax([r.get("auc", 0) for r in fold_data])
        fold_results.append(fold_data[best_idx])

    return pd.DataFrame(fold_results)


def plot_metrics_summary(df, save_path):
    """Generate a box plot of metrics across folds using only matplotlib."""
    metrics = ["acc", "auc", "sens", "spec"]
    labels = ["Accuracy", "AUC", "Sensitivity", "Specificity"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    data = [df[m].dropna().values for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, d in enumerate(data):
        y = d
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, color="black", s=15, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("N-Fold Cross-Validation Performance", fontweight="bold", pad=20)
    ax.set_ylabel("Score")

    # Add mean ± std text
    means = df[metrics].mean()
    stds = df[metrics].std()
    for i, m in enumerate(metrics):
        val = means[m]
        err = stds[m]
        ax.text(
            i + 1,
            0.05,
            f"{val:.3f}\n±{err:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            color="darkblue",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_radar_summary(df, save_path):
    """Generate a radar chart for final average metrics using matplotlib."""
    metrics = ["acc", "auc", "sens", "spec"]
    categories = ["Accuracy", "AUC", "Sensitivity", "Specificity"]

    # If some metrics are missing, fill with 0 for the radar chart
    values = [df[m].mean() if m in df.columns else 0 for m in metrics]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        color="grey",
        size=10,
    )
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, color="#4C72B0", linewidth=2, linestyle="solid")
    ax.fill(angles, values, color="#4C72B0", alpha=0.3)

    plt.title("Average Model Performance", size=16, fontweight="bold", y=1.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Summarize N-fold training results")
    parser.add_argument("log_file", help="Path to the training log file")
    parser.add_argument(
        "--out_dir", default="../figures", help="Base directory for figures"
    )
    args = parser.parse_args()

    log_path = Path(args.log_file)
    uuid = log_path.stem

    # Create session-specific subdirectory
    target_dir = Path(args.out_dir) / uuid
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Parsing log: {log_path.name}")
    df = parse_log(log_path)

    if df.empty:
        print("❌ No matching metrics found in log file.")
        return

    print(f"📊 Found {len(df)} folds.")
    print("\nSummary Statistics:")

    present_metrics = [m for m in ["acc", "auc", "sens", "spec"] if m in df.columns]
    summary = df[present_metrics].agg(["mean", "std"]).T
    print(summary)

    # Save statistics to CSV
    csv_path = target_dir / "summary.csv"
    summary.to_csv(csv_path)
    print(f"💾 Summary saved to {csv_path}")

    # Generate plots
    box_path = target_dir / "metric_boxplot.png"
    plot_metrics_summary(df, box_path)
    print(f"🖼️  Boxplot saved to {box_path}")

    radar_path = target_dir / "radar_chart.png"
    plot_radar_summary(df, radar_path)
    print(f"🖼️  Radar chart saved to {radar_path}")


if __name__ == "__main__":
    main()
