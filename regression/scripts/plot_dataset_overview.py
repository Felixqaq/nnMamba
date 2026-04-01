#!/usr/bin/env python3
"""Generate dataset overview figures for the CT angle regression task."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _shared import (
    default_angle_json,
    default_output_dir,
    default_source_root,
    iter_ct_files,
    load_angle_lookup,
    patient_id_from_filename,
)


def load_angles(source_root: Path, angle_json: Path) -> dict[str, list[float]]:
    """Group angles by semantic angle bucket from the label JSON."""
    lookup = load_angle_lookup(angle_json)
    grouped: dict[str, list[float]] = {}

    for path in iter_ct_files(source_root):
        patient_id = patient_id_from_filename(path)
        meta = lookup.get(patient_id)
        if meta is None:
            continue
        grouped.setdefault(str(meta["angle_group"]), []).append(float(meta["angle"]))

    return grouped


def save_histogram(grouped: dict[str, list[float]], output_dir: Path) -> None:
    """Save a stacked histogram of patient angles."""
    plt.figure(figsize=(11, 6))
    bins = np.arange(100, 186, 5)
    data = [grouped[key] for key in sorted(grouped)]
    labels = [f"{key} (n={len(values)})" for key, values in sorted(grouped.items())]
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]

    plt.hist(
        data,
        bins=bins,
        stacked=True,
        alpha=0.85,
        edgecolor="black",
        color=colors[: len(data)],
        label=labels,
    )
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Number of Patients")
    plt.title("Patient Angle Distribution")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "angle_histogram.png", dpi=300)
    plt.close()


def save_boxplot(grouped: dict[str, list[float]], output_dir: Path) -> None:
    """Save a boxplot of angles by subset."""
    plt.figure(figsize=(9, 6))
    keys = sorted(grouped)
    bp = plt.boxplot(
        [grouped[key] for key in keys], tick_labels=keys, patch_artist=True
    )
    for patch, color in zip(bp["boxes"], ["#fdb462", "#80b1d3", "#b3de69", "#fb8072"]):
        patch.set_facecolor(color)
    plt.ylabel("Angle (degrees)")
    plt.title("Angle Distribution by Subset")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "angle_boxplot.png", dpi=300)
    plt.close()


def save_summary_table(grouped: dict[str, list[float]], output_dir: Path) -> None:
    """Save a paper-style summary table."""
    rows = [["Subset", "Count", "Mean", "Median", "Min", "Max"]]
    for key in sorted(grouped):
        values = np.asarray(grouped[key], dtype=float)
        rows.append(
            [
                key,
                str(len(values)),
                f"{values.mean():.1f}",
                f"{np.median(values):.1f}",
                f"{values.min():.0f}",
                f"{values.max():.0f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")
    table = ax.table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    for col in range(len(rows[0])):
        cell = table[(0, col)]
        cell.set_facecolor("#2f2f2f")
        cell.set_text_props(color="white", weight="bold")
    plt.title("Regression Dataset Overview", fontsize=14, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(output_dir / "angle_table.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create overview figures for the angle regression dataset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_source_root(),
        help="Root folder containing by_angle_all/",
    )
    parser.add_argument(
        "--angle-json",
        type=Path,
        default=default_angle_json(),
        help="JSON file containing patient angle labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir() / "figures",
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    grouped = load_angles(args.source_root, args.angle_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_histogram(grouped, args.output_dir)
    save_boxplot(grouped, args.output_dir)
    save_summary_table(grouped, args.output_dir)

    print(f"Saved figures to: {args.output_dir}")
    for name in ["angle_histogram.png", "angle_boxplot.png", "angle_table.png"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
