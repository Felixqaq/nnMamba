#!/usr/bin/env python3
"""Validate the regression CT dataset and report label coverage."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

from _shared import (
    default_angle_json,
    default_source_root,
    iter_ct_files,
    load_angle_lookup,
    patient_id_from_filename,
)


def inspect_dataset(source_root: Path, angle_json: Path) -> dict:
    """Collect basic dataset integrity statistics."""
    lookup = load_angle_lookup(angle_json)
    records = []
    missing = []

    for path in iter_ct_files(source_root):
        patient_id = patient_id_from_filename(path)
        meta = lookup.get(patient_id)
        if meta is None:
            missing.append(str(path))
            continue

        try:
            image = nib.load(str(path)).get_fdata()
            shape = tuple(int(v) for v in image.shape)
            if image.ndim > 3:
                image = image[..., 0]
            stats = {
                "min": float(np.min(image)),
                "max": float(np.max(image)),
                "mean": float(np.mean(image)),
            }
        except Exception as exc:
            shape = ("error",)
            stats = {"error": str(exc)}

        records.append(
            {
                "patient_id": patient_id,
                "subset": path.parent.name,
                "angle_group": str(meta["angle_group"]),
                "angle": float(meta["angle"]),
                "shape": shape,
                "stats": stats,
            }
        )

    return {
        "total": len(records),
        "missing_labels": missing,
        "subset_counts": Counter(item["subset"] for item in records),
        "angle_group_counts": Counter(item["angle_group"] for item in records),
        "angles": [item["angle"] for item in records],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check regression dataset completeness and basic CT properties."
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
    args = parser.parse_args()

    report = inspect_dataset(args.source_root, args.angle_json)
    angles = np.asarray(report["angles"], dtype=float)

    print(f"Dataset root: {args.source_root}")
    print(f"Samples: {report['total']}")
    print(f"Subsets: {dict(report['subset_counts'])}")
    print(f"Angle groups: {dict(report['angle_group_counts'])}")
    if len(angles) > 0:
        print(
            "Angle range: "
            f"{angles.min():.1f} - {angles.max():.1f} | "
            f"mean={angles.mean():.2f} | median={np.median(angles):.2f}"
        )

    if report["missing_labels"]:
        print(f"Missing labels: {len(report['missing_labels'])}")
        for item in report["missing_labels"][:10]:
            print(f"  - {item}")
    else:
        print("Missing labels: 0")


if __name__ == "__main__":
    main()

