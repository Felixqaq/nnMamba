#!/usr/bin/env python3
"""Build a regression manifest from by_angle_all and the angle label JSON."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from _shared import (
    RegressionRecord,
    default_angle_json,
    default_output_dir,
    default_source_root,
    iter_ct_files,
    load_angle_lookup,
    patient_id_from_filename,
    timestamp,
)


def build_manifest(source_root: Path, angle_json: Path) -> dict:
    """Create a JSON-serializable manifest dictionary."""
    lookup = load_angle_lookup(angle_json)
    records: list[RegressionRecord] = []
    missing_labels: list[str] = []

    for path in iter_ct_files(source_root):
        patient_id = patient_id_from_filename(path)
        meta = lookup.get(patient_id)
        if meta is None:
            missing_labels.append(str(path))
            continue

        records.append(
            RegressionRecord(
                patient_id=patient_id,
                path=path,
                subset=path.parent.name,
                original_group=str(meta["group"]),
                angle_group=str(meta["angle_group"]),
                angle=float(meta["angle"]),
                binary_label=1 if "abnormal" in str(meta["group"]).lower() else 0,
            )
        )

    manifest = {
        "generated_at": timestamp(),
        "source_root": str(source_root),
        "label_source": str(angle_json),
        "total_samples": len(records),
        "subset_counts": dict(Counter(item.subset for item in records)),
        "angle_group_counts": dict(Counter(item.angle_group for item in records)),
        "angle_statistics": {
            "mean": round(float(np.mean([item.angle for item in records])), 3)
            if records
            else None,
            "median": round(float(np.median([item.angle for item in records])), 3)
            if records
            else None,
            "min": round(float(np.min([item.angle for item in records])), 3)
            if records
            else None,
            "max": round(float(np.max([item.angle for item in records])), 3)
            if records
            else None,
        },
        "missing_labels": missing_labels,
        "records": [
            {
                "patient_id": item.patient_id,
                "path": str(item.path),
                "subset": item.subset,
                "original_group": item.original_group,
                "angle_group": item.angle_group,
                "angle": item.angle,
                "binary_label": item.binary_label,
            }
            for item in records
        ],
    }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a manifest for regression training on CT angle prediction."
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
        "--output",
        type=Path,
        default=default_output_dir() / "regression_manifest.json",
        help="Where to write the generated manifest.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.source_root, args.angle_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved manifest to: {args.output}")
    print(f"Samples: {manifest['total_samples']}")
    if manifest["missing_labels"]:
        print(f"Missing labels: {len(manifest['missing_labels'])}")
        for item in manifest["missing_labels"][:10]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
