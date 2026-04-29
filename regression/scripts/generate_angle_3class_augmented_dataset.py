#!/usr/bin/env python3
"""Materialize conservative angle three-class minority augmentations as NIfTI files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REGRESSION_ROOT = REPO_ROOT / "regression"
sys.path.insert(0, str(REGRESSION_ROOT))

from data.manifest import ANGLE_3CLASS_NAMES, build_angle_manifest, save_manifest
from generate_gold_augmented_dataset import (
    _augmented_name,
    _copy_originals,
    _load_volume,
    _random_affine,
    _random_intensity,
    _write_nifti,
)


def generate_augmented_dataset(args: argparse.Namespace) -> dict:
    """Generate a copied+augmented angle three-class dataset and return a summary."""
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    labels_json = args.labels_json.resolve()
    pft_json = args.pft_json.resolve() if args.pft_json is not None else None
    manifest_path = args.manifest.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    _copy_originals(source_root, output_root)

    source_manifest = build_angle_manifest(
        source_root,
        labels_json,
        pft_json=pft_json,
        target_mode="angle_3class",
    )
    records_by_class: dict[int, list] = defaultdict(list)
    for record in source_manifest.records:
        if record.class_index is not None:
            records_by_class[int(record.class_index)].append(record)

    counts = {class_idx: len(records) for class_idx, records in records_by_class.items()}
    target_count = args.target_count or max(counts.values())
    target_classes = (
        set(args.classes)
        if args.classes is not None
        else {class_idx for class_idx, count in counts.items() if count < target_count}
    )
    torch_rng = torch.Generator().manual_seed(args.seed)
    numpy_rng = np.random.default_rng(args.seed)

    generated_records = []
    per_patient_copy_count: Counter[str] = Counter()
    for class_idx in sorted(target_classes):
        class_records = records_by_class.get(class_idx, [])
        if not class_records:
            continue
        needed = max(0, target_count - len(class_records))
        for offset in range(needed):
            source_record = class_records[offset % len(class_records)]
            source_path = Path(source_record.path)
            image, volume = _load_volume(source_path)
            augmented = _random_affine(
                volume,
                torch_rng,
                rotation_degrees=args.rotation_degrees,
                translation_fraction=args.translation_fraction,
                scale_range=tuple(args.scale_range),
            )
            augmented = _random_intensity(
                augmented,
                numpy_rng,
                intensity_scale_range=tuple(args.intensity_scale_range),
                intensity_shift_range=tuple(args.intensity_shift_range),
                noise_std=args.noise_std,
            )
            per_patient_copy_count[source_record.patient_id] += 1
            copy_number = per_patient_copy_count[source_record.patient_id]
            output_name = _augmented_name(source_path.name, copy_number)
            relative_parent = source_path.parent.relative_to(source_root)
            output_path = output_root / relative_parent / output_name
            if output_path.exists() and not args.overwrite:
                continue
            _write_nifti(image, augmented, output_path)
            generated_records.append(
                {
                    "patient_id": source_record.patient_id,
                    "class_index": int(class_idx),
                    "class_label": ANGLE_3CLASS_NAMES[class_idx],
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                }
            )

    augmented_manifest = build_angle_manifest(
        output_root,
        labels_json,
        pft_json=pft_json,
        target_mode="angle_3class",
    )
    save_manifest(augmented_manifest, manifest_path)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "target_count_per_class": target_count,
        "source_class_counts": {
            ANGLE_3CLASS_NAMES[class_idx]: count
            for class_idx, count in sorted(counts.items())
        },
        "augmented_class_counts": augmented_manifest.class_counts,
        "total_records": augmented_manifest.counts["total"],
        "unique_patients": augmented_manifest.counts["unique_patients"],
        "generated_count": len(generated_records),
        "generated_records": generated_records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a materialized angle three-class augmented NIfTI dataset."
    )
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "by_angle_all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "by_angle_all_angle_3class_augmented",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=REPO_ROOT / "patient_angle_classification_by_group.json",
    )
    parser.add_argument("--pft-json", type=Path, default=REPO_ROOT / "pft.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REGRESSION_ROOT
        / "datasets"
        / "generated"
        / "angle_3class_manifest.augmented.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REGRESSION_ROOT
        / "datasets"
        / "generated"
        / "angle_3class_augmented_dataset_summary.json",
    )
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Zero-based class indices to augment. Defaults to all minority classes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rotation-degrees", type=float, default=7.0)
    parser.add_argument("--translation-fraction", type=float, default=0.05)
    parser.add_argument("--scale-range", type=float, nargs=2, default=[0.95, 1.05])
    parser.add_argument(
        "--intensity-scale-range",
        type=float,
        nargs=2,
        default=[0.98, 1.02],
    )
    parser.add_argument(
        "--intensity-shift-range",
        type=float,
        nargs=2,
        default=[-15.0, 15.0],
    )
    parser.add_argument("--noise-std", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = generate_augmented_dataset(parse_args())
    print(f"Output root: {summary['output_root']}")
    print(f"Manifest: {summary['manifest']}")
    print(f"Total records: {summary['total_records']}")
    print(f"Unique patients: {summary['unique_patients']}")
    print(f"Generated augmented files: {summary['generated_count']}")
    print(f"Angle 3-class counts: {summary['augmented_class_counts']}")


if __name__ == "__main__":
    main()
