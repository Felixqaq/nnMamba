#!/usr/bin/env python3
"""Materialize conservative GOLD minority-class augmentations as NIfTI files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REGRESSION_ROOT = REPO_ROOT / "regression"
sys.path.insert(0, str(REGRESSION_ROOT))

from data.manifest import build_angle_manifest, save_manifest


GOLD_STAGE_NAMES = {
    0: "GOLD 1 (Mild)",
    1: "GOLD 2 (Moderate)",
    2: "GOLD 3 (Severe)",
    3: "GOLD 4 (Very Severe)",
}


def _copy_originals(source_root: Path, output_root: Path) -> None:
    """Copy original NIfTI files while preserving immediate subset folders."""
    for source_path in sorted(source_root.rglob("*.nii.gz")):
        relative = source_path.relative_to(source_root)
        target_path = output_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)


def _load_volume(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI image and return a writable 3D float volume."""
    image = nib.load(str(path))
    volume = image.get_fdata(dtype=np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got {volume.shape}: {path}")
    return image, np.asarray(volume, dtype=np.float32)


def _random_affine(
    volume: np.ndarray,
    rng: torch.Generator,
    rotation_degrees: float,
    translation_fraction: float,
    scale_range: tuple[float, float],
) -> np.ndarray:
    """Apply the same conservative 3D affine family used at train time."""
    tensor = torch.from_numpy(volume).unsqueeze(0).float()
    _, depth, height, width = tensor.shape

    angle = float(
        torch.empty(()).uniform_(-rotation_degrees, rotation_degrees, generator=rng)
    )
    scale = float(torch.empty(()).uniform_(*scale_range, generator=rng))
    inv_scale = 1.0 / max(scale, 1e-6)
    radians = np.deg2rad(angle)
    cos_a = float(np.cos(radians)) * inv_scale
    sin_a = float(np.sin(radians)) * inv_scale
    translate = [
        float(
            torch.empty(()).uniform_(
                -translation_fraction,
                translation_fraction,
                generator=rng,
            )
        )
        for _ in range(3)
    ]

    theta = torch.tensor(
        [
            [cos_a, -sin_a, 0.0, translate[0]],
            [sin_a, cos_a, 0.0, translate[1]],
            [0.0, 0.0, inv_scale, translate[2]],
        ],
        dtype=tensor.dtype,
    ).unsqueeze(0)
    grid = F.affine_grid(
        theta,
        size=(1, 1, depth, height, width),
        align_corners=False,
    )
    augmented = F.grid_sample(
        tensor.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return augmented.numpy().astype(np.float32)


def _random_intensity(
    volume: np.ndarray,
    rng: np.random.Generator,
    intensity_scale_range: tuple[float, float],
    intensity_shift_range: tuple[float, float],
    noise_std: float,
) -> np.ndarray:
    """Apply mild raw-HU intensity jitter before saving NIfTI output."""
    scale = float(rng.uniform(*intensity_scale_range))
    shift = float(rng.uniform(*intensity_shift_range))
    augmented = volume * scale + shift
    if noise_std > 0:
        augmented = augmented + rng.normal(0.0, noise_std, size=volume.shape)
    return np.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _augmented_name(original_name: str, copy_number: int) -> str:
    """Insert an augmentation marker after the patient-id prefix."""
    if "_" not in original_name:
        return f"{Path(original_name).stem}_aug{copy_number:03d}.nii.gz"
    patient_id, suffix = original_name.split("_", 1)
    return f"{patient_id}_aug{copy_number:03d}_{suffix}"


def _write_nifti(
    image: nib.Nifti1Image,
    volume: np.ndarray,
    output_path: Path,
) -> None:
    """Save an augmented volume with the original spatial metadata."""
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(volume.astype(np.float32), image.affine, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, str(output_path))


def generate_augmented_dataset(args: argparse.Namespace) -> dict:
    """Generate a copied+augmented GOLD dataset and return a summary."""
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    labels_json = args.labels_json.resolve()
    pft_json = args.pft_json.resolve()
    manifest_path = args.manifest.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    _copy_originals(source_root, output_root)

    source_manifest = build_angle_manifest(
        source_root,
        labels_json,
        pft_json=pft_json,
        target_mode="gold",
    )
    records_by_stage: dict[int, list] = defaultdict(list)
    for record in source_manifest.records:
        if record.gold_stage is not None:
            records_by_stage[int(record.gold_stage)].append(record)

    counts = {stage: len(records) for stage, records in records_by_stage.items()}
    target_count = args.target_count or max(counts.values())
    target_stages = {stage - 1 for stage in args.gold_stages}
    torch_rng = torch.Generator().manual_seed(args.seed)
    numpy_rng = np.random.default_rng(args.seed)

    generated_records = []
    per_patient_copy_count: Counter[str] = Counter()
    for stage in sorted(target_stages):
        stage_records = records_by_stage.get(stage, [])
        if not stage_records:
            continue
        needed = max(0, target_count - len(stage_records))
        for offset in range(needed):
            source_record = stage_records[offset % len(stage_records)]
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
                    "gold_stage": int(stage),
                    "gold_stage_label": GOLD_STAGE_NAMES.get(stage, str(stage)),
                    "source_path": str(source_path),
                    "output_path": str(output_path),
                }
            )

    augmented_manifest = build_angle_manifest(
        output_root,
        labels_json,
        pft_json=pft_json,
        target_mode="gold",
    )
    save_manifest(augmented_manifest, manifest_path)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "target_count_per_gold_stage": target_count,
        "source_gold_stage_counts": {
            GOLD_STAGE_NAMES.get(stage, str(stage)): count
            for stage, count in sorted(counts.items())
        },
        "augmented_gold_stage_counts": augmented_manifest.gold_stage_counts,
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
        description="Create a materialized GOLD augmented NIfTI dataset."
    )
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "by_angle_all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "by_angle_all_gold_augmented",
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
        default=REGRESSION_ROOT / "datasets" / "generated" / "gold_manifest.augmented.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REGRESSION_ROOT
        / "datasets"
        / "generated"
        / "gold_augmented_dataset_summary.json",
    )
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--gold-stages", type=int, nargs="+", default=[2, 3, 4])
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
    print(f"GOLD counts: {summary['augmented_gold_stage_counts']}")


if __name__ == "__main__":
    main()
