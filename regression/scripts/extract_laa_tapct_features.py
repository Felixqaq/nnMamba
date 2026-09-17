#!/usr/bin/env python3
"""Build compact LAA auxiliary features for frozen TAP-CT late fusion.

TAP-CT's pretrained encoder consumes a single CT intensity channel.  Expanding
its patch projection to two or three channels would change the pretrained model
and confound an ablation of the information itself.  This script instead keeps
the TAP-CT embedding fixed and summarizes the precomputed native-resolution
LAA density and lung-occupancy channels for late fusion with the same logistic
probe.

The density-only set contains global distribution summaries plus a 4x4x4 grid
of local density means.  The density-and-occupancy set adds matching occupancy
summaries and regional density/occupancy ratios, including reconstructed global
%LAA-950.  All feature selection and decision-threshold tuning remain confined
to training-only cross-validation in ``fit_calibrated_holdout_tapct.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


QUANTILES = (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapct-features", type=Path, required=True)
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=3, default=(112, 136, 112))
    return parser.parse_args()


def positive_quantiles(array: np.ndarray) -> list[float]:
    positive = array[array > 0]
    if positive.size == 0:
        return [0.0] * len(QUANTILES)
    return [float(value) for value in np.quantile(positive, QUANTILES)]


def block_sums(array: np.ndarray, grid_size: int) -> np.ndarray:
    if array.ndim != 3:
        raise ValueError(f"expected a 3D array, got {array.shape}")
    if any(size % grid_size for size in array.shape):
        raise ValueError(
            f"array shape {array.shape} must be divisible by grid size {grid_size}"
        )
    depth, height, width = array.shape
    reshaped = array.reshape(
        grid_size,
        depth // grid_size,
        grid_size,
        height // grid_size,
        grid_size,
        width // grid_size,
    )
    return reshaped.sum(axis=(1, 3, 5), dtype=np.float64).reshape(-1)


def channel_features(
    array: np.ndarray,
    prefix: str,
    grid_size: int,
) -> tuple[list[float], list[str]]:
    sums = block_sums(array, grid_size)
    block_voxels = float(array.size) / float(grid_size**3)
    values = [
        float(array.mean()),
        float(array.std()),
        float(array.max()),
        float(np.mean(array > 0)),
        float(np.mean(array >= 0.5)),
        *positive_quantiles(array),
        *(sums / block_voxels).tolist(),
    ]
    names = [
        f"{prefix}_mean",
        f"{prefix}_std",
        f"{prefix}_max",
        f"{prefix}_positive_fraction",
        f"{prefix}_ge_0p5_fraction",
        *(f"{prefix}_positive_q{int(q * 100):02d}" for q in QUANTILES),
        *(
            f"{prefix}_grid_{z}_{y}_{x}"
            for z in range(grid_size)
            for y in range(grid_size)
            for x in range(grid_size)
        ),
    ]
    return values, names


def extract_feature_vectors(
    stored: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    density = stored[0].astype(np.float32) / 255.0
    occupancy = stored[1].astype(np.float32) / 255.0

    density_values, density_names = channel_features(
        density, "laa_density", grid_size
    )
    occupancy_values, occupancy_names = channel_features(
        occupancy, "lung_occupancy", grid_size
    )
    density_sums = block_sums(density, grid_size)
    occupancy_sums = block_sums(occupancy, grid_size)
    regional_ratio = np.divide(
        density_sums,
        occupancy_sums,
        out=np.zeros_like(density_sums),
        where=occupancy_sums > 1e-6,
    )
    core = occupancy > 0.5
    ratio_values = [
        float(density.sum(dtype=np.float64) / max(occupancy.sum(dtype=np.float64), 1e-6)),
        float(
            density[core].sum(dtype=np.float64)
            / max(occupancy[core].sum(dtype=np.float64), 1e-6)
        ),
        *regional_ratio.tolist(),
    ]
    ratio_names = [
        "laa950_global_ratio",
        "laa950_core_ratio",
        *(
            f"laa950_ratio_grid_{z}_{y}_{x}"
            for z in range(grid_size)
            for y in range(grid_size)
            for x in range(grid_size)
        ),
    ]

    density_only = np.asarray(density_values, dtype=np.float32)
    density_occupancy = np.asarray(
        density_values + occupancy_values + ratio_values, dtype=np.float32
    )
    if not np.isfinite(density_only).all() or not np.isfinite(density_occupancy).all():
        raise ValueError("non-finite LAA auxiliary feature")
    return (
        density_only,
        density_names,
        density_occupancy,
        density_names + occupancy_names + ratio_names,
    )


def write_bundle(
    output_dir: Path,
    matrix: np.ndarray,
    patient_ids: list[str],
    names: list[str],
    groups: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "features.npz",
        features=matrix.astype(np.float32),
        patient_ids=np.asarray(patient_ids),
        feature_names=np.asarray(names),
    )
    with (output_dir / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "source_group"])
        writer.writeheader()
        for patient_id in patient_ids:
            writer.writerow(
                {"patient_id": patient_id, "source_group": groups[patient_id]}
            )


def main() -> None:
    args = parse_args()
    if args.grid_size < 1:
        raise SystemExit("--grid-size must be positive")
    expected_shape = (2, *tuple(int(value) for value in args.image_size))
    with np.load(args.tapct_features, allow_pickle=False) as data:
        patient_ids = [str(value) for value in data["patient_ids"]]
    metadata_path = args.tapct_features.parent / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig") as handle:
        groups = {
            str(row["patient_id"]): str(row["source_group"]).strip()
            for row in csv.DictReader(handle)
        }
    missing_groups = [patient_id for patient_id in patient_ids if patient_id not in groups]
    if missing_groups:
        raise SystemExit(f"metadata missing {len(missing_groups)} patients")

    density_rows: list[np.ndarray] = []
    occupancy_rows: list[np.ndarray] = []
    density_names: list[str] | None = None
    occupancy_names: list[str] | None = None
    for position, patient_id in enumerate(patient_ids, start=1):
        path = args.density_dir / f"{patient_id}.npy"
        if not path.exists():
            raise SystemExit(f"missing LAA density for {patient_id}: {path}")
        stored = np.load(path, allow_pickle=False)
        if stored.shape != expected_shape or stored.dtype != np.uint8:
            raise SystemExit(
                f"{patient_id}: shape={stored.shape}, dtype={stored.dtype}; "
                f"expected {expected_shape}/uint8"
            )
        density, current_density_names, occupancy, current_occupancy_names = (
            extract_feature_vectors(stored, args.grid_size)
        )
        density_rows.append(density)
        occupancy_rows.append(occupancy)
        density_names = current_density_names
        occupancy_names = current_occupancy_names
        if position % 100 == 0 or position == len(patient_ids):
            print(f"{position}/{len(patient_ids)}", flush=True)

    density_matrix = np.stack(density_rows)
    occupancy_matrix = np.stack(occupancy_rows)
    if density_names is None or occupancy_names is None:
        raise SystemExit("no feature rows were created")
    write_bundle(
        args.out_root / "laa_density",
        density_matrix,
        patient_ids,
        density_names,
        groups,
    )
    write_bundle(
        args.out_root / "laa_density_occupancy",
        occupancy_matrix,
        patient_ids,
        occupancy_names,
        groups,
    )
    design = {
        "tapct_features": str(args.tapct_features),
        "density_dir": str(args.density_dir),
        "n_patients": len(patient_ids),
        "image_size": list(args.image_size),
        "grid_size": args.grid_size,
        "density_feature_count": int(density_matrix.shape[1]),
        "density_occupancy_feature_count": int(occupancy_matrix.shape[1]),
        "fusion": "standardized feature concatenation after frozen TAP-CT encoding",
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "feature_design.json").write_text(
        json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(design, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
