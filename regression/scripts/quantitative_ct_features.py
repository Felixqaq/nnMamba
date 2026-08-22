#!/usr/bin/env python3
"""Compute the classical quantitative-CT emphysema indices for each patient.

This is the comparator the deep embeddings have to beat. %LAA-950 — the fraction
of lung voxels below -950 HU — is the established radiological measure of
emphysema extent, and it needs no learning at all: segment the lung, count
voxels, divide. A reviewer will ask how 1152 learned features compare against
this single number, so the number has to exist.

Note what the two sides actually measure. The label here is airflow obstruction
(FEV1/FVC < 70%), while %LAA-950 measures parenchymal destruction. COPD driven by
small-airway disease rather than emphysema is obstructed with near-normal density,
so this baseline is expected to miss those patients — which is precisely where
learned features could earn their place.

Caveat worth carrying into the paper: %LAA-950 is sensitive to reconstruction
kernel and slice thickness. Sharp kernels (B60f/I70f here) raise image noise and
inflate the low-attenuation tail relative to smooth kernels, so the cohort's
kernel mix is a confounder. build_summary.json records each patient's series, and
the per-patient kernel is emitted below so the effect can be checked.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REGRESSION_ROOT.parent
DEFAULT_SOURCE = REPO_ROOT / "classification/datasets/normal_v_abnormal_fev1fvc70"
DEFAULT_MASKS = REGRESSION_ROOT / "masks/totalseg/lung"

# Thresholds in Hounsfield units. -950 is the standard emphysema cut-point on
# inspiratory CT; -910 and -920 appear in older literature and are reported so a
# reviewer preferring either can read it off directly.
LAA_THRESHOLDS = (-950, -920, -910)
# Percentiles of the lung-density histogram. PD15 (the 15th percentile HU) is the
# other widely used emphysema index and is less noise-sensitive than %LAA.
PERCENTILES = (1, 5, 10, 15, 50)


def kernel_family(description: str) -> str:
    """Coarse reconstruction-kernel group, for the confounding check."""
    d = (description or "").lower()
    if re.search(r"b\s*6\d|br\s*6\d", d):
        return "B60/Br60"
    if re.search(r"i\s*7\d|br\s*59", d):
        return "I70/Br59"
    if re.search(r"br\s*4\d|b\s*3\d", d):
        return "soft"
    return "other"


def features_for(volume: np.ndarray, mask: np.ndarray) -> dict:
    """Density statistics over the masked lung."""
    lung = volume[mask]
    lung = lung[np.isfinite(lung)]
    if lung.size == 0:
        raise ValueError("empty lung mask")

    out: dict[str, float] = {"lung_voxels": float(lung.size)}
    for t in LAA_THRESHOLDS:
        out[f"laa{abs(t)}"] = float(np.mean(lung < t) * 100.0)
    for p in PERCENTILES:
        out[f"pd{p}"] = float(np.percentile(lung, p))
    out["mean_hu"] = float(lung.mean())
    out["std_hu"] = float(lung.std())
    out["skew_hu"] = float(
        np.mean(((lung - lung.mean()) / (lung.std() + 1e-8)) ** 3)
    )
    out["kurtosis_hu"] = float(
        np.mean(((lung - lung.mean()) / (lung.std() + 1e-8)) ** 4)
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    ap.add_argument(
        "--build-summary",
        type=Path,
        default=DEFAULT_SOURCE / "build_summary.json",
        help="used only to attach the reconstruction kernel per patient",
    )
    ap.add_argument("--output", type=Path, default=REGRESSION_ROOT / "embeddings/qct")
    args = ap.parse_args()

    series: dict[str, str] = {}
    if args.build_summary.exists():
        for rec in json.loads(args.build_summary.read_text("utf-8"))["records"]:
            series[rec["patient_id"]] = rec.get("series_description") or ""

    cts = sorted(args.source_dir.glob("*/*.nii.gz"))
    rows: list[dict] = []
    missing_mask: list[str] = []

    for ct_path in cts:
        pid = ct_path.name[:-7].partition("_")[0]
        mask_path = args.masks / f"{pid}.nii.gz"
        if not mask_path.exists():
            missing_mask.append(pid)
            continue

        volume = nib.load(str(ct_path)).get_fdata()
        mask = nib.load(str(mask_path)).get_fdata() > 0.5
        if mask.shape != volume.shape:
            raise SystemExit(
                f"{pid}: mask {mask.shape} does not match CT {volume.shape}"
            )
        row = {
            "patient_id": pid,
            "label": ct_path.parent.name,
            "series_description": series.get(pid, ""),
            "kernel_family": kernel_family(series.get(pid, "")),
        }
        row.update(features_for(volume, mask))
        rows.append(row)
        print(
            f"{pid} {row['label']:8s} LAA950={row['laa950']:6.2f}%  "
            f"PD15={row['pd15']:7.1f} HU  [{row['kernel_family']}]",
            flush=True,
        )

    if not rows:
        raise SystemExit(f"No masks found under {args.masks}")
    if missing_mask:
        print(f"\nWARNING: no mask for {len(missing_mask)} patients: {missing_mask}")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "qct_features.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Same npz layout the probe scripts consume, so the identical nested-CV
    # protocol can be run over these features without special-casing.
    feature_names = [k for k in rows[0] if k not in
                     ("patient_id", "label", "series_description", "kernel_family")]
    X = np.array([[r[k] for k in feature_names] for r in rows], dtype=np.float32)
    np.savez(
        args.output / "features.npz",
        features=X,
        patient_ids=np.array([r["patient_id"] for r in rows]),
        feature_names=np.array(feature_names),
    )
    with (args.output / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patient_id", "source_group"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"patient_id": r["patient_id"], "source_group": r["label"]})

    lab = np.array([r["label"] for r in rows])
    laa = np.array([r["laa950"] for r in rows])
    print(f"\nwrote {csv_path} and features.npz ({X.shape})")
    print(f"  LAA950 Abnormal: median {np.median(laa[lab == 'Abnormal']):.2f}%")
    print(f"  LAA950 Normal  : median {np.median(laa[lab == 'Normal']):.2f}%")


if __name__ == "__main__":
    main()
