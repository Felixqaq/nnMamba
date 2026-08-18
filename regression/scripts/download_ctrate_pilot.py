#!/usr/bin/env python
"""Stream-download a small CT-RATE pilot subset for Normal-vs-Abnormal expansion.

Bandwidth-aware & disk-aware: for each selected patient it downloads ONE full
volume, converts raw stored values to HU (RescaleSlope/Intercept from CT-RATE
metadata), saves a compact HU NIfTI under the (git-ignored) project data dir, and
DELETES the raw ~284 MB file before moving on. Tracks cumulative download bytes
and stops if it would exceed --max-gb (protects a metered connection).

Labels (matches the 54-case cohort): Emphysema==1 -> Abnormal ; all-18-negative -> Normal.
One volume per patient (patient-level, no reconstruction duplicates).

Run in the `merlin` env (has huggingface_hub + nibabel), after `huggingface-cli login`:
    python scripts/download_ctrate_pilot.py --n-per-class 20 --max-gb 14
    python scripts/download_ctrate_pilot.py --n-per-class 1  --max-gb 2   # micro-validate first
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "ibrahimhamamci/CT-RATE"
LABELS_CSV = "dataset/multi_abnormality_labels/train_predicted_labels.csv"
META_CSV = "dataset/metadata/train_metadata.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download a CT-RATE pilot subset")
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="datasets/ctrate_pilot")
    ap.add_argument("--raw-tmp", default="datasets/ctrate_raw_tmp")
    ap.add_argument("--max-gb", type=float, default=14.0, help="Hard cap on cumulative download.")
    ap.add_argument("--inspect-only", action="store_true", help="Print HU stats, do not save/delete raw.")
    return ap.parse_args()


def volume_repo_path(vn: str) -> str:
    """train_1_a_1(.nii.gz) -> dataset/train/train_1/train_1_a/train_1_a_1.nii.gz"""
    vn = vn if vn.endswith(".nii.gz") else vn + ".nii.gz"
    base = vn[:-7]
    parts = base.split("_")
    return f"dataset/train/{'_'.join(parts[:2])}/{'_'.join(parts[:3])}/{vn}"


def patient_of(vn: str) -> str:
    return "_".join(vn[:-7].split("_")[:2]) if vn.endswith(".nii.gz") else "_".join(vn.split("_")[:2])


def to_hu(raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert stored values to HU, unless data already looks like HU."""
    if float(np.nanmin(raw)) <= -500.0:  # already HU (air ~ -1000)
        return raw
    return raw * slope + intercept


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    raw_tmp = Path(args.raw_tmp)
    raw_tmp.mkdir(parents=True, exist_ok=True)

    labels_path = hf_hub_download(REPO, LABELS_CSV, repo_type="dataset")
    meta_path = hf_hub_download(REPO, META_CSV, repo_type="dataset")
    ldf = pd.read_csv(labels_path)
    mdf = pd.read_csv(meta_path).set_index("VolumeName")
    label_cols = [c for c in ldf.columns if c != "VolumeName"]

    ldf["patient"] = ldf["VolumeName"].map(patient_of)
    first = ldf.drop_duplicates("patient", keep="first")  # 1 volume / patient
    emph = list(first[first["Emphysema"] == 1]["VolumeName"])
    norm = list(first[first[label_cols].sum(axis=1) == 0]["VolumeName"])

    rng = random.Random(args.seed)
    sel = {"Abnormal": rng.sample(emph, args.n_per_class), "Normal": rng.sample(norm, args.n_per_class)}
    print(f"Selected {args.n_per_class}/class | pool: {len(emph)} emphysema, {len(norm)} normal")

    total_bytes = 0
    saved = 0
    for group, vols in sel.items():
        gdir = out_dir / group
        gdir.mkdir(parents=True, exist_ok=True)
        for vn in vols:
            if total_bytes / 1e9 >= args.max_gb:
                print(f"\n[STOP] hit --max-gb {args.max_gb}. Downloaded {total_bytes/1e9:.1f} GB, saved {saved}.")
                return
            rp = volume_repo_path(vn)
            raw_file = hf_hub_download(REPO, rp, repo_type="dataset", local_dir=str(raw_tmp))
            fsize = os.path.getsize(raw_file)
            total_bytes += fsize
            img = nib.load(raw_file)
            data = np.asarray(img.get_fdata(), dtype=np.float32)
            slope = float(mdf.loc[vn, "RescaleSlope"]) if vn in mdf.index else 1.0
            inter = float(mdf.loc[vn, "RescaleIntercept"]) if vn in mdf.index else 0.0
            hu = to_hu(data, slope, inter)
            print(
                f"  [{group}] {vn:>18} {fsize/1e6:5.0f}MB shape={data.shape} "
                f"raw[{data.min():.0f},{data.max():.0f}] -> HU[{hu.min():.0f},{hu.max():.0f}] "
                f"(slope={slope:g} inter={inter:g}) | cum {total_bytes/1e9:.1f}GB"
            )
            if not args.inspect_only:
                out = nib.Nifti1Image(hu.astype(np.int16), img.affine)
                nib.save(out, gdir / f"{patient_of(vn)}.nii.gz")
                saved += 1
            # free disk: drop the raw volume immediately
            try:
                os.remove(raw_file)
            except OSError:
                pass

    print(f"\nDone. Saved {saved} volumes to {out_dir} | total downloaded {total_bytes/1e9:.2f} GB")


if __name__ == "__main__":
    main()
