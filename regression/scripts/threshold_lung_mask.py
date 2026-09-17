#!/usr/bin/env python3
"""Can a threshold-based lung mask replace TotalSegmentator at deployment?

A raw `CT < -950` channel is dominated by the air around the patient, not by
emphysema, so it carries almost nothing. Restricting it to the lung fixes that,
but a learned segmenter would drag a GPU into the field. This measures how far
classical thresholding plus morphology gets, against the TotalSegmentator masks
that already exist for part of the cohort, and times it on CPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct-root", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def threshold_lung_mask(ct: np.ndarray) -> np.ndarray:
    """Lung from HU alone: inside the body, low attenuation, largest components.

    No learned model and no GPU -- the same recipe quantitative CT used before
    deep segmenters existed, so a deployment can carry it as plain code.
    """
    # 1. body: everything denser than air, holes filled, largest component only.
    body = ct > -300
    body = ndimage.binary_closing(body, ndimage.generate_binary_structure(3, 1),
                                  iterations=2)
    lab, n = ndimage.label(body)
    if n:
        sizes = ndimage.sum(body, lab, range(1, n + 1))
        body = lab == (int(np.argmax(sizes)) + 1)
    # Fill the air pockets inside the torso slice by slice: a 3D fill leaks
    # through the trachea at the top of the volume.
    for z in range(body.shape[0]):
        body[z] = ndimage.binary_fill_holes(body[z])

    # 2. lung: low attenuation inside the body, minus the table and outside air.
    lung = (ct < -400) & body
    lung = ndimage.binary_opening(lung, ndimage.generate_binary_structure(3, 1))
    lab, n = ndimage.label(lung)
    if n == 0:
        return np.zeros_like(ct, dtype=bool)
    sizes = ndimage.sum(lung, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [int(order[0]) + 1]
    # A second component only if it is a plausible contralateral lung, not noise.
    if len(order) > 1 and sizes[order[1]] > 0.25 * sizes[order[0]]:
        keep.append(int(order[1]) + 1)
    lung = np.isin(lab, keep)
    return ndimage.binary_closing(lung, ndimage.generate_binary_structure(3, 1),
                                  iterations=2)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float("nan")


def main() -> None:
    args = parse_args()
    cts = {}
    for cls in ("Normal", "Abnormal"):
        for p in glob.glob(str(args.ct_root / cls / "*.nii.gz")):
            cts[os.path.basename(p).split("_")[0]] = p

    rows = []
    masks = sorted(glob.glob(str(args.mask_dir / "*.nii.gz")))
    print(f"{'patient':<12}{'Dice':>8}{'LAA950 閾值法':>15}{'LAA950 TotalSeg':>17}"
          f"{'差':>8}{'CPU 秒':>9}")
    print("-" * 72)
    for mp in masks:
        pid = os.path.basename(mp).replace(".nii.gz", "").split("_")[0]
        if pid not in cts or len(rows) >= args.limit:
            continue
        ct = np.asanyarray(nib.load(cts[pid]).dataobj).astype(np.float32)
        ref = np.asanyarray(nib.load(mp).dataobj) > 0
        if ref.shape != ct.shape:
            continue
        t0 = time.perf_counter()
        got = threshold_lung_mask(ct)
        dt = time.perf_counter() - t0
        laa_t = 100.0 * ((ct < -950) & got).sum() / max(got.sum(), 1)
        laa_r = 100.0 * ((ct < -950) & ref).sum() / max(ref.sum(), 1)
        d = dice(got, ref)
        rows.append({"patient_id": pid, "dice": round(d, 4),
                     "laa950_threshold": round(laa_t, 3),
                     "laa950_totalseg": round(laa_r, 3),
                     "seconds_cpu": round(dt, 2)})
        print(f"{pid:<12}{d:>8.4f}{laa_t:>15.2f}{laa_r:>17.2f}"
              f"{laa_t - laa_r:>8.2f}{dt:>9.2f}")

    if rows:
        ds = np.array([r["dice"] for r in rows])
        a = np.array([r["laa950_threshold"] for r in rows])
        b = np.array([r["laa950_totalseg"] for r in rows])
        secs = np.array([r["seconds_cpu"] for r in rows])
        print("-" * 72)
        print(f"n={len(rows)}  Dice 平均 {ds.mean():.4f} (最低 {ds.min():.4f})")
        print(f"%LAA-950 平均絕對差 {np.abs(a - b).mean():.2f} 個百分點")
        if len(rows) > 2:
            print(f"%LAA-950 兩法相關 r = {np.corrcoef(a, b)[0, 1]:.4f}")
        print(f"CPU 時間 中位 {np.median(secs):.2f} s / 病人")
        if args.out:
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
