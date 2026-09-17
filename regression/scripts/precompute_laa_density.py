#!/usr/bin/env python3
"""Precompute an emphysema density channel that survives downsampling.

The loader resizes each CT with `anti_aliasing=True`, a Gaussian blur, and only
then would a -950 HU test be applied. %LAA-950 lives in the low tail of the
histogram, so the blur averages it away: measured against full-resolution
%LAA-950 of 13.01% on 40 cohort patients, the current pipeline retains 2.49%
(r = 0.776) and an isotropic 2 mm 192^3 input retains 2.84% (r = 0.790) for five
times the compute. Even an infeasible 1 mm input reaches only 6.48%.

Reversing the two operations fixes it exactly. Thresholding at native resolution
produces a 0/1 map, and averaging a 0/1 map down is a local density estimate
whose mean over the lung is the global density -- so nothing is lost by the
resize. The same 40 patients reconstruct to 13.12% (r = 1.0000).

This writes two channels per patient at the training resolution:

  channel 0  emphysema density -- the fraction of each output voxel that was
             below the HU threshold inside the lung. Zero outside the lung.
  channel 1  lung occupancy -- the fraction of each output voxel that was lung.
             Needed to turn channel 0 back into %LAA, and useful on its own as a
             soft lung prior.

Both are quantised to uint8 over [0, 1]; the verification pass at the end
reports the error that quantisation and resampling together introduce, so the
cost is measured rather than assumed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage import transform

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from data.dataset import (  # noqa: E402
    _content_center,
    _crop_to_shape,
    _pad_to_shape,
    _resample_to_spacing,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct-root", type=Path, required=True,
                   help="dataset root holding Normal/ and Abnormal/ NIfTI files")
    p.add_argument("--mask-dir", type=Path, required=True,
                   help="directory of <patient_id>.nii.gz lung masks")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-size", type=int, nargs=3, default=(112, 136, 112),
                   help="training resolution, matching data.image_size in the config")
    p.add_argument("--hu", type=float, default=-950.0,
                   help="emphysema threshold; -950 is the standard for %%LAA")
    p.add_argument("--target-spacing", type=float, nargs=3, default=None,
                   help="resample to this voxel size in mm before cropping to "
                        "--image-size, matching data.target_spacing in the config. "
                        "Omit to resize straight to the shape, as the loader does "
                        "when the config sets no target_spacing.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0 means every patient")
    return p.parse_args()


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Quantise a [0, 1] density to uint8 without letting rounding leave the range."""
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def to_target_geometry(volume, source_spacing, target_spacing, target, center, pad_value):
    """Put one array on the loader's grid: resample, crop about `center`, then pad.

    The CT and the two derived maps must land on the same voxels or the channels
    are misaligned, so the crop center is computed once from the CT and passed in
    rather than recomputed per array.
    """
    resampled = _resample_to_spacing(volume, source_spacing, target_spacing)
    cropped = _crop_to_shape(resampled, target, center)
    return _pad_to_shape(cropped, target, pad_value)


def one(job):
    pid, ct_path, mask_path, target, hu, out_dir, spacing = job
    try:
        ct_img = nib.load(ct_path)
        ct = np.asanyarray(ct_img.dataobj).astype(np.float32)
        if ct.ndim > 3:
            ct = ct[..., 0]
        mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
        if mask.shape != ct.shape:
            return {"patient_id": pid,
                    "error": f"mask shape {mask.shape} != ct shape {ct.shape}"}
        lung_voxels = int(mask.sum())
        if lung_voxels < 1000:
            return {"patient_id": pid, "error": f"lung mask has only {lung_voxels} voxels"}

        # Threshold first, at native resolution. This is the whole point.
        indicator = ((ct < hu) & mask).astype(np.float32)
        laa_full = 100.0 * float(indicator.sum()) / float(lung_voxels)

        if spacing is None:
            density = transform.resize(indicator, target, order=1, preserve_range=True,
                                       anti_aliasing=True).astype(np.float32)
            occupancy = transform.resize(mask.astype(np.float32), target, order=1,
                                         preserve_range=True,
                                         anti_aliasing=True).astype(np.float32)
        else:
            source_spacing = tuple(float(z) for z in ct_img.header.get_zooms()[:3])
            # The loader crops around the scanned body it finds in the resampled CT,
            # so the center has to come from the CT and then be reused verbatim.
            resampled_ct = _resample_to_spacing(ct, source_spacing, spacing)
            center = _content_center(resampled_ct)
            density = to_target_geometry(indicator, source_spacing, spacing,
                                         target, center, 0.0)
            occupancy = to_target_geometry(mask.astype(np.float32), source_spacing,
                                           spacing, target, center, 0.0)
            del resampled_ct
        stacked = to_uint8(np.stack([density, occupancy], axis=0))
        np.save(out_dir / f"{pid}.npy", stacked)

        # Read back what was actually written, so the check covers quantisation.
        d = stacked[0].astype(np.float32) / 255.0
        o = stacked[1].astype(np.float32) / 255.0
        sel = o > 0.5
        laa_recon = 100.0 * float(d[sel].sum() / o[sel].sum()) if sel.any() else float("nan")

        return {
            "patient_id": pid,
            "source_shape": list(ct.shape),
            "source_spacing": [round(float(z), 3) for z in ct_img.header.get_zooms()[:3]],
            "lung_voxels": lung_voxels,
            "laa950_full_resolution": round(laa_full, 4),
            "laa950_reconstructed": round(laa_recon, 4),
        }
    except Exception as exc:  # a failure is a row to report, not a crashed batch
        return {"patient_id": pid, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> None:
    args = parse_args()
    target = tuple(int(v) for v in args.image_size)

    cts: dict[str, str] = {}
    for class_dir in ("Normal", "Abnormal"):
        for path in glob.glob(str(args.ct_root / class_dir / "*.nii.gz")):
            cts[os.path.basename(path).split("_")[0]] = path
    if not cts:
        raise SystemExit(f"{args.ct_root}: found no CT volumes under Normal/ or Abnormal/")

    masks = {os.path.basename(p).replace(".nii.gz", ""): p
             for p in glob.glob(str(args.mask_dir / "*.nii.gz"))}
    if not masks:
        raise SystemExit(f"{args.mask_dir}: found no lung masks")

    paired = sorted(set(cts) & set(masks))
    if not paired:
        raise SystemExit("no patient has both a CT and a lung mask")
    unmasked = sorted(set(cts) - set(masks))
    if args.limit:
        paired = paired[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(cts)} 位病人有 CT,{len(paired)} 位有遮罩可處理,"
          f"{len(unmasked)} 位還缺遮罩")
    print(f"輸出 {args.out}  解析度 {target}  閾值 {args.hu} HU  平行度 {args.workers}",
          flush=True)

    spacing = tuple(float(v) for v in args.target_spacing) if args.target_spacing else None
    jobs = [(pid, cts[pid], masks[pid], target, args.hu, args.out, spacing)
            for pid in paired]
    started = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                rate = (time.time() - started) / i
                print(f"  {i}/{len(jobs)}  已耗時 {(time.time()-started)/60:.1f} 分,"
                      f"預計剩餘 {rate*(len(jobs)-i)/60:.1f} 分", flush=True)

    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ct_root": str(args.ct_root),
        "mask_dir": str(args.mask_dir),
        "image_size": list(target),
        "hu_threshold": args.hu,
        "target_spacing": list(spacing) if spacing else None,
        "channels": ["emphysema_density", "lung_occupancy"],
        "dtype": "uint8, scaled by 255 over [0, 1]",
        "n_written": len(ok),
        "n_failed": len(bad),
        "patients_without_mask": unmasked,
        "records": rows,
    }
    (args.out / "laa_density_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成,總耗時 {(time.time()-started)/60:.1f} 分")
    print(f"  寫出 {len(ok)},失敗 {len(bad)}")
    for r in bad[:10]:
        print("   ", r["patient_id"], r["error"][:110])
    if not ok:
        raise SystemExit("沒有任何病人成功,不要拿這份輸出去訓練")

    # Verification: the stored channels must reproduce the full-resolution value.
    full = np.array([r["laa950_full_resolution"] for r in ok])
    recon = np.array([r["laa950_reconstructed"] for r in ok])
    diff = np.abs(recon - full)
    print(f"\n驗證 n={len(ok)}(從實際寫出的 uint8 檔案讀回來重算)")
    print(f"  全解析度 %LAA-950  平均 {full.mean():.3f}%  範圍 {full.min():.2f}-{full.max():.2f}")
    print(f"  重建的  %LAA-950  平均 {recon.mean():.3f}%")
    print(f"  平均絕對差 {diff.mean():.4f} pt,95% 分位 {np.percentile(diff,95):.4f} pt,"
          f"最大 {diff.max():.4f} pt")
    print(f"  相關 r = {np.corrcoef(recon, full)[0,1]:.6f}")
    worst = sorted(ok, key=lambda r: -abs(r["laa950_reconstructed"] - r["laa950_full_resolution"]))
    print("  誤差最大的 3 位:")
    for r in worst[:3]:
        print(f"    {r['patient_id']:10s} 全解析度 {r['laa950_full_resolution']:6.2f}%  "
              f"重建 {r['laa950_reconstructed']:6.2f}%")

    size_mb = sum(f.stat().st_size for f in args.out.glob("*.npy")) / 1e6
    print(f"\n  磁碟佔用 {size_mb:.0f} MB  ({size_mb/max(len(ok),1):.1f} MB/病人)")
    if unmasked:
        print(f"\n  還有 {len(unmasked)} 位缺 TotalSegmentator 遮罩,清單在 "
              f"{args.out / 'laa_density_summary.json'} 的 patients_without_mask")


if __name__ == "__main__":
    main()
