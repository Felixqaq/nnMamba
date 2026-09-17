#!/usr/bin/env python3
"""Bake the target-spacing resample once, so training stops paying for it 80 times.

Measured on this box: training at 512x512x112 with the resample done on the fly
runs at 1.44 s per batch of 2 with the GPU at 1% -- 16.8 minutes an epoch, 22
hours for 80 epochs, essentially all of it spent resampling in three dataloader
workers. The same model needs 132.6 ms per sample on the card. The work is
identical every epoch, so it belongs on disk.

The output is written so that the existing loader needs no change. Each volume is
resampled to the target spacing and cropped, then saved with that spacing in its
header; `load_ct` with the same `target_spacing` finds the factors already equal,
skips the resample, finds the shape already within `image_size`, skips the crop,
and goes straight to normalise-then-pad. Padding is deliberately left to the
loader rather than baked in: it normalises on scanned content and pads afterwards,
so pre-padding here would let the amount of empty space leak into the intensity
statistics.

The emphysema channels are built in the same pass and from the same crop center,
because a center recomputed per array would put the two channels on different
voxels.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    _content_center,
    _crop_to_shape,
    _pad_to_shape,
    _resample_to_spacing,
)

# CT reconstructions in this cohort carry padding sentinels down to -8192 and
# stray highs above 45000, both outside int16. Clipping to the diagnostic CT range
# costs nothing -- the config windows to [-1000, 400] before normalising anyway.
HU_MIN, HU_MAX = -1024.0, 3071.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ct-root", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--out-ct", type=Path, required=True,
                   help="new source_dir; Normal/ and Abnormal/ are recreated inside")
    p.add_argument("--out-density", type=Path, required=True)
    p.add_argument("--target-spacing", type=float, nargs=3, default=(0.7, 0.7, 3.0))
    p.add_argument("--image-size", type=int, nargs=3, default=(512, 512, 112))
    p.add_argument("--hu", type=float, default=-950.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def one(job):
    src, cls, mask_path, spacing, target, hu, out_ct, out_density = job
    pid = os.path.basename(src).split("_")[0]
    try:
        img = nib.load(src)
        ct = np.asanyarray(img.dataobj).astype(np.float32)
        if ct.ndim > 3:
            ct = ct[..., 0]
        source_spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

        resampled = _resample_to_spacing(ct, source_spacing, spacing)
        center = _content_center(resampled)
        cropped = _crop_to_shape(resampled, target, center)

        # Saved unpadded: the loader normalises on content, then pads.
        stored = np.clip(cropped, HU_MIN, HU_MAX).astype(np.int16)
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        out_name = os.path.basename(src)
        if out_name.endswith(".nii.gz"):
            out_name = out_name[: -len(".nii.gz")] + ".nii"
        nib.save(nib.Nifti1Image(stored, affine), str(out_ct / cls / out_name))

        row = {"patient_id": pid, "class": cls,
               "source_shape": list(ct.shape),
               "source_spacing": [round(v, 3) for v in source_spacing],
               "stored_shape": list(stored.shape)}

        if mask_path is None:
            row["density"] = "skipped: no lung mask"
            return row

        mask = np.asanyarray(nib.load(mask_path).dataobj) > 0
        if mask.shape != ct.shape:
            row["density"] = f"skipped: mask shape {mask.shape} != ct {ct.shape}"
            return row
        lung_voxels = int(mask.sum())
        if lung_voxels < 1000:
            row["density"] = f"skipped: lung mask has {lung_voxels} voxels"
            return row

        indicator = ((ct < hu) & mask).astype(np.float32)
        laa_full = 100.0 * float(indicator.sum()) / float(lung_voxels)

        def to_grid(volume, pad_value):
            r = _resample_to_spacing(volume, source_spacing, spacing)
            return _pad_to_shape(_crop_to_shape(r, target, center), target, pad_value)

        density = to_grid(indicator, 0.0)
        occupancy = to_grid(mask.astype(np.float32), 0.0)
        stacked = np.clip(np.rint(np.stack([density, occupancy]) * 255.0),
                          0, 255).astype(np.uint8)
        np.save(out_density / f"{pid}.npy", stacked)

        d = stacked[0].astype(np.float32) / 255.0
        o = stacked[1].astype(np.float32) / 255.0
        sel = o > 0.5
        row["laa950_full_resolution"] = round(laa_full, 4)
        row["laa950_reconstructed"] = round(
            100.0 * float(d[sel].sum() / o[sel].sum()) if sel.any() else float("nan"), 4)
        return row
    except Exception as exc:
        return {"patient_id": pid, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> None:
    args = parse_args()
    spacing = tuple(float(v) for v in args.target_spacing)
    target = tuple(int(v) for v in args.image_size)

    masks = {os.path.basename(p).replace(".nii.gz", ""): p
             for p in glob.glob(str(args.mask_dir / "*.nii.gz"))}
    if not masks:
        raise SystemExit(f"{args.mask_dir}: found no lung masks")

    jobs = []
    for cls in ("Normal", "Abnormal"):
        for src in sorted(glob.glob(str(args.ct_root / cls / "*.nii.gz"))):
            pid = os.path.basename(src).split("_")[0]
            jobs.append((src, cls, masks.get(pid), spacing, target, args.hu,
                         args.out_ct, args.out_density))
    if not jobs:
        raise SystemExit(f"{args.ct_root}: found no CT volumes")
    if args.limit:
        jobs = jobs[: args.limit]

    for cls in ("Normal", "Abnormal"):
        (args.out_ct / cls).mkdir(parents=True, exist_ok=True)
    args.out_density.mkdir(parents=True, exist_ok=True)

    without_mask = sum(1 for j in jobs if j[2] is None)
    print(f"{len(jobs)} 位病人,{without_mask} 位沒有肺遮罩(仍會寫出 CT,但沒有密度通道)")
    print(f"目標 spacing {spacing}  image_size {target}  平行度 {args.workers}")
    print(f"CT -> {args.out_ct}\n密度 -> {args.out_density}", flush=True)

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
    with_density = [r for r in ok if "laa950_reconstructed" in r]

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ct_root": str(args.ct_root),
        "out_ct": str(args.out_ct),
        "out_density": str(args.out_density),
        "target_spacing": list(spacing),
        "image_size": list(target),
        "hu_threshold": args.hu,
        "hu_clip_on_write": [HU_MIN, HU_MAX],
        "padding": "left to the loader; volumes are stored cropped but unpadded",
        "n_written": len(ok),
        "n_with_density": len(with_density),
        "n_failed": len(bad),
        "records": rows,
    }
    (args.out_density / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n完成,總耗時 {(time.time()-started)/60:.1f} 分")
    print(f"  CT 寫出 {len(ok)},密度寫出 {len(with_density)},失敗 {len(bad)}")
    for r in bad[:8]:
        print("   ", r["patient_id"], r["error"][:110])
    for r in ok:
        if "density" in r:
            print("    無密度:", r["patient_id"], r["density"][:80])
    if not ok:
        raise SystemExit("沒有任何病人成功,不要拿這份輸出去訓練")

    if with_density:
        f = np.array([r["laa950_full_resolution"] for r in with_density])
        g = np.array([r["laa950_reconstructed"] for r in with_density])
        d = np.abs(g - f)
        print(f"\n密度通道驗證 n={len(with_density)}")
        print(f"  全解析度 %LAA-950 平均 {f.mean():.3f}%,重建 {g.mean():.3f}%")
        print(f"  平均絕對差 {d.mean():.4f} pt,95% 分位 {np.percentile(d,95):.4f},"
              f"最大 {d.max():.4f}")
        print(f"  相關 r = {np.corrcoef(g, f)[0,1]:.6f}")

    shapes = np.array([r["stored_shape"] for r in ok])
    print(f"\n儲存形狀 中位 {[int(v) for v in np.median(shapes,axis=0)]}  "
          f"最大 {[int(v) for v in shapes.max(axis=0)]}")
    ct_mb = sum(f.stat().st_size for f in args.out_ct.rglob("*.nii")) / 1e6
    de_mb = sum(f.stat().st_size for f in args.out_density.glob("*.npy")) / 1e6
    total_gb = (ct_mb + de_mb) / 1000
    free_gb = shutil.disk_usage(args.out_ct).free / 1e9
    print(f"磁碟 CT {ct_mb/1000:.1f} GB + 密度 {de_mb/1000:.1f} GB = "
          f"{total_gb:.1f} GB(剩餘 {free_gb:.0f} GB)")


if __name__ == "__main__":
    main()
