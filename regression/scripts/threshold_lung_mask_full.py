#!/usr/bin/env python3
"""Full-cohort check of the threshold lung mask, parallel across CPU cores.

The 12-patient sample agreed well (Dice 0.896, %LAA-950 within 0.52 points), but
a deployment decision needs the worst case, not the average -- especially on the
series already flagged as unusual: contrast phases, soft kernels and the short
stacks. Every patient with a TotalSegmentator mask is compared; patients without
one are still processed so the timing reflects a real deployment pass.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage


def threshold_lung_mask(ct: np.ndarray) -> np.ndarray:
    body = ct > -300
    body = ndimage.binary_closing(body, ndimage.generate_binary_structure(3, 1),
                                  iterations=2)
    lab, n = ndimage.label(body)
    if n:
        sizes = ndimage.sum(body, lab, range(1, n + 1))
        body = lab == (int(np.argmax(sizes)) + 1)
    for z in range(body.shape[0]):
        body[z] = ndimage.binary_fill_holes(body[z])

    lung = (ct < -400) & body
    lung = ndimage.binary_opening(lung, ndimage.generate_binary_structure(3, 1))
    lab, n = ndimage.label(lung)
    if n == 0:
        return np.zeros_like(ct, dtype=bool)
    sizes = ndimage.sum(lung, lab, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = [int(order[0]) + 1]
    if len(order) > 1 and sizes[order[1]] > 0.25 * sizes[order[0]]:
        keep.append(int(order[1]) + 1)
    lung = np.isin(lab, keep)
    return ndimage.binary_closing(lung, ndimage.generate_binary_structure(3, 1),
                                  iterations=2)


def one(job):
    pid, ct_path, mask_path, desc = job
    try:
        ct = np.asanyarray(nib.load(ct_path).dataobj).astype(np.float32)
        t0 = time.perf_counter()
        got = threshold_lung_mask(ct)
        dt = time.perf_counter() - t0
        row = {
            "patient_id": pid,
            "series_description": desc,
            "seconds_cpu": round(dt, 2),
            "lung_voxels_threshold": int(got.sum()),
            "laa950_threshold": round(
                100.0 * ((ct < -950) & got).sum() / max(got.sum(), 1), 3),
        }
        if mask_path:
            ref = np.asanyarray(nib.load(mask_path).dataobj) > 0
            if ref.shape == ct.shape:
                s = got.sum() + ref.sum()
                row["dice"] = round(float(2.0 * (got & ref).sum() / s), 4) if s else None
                row["lung_voxels_totalseg"] = int(ref.sum())
                row["laa950_totalseg"] = round(
                    100.0 * ((ct < -950) & ref).sum() / max(ref.sum(), 1), 3)
        return row
    except Exception as exc:  # a failure here is data to report, not a crash
        return {"patient_id": pid, "error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ct-root", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--build-summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cts = {}
    for cls in ("Normal", "Abnormal"):
        for p in glob.glob(str(args.ct_root / cls / "*.nii.gz")):
            cts[os.path.basename(p).split("_")[0]] = p
    masks = {os.path.basename(p).replace(".nii.gz", "").split("_")[0]: p
             for p in glob.glob(str(args.mask_dir / "*.nii.gz"))}
    desc = {}
    if args.build_summary and args.build_summary.exists():
        for r in json.loads(args.build_summary.read_text(encoding="utf-8"))["records"]:
            if r.get("ok"):
                desc[r["patient_id"]] = r.get("series_description", "")

    jobs = [(pid, path, masks.get(pid), desc.get(pid, "")) for pid, path in sorted(cts.items())]
    print(f"{len(jobs)} 位病人,其中 {sum(1 for j in jobs if j[2])} 位有 TotalSegmentator 遮罩可比對")
    print(f"平行度 {args.workers}", flush=True)

    started = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.append(row)
            if i % 50 == 0:
                rate = (time.time() - started) / i
                print(f"  {i}/{len(jobs)}  已耗時 {(time.time()-started)/60:.1f} 分,"
                      f"預計剩餘 {rate*(len(jobs)-i)/60:.1f} 分", flush=True)

    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    errs = [r for r in rows if "error" in r]
    ok = [r for r in rows if "dice" in r and r["dice"] is not None]
    print(f"\n完成,總耗時 {(time.time()-started)/60:.1f} 分")
    print(f"  失敗 {len(errs)}")
    for r in errs[:5]:
        print("   ", r["patient_id"], r["error"][:90])
    if not ok:
        return
    d = np.array([r["dice"] for r in ok])
    a = np.array([r["laa950_threshold"] for r in ok])
    b = np.array([r["laa950_totalseg"] for r in ok])
    s = np.array([r["seconds_cpu"] for r in rows if "seconds_cpu" in r])
    print(f"\n可比對 {len(ok)} 位")
    print(f"  Dice        平均 {d.mean():.4f}  中位 {np.median(d):.4f}  "
          f"5% 分位 {np.percentile(d,5):.4f}  最低 {d.min():.4f}")
    print(f"  Dice < 0.80 的病人: {(d < 0.80).sum()} 位 ({(d<0.80).mean():.1%})")
    print(f"  Dice < 0.70 的病人: {(d < 0.70).sum()} 位")
    print(f"  %LAA-950 平均絕對差 {np.abs(a-b).mean():.2f} pt,"
          f"95% 分位 {np.percentile(np.abs(a-b),95):.2f} pt,最大 {np.abs(a-b).max():.2f} pt")
    print(f"  %LAA-950 相關 r = {np.corrcoef(a,b)[0,1]:.4f}")
    print(f"  CPU 時間 中位 {np.median(s):.2f} s,95% 分位 {np.percentile(s,95):.2f} s")

    worst = sorted(ok, key=lambda r: r["dice"])[:10]
    print("\n  Dice 最差的 10 位:")
    for r in worst:
        print("    {:10s} dice {:.4f}  LAA {:6.2f} vs {:6.2f}  {}".format(
            r["patient_id"], r["dice"], r["laa950_threshold"],
            r["laa950_totalseg"], (r.get("series_description") or "")[:44]))


if __name__ == "__main__":
    main()
