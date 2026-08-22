#!/usr/bin/env python3
"""One row per patient: what each model predicted, and everything needed to read it.

Merges the 3D run's per-fold predictions, the linear probe's out-of-fold
probabilities, the measured FEV1/FVC, the quantitative-CT indices and the
acquisition metadata into a single table, so a disagreement can be traced to a
cause — a borderline label, a failed lung mask, an odd series — rather than just
counted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_model_errors import load_3d, probe_oof, ratio_lookup  # noqa: E402

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REGRESSION_ROOT.parent
COHORT = REPO_ROOT / "classification/datasets/normal_v_abnormal_fev1fvc70"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--features", type=Path,
                    default=REGRESSION_ROOT / "embeddings/tapct_fev1fvc70/features.npz")
    ap.add_argument("--qct", type=Path,
                    default=REGRESSION_ROOT / "embeddings/qct/qct_features.csv")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path,
                    default=REGRESSION_ROOT / "embeddings/per_patient_comparison.csv")
    args = ap.parse_args()

    three_d = load_3d(args.run_dir)
    pids, y, probe_prob = probe_oof(args.features, args.repeats, args.seed)
    ratios = ratio_lookup()

    qct: dict[str, dict] = {}
    if args.qct.exists():
        with args.qct.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                qct[row["patient_id"]] = row

    build = {r["patient_id"]: r for r in
             json.loads((COHORT / "build_summary.json").read_text("utf-8"))["records"]}

    rows = []
    for i, pid in enumerate(pids):
        if pid not in three_d:
            continue
        truth = int(y[i])
        p3d = three_d[pid]["prob_abnormal"]
        pp = float(probe_prob[i])
        ok3 = int(p3d >= 0.5) == truth
        okp = int(pp >= 0.5) == truth
        ratio = ratios.get(pid)
        q = qct.get(pid, {})
        b = build.get(pid, {})
        if ok3 and okp:
            agree = "both_correct"
        elif ok3:
            agree = "only_3D_correct"
        elif okp:
            agree = "only_probe_correct"
        else:
            agree = "both_wrong"
        rows.append({
            "patient_id": pid,
            "truth": "Abnormal" if truth else "Normal",
            "fev1_fvc": ratio,
            "dist_from_70": None if ratio is None else round(abs(ratio - 70.0), 1),
            "gray_zone": None if ratio is None else abs(ratio - 70.0) < 5,
            "prob_3d": round(p3d, 4),
            "correct_3d": ok3,
            "prob_probe": round(pp, 4),
            "correct_probe": okp,
            "agreement": agree,
            "laa950_pct": round(float(q["laa950"]), 2) if q.get("laa950") else None,
            "pd15_hu": round(float(q["pd15"]), 1) if q.get("pd15") else None,
            "lung_litres": round(float(q["lung_voxels"]) * 1e-6, 2) if q.get("lung_voxels") else None,
            "cohort": b.get("cohort"),
            "batch": b.get("batch"),
            "series": b.get("series_description"),
            "slices": b.get("num_slices"),
        })

    order = {"both_wrong": 0, "only_3D_correct": 1, "only_probe_correct": 2,
             "both_correct": 3}
    rows.sort(key=lambda r: (order[r["agreement"]],
                             r["dist_from_70"] if r["dist_from_70"] is not None else 1e9))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["agreement"]] = counts.get(r["agreement"], 0) + 1
    print(f"\n{len(rows)} patients")
    for k in ("both_correct", "only_probe_correct", "only_3D_correct", "both_wrong"):
        print(f"  {k:20s} {counts.get(k, 0):3d}")

    print(f"\n{'patient':>9} {'truth':>9} {'FF':>4} {'d70':>4} "
          f"{'P3D':>6} {'Pprobe':>7} {'LAA950':>7} {'agreement':>19}  series")
    for r in rows:
        if r["agreement"] == "both_correct":
            continue
        ff = "-" if r["fev1_fvc"] is None else f"{r['fev1_fvc']:.0f}"
        d = "-" if r["dist_from_70"] is None else f"{r['dist_from_70']:.0f}"
        laa = "-" if r["laa950_pct"] is None else f"{r['laa950_pct']:.1f}"
        print(f"{r['patient_id']:>9} {r['truth']:>9} {ff:>4} {d:>4} "
              f"{r['prob_3d']:>6.3f} {r['prob_probe']:>7.3f} {laa:>7} "
              f"{r['agreement']:>19}  {(r['series'] or '')[:28]}")

    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
