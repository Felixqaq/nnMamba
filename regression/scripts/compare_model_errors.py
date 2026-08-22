#!/usr/bin/env python3
"""Cross-compare where the 3D model and the linear probe disagree with the label.

Two architectures with almost nothing in common — a from-scratch 3D Mamba trained
on 112x136x112 volumes, and a logistic regression over frozen 224x224 ViT
embeddings — should not fail on the same patients unless the difficulty is in the
data. Agreement on errors, concentrated near the FEV1/FVC = 70 cut, is evidence
that the ceiling belongs to the label rather than to either model.

The 3D side reads the per-fold prediction files the trainer writes (each patient
appears exactly once, in the fold where it was held out). The probe side recomputes
out-of-fold probabilities under the same repeated nested CV used elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_heldout import build_search, load_features  # noqa: E402

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = Path("/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv")
GOLD_JSON = REGRESSION_ROOT / "GOLD_2026_classification.json"


def ratio_lookup() -> dict[str, float]:
    out: dict[str, float] = {}
    for rec in json.loads(GOLD_JSON.read_text("utf-8-sig"))["records"]:
        v = rec.get("fev1_fvc_measured_percent")
        if v is not None:
            out[str(rec["patient_id"])] = float(v)
    if CSV_PATH.exists():
        with io.open(CSV_PATH, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                if row.get("FEV1FVC_pct"):
                    out[row["PatientID"]] = float(row["FEV1FVC_pct"])
    return out


def load_3d(run_dir: Path) -> dict[str, dict]:
    """patient_id -> {truth, prob_abnormal, correct} from the per-fold files."""
    out: dict[str, dict] = {}
    files = sorted(run_dir.glob("fold*_predictions.json"))
    if not files:
        raise SystemExit(f"no fold*_predictions.json under {run_dir}")
    for path in files:
        payload = json.loads(path.read_text("utf-8"))
        for row in payload["predictions"]:
            pid = str(row["patient_id"])
            if pid in out:
                raise SystemExit(f"{pid} appears in more than one fold")
            out[pid] = {
                "truth": 1 if row["true_label"] == "Abnormal" else 0,
                "prob_abnormal": float(row["probabilities"]["Abnormal"]),
                "correct": bool(row["correct"]),
            }
    print(f"3D predictions: {len(out)} patients from {len(files)} folds")
    return out


def probe_oof(features: Path, repeats: int, seed: int):
    X, y, pids = load_features(features)
    acc = np.zeros(len(y))
    for r in range(repeats):
        s = seed + r
        prob = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            g = build_search(s, 5)
            g.fit(X[tr], y[tr])
            prob[te] = g.best_estimator_.predict_proba(X[te])[:, 1]
        acc += prob
    return pids, y, acc / repeats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="figures/<task>/<uuid> directory of the 3D run")
    ap.add_argument("--features", type=Path,
                    default=REGRESSION_ROOT / "embeddings/tapct_fev1fvc70/features.npz")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gray-zone", type=float, default=5.0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    three_d = load_3d(args.run_dir)
    pids, y, probe_prob = probe_oof(args.features, args.repeats, args.seed)
    ratios = ratio_lookup()

    shared = [p for p in pids if p in three_d]
    print(f"patients in both: {len(shared)}\n")
    idx = {p: i for i, p in enumerate(pids)}

    rows = []
    for p in shared:
        i = idx[p]
        if three_d[p]["truth"] != int(y[i]):
            raise SystemExit(f"{p}: label mismatch between the two sources")
        rows.append({
            "patient_id": p,
            "truth": int(y[i]),
            "fev1_fvc": ratios.get(p),
            "d_from_cut": abs(ratios[p] - 70.0) if p in ratios else None,
            "p3d": three_d[p]["prob_abnormal"],
            "pprobe": float(probe_prob[i]),
            "ok3d": int(three_d[p]["prob_abnormal"] >= 0.5) == int(y[i]),
            "okprobe": int(probe_prob[i] >= 0.5) == int(y[i]),
        })

    ok3d = np.array([r["ok3d"] for r in rows])
    okpr = np.array([r["okprobe"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    d = np.array([r["d_from_cut"] if r["d_from_cut"] is not None else np.nan
                  for r in rows])

    print(f"3D    accuracy {ok3d.mean():.4f}   AUC "
          f"{roc_auc_score(truth, [r['p3d'] for r in rows]):.4f}")
    print(f"probe accuracy {okpr.mean():.4f}   AUC "
          f"{roc_auc_score(truth, [r['pprobe'] for r in rows]):.4f}\n")

    both_wrong = (~ok3d) & (~okpr)
    only3d = (~ok3d) & okpr
    onlypr = ok3d & (~okpr)
    both_ok = ok3d & okpr
    print("agreement table")
    print(f"  both correct   : {both_ok.sum():3d}")
    print(f"  only probe ok  : {only3d.sum():3d}   (3D wrong)")
    print(f"  only 3D ok     : {onlypr.sum():3d}   (probe wrong)")
    print(f"  both wrong     : {both_wrong.sum():3d}")

    # If the two models were failing independently, the overlap would be the
    # product of their error rates; a large excess means shared difficulty.
    exp = (1 - ok3d.mean()) * (1 - okpr.mean()) * len(rows)
    print(f"\n  expected overlap if errors were independent: {exp:.1f}")
    print(f"  observed both-wrong                        : {both_wrong.sum()}")
    print(f"  excess factor                              : "
          f"{both_wrong.sum() / exp:.2f}x")

    gz = d < args.gray_zone
    print(f"\nwithin +-{args.gray_zone:g} of the 70 cut: {int(np.nansum(gz))} patients")
    for name, mask in (("3D", ok3d), ("probe", okpr)):
        print(f"  {name:5s} accuracy  gray {mask[gz].mean():.3f} | "
              f"outside {mask[~gz].mean():.3f}")
    print(f"  both wrong in gray zone: {int((both_wrong & gz).sum())} / "
          f"{int(both_wrong.sum())} of all shared errors")

    print("\npatients both models got wrong, nearest the cut first:")
    print(f"{'patient':>9} {'FEV1/FVC':>9} {'truth':>9} {'P3D':>7} {'Pprobe':>8}")
    for r in sorted((r for r, w in zip(rows, both_wrong) if w),
                    key=lambda r: r["d_from_cut"] if r["d_from_cut"] is not None else 1e9):
        print(f"{r['patient_id']:>9} {r['fev1_fvc']:>9.0f} "
              f"{'Abnormal' if r['truth'] else 'Normal':>9} "
              f"{r['p3d']:>7.3f} {r['pprobe']:>8.3f}")

    out = args.output or REGRESSION_ROOT / "embeddings/model_error_comparison.json"
    out.write_text(json.dumps({
        "n": len(rows),
        "accuracy": {"3d": round(float(ok3d.mean()), 4),
                     "probe": round(float(okpr.mean()), 4)},
        "auc": {"3d": round(float(roc_auc_score(truth, [r["p3d"] for r in rows])), 4),
                "probe": round(float(roc_auc_score(truth, [r["pprobe"] for r in rows])), 4)},
        "agreement": {"both_correct": int(both_ok.sum()),
                      "only_probe": int(only3d.sum()),
                      "only_3d": int(onlypr.sum()),
                      "both_wrong": int(both_wrong.sum()),
                      "expected_if_independent": round(float(exp), 2)},
        "gray_zone": args.gray_zone,
        "per_patient": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
