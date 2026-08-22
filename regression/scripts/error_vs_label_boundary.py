#!/usr/bin/env python3
"""Ask whether the model's errors sit where the label itself is ambiguous.

The label is a hard cut at FEV1/FVC = 70%, but the underlying quantity is
continuous and measured with error: a patient at 69 and one at 71 receive
opposite labels while being physiologically indistinguishable. If misclassified
patients cluster near the threshold, the ceiling is partly the label's, not the
model's — and that changes what "improving accuracy" can even mean.

Out-of-fold probabilities come from the same repeated nested CV used everywhere
else, so nothing here is scored on data its model trained on.
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
REPO_ROOT = REGRESSION_ROOT.parent
CSV_PATH = Path("/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv")
GOLD_JSON = REGRESSION_ROOT / "GOLD_2026_classification.json"


def ratio_lookup() -> dict[str, float]:
    """patient_id -> measured FEV1/FVC %, from whichever cohort file has it."""
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


def oof_probabilities(X, y, *, seed, outer=5, inner=5) -> np.ndarray:
    prob = np.zeros(len(y))
    for train_i, test_i in StratifiedKFold(
        outer, shuffle=True, random_state=seed
    ).split(X, y):
        s = build_search(seed, inner)
        s.fit(X[train_i], y[train_i])
        prob[test_i] = s.best_estimator_.predict_proba(X[test_i])[:, 1]
    return prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=REGRESSION_ROOT / "embeddings/tapct_fev1fvc70/features.npz",
    )
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="tapct-s")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    X, y, pids = load_features(args.features)
    prob = np.mean(
        [oof_probabilities(X, y, seed=args.seed + r) for r in range(args.repeats)],
        axis=0,
    )
    pred = (prob >= 0.5).astype(int)
    correct = pred == y

    ratios = ratio_lookup()
    missing = [p for p in pids if p not in ratios]
    if missing:
        print(f"WARNING: no FEV1/FVC for {len(missing)}: {missing[:5]}")
    r = np.array([ratios.get(p, np.nan) for p in pids])
    dist = np.abs(r - 70.0)

    print(f"[{args.label}] n={len(y)}  AUC={roc_auc_score(y, prob):.4f}  "
          f"accuracy={correct.mean():.4f}\n")

    bands = [(0, 5), (5, 10), (10, 20), (20, 100)]
    print(f"{'|FEV1/FVC - 70|':>16} {'n':>5} {'correct':>9} {'accuracy':>10}")
    rows = []
    for lo, hi in bands:
        m = (dist >= lo) & (dist < hi) & np.isfinite(dist)
        if not m.any():
            continue
        acc = float(correct[m].mean())
        rows.append({"band": f"{lo}-{hi}", "n": int(m.sum()), "accuracy": round(acc, 4)})
        print(f"{lo:>7}-{hi:<8} {int(m.sum()):>5} {int(correct[m].sum()):>9} {acc:>10.3f}")

    near = (dist < 5) & np.isfinite(dist)
    far = (dist >= 10) & np.isfinite(dist)
    print(f"\nwithin 5 points of the cut : {near.sum():3d} patients, "
          f"accuracy {correct[near].mean():.3f}")
    print(f"more than 10 points away   : {far.sum():3d} patients, "
          f"accuracy {correct[far].mean():.3f}")

    print(f"\nmisclassified patients, ordered by distance from the 70 cut:")
    print(f"{'patient':>9} {'FEV1/FVC':>9} {'truth':>9} {'P(abn)':>8} {'|d-70|':>7}")
    wrong = np.where(~correct)[0]
    for i in sorted(wrong, key=lambda i: dist[i] if np.isfinite(dist[i]) else 1e9):
        print(f"{pids[i]:>9} {r[i]:>9.0f} "
              f"{'Abnormal' if y[i] else 'Normal':>9} {prob[i]:>8.3f} "
              f"{dist[i]:>7.0f}")

    out = args.output or (
        REGRESSION_ROOT / "embeddings" / f"error_vs_boundary_{args.label}.json"
    )
    out.write_text(json.dumps({
        "label": args.label,
        "n": int(len(y)),
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "accuracy": round(float(correct.mean()), 4),
        "bands": rows,
        "per_patient": [
            {
                "patient_id": pids[i],
                "fev1_fvc": None if not np.isfinite(r[i]) else float(r[i]),
                "truth": int(y[i]),
                "prob_abnormal": round(float(prob[i]), 4),
                "correct": bool(correct[i]),
            }
            for i in range(len(y))
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
