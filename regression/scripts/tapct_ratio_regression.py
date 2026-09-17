#!/usr/bin/env python3
"""Does predicting FEV1/FVC as a number beat classifying it at 70?

The binary label makes a patient at 69 and one at 71 maximally different while
their lungs are not, and both models sit near chance on that band. A regression
target removes that discontinuity: 69 and 71 are two apart. This refits the
frozen TAP-CT features under both objectives on the same training patients and
scores them on the same frozen holdout, so the only thing that changes is what
the head was asked to learn.

Nothing is chosen on the holdout: alpha, C and the decision threshold all come
from training-fold out-of-fold predictions.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLD = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--pft-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--margin", type=float, default=7.0)
    return p.parse_args()


def load_ratios(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {(f or "").strip() for f in (reader.fieldnames or [])}
        for required in ("PatientID", "FEV1FVC_pct"):
            if required not in fields:
                raise SystemExit(f"{path}: no {required!r} column")
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            if row["FEV1FVC_pct"]:
                out[row["PatientID"]] = float(row["FEV1FVC_pct"])
    if not out:
        raise SystemExit(f"{path}: parsed no ratios")
    return out


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_abnormal": int(y.sum()),
        "auc": round(float(roc_auc_score(y, score)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if tp + fn else float("nan"), 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else float("nan"), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def boot_ci(y, score, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[i], score[i]))
    return (round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)) if vals else (float("nan"),) * 2


def choose_threshold(y: np.ndarray, score: np.ndarray) -> float:
    best, best_t = -1.0, 0.0
    for t in np.unique(score):
        s = balanced_accuracy_score(y, (score >= t).astype(int))
        if s > best:
            best, best_t = s, float(t)
    return best_t


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    x_all = np.asarray(data["features"], dtype=np.float64)
    pids = [str(v) for v in data["patient_ids"]]
    index = {p: i for i, p in enumerate(pids)}

    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    train_ids = list(split["training_patient_ids"])
    hold_ids = list(split["validation_patient_ids"])
    ratios = load_ratios(args.pft_csv)
    missing = [p for p in train_ids + hold_ids if p not in index or p not in ratios]
    if missing:
        raise SystemExit(f"missing features or ratio for {len(missing)}: {missing[:5]}")

    xt = x_all[[index[p] for p in train_ids]]
    xh = x_all[[index[p] for p in hold_ids]]
    rt = np.array([ratios[p] for p in train_ids], dtype=float)
    rh = np.array([ratios[p] for p in hold_ids], dtype=float)
    yt = (rt < THRESHOLD).astype(int)
    yh = (rh < THRESHOLD).astype(int)
    print(f"train {len(train_ids)} (abnormal {yt.sum()}), holdout {len(hold_ids)} "
          f"(abnormal {yh.sum()})", flush=True)

    results: dict[str, dict] = {}

    # ---------------- classification baseline ----------------
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    best_c, best_auc = None, -1.0
    for c in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0):
        oof = np.zeros(len(yt))
        for tr, va in skf.split(xt, yt):
            m = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(C=c, max_iter=5000,
                                                   class_weight="balanced",
                                                   random_state=args.seed))])
            m.fit(xt[tr], yt[tr])
            oof[va] = m.predict_proba(xt[va])[:, 1]
        auc = roc_auc_score(yt, oof)
        if auc > best_auc:
            best_auc, best_c, best_oof = auc, c, oof.copy()
    thr_c = choose_threshold(yt, best_oof)
    clf = Pipeline([("s", StandardScaler()),
                    ("c", LogisticRegression(C=best_c, max_iter=5000,
                                             class_weight="balanced",
                                             random_state=args.seed))]).fit(xt, yt)
    sh = clf.predict_proba(xh)[:, 1]
    results["classification"] = {
        "hyperparameter": {"C": best_c},
        "training_oof_auc": round(float(best_auc), 4),
        "threshold": round(float(thr_c), 6),
        "holdout": metrics(yh, sh, (sh >= thr_c).astype(int)),
        "holdout_auc_ci": boot_ci(yh, sh),
        "holdout_score": sh.tolist(),
    }

    # ---------------- ratio regression ----------------
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    best_a, best_auc_r = None, -1.0
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5):
        oof = np.zeros(len(rt))
        for tr, va in kf.split(xt):
            m = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=a))])
            m.fit(xt[tr], rt[tr])
            oof[va] = m.predict(xt[va])
        # score for AUC is "how obstructed", i.e. the negated predicted ratio
        auc = roc_auc_score(yt, -oof)
        if auc > best_auc_r:
            best_auc_r, best_a, best_oof_r = auc, a, oof.copy()
    # two thresholds: the label rule itself, and one calibrated on training OOF
    thr_r = choose_threshold(yt, -best_oof_r)
    reg = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=best_a))]).fit(xt, rt)
    ph = reg.predict(xh)
    results["regression"] = {
        "hyperparameter": {"alpha": best_a},
        "training_oof_auc": round(float(best_auc_r), 4),
        "training_oof_mae": round(float(np.abs(best_oof_r - rt).mean()), 3),
        "threshold_calibrated_on_negated_ratio": round(float(thr_r), 4),
        "holdout_at_label_rule_70": metrics(yh, -ph, (ph < THRESHOLD).astype(int)),
        "holdout_at_calibrated_threshold": metrics(yh, -ph, (-ph >= thr_r).astype(int)),
        "holdout_auc_ci": boot_ci(yh, -ph),
        "holdout_mae": round(float(np.abs(ph - rh).mean()), 3),
        "holdout_predicted_ratio": ph.tolist(),
    }

    # ---------------- the question that motivated this ----------------
    bands = {}
    dist = np.abs(rh - THRESHOLD)
    for name, sel in (("邊界 <7", dist < args.margin), ("明確 >=7", dist >= args.margin)):
        if len(set(yh[sel].tolist())) < 2:
            continue
        bands[name] = {
            "n": int(sel.sum()),
            "n_abnormal": int(yh[sel].sum()),
            "classification_auc": round(float(roc_auc_score(yh[sel], sh[sel])), 4),
            "regression_auc": round(float(roc_auc_score(yh[sel], -ph[sel])), 4),
        }
    results["by_difficulty"] = bands
    results["meta"] = {
        "features": str(args.features),
        "split_json": str(args.split_json),
        "folds": args.folds,
        "seed": args.seed,
        "selection": "alpha, C and thresholds chosen on training-fold OOF only",
        "holdout_used_for_selection": False,
    }
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 凍結 200 人 ===")
    print("{:34s}{:>9}{:>20}{:>10}{:>10}{:>10}".format(
        "", "AUC", "95% CI", "balacc", "敏感度", "特異度"))
    c = results["classification"]["holdout"]
    lo, hi = results["classification"]["holdout_auc_ci"]
    print("{:34s}{:>9.4f}   [{:.4f}, {:.4f}]{:>10.4f}{:>10.4f}{:>10.4f}".format(
        "分類 (現行做法)", c["auc"], lo, hi, c["balanced_accuracy"],
        c["sensitivity"], c["specificity"]))
    for key, label in (("holdout_at_label_rule_70", "迴歸,切 70"),
                       ("holdout_at_calibrated_threshold", "迴歸,校準閾值")):
        r = results["regression"][key]
        lo, hi = results["regression"]["holdout_auc_ci"]
        print("{:34s}{:>9.4f}   [{:.4f}, {:.4f}]{:>10.4f}{:>10.4f}{:>10.4f}".format(
            label, r["auc"], lo, hi, r["balanced_accuracy"],
            r["sensitivity"], r["specificity"]))

    print("\n=== 分難度看 AUC(這才是重點) ===")
    print("  {:12s}{:>6}{:>8}{:>12}{:>12}".format("族群", "n", "異常", "分類", "迴歸"))
    for name, b in bands.items():
        print("  {:12s}{:>6}{:>8}{:>12.4f}{:>12.4f}".format(
            name, b["n"], b["n_abnormal"], b["classification_auc"], b["regression_auc"]))
    print(f"\n  迴歸的比值預測誤差: 訓練 OOF MAE {results['regression']['training_oof_mae']}"
          f",凍結 200 MAE {results['regression']['holdout_mae']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
