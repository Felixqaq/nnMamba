#!/usr/bin/env python3
"""Linear probe on frozen CT embeddings, with an honest evaluation protocol.

Why this exists rather than train_embedding_probe.py: that script reports
cross-validated scores, and its target modes are angle/GOLD specific. The failure
this one is built to avoid is the one the 3D run walked into — picking the best
epoch (or the best hyper-parameter) on the same split whose score you then report.
With 182 patients and a ~37-case fold, the maximum over 20 noisy evaluations lands
near 0.70 even when true performance is chance.

Protocol:
  * a stratified held-out TEST set is split off once and never touched until the
    final line of the run;
  * C (and any other hyper-parameter) is chosen by cross-validation *inside* the
    development set only;
  * the test score is computed once, from a single refit on the whole dev set.

A permutation test on the labels gives the null distribution, so "0.62" can be
read against what shuffled labels achieve on this very cohort.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REGRESSION_ROOT / "embeddings" / "tapct_fev1fvc70" / "features.npz"
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, patient_ids) with y=1 for Abnormal.

    Labels come from metadata.csv's source_group (the parent folder the CT was
    read from), not from the npz arrays: the extractor writes whatever target mode
    it ran under into keys literally named "angle_3class", so under
    --target-mode normal_v_abnormal that key holds Normal/Abnormal indices and
    reading it by name would silently invert the classes.
    """
    import csv

    data = np.load(path, allow_pickle=True)
    X = np.asarray(data["features"], dtype=np.float64)
    pids = [str(v) for v in data["patient_ids"]]

    meta_path = path.parent / "metadata.csv"
    if not meta_path.exists():
        raise SystemExit(f"Need {meta_path} to resolve labels")
    groups: dict[str, str] = {}
    with meta_path.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            groups[str(row["patient_id"])] = str(row["source_group"]).strip()

    missing = [p for p in pids if p not in groups]
    if missing:
        raise SystemExit(f"No source_group for {len(missing)} patients: {missing[:5]}")
    unknown = {g for g in groups.values() if g.lower() not in ("normal", "abnormal")}
    if unknown:
        raise SystemExit(f"Unexpected source_group values: {sorted(unknown)}")

    y = np.array([1 if groups[p].lower() == "abnormal" else 0 for p in pids], dtype=int)
    return X, y, pids


def evaluate(y_true: np.ndarray, prob: np.ndarray, thresh: float = 0.5) -> dict:
    pred = (prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_abnormal": int(y_true.sum()),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "auc": round(float(roc_auc_score(y_true, prob)), 4),
        "sensitivity": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
        "specificity": round(float(tn / (tn + fp)) if tn + fp else 0.0, 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "majority_baseline_accuracy": round(
            float(max(y_true.mean(), 1 - y_true.mean())), 4
        ),
    }


def build_search(seed: int, inner_splits: int) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000, class_weight="balanced", solver="liblinear"
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipe,
        {"clf__C": C_GRID},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(inner_splits, shuffle=True, random_state=seed),
        n_jobs=-1,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    X, y, pids = load_features(args.features)
    print(f"features : {X.shape}  Abnormal={int(y.sum())}  Normal={int((1 - y).sum())}")

    idx = np.arange(len(y))
    dev_i, test_i = train_test_split(
        idx, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"dev      : {len(dev_i)}  (Abnormal={int(y[dev_i].sum())})")
    print(f"held-out : {len(test_i)}  (Abnormal={int(y[test_i].sum())})  "
          f"— untouched until the final fit\n")

    # --- model selection: development set only -----------------------------
    search = build_search(args.seed, args.inner_splits)
    search.fit(X[dev_i], y[dev_i])
    print(f"chosen C            : {search.best_params_['clf__C']}")
    print(f"inner CV bal-acc    : {search.best_score_:.4f}  (dev set, NOT a result)")
    for C, m in zip(C_GRID, search.cv_results_["mean_test_score"]):
        print(f"    C={C:<8} bal-acc={m:.4f}")

    # --- the one and only look at the held-out set --------------------------
    prob = search.best_estimator_.predict_proba(X[test_i])[:, 1]
    test_metrics = evaluate(y[test_i], prob)
    print("\n=== HELD-OUT TEST (single evaluation) ===")
    for k, v in test_metrics.items():
        print(f"  {k:28s}: {v}")

    # --- null distribution --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    null = []
    for _ in range(args.permutations):
        y_shuf = rng.permutation(y[dev_i])
        s = build_search(args.seed, args.inner_splits)
        s.fit(X[dev_i], y_shuf)
        p = s.best_estimator_.predict_proba(X[test_i])[:, 1]
        null.append(balanced_accuracy_score(y[test_i], (p >= 0.5).astype(int)))
    null = np.asarray(null)
    observed = test_metrics["balanced_accuracy"]
    null_summary = None
    p_value = None
    if len(null):
        p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        null_summary = {
            "n": int(len(null)),
            "mean": round(float(null.mean()), 4),
            "p95": round(float(np.quantile(null, 0.95)), 4),
            "max": round(float(null.max()), 4),
            "p_value": round(p_value, 4),
        }
        print(f"\npermutation null (n={len(null)}): mean={null.mean():.4f} "
              f"p95={np.quantile(null, 0.95):.4f} max={null.max():.4f}")
        print(f"observed balanced accuracy = {observed:.4f}   p = {p_value:.4f}")
        print(
            "\nVERDICT: "
            + (
                "above chance (p < 0.05)"
                if p_value < 0.05
                else "NOT distinguishable from chance on this cohort"
            )
        )
    else:
        print("\npermutation test skipped (--permutations 0)")

    out = args.output or (
        REGRESSION_ROOT
        / "embeddings"
        / f"linear_probe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "features": str(args.features),
                "n_total": int(len(y)),
                "test_size": args.test_size,
                "seed": args.seed,
                "chosen_C": search.best_params_["clf__C"],
                "dev_inner_cv_balanced_accuracy": round(float(search.best_score_), 4),
                "held_out": test_metrics,
                "held_out_patient_ids": [pids[i] for i in test_i],
                "permutation_null": null_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
