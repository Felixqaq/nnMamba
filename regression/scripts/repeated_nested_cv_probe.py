#!/usr/bin/env python3
"""Repeated nested cross-validation for a frozen-embedding linear probe.

Replaces the single 80/20 split of linear_probe_heldout.py. That protocol is
honest but wasteful at n=182: only 37 patients are ever scored, so the estimate
swings ~0.13 in balanced accuracy purely on which patients land in the test set —
two encoders of very different size were observed to rise and fall together
across the same seeds, which is the signature of split noise, not model quality.

Here every patient is scored exactly once per repeat:

    outer StratifiedKFold(5)          <- the fold being scored is held out
      └─ inner GridSearchCV(5) over C <- selection sees outer-train only
    repeated N times with different shuffles

Model selection never touches the outer test fold, so the leakage the 3D run
suffered (best-of-20-evaluations on the reported split) cannot occur here.

Reported per run:
  * per-repeat metrics computed on the pooled out-of-fold predictions;
  * the spread across repeats — how much re-partitioning the SAME patients moves
    the estimate. This is not a confidence interval and must not be quoted as one;
  * a patient-level bootstrap 95% CI, which is the interval to put in a paper.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_heldout import build_search, evaluate, load_features  # noqa: E402

REGRESSION_ROOT = Path(__file__).resolve().parents[1]


def nested_oof_probabilities(
    X: np.ndarray,
    y: np.ndarray,
    *,
    repeat_seed: int,
    outer_splits: int,
    inner_splits: int,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold P(abnormal) for every patient, plus the C chosen per fold."""
    prob = np.zeros(len(y), dtype=float)
    chosen: list[float] = []
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=repeat_seed)
    for train_i, test_i in outer.split(X, y):
        search = build_search(repeat_seed, inner_splits)
        search.fit(X[train_i], y[train_i])
        prob[test_i] = search.best_estimator_.predict_proba(X[test_i])[:, 1]
        chosen.append(float(search.best_params_["clf__C"]))
    return prob, chosen


def bootstrap_ci(
    y: np.ndarray, prob: np.ndarray, *, n: int, seed: int
) -> dict[str, dict]:
    """Patient-level bootstrap CI — the uncertainty that belongs in a paper.

    The spread across repeats only says how much the estimate moves when the same
    182 patients are re-partitioned; it says nothing about sampling a different
    182 patients, and is several times too narrow to quote as a confidence
    interval. Resampling patients with replacement gives the latter.
    """
    rng = np.random.default_rng(seed)
    keys = ("balanced_accuracy", "auc", "accuracy", "sensitivity", "specificity")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    idx = np.arange(len(y))
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:      # a degenerate resample cannot be scored
            continue
        m = evaluate(y[take], prob[take])
        for k in keys:
            draws[k].append(m[k])
    return {
        k: {
            "lo95": round(float(np.quantile(v, 0.025)), 4),
            "hi95": round(float(np.quantile(v, 0.975)), 4),
        }
        for k, v in draws.items()
        if v
    }


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "sd": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features",
        type=Path,
        default=REGRESSION_ROOT / "embeddings/tapct_fev1fvc70/features.npz",
    )
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="label permutations for the null; each runs one full nested CV repeat",
    )
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="patient-level bootstrap resamples for the reported CI")
    ap.add_argument(
        "--only-features",
        default="",
        help="comma-separated feature_names to keep; use to score a single "
        "clinical index (e.g. laa950) under the identical protocol",
    )
    ap.add_argument("--label", default=None, help="name for this run in the output")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    X, y, pids = load_features(args.features)
    name = args.label or args.features.parent.name

    wanted = [f.strip() for f in args.only_features.split(",") if f.strip()]
    if wanted:
        stored = np.load(args.features, allow_pickle=True)
        if "feature_names" not in stored.files:
            raise SystemExit(f"{args.features} has no feature_names to select from")
        available = [str(n) for n in stored["feature_names"]]
        missing = [f for f in wanted if f not in available]
        if missing:
            raise SystemExit(f"unknown features {missing}; have {available}")
        X = X[:, [available.index(f) for f in wanted]]
        print(f"restricted to {len(wanted)} feature(s): {wanted}")
    print(f"[{name}] features {X.shape}  Abnormal={int(y.sum())} Normal={int((1-y).sum())}")
    print(f"protocol: {args.repeats} x (outer {args.outer_splits}-fold, "
          f"inner {args.inner_splits}-fold C search)\n")

    per_repeat: list[dict] = []
    all_chosen: list[float] = []
    first_prob: np.ndarray | None = None
    t0 = time.time()
    for r in range(args.repeats):
        seed = args.seed + r
        prob, chosen = nested_oof_probabilities(
            X, y,
            repeat_seed=seed,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
        )
        if first_prob is None:
            first_prob = prob
        m = evaluate(y, prob)
        m["repeat_seed"] = seed
        per_repeat.append(m)
        all_chosen.extend(chosen)
        print(f"  repeat {r + 1:2d} (seed {seed}): "
              f"bal_acc={m['balanced_accuracy']:.4f} auc={m['auc']:.4f} "
              f"sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}")
    elapsed = time.time() - t0
    print(f"\n{args.repeats} repeats in {elapsed:.1f}s "
          f"({elapsed / args.repeats:.1f}s each)\n")

    summary = {
        metric: summarize([m[metric] for m in per_repeat])
        for metric in ("balanced_accuracy", "auc", "accuracy", "macro_f1",
                       "sensitivity", "specificity")
    }
    print(f"=== {name}: every patient scored once per repeat (n={len(y)}) ===")
    for metric, s in summary.items():
        print(f"  {metric:20s} {s['mean']:.4f} ± {s['sd']:.4f}   "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print(f"  {'majority baseline':20s} {per_repeat[0]['majority_baseline_accuracy']:.4f} "
          "(accuracy of always predicting Normal)")
    uniq, cnt = np.unique(all_chosen, return_counts=True)
    print(f"  C chosen across {len(all_chosen)} outer folds: "
          + ", ".join(f"{c:g}x{n}" for c, n in zip(uniq, cnt)))

    ci = bootstrap_ci(y, first_prob, n=args.bootstrap, seed=args.seed)
    print(f"\n  patient-level bootstrap 95% CI (n={args.bootstrap}, repeat 1 predictions)")
    print("  — quote THIS, not the +- across repeats")
    for metric, b in ci.items():
        print(f"    {metric:20s} [{b['lo95']:.4f}, {b['hi95']:.4f}]")

    null_summary = None
    if args.permutations:
        print(f"\nrunning {args.permutations} label permutations "
              f"(~{elapsed / args.repeats * args.permutations / 60:.0f} min)...",
              flush=True)
        rng = np.random.default_rng(args.seed)
        null_scores = []
        for i in range(args.permutations):
            # Each permutation is scored against the labels that run actually saw.
            yp = rng.permutation(y)
            prob, _ = nested_oof_probabilities(
                X, yp,
                repeat_seed=args.seed,
                outer_splits=args.outer_splits,
                inner_splits=args.inner_splits,
            )
            null_scores.append(evaluate(yp, prob)["balanced_accuracy"])
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{args.permutations}", flush=True)
        null_arr = np.asarray(null_scores)
        observed = summary["balanced_accuracy"]["mean"]
        p_value = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))
        null_summary = {
            "n": int(len(null_arr)),
            "mean": round(float(null_arr.mean()), 4),
            "p95": round(float(np.quantile(null_arr, 0.95)), 4),
            "max": round(float(null_arr.max()), 4),
            "p_value": round(p_value, 4),
        }
        print(f"\npermutation null: mean={null_arr.mean():.4f} "
              f"p95={np.quantile(null_arr, 0.95):.4f} max={null_arr.max():.4f}")
        print(f"observed mean balanced accuracy = {observed:.4f}   p = {p_value:.4f}")
        print("VERDICT: " + ("above chance (p < 0.05)" if p_value < 0.05
                             else "NOT distinguishable from chance"))

    out = args.output or (
        REGRESSION_ROOT / "embeddings"
        / f"nested_cv_{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": name,
        "features": str(args.features),
        "n_patients": int(len(y)),
        "n_abnormal": int(y.sum()),
        "protocol": {
            "repeats": args.repeats,
            "outer_splits": args.outer_splits,
            "inner_splits": args.inner_splits,
            "base_seed": args.seed,
        },
        "summary": summary,
        "bootstrap_ci": ci,
        "per_repeat": per_repeat,
        "permutation_null": null_summary,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
