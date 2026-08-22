#!/usr/bin/env python3
"""Paired comparison of feature sets on identical patients and identical folds.

Comparing two models by asking whether their bootstrap CIs overlap is the wrong
test and badly under-powered: both are scored on the same 182 patients, so most
of the uncertainty is shared and cancels. The paired bootstrap here resamples
patients once per draw and recomputes *both* models on that same resample, so the
distribution is of the difference itself.

Both models also see the identical outer folds, so no part of the gap can come
from one having had a luckier partition.
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_heldout import build_search, load_features  # noqa: E402

REGRESSION_ROOT = Path(__file__).resolve().parents[1]


def oof_probabilities(X, y, *, seed: int, outer: int, inner: int) -> np.ndarray:
    """Out-of-fold P(abnormal); fold assignment depends only on (y, seed)."""
    prob = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(outer, shuffle=True, random_state=seed)
    for train_i, test_i in splitter.split(X, y):
        search = build_search(seed, inner)
        search.fit(X[train_i], y[train_i])
        prob[test_i] = search.best_estimator_.predict_proba(X[test_i])[:, 1]
    return prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="named feature set, repeatable; compared pairwise",
    )
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    sets: dict[str, Path] = {}
    for item in args.set:
        label, _, path = item.partition("=")
        sets[label] = Path(path)

    # Align every set to a common patient order before anything else.
    loaded = {label: load_features(path) for label, path in sets.items()}
    common = sorted(set.intersection(*[set(v[2]) for v in loaded.values()]))
    print(f"feature sets: {list(sets)}   common patients: {len(common)}")

    order: dict[str, np.ndarray] = {}
    y_ref = None
    for label, (X, y, pids) in loaded.items():
        pos = {p: i for i, p in enumerate(pids)}
        take = [pos[p] for p in common]
        order[label] = X[take]
        y_here = y[take]
        if y_ref is None:
            y_ref = y_here
        elif not np.array_equal(y_ref, y_here):
            raise SystemExit(f"{label} disagrees on labels for the shared patients")
    y = y_ref
    print(f"Abnormal={int(y.sum())} Normal={int((1 - y).sum())}\n")

    # Average the out-of-fold probabilities over repeats so the comparison is not
    # hostage to one partition; the same seeds are used for every feature set.
    probs: dict[str, np.ndarray] = {}
    for label, X in order.items():
        acc = np.zeros(len(y))
        for r in range(args.repeats):
            acc += oof_probabilities(
                X, y,
                seed=args.seed + r,
                outer=args.outer_splits,
                inner=args.inner_splits,
            )
        probs[label] = acc / args.repeats
        print(f"  {label:16s} AUC over averaged OOF = "
              f"{roc_auc_score(y, probs[label]):.4f}")

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    resamples = []
    for _ in range(args.bootstrap):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        resamples.append(take)

    results = []
    print(f"\npaired bootstrap ({len(resamples)} resamples)")
    for a, b in itertools.combinations(sets, 2):
        diffs = np.array([
            roc_auc_score(y[t], probs[a][t]) - roc_auc_score(y[t], probs[b][t])
            for t in resamples
        ])
        observed = roc_auc_score(y, probs[a]) - roc_auc_score(y, probs[b])
        # Two-sided p: how often the difference crosses zero.
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        p = float(min(1.0, max(p, 1.0 / (len(diffs) + 1))))
        row = {
            "a": a,
            "b": b,
            "delta_auc": round(float(observed), 4),
            "ci95": [round(float(np.quantile(diffs, 0.025)), 4),
                     round(float(np.quantile(diffs, 0.975)), 4)],
            "p_value": round(p, 4),
        }
        results.append(row)
        star = "  *" if p < 0.05 else ""
        print(f"  {a:16s} vs {b:16s} ΔAUC={observed:+.4f} "
              f"95% CI [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]  "
              f"p={p:.4f}{star}")

    out = args.output or (
        REGRESSION_ROOT / "embeddings"
        / f"paired_comparison_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    out.write_text(json.dumps({
        "n_patients": len(common),
        "n_abnormal": int(y.sum()),
        "repeats": args.repeats,
        "bootstrap": len(resamples),
        "auc": {k: round(float(roc_auc_score(y, v)), 4) for k, v in probs.items()},
        "pairwise": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
