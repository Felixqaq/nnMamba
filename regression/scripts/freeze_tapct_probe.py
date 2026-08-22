#!/usr/bin/env python3
"""Fit the final TAP-CT linear probe on the whole cohort and export it for copd-ct-app.

repeated_nested_cv_probe.py measures how well this probe generalises; it never keeps
a fitted model. This script produces the deployable artefact: the same pipeline
(StandardScaler + balanced LogisticRegression, C chosen by the identical inner CV)
refit on every labelled patient, exported as four plain arrays.

Plain arrays rather than a joblib pickle on purpose — the app then needs no sklearn
at all (P(Abnormal) = sigmoid(w . (x - mu)/sigma + b)), and the offline Windows
bundle does not have to track sklearn version compatibility for a linear model.

Like the all-data ensemble release, the exported model has NO measured performance of
its own: every patient it saw is a training patient. The nested-CV numbers copied into
metrics.json are a reference for the *protocol*, not a score for this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linear_probe_heldout import build_search, load_features  # noqa: E402

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REGRESSION_ROOT / "embeddings" / "tapct_fev1fvc70" / "features.npz"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--nested-cv",
        type=Path,
        default=None,
        help="nested_cv_*.json whose summary is copied into metrics.json as the "
        "reference score for this protocol",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    X, y, pids = load_features(args.features)
    print(f"cohort {X.shape}  Abnormal={int(y.sum())}  Normal={int((1 - y).sum())}")

    search = build_search(args.seed, args.inner_splits)
    search.fit(X, y)
    C = float(search.best_params_["clf__C"])
    print(f"C={C:g} chosen by inner {args.inner_splits}-fold CV "
          f"(balanced accuracy {search.best_score_:.4f})")

    pipe = search.best_estimator_
    scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    mu = scaler.mean_.astype(np.float64)
    sigma = scaler.scale_.astype(np.float64)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])

    # class_index 1 is Abnormal in load_features; the app reports P(Abnormal), so a
    # flipped class order here would inverse every prediction with no error raised.
    assert list(clf.classes_) == [0, 1], f"unexpected class order {clf.classes_}"

    # The exported arrays must reproduce sklearn exactly, or the app is running a
    # different model from the one that was just validated.
    mine = 1.0 / (1.0 + np.exp(-(((X - mu) / sigma) @ w + b)))
    ref = pipe.predict_proba(X)[:, 1]
    drift = float(np.abs(mine - ref).max())
    print(f"numpy vs sklearn on all {len(y)} patients: max |dP| = {drift:.3e}")
    assert drift < 1e-9, "exported arrays do not reproduce the fitted pipeline"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "probe.npz",
        mean=mu.astype(np.float32),
        scale=sigma.astype(np.float32),
        coef=w.astype(np.float32),
        intercept=np.float32(b),
        class_names=np.array(["Normal", "Abnormal"]),  # index 0, index 1
    )

    reference = None
    if args.nested_cv and args.nested_cv.exists():
        nested = json.loads(args.nested_cv.read_text())
        reference = {
            "auc": nested["summary"]["auc"]["mean"],
            "balanced_accuracy": nested["summary"]["balanced_accuracy"]["mean"],
            "sensitivity": nested["summary"]["sensitivity"]["mean"],
            "specificity": nested["summary"]["specificity"]["mean"],
            "bootstrap_ci": nested.get("bootstrap_ci"),
            "source": "{}x repeated nested CV, n={}, {}".format(
                nested["protocol"]["repeats"], nested["n_patients"], nested["label"]
            ),
        }

    (args.output_dir / "metrics.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "encoder": "fomofo/tap-ct-s-3d",
        "features": str(args.features),
        "n_training_cases": int(len(y)),
        "n_abnormal": int(y.sum()),
        "label_rule": "Abnormal = FEV1/FVC < 70% (GOLD fixed ratio)",
        "C": C,
        "held_out": False,
        "note": "Fitted on 100% of the labelled cohort: no held-out set. This release "
                "has NO measured performance of its own. Quote the nested-CV reference "
                "below, labelled as a reference.",
        "nested_cv_reference": reference,
    }, indent=2), encoding="utf-8")
    print(f"wrote probe.npz and metrics.json into {args.output_dir}")


if __name__ == "__main__":
    main()
