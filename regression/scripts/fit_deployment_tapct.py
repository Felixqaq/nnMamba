#!/usr/bin/env python3
"""Phase B for TAP-CT -- fit the deployment probe on every available patient.

The Phase A run fitted the probe on the 512 training patients only and never
saved a model object, so there was nothing to hand a new hospital even though
the probe outscores the 3D ensemble. This refits the same pipeline on all 712
and serialises it.

Nothing is selected here. C and the decision threshold are read from the Phase A
calibration, which chose both on training-fold out-of-fold predictions with the
frozen 200 excluded. The expected field performance of this model is therefore
the Phase A frozen-200 number, not anything this script can compute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--phase-a-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def load_labels(features_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Labels come from metadata.csv's source_group, never from an npz key.

    The extractor writes whatever target mode it ran under into keys literally
    named "angle_3class", so reading the npz by name silently inverts classes.
    """
    data = np.load(features_path, allow_pickle=True)
    features = np.asarray(data["features"], dtype=np.float64)
    pids = [str(v) for v in data["patient_ids"]]

    meta = features_path.parent / "metadata.csv"
    groups: dict[str, str] = {}
    with io.open(meta, encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            groups[row["patient_id"]] = row["source_group"]
    missing = [p for p in pids if p not in groups]
    if missing:
        raise SystemExit(f"no source_group for {len(missing)} patients: {missing[:5]}")
    labels = np.array([1 if groups[p] == "Abnormal" else 0 for p in pids], dtype=int)
    return features, labels, pids


def sha256_of(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    phase_a = json.loads(args.phase_a_json.read_text(encoding="utf-8-sig"))
    meta_a = phase_a.get("meta", {})
    c_value = meta_a.get("selected_C")
    threshold = meta_a.get("selected_threshold")
    if c_value is None or threshold is None:
        raise SystemExit(
            f"phase A json lacks selected_C/selected_threshold; keys={sorted(meta_a)}"
        )
    c_value, threshold = float(c_value), float(threshold)

    features, labels, pids = load_labels(args.features)
    index = {pid: i for i, pid in enumerate(pids)}
    all_ids = sorted(
        set(split["training_patient_ids"]) | set(split["validation_patient_ids"])
    )
    missing = sorted(set(all_ids) - set(index))
    if missing:
        raise SystemExit(f"features missing for {len(missing)} patients: {missing[:5]}")
    rows = [index[p] for p in all_ids]
    x, y = features[rows], labels[rows]

    print(f"deployment fit: {len(all_ids)} patients "
          f"(Abnormal={int(y.sum())} Normal={int((1 - y).sum())})", flush=True)
    print(f"inherited C={c_value}  threshold={threshold:.8f}", flush=True)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=c_value,
            max_iter=5000,
            class_weight="balanced",
            random_state=args.seed,
        )),
    ])
    model.fit(x, y)

    model_path = args.out / "tapct_deployment_probe.joblib"
    dump(model, model_path)

    # In-sample only, written as a sanity check that the fit converged. It is not
    # performance and must never be quoted as such.
    insample = model.predict_proba(x)[:, 1]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "TAP-CT deployment probe for a new hospital",
        "encoder": "fomofo/tap-ct-s-3d (frozen, never fine-tuned)",
        "trainable_part": "StandardScaler + LogisticRegression on 1152-d embeddings",
        "performance_note": (
            "Fitted on every available patient, so no honest score exists here. "
            "Quote the Phase A frozen-200 result: AUC 0.7089, balanced accuracy "
            "0.6441, sensitivity 0.6667, specificity 0.6214."
        ),
        "C": c_value,
        "decision_threshold": threshold,
        "selection_source": str(args.phase_a_json),
        "class_weight": "balanced",
        "n_training_patients": len(all_ids),
        "n_abnormal": int(y.sum()),
        "n_normal": int((1 - y).sum()),
        "training_patient_ids_sha256": sha256_of(all_ids),
        "model_file": model_path.name,
        "feature_source": str(args.features),
        "embedding_spec": {
            "model_id": "fomofo/tap-ct-s-3d",
            "resize_dim": 224,
            "depth_window": 12,
            "depth_stride": 6,
            "pooling": "mean_std_max",
            "dtype": "float16",
        },
        "insample_sanity": {
            "mean_prob_abnormal_true": float(insample[y == 1].mean()),
            "mean_prob_abnormal_false": float(insample[y == 0].mean()),
        },
        "label_rule": "Abnormal if FEV1/FVC < 70% (strict)",
    }
    (args.out / "tapct_deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print("TAPCT_DEPLOYMENT_OK", flush=True)


if __name__ == "__main__":
    main()
