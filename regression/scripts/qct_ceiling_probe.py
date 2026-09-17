#!/usr/bin/env python3
"""How much of FEV1/FVC < 70 is reachable from CT at all?

Every model tried so far lands between AUC 0.64 and 0.73 on the frozen 200,
across architectures that differ by orders of magnitude in capacity and
pretraining -- a from-scratch 3D CNN, a Mamba hybrid, and a transformer
pretrained on ~105k CTs. In every one of them the borderline band sits near 0.60
while the clear cases sit near 0.82. That pattern is what a task ceiling looks
like, not what a modelling gap looks like.

This prices the ceiling with the oldest tool available: quantitative CT. %LAA-950
-- the fraction of lung below -950 HU, computed here at native resolution -- is
the standard imaging readout for emphysema extent and the feature the clinical
literature regresses lung function on. If plain logistic regression on it lands
where the networks land, then the networks are rediscovering it and nothing is
being left on the table by the architecture. If it lands well below them, they
are learning something extra and the architecture work is worth continuing.

Selection follows the project protocol: the decision threshold comes from
training-fold out-of-fold predictions only. The frozen holdout is scored once and
reported separately, and the report says plainly how many times that holdout has
now been looked at.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLD = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density-summary", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--pft-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--margin", type=float, default=7.0,
                   help="half-width of the borderline band, in ratio points either "
                        "side of whichever cutoff the chosen label uses")
    p.add_argument("--label", choices=("fixed70", "gli"), default="fixed70",
                   help="fixed70 applies the GOLD fixed ratio; gli uses the "
                        "age/sex/height-adjusted GLI lower limit of normal, which "
                        "removes the fixed ratio's known bias against the elderly")
    return p.parse_args()


def label_and_distance(clinical: dict, pids: list[str], mode: str):
    """Return the binary label and each patient's distance from its own cutoff.

    The fixed ratio compares everyone to 70. GLI compares each patient to a
    personal lower limit computed from age, sex and height, so 'borderline' has
    to be measured against that limit rather than against a shared number --
    otherwise the two labels would not be scored on comparable bands.
    """
    y, dist = [], []
    for pid in pids:
        row = clinical[pid]
        ratio = float(row["FEV1FVC_pct"])
        if mode == "fixed70":
            y.append(int(ratio < THRESHOLD))
            dist.append(abs(ratio - THRESHOLD))
            continue
        flag = (row.get("Obstruction_GLI") or "").strip().upper()
        if flag not in ("Y", "N"):
            raise SystemExit(f"{pid}: Obstruction_GLI is {flag!r}, expected Y or N")
        lln = (row.get("FEV1FVC_LLN_GLI") or "").strip()
        if not lln:
            raise SystemExit(f"{pid}: no FEV1FVC_LLN_GLI, cannot place the GLI band")
        y.append(int(flag == "Y"))
        dist.append(abs(ratio - float(lln)))
    return np.asarray(y, dtype=int), np.asarray(dist, dtype=float)


def load_clinical(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = {(f or "").strip() for f in (reader.fieldnames or [])}
        for required in ("PatientID", "FEV1FVC_pct"):
            if required not in fields:
                raise SystemExit(f"{path}: no {required!r} column")
        for row in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            rows[row["PatientID"]] = row
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def load_qct(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "records" not in payload:
        raise SystemExit(f"{path}: no 'records' key")
    out: dict[str, dict] = {}
    for rec in payload["records"]:
        if "error" in rec:
            continue
        for key in ("laa950_full_resolution", "lung_voxels", "source_spacing"):
            if key not in rec:
                raise SystemExit(f"{path}: record {rec.get('patient_id')} lacks {key!r}")
        voxel_ml = float(np.prod(rec["source_spacing"])) / 1000.0
        out[rec["patient_id"]] = {
            "laa950": float(rec["laa950_full_resolution"]),
            "lung_ml": float(rec["lung_voxels"]) * voxel_ml,
        }
    if not out:
        raise SystemExit(f"{path}: parsed no usable records")
    return out


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


FEATURE_SETS = {
    "%LAA-950 單獨": ["laa950"],
    "+ 肺容積": ["laa950", "lung_ml"],
    "+ 年齡性別身高": ["laa950", "lung_ml", "age", "sex", "height"],
    "只有年齡性別身高 (無影像)": ["age", "sex", "height"],
}


def build_matrix(pids, qct, clinical, names):
    rows = []
    for pid in pids:
        q = qct[pid]
        c = clinical[pid]
        value = {
            "laa950": q["laa950"],
            "lung_ml": q["lung_ml"],
            "age": as_float(c.get("Age", "")),
            "sex": 1.0 if (c.get("Sex", "") or "").upper().startswith("M") else 0.0,
            "height": as_float(c.get("Height_cm", "")),
        }
        rows.append([value[n] for n in names])
    matrix = np.asarray(rows, dtype=float)
    # Impute a missing covariate with the column median rather than dropping the
    # patient, so every feature set is scored on exactly the same people.
    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        if np.isnan(column).any():
            column[np.isnan(column)] = np.nanmedian(column)
    return matrix


def choose_threshold(y, score):
    best, best_t = -1.0, 0.5
    for t in np.unique(score):
        s = balanced_accuracy_score(y, (score >= t).astype(int))
        if s > best:
            best, best_t = s, float(t)
    return best_t


def metrics(y, score, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
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
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def main() -> None:
    args = parse_args()
    qct = load_qct(args.density_summary)
    clinical = load_clinical(args.pft_csv)
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))

    train_ids = [p for p in split["training_patient_ids"] if p in qct and p in clinical]
    hold_ids = [p for p in split["validation_patient_ids"] if p in qct and p in clinical]
    dropped_tr = len(split["training_patient_ids"]) - len(train_ids)
    dropped_ho = len(split["validation_patient_ids"]) - len(hold_ids)

    yt, _ = label_and_distance(clinical, train_ids, args.label)
    yh, dist_h = label_and_distance(clinical, hold_ids, args.label)

    cutoff = "固定比值 70" if args.label == "fixed70" else "GLI 個人化下限 (LLN)"
    print(f"標籤: {args.label}  ({cutoff})")
    print(f"訓練 {len(train_ids)} 位(缺 QCT 而略過 {dropped_tr} 位),"
          f"凍結 {len(hold_ids)} 位(略過 {dropped_ho} 位)")
    print(f"訓練異常 {int(yt.sum())},凍結異常 {int(yh.sum())}")
    print(f"凍結中邊界(距離自己的切點 <{args.margin:g} 點)"
          f" {int((dist_h < args.margin).sum())} 位\n", flush=True)

    results = {}
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    print("{:28s}{:>11}{:>11}{:>21}{:>11}{:>11}".format(
        "特徵", "訓練OOF", "凍結AUC", "凍結95%CI", "平衡準確", "敏感度"))
    print("-" * 93)

    for name, feats in FEATURE_SETS.items():
        xt = build_matrix(train_ids, qct, clinical, feats)
        xh = build_matrix(hold_ids, qct, clinical, feats)

        best_c, best_auc, best_oof = None, -1.0, None
        for c in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            oof = np.zeros(len(yt))
            for tr, va in skf.split(xt, yt):
                model = Pipeline([("s", StandardScaler()),
                                  ("c", LogisticRegression(C=c, max_iter=5000,
                                                           class_weight="balanced",
                                                           random_state=args.seed))])
                model.fit(xt[tr], yt[tr])
                oof[va] = model.predict_proba(xt[va])[:, 1]
            auc = roc_auc_score(yt, oof)
            if auc > best_auc:
                best_auc, best_c, best_oof = auc, c, oof.copy()

        thr = choose_threshold(yt, best_oof)
        final = Pipeline([("s", StandardScaler()),
                          ("c", LogisticRegression(C=best_c, max_iter=5000,
                                                   class_weight="balanced",
                                                   random_state=args.seed))]).fit(xt, yt)
        sh = final.predict_proba(xh)[:, 1]
        m = metrics(yh, sh, (sh >= thr).astype(int))
        ci = boot_ci(yh, sh)

        bands = {}
        for label, sel in (("邊界 <7", dist_h < args.margin),
                           ("明確 >=7", dist_h >= args.margin)):
            if len(set(yh[sel].tolist())) >= 2:
                bands[label] = {"n": int(sel.sum()),
                                "auc": round(float(roc_auc_score(yh[sel], sh[sel])), 4)}

        results[name] = {"features": feats, "C": best_c,
                         "training_oof_auc": round(float(best_auc), 4),
                         "threshold_from_training_oof": round(float(thr), 6),
                         "holdout": m, "holdout_auc_95ci": ci, "by_band": bands}

        print("{:28s}{:>11.4f}{:>11.4f}{:>21}{:>11.4f}{:>11.4f}".format(
            name, best_auc, m["auc"], f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            m["balanced_accuracy"], m["sensitivity"]))

    print("\n分難度看 AUC(邊界才是決定天花板的地方)")
    print("  {:28s}{:>14}{:>14}".format("特徵", "邊界 <7", "明確 >=7"))
    for name, r in results.items():
        b = r["by_band"]
        print("  {:28s}{:>14}{:>14}".format(
            name,
            f"{b['邊界 <7']['auc']:.4f} (n={b['邊界 <7']['n']})" if "邊界 <7" in b else "-",
            f"{b['明確 >=7']['auc']:.4f} (n={b['明確 >=7']['n']})" if "明確 >=7" in b else "-"))

    results["_meta"] = {
        "label": args.label,
        "cutoff": cutoff,
        "n_train_abnormal": int(yt.sum()),
        "n_holdout_abnormal": int(yh.sum()),
        "borderline_margin_points": args.margin,
        "selection": "C and threshold chosen on training-fold OOF only",
        "holdout_used_for_selection": False,
        "laa950_source": "computed at native CT resolution, before any resize",
        "caution": (
            "This is one more evaluation on a frozen holdout that has now been "
            "scored many times across models. Treat the ranking as indicative; "
            "the training OOF column is the one that is safe to select on."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
