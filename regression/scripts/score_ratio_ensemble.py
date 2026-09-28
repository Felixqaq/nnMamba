#!/usr/bin/env python3
"""Score an ensemble of two-stage ratio regressors, both ways it can be read.

A ratio ensemble admits two decision rules and they do not always agree:

  mean ratio     average the predicted FEV1/FVC across members, then compare the
                 average with the cutoff. One number per patient, and it degrades
                 smoothly -- a patient at 69.4 is visibly near the line.
  majority vote  compare each member with the cutoff, then take the majority.
                 This is what the earlier classification ensemble did.

Both are reported here because the classification ensemble's displayed
`mean_prob_abnormal` was not what its decision used: members saturated to 0.0002
or 0.9998, so a patient could be called Abnormal on a 3/5 vote while the mean
probability read 0.19. A ratio model does not saturate the same way, but the two
rules can still disagree, and a clinician should be shown the rule that decided.

Nothing here fits anything. The cutoff is the clinical 70, or each patient's GLI
lower limit; no threshold is tuned on the patients being scored.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from core.checkpoints import load_model_weights  # noqa: E402
from models import build_model  # noqa: E402

FIXED_CUTOFF = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--which", choices=("validation", "training"), default="validation",
                   help="which half of the split to score")
    p.add_argument("--pft-csv", type=Path, required=True)
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "datasets/generated/ensemble_score_manifest.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--margin", type=float, default=7.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


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


def metrics(y, score, pred):
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
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def bands(y, score, dist, margin):
    out = {}
    for name, sel in (("borderline", dist < margin), ("clear", dist >= margin)):
        if len(set(y[sel].tolist())) >= 2:
            out[name] = {"n": int(sel.sum()),
                         "auc": round(float(roc_auc_score(y[sel], score[sel])), 4)}
    return out


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    clinical = load_clinical(args.pft_csv)
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    key = "validation_patient_ids" if args.which == "validation" else "training_patient_ids"
    pids = [p for p in split[key]]
    missing = [p for p in pids if p not in clinical]
    if missing:
        raise SystemExit(f"{len(missing)} scored patients absent from {args.pft_csv}: "
                         f"{missing[:5]}")

    config = Config.from_yaml(str(args.config))
    config = replace(config, data=replace(config.data,
                                          source_dir=str(args.source_dir),
                                          manifest=str(args.manifest),
                                          cache_data=False))
    configure_torch_runtime()
    helper = LoaderHelper(config)
    index = {pid: i for i, pid in enumerate(helper.patient_ids)}
    absent = [p for p in pids if p not in index]
    if absent:
        raise SystemExit(f"{len(absent)} scored patients absent from {args.source_dir}: "
                         f"{absent[:5]}")
    idx = [index[p] for p in pids]
    loader = helper._build_loader(
        idx, batch_size=int(helper.val_batch_size or helper.batch_size),
        shuffle=False, drop_last=False, augmentation=None)

    truth = np.array([float(clinical[p]["FEV1FVC_pct"]) for p in pids])
    gli = []
    for p in pids:
        value = (clinical[p].get("FEV1FVC_LLN_GLI") or "").strip()
        gli.append(float(value) if value else np.nan)
    gli = np.array(gli)
    use_amp = bool(config.training.amp)

    per_member = {}
    member_ratio = []
    for path in args.checkpoints:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
        load_model_weights(model, blob["state_dict"], str(path))
        model.eval()
        mean, std = float(blob["ratio_mean"]), float(blob["ratio_std"])
        preds: dict[str, float] = {}
        for batch in loader:
            ct = batch["ct"].to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16,
                    enabled=bool(use_amp and device.type == "cuda")):
                out = model(ct).float().view(-1)
            for pid, value in zip(batch["patient_id"], out.cpu().numpy() * std + mean):
                preds[pid] = float(value)
        gap = [p for p in pids if p not in preds]
        if gap:
            raise SystemExit(f"{path}: no prediction for {len(gap)}: {gap[:5]}")
        arr = np.array([preds[p] for p in pids])
        member_ratio.append(arr)
        y = (truth < FIXED_CUTOFF).astype(int)
        per_member[str(path)] = {
            "seed": blob.get("seed"),
            "fixed70": metrics(y, -arr, (arr < FIXED_CUTOFF).astype(int)),
            "ratio_mae": round(float(np.abs(arr - truth).mean()), 4),
        }
        print("  member seed {:>4}  auc {:.4f}  MAE {:.3f}".format(
            str(blob.get("seed")), per_member[str(path)]["fixed70"]["auc"],
            per_member[str(path)]["ratio_mae"]), flush=True)

    stack = np.stack(member_ratio)                     # (members, patients)
    mean_ratio = stack.mean(axis=0)
    n_members = stack.shape[0]
    need = n_members // 2 + 1

    results = {"meta": {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(args.config),
        "checkpoints": [str(p) for p in args.checkpoints],
        "n_members": n_members,
        "vote_rule": f"Abnormal if >= {need} of {n_members} members put the ratio "
                     f"below the cutoff",
        "split_json": str(args.split_json),
        "scored_half": args.which,
        "nothing_fitted_here": True,
    }, "per_member": per_member}

    for tag, cutoff in (("fixed70", np.full(len(pids), FIXED_CUTOFF)), ("gli", gli)):
        if np.isnan(cutoff).any():
            results[tag] = {"skipped": "some patients have no GLI lower limit"}
            continue
        y = (truth < cutoff).astype(int)
        dist = np.abs(truth - cutoff)
        votes = (stack < cutoff[None, :]).sum(axis=0)

        block = {
            "mean_ratio": metrics(y, -mean_ratio, (mean_ratio < cutoff).astype(int)),
            "majority_vote": metrics(y, votes.astype(float), (votes >= need).astype(int)),
        }
        block["mean_ratio"]["auc_95ci"] = boot_ci(y, -mean_ratio)
        block["mean_ratio"]["by_band"] = bands(y, -mean_ratio, dist, args.margin)
        block["majority_vote"]["by_band"] = bands(y, votes.astype(float), dist, args.margin)
        disagree = int(((mean_ratio < cutoff) != (votes >= need)).sum())
        block["rules_disagree_on"] = disagree
        block["ratio_mae"] = round(float(np.abs(mean_ratio - truth).mean()), 4)
        results[tag] = block

    results["patients"] = {
        p: {"true_ratio": float(truth[i]),
            "mean_predicted_ratio": round(float(mean_ratio[i]), 2),
            "member_ratios": [round(float(v), 2) for v in stack[:, i]],
            "votes_for_abnormal_at_70": int((stack[:, i] < FIXED_CUTOFF).sum()),
            "vote_text": f"{int((stack[:, i] < FIXED_CUTOFF).sum())}/{n_members}"}
        for i, p in enumerate(pids)}
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{n_members} 個成員,{args.which} 共 {len(pids)} 人")
    for tag, label in (("fixed70", "固定比值 70"), ("gli", "GLI 個人化下限")):
        block = results[tag]
        if "skipped" in block:
            print(f"\n{label}: {block['skipped']}")
            continue
        print(f"\n{label}")
        print("  {:14s}{:>9}{:>10}{:>9}{:>9}{:>11}".format(
            "讀法", "AUC", "balacc", "敏感度", "特異度", "邊界AUC"))
        for rule, name in (("mean_ratio", "平均比值"), ("majority_vote", "多數投票")):
            m = block[rule]
            print("  {:14s}{:>9.4f}{:>10.4f}{:>9.4f}{:>9.4f}{:>11.4f}".format(
                name, m["auc"], m["balanced_accuracy"], m["sensitivity"],
                m["specificity"], m["by_band"].get("borderline", {}).get("auc", float("nan"))))
        print(f"  兩種讀法判定不同的病人: {block['rules_disagree_on']} 位")
    print(f"\n比值 MAE {results['fixed70']['ratio_mae']}")
    print(f"總耗時 {(time.time()-started)/60:.1f} 分  ->  {args.out}")


if __name__ == "__main__":
    main()
