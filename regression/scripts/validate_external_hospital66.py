#!/usr/bin/env python3
"""Score trained ratio regressors on the hospital_66 cohort, which no run has seen.

The frozen 200 has now been scored more than twenty times, so the figure it
produces carries the optimism of every configuration that was compared on it.
The hospital_66 patients are a separate cohort with measured spirometry that
never entered any split: not trained on, not evaluated on, and not consulted
when any hyperparameter was chosen. Running the existing checkpoints over them
costs no training at all and gives the first genuinely external number.

Two properties of this cohort must be read alongside the result. It is close to
balanced -- 32 obstructed against 33 normal, where the frozen holdout is 60/140 --
and only 9% of it sits within 7 points of the cutoff against 45% there. It is
therefore an easier cohort, and a higher AUC here does not by itself mean a
better model. With 65 patients the confidence interval is also about twice as
wide as on the holdout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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
from models import build_model  # noqa: E402

FIXED_CUTOFF = 70.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--source-dir", type=Path, required=True,
                   help="dataset root containing the cohort, e.g. the full 778 build")
    p.add_argument("--build-summary", type=Path, required=True,
                   help="build_summary.json, which carries fev1_fvc_pct and batch")
    p.add_argument("--exclude-split", type=Path, required=True,
                   help="split.json whose patients must be left out, so nothing "
                        "any model trained on can leak into this evaluation")
    p.add_argument("--batch", default="hospital66")
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "datasets/generated/external_manifest.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--margin", type=float, default=7.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_external(summary_path: Path, batch: str, excluded: set[str]) -> dict[str, float]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if "records" not in payload:
        raise SystemExit(f"{summary_path}: no 'records' key")
    out: dict[str, float] = {}
    skipped_used = 0
    for rec in payload["records"]:
        if rec.get("batch") != batch or not rec.get("ok"):
            continue
        pid = rec["patient_id"]
        if pid in excluded:
            skipped_used += 1
            continue
        value = rec.get("fev1_fvc_pct")
        if value in (None, ""):
            raise SystemExit(f"{pid}: batch {batch} record has no fev1_fvc_pct")
        if rec.get("pft_source") != "measured":
            raise SystemExit(f"{pid}: pft_source is {rec.get('pft_source')!r}, "
                             "only measured spirometry may be scored here")
        out[pid] = float(value)
    if not out:
        raise SystemExit(f"{summary_path}: no usable {batch} patients outside the split")
    print(f"{batch}: {len(out)} 位可用,{skipped_used} 位因為在 split 裡而排除", flush=True)
    return out


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


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    split = json.loads(args.exclude_split.read_text(encoding="utf-8-sig"))
    excluded = set(split["training_patient_ids"]) | set(split["validation_patient_ids"])
    ratios = load_external(args.build_summary, args.batch, excluded)

    config = Config.from_yaml(str(args.config))
    config = replace(config, data=replace(config.data,
                                          source_dir=str(args.source_dir),
                                          manifest=str(args.manifest),
                                          cache_data=False))
    configure_torch_runtime()
    helper = LoaderHelper(config)

    index = {pid: i for i, pid in enumerate(helper.patient_ids)}
    missing = sorted(p for p in ratios if p not in index)
    if missing:
        raise SystemExit(f"{len(missing)} external patients absent from "
                         f"{args.source_dir}: {missing[:5]}")
    pids = sorted(ratios)
    external_idx = [index[p] for p in pids]
    # One fold whose "training" half is unused; only the second half is read.
    helper.fold_indices = [(external_idx, external_idx)]
    helper.k_folds = 1

    loader = helper._build_loader(
        external_idx,
        batch_size=int(helper.val_batch_size or helper.batch_size),
        shuffle=False, drop_last=False, augmentation=None)

    truth = np.array([ratios[p] for p in pids], dtype=float)
    y = (truth < FIXED_CUTOFF).astype(int)
    dist = np.abs(truth - FIXED_CUTOFF)
    use_amp = bool(config.training.amp)

    per_model = {}
    stacked = []
    for path in args.checkpoints:
        blob = torch.load(path, map_location="cpu")
        model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
        model.load_state_dict(blob["state_dict"])
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
            raise SystemExit(f"{path.name}: no prediction for {len(gap)}: {gap[:5]}")

        arr = np.array([preds[p] for p in pids])
        stacked.append(arr)
        block = metrics(y, -arr, (arr < FIXED_CUTOFF).astype(int))
        block["auc_95ci"] = boot_ci(y, -arr)
        block["ratio_mae"] = round(float(np.abs(arr - truth).mean()), 4)
        block["ratio_r"] = round(float(np.corrcoef(arr, truth)[0, 1]), 4)
        block["by_band"] = {}
        for name, sel in (("borderline", dist < args.margin),
                          ("clear", dist >= args.margin)):
            if len(set(y[sel].tolist())) >= 2:
                block["by_band"][name] = {
                    "n": int(sel.sum()),
                    "auc": round(float(roc_auc_score(y[sel], -arr[sel])), 4)}
        block["checkpoint"] = str(path)
        block["training_seed"] = blob.get("seed")
        per_model[path.parent.name or path.stem] = block
        print("  {:22s} auc {:.4f}  bal {:.4f}  sen {:.4f}  MAE {:.3f}".format(
            path.parent.name or path.stem, block["auc"], block["balanced_accuracy"],
            block["sensitivity"], block["ratio_mae"]), flush=True)

    mean_pred = np.mean(np.stack(stacked), axis=0)
    ens = metrics(y, -mean_pred, (mean_pred < FIXED_CUTOFF).astype(int))
    ens["auc_95ci"] = boot_ci(y, -mean_pred)
    ens["ratio_mae"] = round(float(np.abs(mean_pred - truth).mean()), 4)

    results = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cohort": args.batch,
            "n": len(pids),
            "config": str(args.config),
            "checkpoints": [str(p) for p in args.checkpoints],
            "excluded_split": str(args.exclude_split),
            "seen_during_training_or_selection": False,
            "caution": ("this cohort is near balanced and only 9% borderline, so it "
                        "is easier than the frozen 200; a higher AUC here is not by "
                        "itself evidence of a better model"),
        },
        "cohort_summary": {
            "n_abnormal": int(y.sum()), "n_normal": int((1 - y).sum()),
            "ratio_min": float(truth.min()), "ratio_median": float(np.median(truth)),
            "ratio_max": float(truth.max()),
            "n_borderline": int((dist < args.margin).sum()),
        },
        "per_model": per_model,
        "mean_of_models": ens,
        "patients": {p: {"predicted_ratio": round(float(mean_pred[i]), 3),
                         "true_ratio": truth[i]} for i, p in enumerate(pids)},
    }
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    aucs = [b["auc"] for b in per_model.values()]
    print(f"\n外部世代 {args.batch},n={len(pids)}(異常 {int(y.sum())} / 正常 {int((1-y).sum())})")
    print(f"  單模型 AUC 平均 {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  三模型平均預測 AUC {ens['auc']:.4f}  95% CI {ens['auc_95ci']}")
    print(f"  平衡準確率 {ens['balanced_accuracy']:.4f}  敏感度 {ens['sensitivity']:.4f}"
          f"  特異度 {ens['specificity']:.4f}")
    print(f"  比值 MAE {ens['ratio_mae']:.3f}")
    print("\n對照:凍結 200(已被評分二十次以上)fixed70 AUC 0.7487 ± 0.0120")
    print("  但那份有 45% 邊界病例,這份只有 9%,難度不同,不能直接比大小。")
    print(f"\n總耗時 {(time.time()-started)/60:.1f} 分  ->  {args.out}")


if __name__ == "__main__":
    main()
