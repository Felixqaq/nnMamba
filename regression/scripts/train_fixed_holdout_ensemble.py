#!/usr/bin/env python3
"""Train five Mamba members on a frozen train set and score one fixed holdout.

There is deliberately no validation loader in the optimization loop, no early
stopping and no checkpoint selection. The fixed epoch budget must be chosen from
earlier experiments before this script is run. Every non-holdout patient is used
for training; the holdout is evaluated once after all members finish.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402

# The epoch budget the mamba runs inherited: the median best epoch across the
# prior 25 cross-validation models. A run that overrides it is not entitled to
# that justification, so the metadata records which of the two it actually used.
DEFAULT_EPOCHS = 45


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "datasets/generated/doctor_validation_manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_fixed_split(helper: LoaderHelper, split: dict) -> tuple[list[int], list[int]]:
    index_by_id = {pid: index for index, pid in enumerate(helper.patient_ids)}
    if len(index_by_id) != len(helper.patient_ids):
        raise SystemExit("dataset contains duplicate patient IDs")
    train_ids = list(split["training_patient_ids"])
    holdout_ids = list(split["validation_patient_ids"])
    expected = set(train_ids) | set(holdout_ids)
    if set(train_ids) & set(holdout_ids):
        raise SystemExit("train and holdout IDs overlap")
    missing = sorted(expected - set(index_by_id))
    extra = sorted(set(index_by_id) - expected)
    if missing or extra:
        raise SystemExit(f"split/dataset mismatch: missing={missing} extra={extra}")
    train_indices = [index_by_id[pid] for pid in train_ids]
    holdout_indices = [index_by_id[pid] for pid in holdout_ids]
    helper.fold_indices = [(train_indices, holdout_indices)]
    helper.k_folds = 1
    return train_indices, holdout_indices


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def train_member(
    config: Config,
    helper: LoaderHelper,
    seed: int,
    epochs: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
    train_loader = helper.get_train_dl(0, shuffle=True)
    optimizer = build_optimizer(
        model,
        float(config.training.learning_rate),
        float(config.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    if not getattr(augmentation, "defer_to_device", False):
        augmentation = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in tqdm(train_loader, leave=False, desc=f"seed {seed} epoch {epoch}/{epochs}"):
            ct = batch["ct"].to(device, non_blocking=True)
            if augmentation is not None:
                ct = augmentation.apply_batch(ct, batch.get("augment"))
            target = batch["label"].to(device, non_blocking=True).long().view(-1)
            optimizer.zero_grad(set_to_none=True)

            def compute_loss(use_amp: bool) -> torch.Tensor:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    return loss_fn(model(ct), target)

            use_amp = bool(config.training.amp and device.type == "cuda")
            loss = compute_loss(use_amp)
            if not torch.isfinite(loss) and use_amp:
                print("Non-finite loss under bf16; retrying this batch in fp32", flush=True)
                loss = compute_loss(False)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss after fp32 retry")
            scaler.scale(loss).backward()
            clip = float(config.training.clip_grad_norm)
            if clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            batches += 1
        scheduler.step()
        print(
            f"seed={seed} epoch={epoch}/{epochs} loss={total_loss / max(batches, 1):.6f} "
            f"lr={scheduler.get_last_lr()[0]:.8g}",
            flush=True,
        )
    model.eval()
    return model


def predict_member(
    model: nn.Module,
    helper: LoaderHelper,
    device: torch.device,
    use_amp: bool,
) -> dict[str, dict]:
    class_names = helper.get_class_names()
    abnormal_index = class_names.index("Abnormal")
    output: dict[str, dict] = {}
    for batch in helper.get_test_dl(0, shuffle=False):
        ct = batch["ct"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            logits = model(ct)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for index, pid in enumerate(batch["patient_id"]):
            output[str(pid)] = {
                "pred_label": class_names[int(predictions[index])],
                "prob_abnormal": float(probabilities[index, abnormal_index]),
            }
    return output


def metrics_for(patients: dict[str, dict]) -> dict:
    labels = np.array([1 if row["true_label"] == "Abnormal" else 0 for row in patients.values()])
    preds = np.array([1 if row["pred_label"] == "Abnormal" else 0 for row in patients.values()])
    probs = np.array([row["mean_prob_abnormal"] for row in patients.values()])
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, preds)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 5),
        "auc_mean_member_probability": round(float(roc_auc_score(labels, probs)), 5),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> None:
    args = parse_args()
    configure_torch_runtime()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = Config.from_yaml(str(args.config))
    config = replace(
        config,
        data=replace(
            config.data,
            source_dir=str(args.source_dir),
            manifest=str(args.manifest),
        ),
    )
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    helper = LoaderHelper(config)
    train_indices, holdout_indices = set_fixed_split(helper, split)
    class_names = helper.get_class_names()
    print(
        f"fixed split: train={len(train_indices)} holdout={len(holdout_indices)} "
        f"classes={class_names}; NO validation/early stopping",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    member_predictions: list[dict[str, dict]] = []
    seeds = [args.base_seed + index for index in range(args.members)]
    for member_index, seed in enumerate(seeds, start=1):
        checkpoint = args.out / f"member_{member_index:02d}_seed{seed}.pth"
        predictions_path = args.out / f"member_{member_index:02d}_predictions.json"
        if checkpoint.exists() and predictions_path.exists():
            print(f"member {member_index}/{args.members}: reusing completed seed={seed}")
            member_predictions.append(json.loads(predictions_path.read_text(encoding="utf-8")))
            continue
        print(f"member {member_index}/{args.members}: training seed={seed}", flush=True)
        model = train_member(config, helper, seed, args.epochs, device)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "epochs": args.epochs,
                "training_patient_ids": split["training_patient_ids"],
                "holdout_patient_ids": split["validation_patient_ids"],
            },
            checkpoint,
        )
        predictions = predict_member(model, helper, device, bool(config.training.amp))
        predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        member_predictions.append(predictions)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    true_by_id = {
        pid: helper.class_names[int(helper.targets[index])]
        for index, pid in enumerate(helper.patient_ids)
    }
    patients: dict[str, dict] = {}
    for pid in split["validation_patient_ids"]:
        members = [prediction[pid] for prediction in member_predictions]
        votes = sum(row["pred_label"] == "Abnormal" for row in members)
        patients[pid] = {
            "true_label": true_by_id[pid],
            "votes_for_abnormal": votes,
            "vote_text": f"{votes}/{args.members}",
            "pred_label": "Abnormal" if votes >= args.members // 2 + 1 else "Normal",
            "mean_prob_abnormal": float(np.mean([row["prob_abnormal"] for row in members])),
            "member_predictions": [row["pred_label"] for row in members],
            "member_prob_abnormal": [row["prob_abnormal"] for row in members],
        }
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config),
            "split_json": str(args.split_json),
            "n_training": len(train_indices),
            "n_holdout": len(holdout_indices),
            "members": args.members,
            "epochs": args.epochs,
            "seeds": seeds,
            "validation_used_during_training": False,
            # Only the default budget carries the CV justification. Anything else
            # was chosen by hand, and saying so keeps a run from claiming a
            # provenance it does not have.
            "epoch_basis": (
                "median best epoch (45) from the prior 25 CV models"
                if args.epochs == DEFAULT_EPOCHS
                else f"{args.epochs} passed on the command line, not selected on CV"
            ),
        },
        "metrics": metrics_for(patients),
        "patients": patients,
    }
    output = args.out / "mamba5_holdout_predictions.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
