#!/usr/bin/env python3
"""Train a hard-example-aware Mamba ensemble without tuning on the holdout.

The frozen holdout is never used for epoch selection or hard-example mining.
First, five-fold out-of-fold (OOF) training is performed only within the
training cohort. A single epoch is selected from pooled OOF balanced accuracy.
Training-patient sampling weights then combine clinical boundary proximity,
OOF error and OOF uncertainty. Five final members are trained on all training
patients and the frozen holdout is read only after every final checkpoint exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402


class HardAwareBalancedViewSampler(Sampler[int]):
    """Balance classes, retain broad coverage, and add weighted hard repeats."""

    def __init__(
        self,
        targets: np.ndarray,
        weights: np.ndarray,
        views_per_sample: int,
        seed: int,
        hard_extra_fraction: float,
    ) -> None:
        targets = np.asarray(targets, dtype=int)
        weights = np.asarray(weights, dtype=float)
        if targets.ndim != 1 or weights.ndim != 1 or len(targets) != len(weights):
            raise ValueError("targets and weights must be aligned 1D arrays")
        if len(targets) == 0 or np.any(~np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("sampler requires non-empty, finite, positive weights")
        self.targets = targets
        self.weights = weights
        self.views_per_sample = int(views_per_sample)
        self.rng = np.random.default_rng(int(seed))
        self.class_positions = {
            int(class_index): np.flatnonzero(targets == class_index).astype(int)
            for class_index in np.unique(targets)
        }
        class_counts = [len(positions) for positions in self.class_positions.values()]
        self.samples_per_class = int(min(class_counts))
        self.extra_per_class = int(round(self.samples_per_class * hard_extra_fraction))
        self.num_samples = (
            (self.samples_per_class + self.extra_per_class)
            * len(self.class_positions)
            * self.views_per_sample
        )

    def __iter__(self):
        sampled_view_positions: list[int] = []
        for positions in self.class_positions.values():
            class_weights = self.weights[positions]
            probabilities = class_weights / class_weights.sum()
            core = self.rng.choice(
                positions,
                size=self.samples_per_class,
                replace=False,
                p=probabilities,
            )
            if self.extra_per_class:
                extras = self.rng.choice(
                    positions,
                    size=self.extra_per_class,
                    replace=True,
                    p=probabilities,
                )
                selected = np.concatenate([core, extras])
            else:
                selected = core
            for base_position in selected:
                start = int(base_position) * self.views_per_sample
                sampled_view_positions.extend(
                    range(start, start + self.views_per_sample)
                )
        self.rng.shuffle(sampled_view_positions)
        return iter(sampled_view_positions)

    def __len__(self) -> int:
        return int(self.num_samples)


class FrozenHoldoutGuard(Dataset):
    """Cache only training CTs and reject holdout reads until explicitly unlocked."""

    def __init__(
        self,
        dataset: Dataset,
        training_indices: list[int],
        holdout_indices: list[int],
    ) -> None:
        self.dataset = dataset
        self.training_indices = set(int(index) for index in training_indices)
        self.holdout_indices = set(int(index) for index in holdout_indices)
        if self.training_indices & self.holdout_indices:
            raise ValueError("training and holdout cache indices overlap")
        if self.training_indices | self.holdout_indices != set(range(len(dataset))):
            raise ValueError("guard indices must cover the full dataset exactly")
        self.cached_data: list[dict[str, Any] | None] = [None] * len(dataset)
        self.holdout_unlocked = False
        for index in tqdm(
            sorted(self.training_indices), desc="Caching training230 CTs", leave=False
        ):
            self.cached_data[index] = dataset[index]

    def unlock_and_cache_holdout(self) -> None:
        """Permit formal inference and cache holdout CTs after training is complete."""
        if self.holdout_unlocked:
            return
        for index in tqdm(
            sorted(self.holdout_indices), desc="Caching formal200 CTs", leave=False
        ):
            self.cached_data[index] = self.dataset[index]
        self.holdout_unlocked = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index in self.holdout_indices and not self.holdout_unlocked:
            raise RuntimeError(
                "formal holdout CT access attempted before all final checkpoints existed"
            )
        sample = self.cached_data[index]
        if sample is None:
            raise RuntimeError(f"guarded CT index {index} was not cached")
        return dict(sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pft-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=20260827)
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--selection-epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--hard-extra-fraction", type=float, default=0.35)
    parser.add_argument(
        "--hard-definition-margin",
        type=float,
        default=5.0,
        help="mark FEV1/FVC cases with |value-70| below this margin as hard",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hash_ids(patient_ids: list[str]) -> str:
    text = "\n".join(patient_ids).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def hash_mapping(values: dict[str, float]) -> str:
    text = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def read_pft_ratios(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, float] = {}
    for raw_row in rows:
        row = {
            (key or "").strip(): (value or "").strip()
            for key, value in raw_row.items()
        }
        patient_id = str(row.get("PatientID", "")).strip()
        ratio = str(row.get("FEV1FVC_pct", "")).strip()
        if patient_id and ratio:
            if patient_id in output:
                raise SystemExit(f"duplicate PFT PatientID: {patient_id}")
            output[patient_id] = float(ratio)
    return output


def set_fixed_split(helper: LoaderHelper, split: dict[str, Any]) -> tuple[list[int], list[int]]:
    index_by_id = {patient_id: index for index, patient_id in enumerate(helper.patient_ids)}
    if len(index_by_id) != len(helper.patient_ids):
        raise SystemExit("dataset contains duplicate patient IDs")
    train_ids = [str(patient_id) for patient_id in split["training_patient_ids"]]
    holdout_ids = [str(patient_id) for patient_id in split["validation_patient_ids"]]
    if set(train_ids) & set(holdout_ids):
        raise SystemExit("train and holdout IDs overlap")
    expected = set(train_ids) | set(holdout_ids)
    missing = sorted(expected - set(index_by_id))
    extra = sorted(set(index_by_id) - expected)
    if missing or extra:
        raise SystemExit(f"split/dataset mismatch: missing={missing} extra={extra}")
    return [index_by_id[patient_id] for patient_id in train_ids], [
        index_by_id[patient_id] for patient_id in holdout_ids
    ]


def build_optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def build_hard_train_loader(
    helper: LoaderHelper,
    indices: list[int],
    weights_by_id: dict[str, float],
    seed: int,
    hard_extra_fraction: float,
) -> DataLoader:
    views_per_sample = helper._views_per_sample()
    if not helper._should_balance_then_augment():
        raise SystemExit(
            "hard-aware training requires balanced_sampling=true, "
            "balance_then_augment=true and views_per_sample>1"
        )
    base_indices = list(indices)
    expanded_indices, augment_flags = helper._build_balanced_view_train_indices(base_indices)
    base_targets = helper.targets[base_indices].astype(int)
    weights = np.asarray(
        [weights_by_id[helper.patient_ids[index]] for index in base_indices],
        dtype=float,
    )
    sampler = HardAwareBalancedViewSampler(
        base_targets,
        weights,
        views_per_sample=views_per_sample,
        seed=seed,
        hard_extra_fraction=hard_extra_fraction,
    )
    return helper._build_loader(
        expanded_indices,
        batch_size=helper.batch_size,
        shuffle=False,
        drop_last=True,
        augmentation=helper.train_augmentation,
        augment_flags=augment_flags,
        sampler=sampler,
    )


def build_eval_loader(helper: LoaderHelper, indices: list[int]) -> DataLoader:
    return helper._build_loader(
        list(indices),
        batch_size=helper.val_batch_size,
        shuffle=False,
        drop_last=False,
        augmentation=None,
    )


def predict_indices(
    model: nn.Module,
    helper: LoaderHelper,
    indices: list[int],
    device: torch.device,
    use_amp: bool,
) -> dict[str, dict[str, Any]]:
    class_names = helper.get_class_names()
    abnormal_index = class_names.index("Abnormal")
    output: dict[str, dict[str, Any]] = {}
    model.eval()
    for batch in build_eval_loader(helper, indices):
        ct = batch["ct"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            logits = model(ct)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        labels = batch["label"].long().view(-1).cpu().numpy()
        for row_index, patient_id in enumerate(batch["patient_id"]):
            patient_id = str(patient_id)
            if patient_id in output:
                raise RuntimeError(f"duplicate prediction for {patient_id}")
            output[patient_id] = {
                "true_label": class_names[int(labels[row_index])],
                "pred_label": class_names[int(predictions[row_index])],
                "prob_abnormal": float(probabilities[row_index, abnormal_index]),
            }
    return output


def classification_metrics(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = np.asarray(
        [1 if row["true_label"] == "Abnormal" else 0 for row in predictions.values()],
        dtype=int,
    )
    predicted = np.asarray(
        [1 if row["pred_label"] == "Abnormal" else 0 for row in predictions.values()],
        dtype=int,
    )
    probabilities = np.asarray(
        [float(row["prob_abnormal"]) for row in predictions.values()], dtype=float
    )
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else None
    return {
        "n": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, predicted)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predicted)), 5),
        "auc": round(auc, 5) if auc is not None else None,
        "sensitivity": round(float(tp / (tp + fn)), 5) if tp + fn else None,
        "specificity": round(float(tn / (tn + fp)), 5) if tn + fp else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_model(
    config: Config,
    helper: LoaderHelper,
    train_indices: list[int],
    weights_by_id: dict[str, float],
    seed: int,
    epochs: int,
    scheduler_horizon: int,
    device: torch.device,
    hard_extra_fraction: float,
    eval_indices: list[int] | None = None,
    eval_interval: int = 5,
) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    set_seed(seed)
    model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
    train_loader = build_hard_train_loader(
        helper,
        train_indices,
        weights_by_id,
        seed=seed,
        hard_extra_fraction=hard_extra_fraction,
    )
    optimizer = build_optimizer(
        model,
        float(config.training.learning_rate),
        float(config.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, scheduler_horizon))
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    augmentation = getattr(getattr(train_loader, "dataset", None), "augmentation", None)
    if not getattr(augmentation, "defer_to_device", False):
        augmentation = None
    history: dict[str, dict[str, Any]] = {}

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
                print("Non-finite loss under bf16; retrying batch in fp32", flush=True)
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
        mean_loss = total_loss / max(batches, 1)
        print(
            f"seed={seed} epoch={epoch}/{epochs} loss={mean_loss:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.8g}",
            flush=True,
        )
        if eval_indices is not None and (epoch % eval_interval == 0 or epoch == epochs):
            predictions = predict_indices(
                model,
                helper,
                eval_indices,
                device,
                bool(config.training.amp),
            )
            metrics = classification_metrics(predictions)
            history[str(epoch)] = {"metrics": metrics, "predictions": predictions}
            print(
                f"internal-only epoch={epoch} accuracy={metrics['accuracy']:.5f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.5f}",
                flush=True,
            )
    model.eval()
    return model, history


def clinical_training_weights(
    train_ids: list[str], pft_ratios: dict[str, float]
) -> dict[str, float]:
    missing = sorted(set(train_ids) - set(pft_ratios))
    if missing:
        raise SystemExit(f"training patients missing PFT ratios: {missing[:10]}")
    return {
        patient_id: 1.0 + max(0.0, 1.0 - abs(pft_ratios[patient_id] - 70.0) / 7.0)
        for patient_id in train_ids
    }


def mine_hard_examples(
    train_ids: list[str],
    pft_ratios: dict[str, float],
    oof_predictions: dict[str, dict[str, Any]],
    hard_definition_margin: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if set(oof_predictions) != set(train_ids):
        missing = sorted(set(train_ids) - set(oof_predictions))
        extra = sorted(set(oof_predictions) - set(train_ids))
        raise RuntimeError(f"OOF coverage mismatch: missing={missing} extra={extra}")
    weights: dict[str, float] = {}
    report: list[dict[str, Any]] = []
    for patient_id in train_ids:
        row = oof_predictions[patient_id]
        ratio = float(pft_ratios[patient_id])
        margin = abs(ratio - 70.0)
        boundary_closeness = max(0.0, 1.0 - margin / 7.0)
        probability = float(row["prob_abnormal"])
        uncertainty = max(0.0, 1.0 - 2.0 * abs(probability - 0.5))
        incorrect = row["pred_label"] != row["true_label"]
        weight = min(
            3.0,
            1.0 + boundary_closeness + 0.75 * float(incorrect) + 0.5 * uncertainty,
        )
        reasons: list[str] = []
        if margin < hard_definition_margin:
            reasons.append(
                f"FEV1/FVC within {hard_definition_margin:g} points of 70"
            )
        if incorrect:
            reasons.append("OOF misclassification")
        if uncertainty >= 0.7:
            reasons.append("OOF probability near 0.5")
        weights[patient_id] = float(weight)
        report.append(
            {
                "Patient_ID": patient_id,
                "True_Label": row["true_label"],
                "FEV1_FVC_pct": ratio,
                "Boundary_Margin": margin,
                "OOF_Prediction": row["pred_label"],
                "OOF_Abnormal_Probability": probability,
                "OOF_Correct": not incorrect,
                "OOF_Uncertainty": uncertainty,
                "Hard_Example": bool(reasons),
                "Hard_Reasons": "; ".join(reasons),
                "Sampling_Weight": weight,
            }
        )
    return weights, report


def write_hardness_csv(path: Path, report: list[dict[str, Any]]) -> None:
    fields = list(report[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report)


def aggregate_member_predictions(
    member_predictions: list[dict[str, dict[str, Any]]],
    holdout_ids: list[str],
) -> dict[str, dict[str, Any]]:
    patients: dict[str, dict[str, Any]] = {}
    member_count = len(member_predictions)
    for patient_id in holdout_ids:
        members = [predictions[patient_id] for predictions in member_predictions]
        votes = sum(row["pred_label"] == "Abnormal" for row in members)
        patients[patient_id] = {
            "true_label": members[0]["true_label"],
            "votes_for_abnormal": votes,
            "vote_text": f"{votes}/{member_count}",
            "pred_label": "Abnormal" if votes >= member_count // 2 + 1 else "Normal",
            "mean_prob_abnormal": float(
                np.mean([float(row["prob_abnormal"]) for row in members])
            ),
            "member_predictions": [row["pred_label"] for row in members],
            "member_prob_abnormal": [float(row["prob_abnormal"]) for row in members],
        }
    return patients


def aggregate_metrics(patients: dict[str, dict[str, Any]]) -> dict[str, Any]:
    predictions = {
        patient_id: {
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "prob_abnormal": row["mean_prob_abnormal"],
        }
        for patient_id, row in patients.items()
    }
    metrics = classification_metrics(predictions)
    metrics["auc_mean_member_probability"] = metrics.pop("auc")
    return metrics


def main() -> None:
    args = parse_args()
    if args.selection_folds < 2 or args.selection_epochs < 1 or args.eval_interval < 1:
        raise SystemExit("selection folds/epochs and eval interval must be positive")
    if not 0.0 <= args.hard_extra_fraction <= 1.0:
        raise SystemExit("hard-extra-fraction must be between 0 and 1")
    if args.hard_definition_margin <= 0.0:
        raise SystemExit("hard-definition-margin must be positive")
    configure_torch_runtime()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = Config.from_yaml(str(args.config))
    config = replace(
        config,
        data=replace(
            config.data,
            source_dir=str(args.source_dir),
            manifest=str(args.manifest),
            cache_data=False,
        ),
    )
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    helper = LoaderHelper(config)
    train_indices, holdout_indices = set_fixed_split(helper, split)
    guarded_dataset = FrozenHoldoutGuard(helper.train_ds, train_indices, holdout_indices)
    helper.train_ds = guarded_dataset
    train_ids = [str(patient_id) for patient_id in split["training_patient_ids"]]
    holdout_ids = [str(patient_id) for patient_id in split["validation_patient_ids"]]
    if len(train_ids) != 230 or len(holdout_ids) != 200:
        raise SystemExit(f"expected frozen 230/200 split, got {len(train_ids)}/{len(holdout_ids)}")
    train_id_set = set(train_ids)
    holdout_id_set = set(holdout_ids)
    pft_ratios = read_pft_ratios(args.pft_csv)
    initial_weights = clinical_training_weights(train_ids, pft_ratios)
    args.out.mkdir(parents=True, exist_ok=True)

    signature_payload = {
        "training_ids_sha256": hash_ids(train_ids),
        "holdout_ids_sha256": hash_ids(holdout_ids),
        "selection_seed": args.selection_seed,
        "selection_folds": args.selection_folds,
        "selection_epochs": args.selection_epochs,
        "eval_interval": args.eval_interval,
        "hard_extra_fraction": args.hard_extra_fraction,
        "hard_definition_margin": args.hard_definition_margin,
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "model": str(config.model.name),
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    print(
        f"frozen split: train={len(train_ids)} holdout={len(holdout_ids)} "
        f"holdout_sha256={signature_payload['holdout_ids_sha256']}",
        flush=True,
    )
    print(
        "SELECTION/MINING POLICY: only the 230 training patients are accessible; "
        "the frozen 200 are excluded from all folds and sampling weights.",
        flush=True,
    )

    train_targets = helper.targets[train_indices].astype(int)
    splitter = StratifiedKFold(
        n_splits=args.selection_folds,
        shuffle=True,
        random_state=args.selection_seed,
    )
    fold_payloads: list[dict[str, Any]] = []
    train_indices_array = np.asarray(train_indices, dtype=int)
    for fold_index, (train_positions, val_positions) in enumerate(
        splitter.split(train_indices_array, train_targets), start=1
    ):
        inner_train_indices = train_indices_array[train_positions].astype(int).tolist()
        inner_val_indices = train_indices_array[val_positions].astype(int).tolist()
        inner_train_ids = {helper.patient_ids[index] for index in inner_train_indices}
        inner_val_ids = {helper.patient_ids[index] for index in inner_val_indices}
        if inner_train_ids & inner_val_ids:
            raise RuntimeError("patient leakage inside training-only CV")
        if (inner_train_ids | inner_val_ids) != train_id_set:
            raise RuntimeError("training-only CV does not cover exactly the 230 training patients")
        if (inner_train_ids | inner_val_ids) & holdout_id_set:
            raise RuntimeError("frozen holdout leaked into training-only CV")
        fold_path = args.out / f"selection_fold_{fold_index:02d}.json"
        if fold_path.exists():
            fold_payload = json.loads(fold_path.read_text(encoding="utf-8"))
            if fold_payload.get("run_signature") != run_signature:
                raise SystemExit(f"stale selection artifact with different settings: {fold_path}")
            print(f"selection fold {fold_index}: reusing {fold_path}", flush=True)
        else:
            fold_seed = args.base_seed + 1000 + fold_index
            print(
                f"selection fold {fold_index}/{args.selection_folds}: "
                f"train={len(inner_train_indices)} val={len(inner_val_indices)} seed={fold_seed}",
                flush=True,
            )
            model, history = train_model(
                config,
                helper,
                inner_train_indices,
                initial_weights,
                seed=fold_seed,
                epochs=args.selection_epochs,
                scheduler_horizon=args.selection_epochs,
                device=device,
                hard_extra_fraction=args.hard_extra_fraction,
                eval_indices=inner_val_indices,
                eval_interval=args.eval_interval,
            )
            fold_payload = {
                "run_signature": run_signature,
                "fold": fold_index,
                "seed": fold_seed,
                "n_train": len(inner_train_indices),
                "n_internal_validation": len(inner_val_indices),
                "training_patient_ids": sorted(inner_train_ids),
                "internal_validation_patient_ids": sorted(inner_val_ids),
                "formal_holdout_patient_ids_used": [],
                "history": history,
            }
            fold_path.write_text(
                json.dumps(fold_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_payloads.append(fold_payload)

    common_epochs = sorted(
        set.intersection(
            *[set(payload["history"]) for payload in fold_payloads]
        ),
        key=int,
    )
    if not common_epochs:
        raise RuntimeError("selection folds have no common evaluation epochs")
    selection_curve: dict[str, dict[str, Any]] = {}
    pooled_by_epoch: dict[str, dict[str, dict[str, Any]]] = {}
    for epoch in common_epochs:
        pooled: dict[str, dict[str, Any]] = {}
        for payload in fold_payloads:
            for patient_id, row in payload["history"][epoch]["predictions"].items():
                if patient_id in pooled:
                    raise RuntimeError(f"duplicate OOF prediction for {patient_id} at epoch {epoch}")
                pooled[patient_id] = row
        if set(pooled) != train_id_set:
            raise RuntimeError(f"OOF predictions at epoch {epoch} do not cover training230")
        pooled_by_epoch[epoch] = pooled
        selection_curve[epoch] = classification_metrics(pooled)

    selected_epoch_key = max(
        common_epochs,
        key=lambda epoch: (
            float(selection_curve[epoch]["balanced_accuracy"]),
            float(selection_curve[epoch]["auc"] or -1.0),
            -int(epoch),
        ),
    )
    selected_epoch = int(selected_epoch_key)
    oof_predictions = pooled_by_epoch[selected_epoch_key]
    final_weights, hardness_report = mine_hard_examples(
        train_ids,
        pft_ratios,
        oof_predictions,
        hard_definition_margin=args.hard_definition_margin,
    )
    if set(final_weights) & holdout_id_set:
        raise RuntimeError("formal holdout leaked into final sampling weights")
    weights_hash = hash_mapping(final_weights)
    write_hardness_csv(args.out / "training230_hardness.csv", hardness_report)
    hard_count = sum(bool(row["Hard_Example"]) for row in hardness_report)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_signature": run_signature,
        "policy": "training-only 5-fold OOF epoch selection and hard-example mining",
        "formal_holdout_patient_ids_used_for_selection_or_mining": [],
        "n_training": len(train_ids),
        "n_formal_holdout": len(holdout_ids),
        "training_ids_sha256": hash_ids(train_ids),
        "formal_holdout_ids_sha256": hash_ids(holdout_ids),
        "selection_metric": "pooled training-only OOF balanced_accuracy; AUC tie-break; earlier epoch final tie-break",
        "selection_curve": selection_curve,
        "selected_epoch": selected_epoch,
        "selected_epoch_oof_metrics": selection_curve[selected_epoch_key],
        "hard_example_definition": (
            f"FEV1/FVC margin<{args.hard_definition_margin:g} OR OOF error "
            "OR OOF uncertainty>=0.7"
        ),
        "hard_example_count": hard_count,
        "sampling_weight_min": min(final_weights.values()),
        "sampling_weight_max": max(final_weights.values()),
        "sampling_weights_sha256": weights_hash,
        "scheduler": f"CosineAnnealingLR(T_max={args.selection_epochs}); identical trajectory in CV and final training",
    }
    (args.out / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"selected_epoch={selected_epoch}; internal OOF={selection_curve[selected_epoch_key]}; "
        f"hard_examples={hard_count}/{len(train_ids)}",
        flush=True,
    )

    seeds = [args.base_seed + index for index in range(args.members)]
    checkpoint_paths: list[Path] = []
    for member_index, seed in enumerate(seeds, start=1):
        checkpoint = args.out / f"member_{member_index:02d}_seed{seed}.pth"
        metadata_path = args.out / f"member_{member_index:02d}_seed{seed}.json"
        expected_metadata = {
            "run_signature": run_signature,
            "sampling_weights_sha256": weights_hash,
            "seed": seed,
            "epochs": selected_epoch,
            "scheduler_horizon": args.selection_epochs,
            "training_ids_sha256": hash_ids(train_ids),
            "formal_holdout_ids_sha256": hash_ids(holdout_ids),
            "formal_holdout_evaluated_during_training": False,
        }
        if checkpoint.exists() and metadata_path.exists():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing_metadata != expected_metadata:
                raise SystemExit(f"stale final member artifact with different settings: {checkpoint}")
            print(f"final member {member_index}/{args.members}: reusing seed={seed}", flush=True)
        else:
            print(
                f"final member {member_index}/{args.members}: train all 230, seed={seed}, "
                f"epochs={selected_epoch}; formal200 remains unread",
                flush=True,
            )
            model, _ = train_model(
                config,
                helper,
                train_indices,
                final_weights,
                seed=seed,
                epochs=selected_epoch,
                scheduler_horizon=args.selection_epochs,
                device=device,
                hard_extra_fraction=args.hard_extra_fraction,
                eval_indices=None,
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    **expected_metadata,
                    "training_patient_ids": train_ids,
                    "formal_holdout_patient_ids": holdout_ids,
                },
                checkpoint,
            )
            metadata_path.write_text(
                json.dumps(expected_metadata, indent=2), encoding="utf-8"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        checkpoint_paths.append(checkpoint)

    print(
        "All five final checkpoints exist. Beginning the single formal holdout read-out now.",
        flush=True,
    )
    guarded_dataset.unlock_and_cache_holdout()
    member_predictions: list[dict[str, dict[str, Any]]] = []
    for member_index, (seed, checkpoint) in enumerate(
        zip(seeds, checkpoint_paths, strict=True), start=1
    ):
        model = build_model(config.model, output_dim=config.model_output_dim()).to(device)
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["state_dict"])
        predictions = predict_indices(
            model,
            helper,
            holdout_indices,
            device,
            bool(config.training.amp),
        )
        if set(predictions) != holdout_id_set:
            raise RuntimeError(f"member {member_index} formal holdout coverage mismatch")
        predictions_path = args.out / f"member_{member_index:02d}_formal_predictions.json"
        predictions_path.write_text(
            json.dumps(predictions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        member_predictions.append(predictions)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    patients = aggregate_member_predictions(member_predictions, holdout_ids)
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": str(args.config),
            "split_json": str(args.split_json),
            "n_training": len(train_ids),
            "n_holdout": len(holdout_ids),
            "members": args.members,
            "epochs": selected_epoch,
            "seeds": seeds,
            "training_only_selection_folds": args.selection_folds,
            "training_only_selection_epochs": args.selection_epochs,
            "validation_used_during_training": False,
            "validation_used_for_epoch_selection": False,
            "validation_used_for_hard_example_mining": False,
            "holdout_evaluated_only_after_all_final_checkpoints": True,
            "formal_holdout_ids_sha256": hash_ids(holdout_ids),
            "hard_example_count": hard_count,
            "selection_summary": str(args.out / "selection_summary.json"),
            "hardness_report": str(args.out / "training230_hardness.csv"),
        },
        "metrics": aggregate_metrics(patients),
        "patients": patients,
    }
    output = args.out / "mamba5_hard_holdout_predictions.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
