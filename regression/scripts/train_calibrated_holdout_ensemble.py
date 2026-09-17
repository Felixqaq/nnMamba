#!/usr/bin/env python3
"""Calibrate on training-only OOF predictions, then score one fixed holdout.

The fixed holdout is never used for early stopping, seed selection, threshold
selection, or hyperparameter selection. Five internal folds produce one
out-of-fold probability for every training patient. Their balanced-accuracy
optimal threshold is frozen before five fresh models are trained on the complete
training cohort for a fixed epoch budget and evaluated on the holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from core.runtime import configure_torch_runtime  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402

from train_fixed_holdout_ensemble import predict_member, train_member  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "datasets/generated/doctor_validation_manifest.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--calibration-seed", type=int, default=20260829)
    parser.add_argument("--calibration-base-seed", type=int, default=102)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def binary_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "threshold": float(threshold),
        "accuracy": round(float(accuracy_score(truth, predictions)), 5),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truth, predictions)), 5
        ),
        "auc": round(float(roc_auc_score(truth, probabilities)), 5),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    """Maximize OOF balanced accuracy, breaking ties by accuracy then proximity to 0.5."""
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0], dtype=np.float64),
                probabilities.astype(np.float64),
            ]
        )
    )
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(truth, predictions))
        accuracy = float(accuracy_score(truth, predictions))
        key = (balanced, accuracy, -abs(float(threshold) - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def index_split(
    helper: LoaderHelper,
    split: dict,
) -> tuple[dict[str, int], list[int], list[int]]:
    index_by_id = {pid: index for index, pid in enumerate(helper.patient_ids)}
    if len(index_by_id) != len(helper.patient_ids):
        raise SystemExit("dataset contains duplicate patient IDs")
    training_ids = list(split["training_patient_ids"])
    holdout_ids = list(split["validation_patient_ids"])
    expected = set(training_ids) | set(holdout_ids)
    missing = sorted(expected - set(index_by_id))
    extra = sorted(set(index_by_id) - expected)
    overlap = sorted(set(training_ids) & set(holdout_ids))
    if missing or extra or overlap:
        raise SystemExit(
            f"split/dataset mismatch: missing={missing} extra={extra} overlap={overlap}"
        )
    return (
        index_by_id,
        [index_by_id[pid] for pid in training_ids],
        [index_by_id[pid] for pid in holdout_ids],
    )


def threshold_predictions(
    raw: dict[str, dict],
    threshold: float,
) -> dict[str, dict]:
    return {
        pid: {
            "prob_abnormal": float(row["prob_abnormal"]),
            "pred_label": (
                "Abnormal" if float(row["prob_abnormal"]) >= threshold else "Normal"
            ),
        }
        for pid, row in raw.items()
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    seed: int,
    epochs: int,
    training_patient_ids: list[str],
    evaluation_patient_ids: list[str],
    phase: str,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "training_patient_ids": training_patient_ids,
            "evaluation_patient_ids": evaluation_patient_ids,
            "phase": phase,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.calibration_folds < 2:
        raise SystemExit("--calibration-folds must be at least 2")
    if args.members < 1 or args.members % 2 == 0:
        raise SystemExit("--members must be a positive odd number")

    configure_torch_runtime()
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    config = Config.from_yaml(str(args.config))
    config = replace(
        config,
        data=replace(
            config.data,
            source_dir=str(args.source_dir),
            manifest=str(args.manifest),
        ),
    )
    if not bool(config.data.balanced_sampling):
        raise SystemExit("config.data.balanced_sampling must be true for this run")

    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    helper = LoaderHelper(config)
    index_by_id, train_indices, holdout_indices = index_split(helper, split)
    class_names = helper.get_class_names()
    abnormal_index = class_names.index("Abnormal")
    true_by_id = {
        pid: class_names[int(helper.targets[index])]
        for pid, index in index_by_id.items()
    }
    train_targets = np.asarray(
        [int(helper.targets[index]) == abnormal_index for index in train_indices],
        dtype=int,
    )
    if np.bincount(train_targets, minlength=2).min() < args.calibration_folds:
        raise SystemExit("not enough cases per class for calibration folds")

    args.out.mkdir(parents=True, exist_ok=True)
    calibration_dir = args.out / "calibration_folds"
    final_dir = args.out / "full_training_members"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=args.calibration_folds,
        shuffle=True,
        random_state=args.calibration_seed,
    )
    oof_raw: dict[str, dict] = {}
    fold_audit: list[dict[str, object]] = []
    for fold_index, (fit_positions, oof_positions) in enumerate(
        splitter.split(np.arange(len(train_indices)), train_targets), start=1
    ):
        fit_indices = [train_indices[int(position)] for position in fit_positions]
        oof_indices = [train_indices[int(position)] for position in oof_positions]
        fit_ids = [helper.patient_ids[index] for index in fit_indices]
        oof_ids = [helper.patient_ids[index] for index in oof_indices]
        helper.fold_indices = [(fit_indices, oof_indices)]
        helper.k_folds = 1
        seed = args.calibration_base_seed + fold_index - 1
        checkpoint = calibration_dir / f"fold_{fold_index:02d}_seed{seed}.pth"
        predictions_path = calibration_dir / f"fold_{fold_index:02d}_oof.json"
        if checkpoint.exists() and predictions_path.exists():
            print(
                f"calibration fold {fold_index}/{args.calibration_folds}: "
                f"reusing seed={seed}",
                flush=True,
            )
            fold_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        else:
            print(
                f"calibration fold {fold_index}/{args.calibration_folds}: "
                f"train={len(fit_indices)} oof={len(oof_indices)} seed={seed}",
                flush=True,
            )
            model = train_member(config, helper, seed, args.epochs, device)
            fold_predictions = predict_member(
                model, helper, device, bool(config.training.amp)
            )
            save_checkpoint(
                checkpoint,
                model,
                seed=seed,
                epochs=args.epochs,
                training_patient_ids=fit_ids,
                evaluation_patient_ids=oof_ids,
                phase="training_only_threshold_calibration",
            )
            predictions_path.write_text(
                json.dumps(fold_predictions, indent=2), encoding="utf-8"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if set(fold_predictions) != set(oof_ids):
            raise SystemExit(f"calibration fold {fold_index} prediction IDs mismatch")
        duplicate_ids = sorted(set(oof_raw) & set(fold_predictions))
        if duplicate_ids:
            raise SystemExit(f"duplicate OOF predictions: {duplicate_ids}")
        oof_raw.update(fold_predictions)
        fold_audit.append(
            {
                "fold": fold_index,
                "seed": seed,
                "n_fit": len(fit_indices),
                "n_oof": len(oof_indices),
                "fit_abnormal": int(train_targets[fit_positions].sum()),
                "oof_abnormal": int(train_targets[oof_positions].sum()),
                "fit_patient_ids": fit_ids,
                "oof_patient_ids": oof_ids,
            }
        )

    training_ids = list(split["training_patient_ids"])
    if set(oof_raw) != set(training_ids):
        raise SystemExit("OOF predictions do not cover the complete training cohort")
    oof_truth = np.asarray(
        [1 if true_by_id[pid] == "Abnormal" else 0 for pid in training_ids],
        dtype=int,
    )
    oof_probabilities = np.asarray(
        [float(oof_raw[pid]["prob_abnormal"]) for pid in training_ids],
        dtype=np.float64,
    )
    threshold = choose_threshold(oof_truth, oof_probabilities)
    oof_metrics = binary_metrics(oof_truth, oof_probabilities, threshold)
    oof_patients = {
        pid: {
            "true_label": true_by_id[pid],
            "prob_abnormal": float(oof_raw[pid]["prob_abnormal"]),
            "pred_label": (
                "Abnormal"
                if float(oof_raw[pid]["prob_abnormal"]) >= threshold
                else "Normal"
            ),
        }
        for pid in training_ids
    }
    calibration_payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_training": len(train_indices),
            "folds": args.calibration_folds,
            "fold_split_seed": args.calibration_seed,
            "model_seeds": [
                args.calibration_base_seed + index
                for index in range(args.calibration_folds)
            ],
            "epochs": args.epochs,
            "balanced_sampling": True,
            "fixed_holdout_touched": False,
            "threshold_selection": (
                "maximize balanced accuracy on one OOF probability per training patient; "
                "ties use accuracy then proximity to 0.5"
            ),
            "folds_audit": fold_audit,
        },
        "selected_threshold": threshold,
        "oof_metrics": oof_metrics,
        "patients": oof_patients,
    }
    calibration_output = args.out / "training_oof_calibration.json"
    calibration_output.write_text(
        json.dumps(calibration_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"frozen training-only threshold={threshold:.8f}")
    print(json.dumps(oof_metrics, indent=2), flush=True)

    helper.fold_indices = [(train_indices, holdout_indices)]
    helper.k_folds = 1
    member_predictions: list[dict[str, dict]] = []
    member_metrics: list[dict[str, object]] = []
    seeds = [args.base_seed + index for index in range(args.members)]
    for member_index, seed in enumerate(seeds, start=1):
        checkpoint = final_dir / f"member_{member_index:02d}_seed{seed}.pth"
        predictions_path = final_dir / f"member_{member_index:02d}_predictions.json"
        if checkpoint.exists() and predictions_path.exists():
            print(
                f"full member {member_index}/{args.members}: reusing seed={seed}",
                flush=True,
            )
            raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        else:
            print(
                f"full member {member_index}/{args.members}: "
                f"train={len(train_indices)} holdout={len(holdout_indices)} seed={seed}",
                flush=True,
            )
            model = train_member(config, helper, seed, args.epochs, device)
            raw_predictions = predict_member(
                model, helper, device, bool(config.training.amp)
            )
            save_checkpoint(
                checkpoint,
                model,
                seed=seed,
                epochs=args.epochs,
                training_patient_ids=training_ids,
                evaluation_patient_ids=list(split["validation_patient_ids"]),
                phase="full_training_fixed_holdout_evaluation",
            )
            predictions_path.write_text(
                json.dumps(raw_predictions, indent=2), encoding="utf-8"
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if set(raw_predictions) != set(split["validation_patient_ids"]):
            raise SystemExit(f"full member {member_index} holdout IDs mismatch")
        predictions = threshold_predictions(raw_predictions, threshold)
        member_predictions.append(predictions)
        truth = np.asarray(
            [
                1 if true_by_id[pid] == "Abnormal" else 0
                for pid in split["validation_patient_ids"]
            ],
            dtype=int,
        )
        probabilities = np.asarray(
            [predictions[pid]["prob_abnormal"] for pid in split["validation_patient_ids"]],
            dtype=np.float64,
        )
        member_metrics.append(
            {
                "member": member_index,
                "seed": seed,
                **binary_metrics(truth, probabilities, threshold),
            }
        )

    patients: dict[str, dict] = {}
    for pid in split["validation_patient_ids"]:
        members = [prediction[pid] for prediction in member_predictions]
        votes = sum(row["pred_label"] == "Abnormal" for row in members)
        patients[pid] = {
            "true_label": true_by_id[pid],
            "votes_for_abnormal": votes,
            "vote_text": f"{votes}/{args.members}",
            "pred_label": (
                "Abnormal" if votes >= args.members // 2 + 1 else "Normal"
            ),
            "mean_prob_abnormal": float(
                np.mean([row["prob_abnormal"] for row in members])
            ),
            "member_predictions": [row["pred_label"] for row in members],
            "member_prob_abnormal": [row["prob_abnormal"] for row in members],
        }
    holdout_truth = np.asarray(
        [1 if row["true_label"] == "Abnormal" else 0 for row in patients.values()],
        dtype=int,
    )
    holdout_predictions = np.asarray(
        [1 if row["pred_label"] == "Abnormal" else 0 for row in patients.values()],
        dtype=int,
    )
    holdout_probabilities = np.asarray(
        [row["mean_prob_abnormal"] for row in patients.values()], dtype=np.float64
    )
    tn, fp, fn, tp = confusion_matrix(
        holdout_truth, holdout_predictions, labels=[0, 1]
    ).ravel()
    metrics = {
        "n": int(len(holdout_truth)),
        "accuracy": round(float(accuracy_score(holdout_truth, holdout_predictions)), 5),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(holdout_truth, holdout_predictions)), 5
        ),
        "auc_mean_member_probability": round(
            float(roc_auc_score(holdout_truth, holdout_probabilities)), 5
        ),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
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
            "balanced_sampling": True,
            "calibration_json": str(calibration_output),
            "training_oof_threshold": threshold,
            "validation_used_during_training_or_calibration": False,
            "holdout_evaluated_after_threshold_frozen": True,
        },
        "metrics": metrics,
        "member_metrics": member_metrics,
        "patients": patients,
    }
    output = args.out / "mamba5_holdout_predictions.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
