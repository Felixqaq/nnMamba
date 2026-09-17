#!/usr/bin/env python3
"""Evaluate a frozen TAP-CT multi-task head on the fixed COPD holdout.

The shared MLP predicts both the binary airflow-obstruction label and the
continuous FEV1/FVC ratio.  Loss weights, stopping epochs, and classification
thresholds are selected using training-fold predictions only; the fixed
holdout is evaluated once after model selection.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn


RATIO_CUTOFF = 70.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--pft-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--lambda-grid", default="0,0.03,0.1,0.3,1,3")
    parser.add_argument("--margin", type=float, default=7.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def parse_float_grid(text: str) -> list[float]:
    """Return a sorted, unique grid of non-negative values."""
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] < 0:
        raise SystemExit("--lambda-grid must contain non-negative values")
    return values


def load_ratios(path: Path) -> dict[str, float]:
    """Load patient-level measured FEV1/FVC percentages."""
    ratios: dict[str, float] = {}
    with io.open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {
                (key or "").strip(): (value or "").strip()
                for key, value in raw_row.items()
            }
            patient_id = row.get("PatientID", "")
            ratio = row.get("FEV1FVC_pct", "")
            if patient_id and ratio:
                ratios[patient_id] = float(ratio)
    if not ratios:
        raise SystemExit(f"{path}: parsed no FEV1/FVC ratios")
    return ratios


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiTaskHead(nn.Module):
    """Small shared head for binary classification and ratio regression."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.classifier(hidden).squeeze(1), self.regressor(hidden).squeeze(1)


def choose_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """Choose a training-only threshold by balanced accuracy."""
    candidates = np.unique(
        np.concatenate([np.asarray([scores.min() - 1e-8, scores.max() + 1e-8]), scores])
    )
    best_threshold = float(np.median(scores))
    best_key = (-1.0, -1.0, float("-inf"))
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        key = (
            float(balanced_accuracy_score(truth, predictions)),
            float(accuracy_score(truth, predictions)),
            -abs(float(threshold) - float(np.median(scores))),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Calculate fixed-threshold binary metrics."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(truth)),
        "n_abnormal": int(truth.sum()),
        "threshold": float(threshold),
        "auc": round(float(roc_auc_score(truth, scores)), 5),
        "accuracy": round(float(accuracy_score(truth, predictions)), 5),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truth, predictions)), 5
        ),
        "sensitivity": round(float(tp / (tp + fn)), 5),
        "specificity": round(float(tn / (tn + fp)), 5),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_auc_ci(
    truth: np.ndarray,
    scores: np.ndarray,
    *,
    samples: int = 4000,
    seed: int = 11,
) -> list[float]:
    """Patient-bootstrap the AUC confidence interval."""
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, len(truth), len(truth))
        if len(np.unique(truth[indices])) < 2:
            continue
        values.append(float(roc_auc_score(truth[indices], scores[indices])))
    return [
        round(float(np.percentile(values, 2.5)), 5),
        round(float(np.percentile(values, 97.5)), 5),
    ]


def predict(
    model: MultiTaskHead,
    features: np.ndarray,
    *,
    ratio_mean: float,
    ratio_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return abnormal probabilities and FEV1/FVC predictions."""
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits, normalized_ratios = model(tensor)
    probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
    ratios = (
        normalized_ratios.cpu().numpy().astype(np.float64) * ratio_std + ratio_mean
    )
    return probabilities, ratios


def train_with_validation(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_ratios: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    valid_ratios: np.ndarray,
    *,
    regression_weight: float,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[MultiTaskHead, int, float]:
    """Fit one fold and early-stop using only that training fold's validation part."""
    set_seed(seed)
    model = MultiTaskHead(
        train_features.shape[1], args.hidden_dim, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(train_labels.sum())
    negatives = float(len(train_labels) - positives)
    positive_weight = torch.tensor(
        [negatives / max(positives, 1.0)], dtype=torch.float32, device=device
    )
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    regression_loss = nn.SmoothL1Loss(beta=0.5)

    x_train = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_labels, dtype=torch.float32, device=device)
    ratio_mean = float(train_ratios.mean())
    ratio_std = max(float(train_ratios.std()), 1e-6)
    r_train = torch.as_tensor(
        (train_ratios - ratio_mean) / ratio_std,
        dtype=torch.float32,
        device=device,
    )

    best_state = deepcopy(model.state_dict())
    best_auc = -1.0
    best_epoch = 1
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, predicted_ratios = model(x_train)
        loss = classification_loss(logits, y_train)
        if regression_weight > 0:
            loss = loss + regression_weight * regression_loss(predicted_ratios, r_train)
        loss.backward()
        optimizer.step()

        valid_probabilities, valid_ratio_predictions = predict(
            model,
            valid_features,
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            device=device,
        )
        valid_auc = float(roc_auc_score(valid_labels, valid_probabilities))
        valid_mae = float(mean_absolute_error(valid_ratios, valid_ratio_predictions))
        improved = valid_auc > best_auc + 1e-5
        if improved:
            best_auc = valid_auc
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch >= args.minimum_epochs and stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, valid_mae


def train_fixed_epochs(
    features: np.ndarray,
    labels: np.ndarray,
    ratios: np.ndarray,
    *,
    regression_weight: float,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> MultiTaskHead:
    """Refit the selected setup on all training patients."""
    set_seed(args.seed)
    model = MultiTaskHead(features.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positives = float(labels.sum())
    positive_weight = torch.tensor(
        [(len(labels) - positives) / max(positives, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    regression_loss = nn.SmoothL1Loss(beta=0.5)
    x_train = torch.as_tensor(features, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels, dtype=torch.float32, device=device)
    ratio_mean = float(ratios.mean())
    ratio_std = max(float(ratios.std()), 1e-6)
    r_train = torch.as_tensor(
        (ratios - ratio_mean) / ratio_std,
        dtype=torch.float32,
        device=device,
    )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, predicted_ratios = model(x_train)
        loss = classification_loss(logits, y_train)
        if regression_weight > 0:
            loss = loss + regression_weight * regression_loss(predicted_ratios, r_train)
        loss.backward()
        optimizer.step()
    return model


def main() -> None:
    """Tune the multi-task loss on training folds and score the fixed holdout."""
    args = parse_args()
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    regression_weights = parse_float_grid(args.lambda_grid)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    feature_bundle = np.load(args.features, allow_pickle=False)
    all_features = np.asarray(feature_bundle["features"], dtype=np.float32)
    patient_ids = [str(value) for value in feature_bundle["patient_ids"]]
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    train_ids = [str(value) for value in split["training_patient_ids"]]
    holdout_ids = [str(value) for value in split["validation_patient_ids"]]
    ratios_by_id = load_ratios(args.pft_csv)
    missing = [
        patient_id
        for patient_id in train_ids + holdout_ids
        if patient_id not in index_by_id or patient_id not in ratios_by_id
    ]
    if missing:
        raise SystemExit(f"missing features or ratios for {len(missing)}: {missing[:5]}")

    train_features_raw = all_features[[index_by_id[item] for item in train_ids]]
    holdout_features_raw = all_features[[index_by_id[item] for item in holdout_ids]]
    train_ratios = np.asarray([ratios_by_id[item] for item in train_ids], dtype=np.float64)
    holdout_ratios = np.asarray(
        [ratios_by_id[item] for item in holdout_ids], dtype=np.float64
    )
    train_labels = (train_ratios < RATIO_CUTOFF).astype(int)
    holdout_labels = (holdout_ratios < RATIO_CUTOFF).astype(int)
    print(
        f"device={device} train={len(train_ids)} abnormal={train_labels.sum()} "
        f"holdout={len(holdout_ids)} abnormal={holdout_labels.sum()}"
    )

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_indices = list(splitter.split(train_features_raw, train_labels))
    candidates: list[dict[str, object]] = []
    oof_by_weight: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for regression_weight in regression_weights:
        oof_probabilities = np.full(len(train_ids), np.nan, dtype=np.float64)
        oof_ratios = np.full(len(train_ids), np.nan, dtype=np.float64)
        best_epochs: list[int] = []
        fold_auc: list[float] = []
        for fold_number, (fit_indices, valid_indices) in enumerate(fold_indices, start=1):
            scaler = StandardScaler()
            fit_features = scaler.fit_transform(
                train_features_raw[fit_indices]
            ).astype(np.float32)
            valid_features = scaler.transform(
                train_features_raw[valid_indices]
            ).astype(np.float32)
            model, best_epoch, _ = train_with_validation(
                fit_features,
                train_labels[fit_indices],
                train_ratios[fit_indices],
                valid_features,
                train_labels[valid_indices],
                train_ratios[valid_indices],
                regression_weight=regression_weight,
                args=args,
                seed=args.seed + fold_number,
                device=device,
            )
            ratio_mean = float(train_ratios[fit_indices].mean())
            ratio_std = max(float(train_ratios[fit_indices].std()), 1e-6)
            probabilities, predicted_ratios = predict(
                model,
                valid_features,
                ratio_mean=ratio_mean,
                ratio_std=ratio_std,
                device=device,
            )
            oof_probabilities[valid_indices] = probabilities
            oof_ratios[valid_indices] = predicted_ratios
            best_epochs.append(best_epoch)
            fold_auc.append(
                float(roc_auc_score(train_labels[valid_indices], probabilities))
            )
        threshold = choose_threshold(train_labels, oof_probabilities)
        candidate_metrics = classification_metrics(
            train_labels, oof_probabilities, threshold
        )
        candidate = {
            "regression_loss_weight": regression_weight,
            "oof_auc": candidate_metrics["auc"],
            "oof_accuracy": candidate_metrics["accuracy"],
            "oof_balanced_accuracy": candidate_metrics["balanced_accuracy"],
            "oof_threshold": threshold,
            "oof_ratio_mae": (
                round(float(mean_absolute_error(train_ratios, oof_ratios)), 5)
                if regression_weight > 0
                else None
            ),
            "mean_fold_auc": round(float(np.mean(fold_auc)), 5),
            "fold_auc": fold_auc,
            "fold_best_epochs": best_epochs,
            "refit_epochs": int(round(float(np.median(best_epochs)))),
        }
        candidates.append(candidate)
        oof_by_weight[regression_weight] = (oof_probabilities, oof_ratios)
        print(json.dumps(candidate, ensure_ascii=False), flush=True)

    selected = max(
        candidates,
        key=lambda item: (
            float(item["oof_auc"]),
            float(item["oof_balanced_accuracy"]),
            -float(item["regression_loss_weight"]),
        ),
    )
    selected_weight = float(selected["regression_loss_weight"])
    selected_threshold = float(selected["oof_threshold"])
    selected_epochs = int(selected["refit_epochs"])
    selected_oof_probabilities, selected_oof_ratios = oof_by_weight[selected_weight]

    final_scaler = StandardScaler()
    train_features = final_scaler.fit_transform(train_features_raw).astype(np.float32)
    holdout_features = final_scaler.transform(holdout_features_raw).astype(np.float32)
    final_model = train_fixed_epochs(
        train_features,
        train_labels,
        train_ratios,
        regression_weight=selected_weight,
        epochs=selected_epochs,
        args=args,
        device=device,
    )
    ratio_mean = float(train_ratios.mean())
    ratio_std = max(float(train_ratios.std()), 1e-6)
    holdout_probabilities, predicted_holdout_ratios = predict(
        final_model,
        holdout_features,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        device=device,
    )

    oof_ratio_score = -selected_oof_ratios
    holdout_ratio_score = -predicted_holdout_ratios
    calibrated_ratio_score_threshold = (
        choose_threshold(train_labels, oof_ratio_score) if selected_weight > 0 else None
    )
    classification_holdout = classification_metrics(
        holdout_labels, holdout_probabilities, selected_threshold
    )
    ratio_at_70 = (
        classification_metrics(holdout_labels, holdout_ratio_score, -RATIO_CUTOFF)
        if selected_weight > 0
        else None
    )
    ratio_calibrated = (
        classification_metrics(
            holdout_labels, holdout_ratio_score, calibrated_ratio_score_threshold
        )
        if calibrated_ratio_score_threshold is not None
        else None
    )

    distance = np.abs(holdout_ratios - RATIO_CUTOFF)
    difficulty: dict[str, object] = {}
    for name, mask in (
        (f"borderline_abs_distance_lt_{args.margin:g}", distance < args.margin),
        (f"clear_abs_distance_ge_{args.margin:g}", distance >= args.margin),
    ):
        difficulty[name] = {
            "n": int(mask.sum()),
            "n_abnormal": int(holdout_labels[mask].sum()),
            "classification_head_auc": round(
                float(roc_auc_score(holdout_labels[mask], holdout_probabilities[mask])),
                5,
            ),
            "ratio_head_auc": (
                round(
                    float(roc_auc_score(holdout_labels[mask], holdout_ratio_score[mask])),
                    5,
                )
                if selected_weight > 0
                else None
            ),
        }

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "features": str(args.features),
            "split_json": str(args.split_json),
            "pft_csv": str(args.pft_csv),
            "n_training": len(train_ids),
            "n_holdout": len(holdout_ids),
            "device": str(device),
            "architecture": {
                "input_dim": int(train_features.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "classification_loss": "class-balanced BCEWithLogitsLoss",
                "regression_loss": "SmoothL1Loss on training-standardized ratio",
            },
            "selection": (
                "regression loss weight, refit epochs, and thresholds selected "
                "with training-fold data only"
            ),
            "holdout_used_for_selection": False,
            "seed": args.seed,
            "folds": args.folds,
        },
        "candidates": candidates,
        "selected": selected,
        "training_oof": {
            "classification_head": classification_metrics(
                train_labels, selected_oof_probabilities, selected_threshold
            ),
            "ratio_mae": (
                round(float(mean_absolute_error(train_ratios, selected_oof_ratios)), 5)
                if selected_weight > 0
                else None
            ),
            "ratio_calibrated_cutoff": (
                -float(calibrated_ratio_score_threshold)
                if calibrated_ratio_score_threshold is not None
                else None
            ),
        },
        "holdout": {
            "classification_head": classification_holdout,
            "classification_head_auc_95ci": bootstrap_auc_ci(
                holdout_labels, holdout_probabilities
            ),
            "ratio_head_at_70": ratio_at_70,
            "ratio_head_at_training_calibrated_cutoff": ratio_calibrated,
            "ratio_head_auc_95ci": (
                bootstrap_auc_ci(holdout_labels, holdout_ratio_score)
                if selected_weight > 0
                else None
            ),
            "ratio_mae": (
                round(
                    float(mean_absolute_error(holdout_ratios, predicted_holdout_ratios)),
                    5,
                )
                if selected_weight > 0
                else None
            ),
            "by_difficulty": difficulty,
        },
        "patients": {
            patient_id: {
                "true_ratio": float(true_ratio),
                "true_label": "Abnormal" if true_label else "Normal",
                "prob_abnormal": float(probability),
                "predicted_ratio": (
                    float(predicted_ratio) if selected_weight > 0 else None
                ),
            }
            for patient_id, true_ratio, true_label, probability, predicted_ratio in zip(
                holdout_ids,
                holdout_ratios,
                holdout_labels,
                holdout_probabilities,
                predicted_holdout_ratios,
                strict=True,
            )
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== selected ===")
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    print("\n=== fixed holdout 200 ===")
    print(json.dumps(payload["holdout"], indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
