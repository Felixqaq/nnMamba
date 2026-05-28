"""Evaluation metrics and utilities for nnMamba regression/classification."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CLASSIFICATION_TASK_TYPES = {"gold", "angle_3class", "angle_binary_extreme"}


@dataclass
class RegressionMetrics:
    """Container for regression evaluation metrics."""

    mae: float
    rmse: float
    r2: float
    pearson: float
    mean_error: float
    mse: float
    labels: torch.Tensor | None = None
    preds: torch.Tensor | None = None
    sample_indices: list[int] | None = None
    num_valid_samples: int = 0
    num_invalid_samples: int = 0
    fold: int | None = None
    best_epoch: int | None = None

    @property
    def pearson_r(self) -> float:
        """Compatibility alias used by some callers."""
        return self.pearson


@dataclass
class ClassificationMetrics:
    """Container for multiclass classification evaluation metrics."""

    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    balanced_accuracy: float
    sensitivity: float = 0.0
    specificity: float = 0.0
    labels: torch.Tensor | None = None
    preds: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    sample_indices: list[int] | None = None
    confusion_matrix: list[list[int]] | None = None
    num_valid_samples: int = 0
    num_invalid_samples: int = 0
    fold: int | None = None
    best_epoch: int | None = None


def _extract_input_tensor(batch: dict[str, Any]) -> torch.Tensor:
    """Get model input from a batch."""
    x = batch.get("mri")
    if x is None:
        x = batch.get("image")
    if x is None:
        x = batch.get("ct")
    if x is None:
        raise KeyError("Batch is missing an input tensor key such as 'mri'.")
    return x


def _extract_model_input(batch: dict[str, Any], device: torch.device):
    """Move model inputs to device, including optional late-fusion embeddings."""
    x = _extract_input_tensor(batch).to(device, non_blocking=True)
    embedding = batch.get("tapct_embedding")
    if embedding is None:
        return x
    model_input = {
        "ct": x,
        "tapct_embedding": embedding.to(device, non_blocking=True),
    }
    if batch.get("tapct_mask") is not None:
        model_input["tapct_mask"] = batch["tapct_mask"].to(device, non_blocking=True)
    return model_input


def _extract_regression_target(batch: dict[str, Any]) -> torch.Tensor:
    """Get regression target from a batch."""
    target = batch.get("angle")
    if target is None:
        target = batch.get("target")
    if target is None:
        raise KeyError("Batch is missing a regression target key.")
    return target


def _extract_classification_target(batch: dict[str, Any]) -> torch.Tensor:
    """Get classification target from a batch."""
    target = batch.get("label")
    if target is None:
        target = batch.get("target")
    if target is None:
        raise KeyError("Batch is missing a classification target key.")
    return target


def _extract_regression_predictions(output: Any) -> torch.Tensor:
    """Convert model output into a flat regression tensor."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        for key in ("pred", "prediction", "output", "angle", "target"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("Cannot infer prediction tensor from model output.")

    if not torch.is_tensor(output):
        output = torch.as_tensor(output)

    return output.float().view(-1)


def _extract_classification_logits(output: Any) -> torch.Tensor:
    """Convert model output into a 2D classification-logit tensor."""
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        for key in ("logits", "pred", "prediction", "output"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("Cannot infer classification logits from model output.")

    if not torch.is_tensor(output):
        output = torch.as_tensor(output)

    output = output.float()
    if output.ndim == 1:
        output = output.unsqueeze(0)
    if output.ndim != 2:
        raise ValueError(f"Expected 2D logits tensor, got shape={tuple(output.shape)}")
    return output


def encode_ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Expand zero-based class ids into cumulative ordinal threshold targets."""
    labels = labels.long().view(-1)
    thresholds = torch.arange(
        max(0, int(num_classes) - 1),
        device=labels.device,
        dtype=labels.dtype,
    )
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def decode_ordinal_logits(
    logits: torch.Tensor,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode cumulative threshold logits into class ids and class probabilities."""
    logits = _extract_classification_logits(logits)
    expected_thresholds = max(1, int(num_classes) - 1)
    if logits.shape[1] != expected_thresholds:
        raise ValueError(
            "Ordinal logits must emit one threshold per adjacent class boundary: "
            f"got {logits.shape[1]}, expected {expected_thresholds}."
        )

    threshold_probs = torch.sigmoid(logits)
    threshold_probs = torch.cummin(threshold_probs, dim=1).values
    preds = (threshold_probs >= 0.5).sum(dim=1).long()

    stage_scores = []
    ones = torch.ones(threshold_probs.shape[0], device=threshold_probs.device)
    for stage in range(int(num_classes)):
        lower = threshold_probs[:, :stage].prod(dim=1) if stage else ones
        upper = (
            (1.0 - threshold_probs[:, stage:]).prod(dim=1)
            if stage < threshold_probs.shape[1]
            else ones
        )
        stage_scores.append(lower * upper)
    class_probs = torch.stack(stage_scores, dim=1)
    class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return preds, class_probs


def _get_batch_indices(
    batch: dict[str, Any],
    batch_size: int,
    running_index: int,
) -> tuple[list[int], int]:
    """Return dataset/sample indices for the current batch."""
    if "index" in batch:
        indices = batch["index"]
        if torch.is_tensor(indices):
            return indices.view(-1).tolist(), running_index
        return list(indices), running_index

    indices = list(range(running_index, running_index + batch_size))
    return indices, running_index + batch_size


def get_regression_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Run inference for regression labels and predictions."""
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_indices: list[int] = []
    running_index = 0

    with torch.no_grad():
        for batch in dataloader:
            x = _extract_model_input(batch, device)
            target = _extract_regression_target(batch).to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
                preds = _extract_regression_predictions(model(x))

            if abs(target_std) > 1e-8:
                preds = preds * target_std + target_mean
            else:
                preds = preds + target_mean
            labels = target.float().view(-1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            indices, running_index = _get_batch_indices(batch, labels.numel(), running_index)
            all_indices.extend(indices)

    if not all_preds:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty, []

    return torch.cat(all_labels), torch.cat(all_preds), all_indices


def get_classification_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    num_classes: int = 1,
    classification_mode: str = "multiclass",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Run inference for classification labels, argmax predictions, and probabilities."""
    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    all_indices: list[int] = []
    running_index = 0

    with torch.no_grad():
        for batch in dataloader:
            x = _extract_model_input(batch, device)
            target = _extract_classification_target(batch).to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
                logits = _extract_classification_logits(model(x))

            if classification_mode == "ordinal":
                preds, probs = decode_ordinal_logits(logits, num_classes)
            else:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            labels = target.long().view(-1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())
            all_probs.append(probs.detach().cpu())

            indices, running_index = _get_batch_indices(batch, labels.numel(), running_index)
            all_indices.extend(indices)

    if not all_preds:
        empty_labels = torch.tensor([], dtype=torch.long)
        empty_probs = torch.empty((0, 0), dtype=torch.float32)
        return empty_labels, empty_labels, empty_probs, []

    return (
        torch.cat(all_labels),
        torch.cat(all_preds),
        torch.cat(all_probs),
        all_indices,
    )


def compute_regression_metrics(
    labels: torch.Tensor,
    preds: torch.Tensor,
    sample_indices: list[int] | None = None,
) -> RegressionMetrics:
    """Calculate regression metrics from labels and predictions."""
    labels = labels.float().view(-1).cpu()
    preds = preds.float().view(-1).cpu()

    if labels.numel() == 0 or preds.numel() == 0:
        return RegressionMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    finite_mask = torch.isfinite(labels) & torch.isfinite(preds)
    num_valid = int(finite_mask.sum().item())
    num_invalid = int((~finite_mask).sum().item())

    if sample_indices is not None and len(sample_indices) == len(finite_mask):
        sample_indices = [
            idx for idx, keep in zip(sample_indices, finite_mask.tolist()) if keep
        ]

    labels = labels[finite_mask]
    preds = preds[finite_mask]

    if labels.numel() == 0:
        return RegressionMetrics(
            mae=math.inf,
            rmse=math.inf,
            r2=-math.inf,
            pearson=0.0,
            mean_error=math.inf,
            mse=math.inf,
            labels=labels,
            preds=preds,
            sample_indices=sample_indices,
            num_valid_samples=num_valid,
            num_invalid_samples=num_invalid,
        )

    errors = preds - labels
    mse = torch.mean(errors**2)
    mae = torch.mean(errors.abs())
    rmse = torch.sqrt(mse)
    mean_error = torch.mean(errors)

    ss_res = torch.sum(errors**2)
    ss_tot = torch.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if torch.isclose(ss_tot, torch.tensor(0.0)) else (1 - ss_res / ss_tot)

    if labels.numel() < 2:
        pearson = torch.tensor(0.0)
    else:
        label_std = labels.std(unbiased=False)
        pred_std = preds.std(unbiased=False)
        if torch.isclose(label_std, torch.tensor(0.0)) or torch.isclose(
            pred_std, torch.tensor(0.0)
        ):
            pearson = torch.tensor(0.0)
        else:
            pearson = torch.corrcoef(torch.stack([labels, preds]))[0, 1]

    return RegressionMetrics(
        mae=round(float(mae), 5),
        rmse=round(float(rmse), 5),
        r2=round(float(r2), 5),
        pearson=round(float(pearson), 5),
        mean_error=round(float(mean_error), 5),
        mse=round(float(mse), 5),
        labels=labels,
        preds=preds,
        sample_indices=sample_indices,
        num_valid_samples=num_valid,
        num_invalid_samples=num_invalid,
    )


def compute_classification_metrics(
    labels: torch.Tensor,
    preds: torch.Tensor,
    probs: torch.Tensor,
    num_classes: int,
    sample_indices: list[int] | None = None,
) -> ClassificationMetrics:
    """Calculate multiclass classification metrics."""
    labels = labels.long().view(-1).cpu()
    preds = preds.long().view(-1).cpu()
    probs = probs.float().cpu()

    if labels.numel() == 0 or preds.numel() == 0:
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    finite_mask = torch.isfinite(probs).all(dim=1)
    num_valid = int(finite_mask.sum().item())
    num_invalid = int((~finite_mask).sum().item())

    if sample_indices is not None and len(sample_indices) == len(finite_mask):
        sample_indices = [
            idx for idx, keep in zip(sample_indices, finite_mask.tolist()) if keep
        ]

    labels = labels[finite_mask]
    preds = preds[finite_mask]
    probs = probs[finite_mask]

    if labels.numel() == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            macro_f1=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            balanced_accuracy=0.0,
            labels=labels,
            preds=preds,
            probs=probs,
            sample_indices=sample_indices,
            confusion_matrix=[[0] * num_classes for _ in range(num_classes)],
            num_valid_samples=num_valid,
            num_invalid_samples=num_invalid,
        )

    labels_np = labels.numpy().astype(int)
    preds_np = preds.numpy().astype(int)
    class_ids = list(range(num_classes))
    cm = confusion_matrix(labels_np, preds_np, labels=class_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_np,
        preds_np,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    class_totals = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_recalls = np.divide(
            np.diag(cm),
            class_totals,
            out=np.zeros(len(class_ids), dtype=float),
            where=class_totals != 0,
        )
    sensitivity, specificity = _classification_sensitivity_specificity(cm)

    return ClassificationMetrics(
        accuracy=round(float(accuracy_score(labels_np, preds_np)), 5),
        macro_f1=round(float(f1), 5),
        macro_precision=round(float(precision), 5),
        macro_recall=round(float(recall), 5),
        balanced_accuracy=round(float(class_recalls.mean()), 5),
        sensitivity=round(float(sensitivity), 5),
        specificity=round(float(specificity), 5),
        labels=labels,
        preds=preds,
        probs=probs,
        sample_indices=sample_indices,
        confusion_matrix=cm.astype(int).tolist(),
        num_valid_samples=num_valid,
        num_invalid_samples=num_invalid,
    )


def _classification_sensitivity_specificity(cm: np.ndarray) -> tuple[float, float]:
    """Return binary class-0 or multiclass macro sensitivity/specificity."""
    if cm.size == 0:
        return 0.0, 0.0

    cm = np.asarray(cm, dtype=float)
    total = float(cm.sum())
    true_positives = np.diag(cm)
    false_negatives = cm.sum(axis=1) - true_positives
    false_positives = cm.sum(axis=0) - true_positives
    true_negatives = total - true_positives - false_negatives - false_positives

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.divide(
            true_positives,
            true_positives + false_negatives,
            out=np.zeros_like(true_positives, dtype=float),
            where=(true_positives + false_negatives) != 0,
        )
        specificities = np.divide(
            true_negatives,
            true_negatives + false_positives,
            out=np.zeros_like(true_negatives, dtype=float),
            where=(true_negatives + false_positives) != 0,
        )

    if cm.shape == (2, 2):
        # Angle binary manifests list the abnormal endpoint first.
        return float(sensitivities[0]), float(specificities[0])

    return float(sensitivities.mean()), float(specificities.mean())


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str = "regression",
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_amp: bool = False,
    num_classes: int = 1,
    classification_mode: str = "multiclass",
) -> RegressionMetrics | ClassificationMetrics:
    """Evaluate model on a dataloader."""
    if task_type in CLASSIFICATION_TASK_TYPES:
        labels, preds, probs, indices = get_classification_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            use_amp=use_amp,
            num_classes=num_classes,
            classification_mode=classification_mode,
        )
        return compute_classification_metrics(
            labels=labels,
            preds=preds,
            probs=probs,
            num_classes=num_classes,
            sample_indices=indices,
        )

    labels, preds, indices = get_regression_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=use_amp,
    )
    return compute_regression_metrics(labels, preds, indices)


def save_predictions(
    metrics: RegressionMetrics | ClassificationMetrics,
    dataset,
    fold_indices: list[int],
    save_path: Path,
    fold: int,
    task_type: str = "regression",
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Persist per-sample predictions for a fold."""
    save_path.mkdir(parents=True, exist_ok=True)
    class_names = class_names or []

    if task_type in CLASSIFICATION_TASK_TYPES:
        if (
            getattr(metrics, "labels", None) is None
            or getattr(metrics, "preds", None) is None
            or getattr(metrics, "probs", None) is None
        ):
            return {}

        rows: list[dict[str, Any]] = []
        labels = metrics.labels
        preds = metrics.preds
        probs = metrics.probs

        for i, (true_label, pred_label, prob_vector) in enumerate(
            zip(labels, preds, probs)
        ):
            dataset_idx = fold_indices[i] if i < len(fold_indices) else i
            record = getattr(dataset, "records", [])[dataset_idx]
            true_idx = int(true_label.item())
            pred_idx = int(pred_label.item())
            prob_values = prob_vector.tolist()
            rows.append(
                {
                    "patient_id": record.patient_id,
                    "path": str(record.path),
                    "source_group": record.source_group,
                    "class_label": getattr(record, "class_label", None),
                    "gold_stage_label": record.gold_stage_label,
                    "true_label_index": true_idx,
                    "true_label": (
                        class_names[true_idx] if true_idx < len(class_names) else true_idx
                    ),
                    "pred_label_index": pred_idx,
                    "pred_label": (
                        class_names[pred_idx] if pred_idx < len(class_names) else pred_idx
                    ),
                    "confidence": round(float(max(prob_values)), 5),
                    "probabilities": {
                        (
                            class_names[class_idx]
                            if class_idx < len(class_names)
                            else str(class_idx)
                        ): round(float(value), 5)
                        for class_idx, value in enumerate(prob_values)
                    },
                    "correct": true_idx == pred_idx,
                }
            )

        rows.sort(key=lambda item: (item["correct"], -item["confidence"]))
        payload = {
            "fold": fold,
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "macro_precision": metrics.macro_precision,
            "macro_recall": metrics.macro_recall,
            "balanced_accuracy": metrics.balanced_accuracy,
            "sensitivity": metrics.sensitivity,
            "specificity": metrics.specificity,
            "class_names": class_names,
            "confusion_matrix": metrics.confusion_matrix,
            "predictions": rows,
        }
        with open(
            save_path / f"fold{fold}_predictions.json", "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return payload

    if getattr(metrics, "labels", None) is None or getattr(metrics, "preds", None) is None:
        return {}

    rows = []
    for i, (true_angle, pred_angle) in enumerate(zip(metrics.labels, metrics.preds)):
        dataset_idx = fold_indices[i] if i < len(fold_indices) else i
        record = getattr(dataset, "records", [])[dataset_idx]
        signed_error = float(pred_angle.item() - true_angle.item())
        rows.append(
            {
                "patient_id": record.patient_id,
                "path": str(record.path),
                "source_group": record.source_group,
                "true_angle": round(float(true_angle.item()), 5),
                "predicted_angle": round(float(pred_angle.item()), 5),
                "signed_error": round(signed_error, 5),
                "absolute_error": round(abs(signed_error), 5),
            }
        )

    rows.sort(key=lambda item: item["absolute_error"], reverse=True)
    payload = {
        "fold": fold,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "pearson": metrics.pearson,
        "mean_error": metrics.mean_error,
        "predictions": rows,
    }

    with open(save_path / f"fold{fold}_predictions.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload
