"""Evaluation metrics and utilities for nnMamba regression."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

    @property
    def pearson_r(self) -> float:
        """Compatibility alias used by some callers."""
        return self.pearson


def _extract_inputs(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Get model input and target from a batch.

    The regression pipeline may expose targets as ``angle``, ``label``, or
    ``target``. This helper keeps the evaluator tolerant while the data layer is
    being finalized.
    """

    x = batch.get("mri")
    if x is None:
        x = batch.get("image")
    if x is None:
        x = batch.get("ct")
    if x is None:
        raise KeyError("Batch is missing an input tensor key such as 'mri'.")

    target = batch.get("angle")
    if target is None:
        target = batch.get("label")
    if target is None:
        target = batch.get("target")
    if target is None:
        raise KeyError("Batch is missing a regression target key.")

    return x, target


def _extract_predictions(output: Any) -> torch.Tensor:
    """Convert model output into a flat prediction tensor."""

    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        for key in ("pred", "prediction", "output", "angle"):
            if key in output:
                output = output[key]
                break
        else:
            raise KeyError("Cannot infer prediction tensor from model output.")

    if not torch.is_tensor(output):
        output = torch.as_tensor(output)

    return output.float().view(-1)


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_amp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Run inference to get labels, predictions, and sample indices."""

    model.eval()
    all_labels: list[torch.Tensor] = []
    all_preds: list[torch.Tensor] = []
    all_indices: list[int] = []
    running_index = 0

    with torch.no_grad():
        for batch in dataloader:
            x, target = _extract_inputs(batch)
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
                preds = _extract_predictions(model(x))
            if abs(target_std) > 1e-8:
                preds = preds * target_std + target_mean
            else:
                preds = preds + target_mean
            labels = target.float().view(-1)

            all_labels.append(labels.detach().cpu())
            all_preds.append(preds.detach().cpu())

            batch_size = labels.numel()
            if "index" in batch:
                indices = batch["index"]
                if torch.is_tensor(indices):
                    all_indices.extend(indices.view(-1).tolist())
                else:
                    all_indices.extend(list(indices))
            else:
                all_indices.extend(range(running_index, running_index + batch_size))
                running_index += batch_size

    if not all_preds:
        empty = torch.tensor([], dtype=torch.float32)
        return empty, empty, []

    return torch.cat(all_labels), torch.cat(all_preds), all_indices


def compute_metrics(
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    use_amp: bool = False,
) -> RegressionMetrics:
    """Evaluate model on a dataloader."""

    labels, preds, indices = get_predictions(
        model,
        dataloader,
        device,
        target_mean=target_mean,
        target_std=target_std,
        use_amp=use_amp,
    )
    return compute_metrics(labels, preds, indices)


def save_predictions(
    metrics: RegressionMetrics,
    dataset,
    fold_indices: list[int],
    save_path: Path,
    fold: int,
) -> dict[str, Any]:
    """Persist per-sample regression predictions for a fold."""
    if metrics.labels is None or metrics.preds is None:
        return {}

    save_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

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
