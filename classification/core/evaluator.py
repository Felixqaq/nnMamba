"""Evaluation metrics and utilities."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, Accuracy, Specificity, Recall


@dataclass
class Metrics:
    """Container for evaluation metrics."""

    accuracy: float
    sensitivity: float
    specificity: float
    auc: float
    labels: torch.Tensor = None
    preds: torch.Tensor = None
    threshold: float = 0.5
    sample_indices: list = None


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """Run inference to get labels, predictions, and sample indices."""
    model.eval()
    all_labels = []
    all_preds = []
    all_indices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch["mri"].to(device)
            y = batch["label"].to(device)

            out = model(x).sigmoid().detach()
            all_labels.append(y.flatten())
            all_preds.append(out.flatten())

            # Track batch indices
            batch_size = x.size(0)
            start_idx = batch_idx * dataloader.batch_size
            all_indices.extend(range(start_idx, start_idx + batch_size))

    if not all_preds:
        return torch.tensor([]), torch.tensor([]), []

    return torch.cat(all_labels), torch.cat(all_preds), all_indices


def save_misclassified(
    metrics: "Metrics",
    dataset,
    fold_indices: list,
    labels: list[str],
    save_path: Path,
    fold: int,
) -> dict:
    """Save misclassified samples to JSON file."""
    if metrics.labels is None or metrics.preds is None:
        return {}

    threshold = metrics.threshold
    pred_labels = (metrics.preds >= threshold).int()
    true_labels = metrics.labels.int()

    misclassified = {
        "fold": fold,
        "threshold": threshold,
        "false_negatives": [],
        "false_positives": [],
    }

    for i, (true, pred, prob) in enumerate(
        zip(true_labels, pred_labels, metrics.preds)
    ):
        if true != pred:
            idx = fold_indices[i]
            path = str(dataset.directories[idx])
            entry = {
                "path": path,
                "filename": Path(path).name,
                "true_label": labels[true.item()],
                "pred_label": labels[pred.item()],
                "prob_abnormal": round(prob.item(), 4),
            }
            if true == 1 and pred == 0:
                misclassified["false_negatives"].append(entry)
            else:
                misclassified["false_positives"].append(entry)

    save_path.mkdir(parents=True, exist_ok=True)
    json_path = save_path / f"fold{fold}_errors.json"
    with open(json_path, "w") as f:
        json.dump(misclassified, f, indent=2)

    return misclassified


def find_optimal_threshold(labels: torch.Tensor, preds: torch.Tensor) -> float:
    """Find optimal threshold using Youden's J statistic."""
    if len(torch.unique(labels)) < 2:
        return 0.5

    # Move to CPU for sklearn/numpy style calculation or use torchmetrics
    from torchmetrics import ROC

    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(preds, labels.long())

    # J = Sensitivity + Specificity - 1
    # Specificity = 1 - FPR
    # J = TPR + (1 - FPR) - 1 = TPR - FPR
    J = tpr - fpr
    best_idx = torch.argmax(J)
    return thresholds[best_idx].item()


def compute_metrics(
    labels: torch.Tensor,
    preds: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> Metrics:
    """Calculate metrics from labels and predictions."""
    if len(preds) == 0:
        return Metrics(0.0, 0.0, 0.0, 0.5)

    preds = preds.to(device)
    labels = labels.to(device)

    auroc = AUROC(task="binary").to(device)
    accuracy_metric = Accuracy(task="binary", threshold=threshold).to(device)
    specificity_metric = Specificity(task="binary", threshold=threshold).to(device)
    sensitivity_metric = Recall(task="binary", threshold=threshold).to(device)

    unique_labels = torch.unique(labels)
    auc = 0.5 if len(unique_labels) < 2 else auroc(preds, labels).item()

    return Metrics(
        accuracy=round(accuracy_metric(preds, labels).item(), 5),
        sensitivity=round(sensitivity_metric(preds, labels).item(), 5),
        specificity=round(specificity_metric(preds, labels).item(), 5),
        auc=round(auc, 5),
        labels=labels.cpu(),
        preds=preds.cpu(),
        threshold=threshold,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Metrics:
    """Evaluate model on a dataloader."""
    labels, preds = get_predictions(model, dataloader, device)
    # Ensure they are on the correct device for metric calculation?
    # compute_metrics handles device transfer.
    # Note: get_predictions output is on the same device as the model output (i.e., 'device' arg).
    return compute_metrics(labels, preds, threshold, device)
