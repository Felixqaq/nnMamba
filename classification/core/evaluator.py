"""Evaluation metrics and utilities."""

from dataclasses import dataclass
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


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference to get labels and predictions."""
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["mri"].to(device)
            y = batch["label"].to(device)

            out = model(x).sigmoid().detach()
            all_labels.append(y.flatten())
            all_preds.append(out.flatten())

    if not all_preds:
        return torch.tensor([]), torch.tensor([])

    return torch.cat(all_labels), torch.cat(all_preds)


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
