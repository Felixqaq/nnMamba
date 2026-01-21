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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Metrics:
    """Evaluate model on a dataloader.

    Args:
        model: PyTorch model
        dataloader: Validation/test dataloader
        device: Target device
        threshold: Classification threshold

    Returns:
        Metrics dataclass with accuracy, sensitivity, specificity, auc
    """
    model.eval()

    auroc = AUROC(task="binary").to(device)
    accuracy_metric = Accuracy(task="binary", threshold=threshold).to(device)
    specificity_metric = Specificity(task="binary", threshold=threshold).to(device)
    sensitivity_metric = Recall(task="binary", threshold=threshold).to(device)

    all_labels = torch.tensor([]).to(device)
    all_preds = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch in dataloader:
            x = batch["mri"].to(device)
            y = batch["label"].to(device)

            out = model(x).sigmoid().detach()
            all_labels = torch.cat((all_labels, y.flatten()), 0)
            all_preds = torch.cat((all_preds, out.flatten()), 0)

    if len(all_preds) == 0:
        return Metrics(0.0, 0.0, 0.0, 0.5)

    unique_labels = torch.unique(all_labels)
    auc = 0.5 if len(unique_labels) < 2 else auroc(all_preds, all_labels).item()

    return Metrics(
        accuracy=round(accuracy_metric(all_preds, all_labels).item(), 5),
        sensitivity=round(sensitivity_metric(all_preds, all_labels).item(), 5),
        specificity=round(specificity_metric(all_preds, all_labels).item(), 5),
        auc=round(auc, 5),
        labels=all_labels.cpu(),
        preds=all_preds.cpu(),
    )
