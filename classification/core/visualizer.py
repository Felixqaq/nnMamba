"""Training visualization utilities."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import ROC, ConfusionMatrix, PrecisionRecallCurve
import torch


def plot_training_curves(
    train_loss: list[float],
    test_metrics: dict[str, list[float]],
    train_metrics: dict[str, list[float]],
    eval_epochs: list[int],
    save_dir: Path,
    uuid: str,
    fold: int,
) -> None:
    """Generate and save all training visualization plots.

    Args:
        train_loss: Loss values per epoch
        test_metrics: Dict with 'acc', 'auc', 'sens', 'spec' lists
        train_metrics: Dict with 'acc', 'auc' lists
        eval_epochs: Epochs where evaluation was performed
        save_dir: Directory to save figures
        uuid: Model identifier
        fold: Current fold number
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)

    # Loss curve
    _plot_single(
        epochs,
        train_loss,
        "Training Loss",
        "Epoch",
        "Loss",
        save_dir / f"fold{fold}_loss.png",
    )

    if not eval_epochs:
        return

    # AUC curve (train vs test)
    _plot_comparison(
        eval_epochs,
        train_metrics["auc"],
        test_metrics["auc"],
        "Train AUC",
        "Test AUC",
        "AUC Curve (Train vs Test)",
        save_dir / f"fold{fold}_auc.png",
    )

    # Accuracy curve
    _plot_comparison(
        eval_epochs,
        train_metrics["acc"],
        test_metrics["acc"],
        "Train Accuracy",
        "Test Accuracy",
        "Accuracy Curve (Train vs Test)",
        save_dir / f"fold{fold}_accuracy.png",
    )

    # Sensitivity & Specificity
    _plot_comparison(
        eval_epochs,
        test_metrics["sens"],
        test_metrics["spec"],
        "Sensitivity",
        "Specificity",
        "Sensitivity & Specificity",
        save_dir / f"fold{fold}_sens_spec.png",
        colors=("blue", "magenta"),
    )

    # Summary plot
    _plot_summary(epochs, train_loss, eval_epochs, test_metrics, uuid, fold, save_dir)


def _plot_single(
    x: range, y: list, title: str, xlabel: str, ylabel: str, path: Path
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, "b-", linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_comparison(
    x: list,
    y1: list,
    y2: list,
    label1: str,
    label2: str,
    title: str,
    path: Path,
    colors: tuple = ("green", "red"),
) -> None:
    if not y1 or not y2:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, f"{colors[0][0]}-o", linewidth=2, label=label1, markersize=6)
    plt.plot(x, y2, f"{colors[1][0]}-s", linewidth=2, label=label2, markersize=6)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_summary(
    epochs: range,
    train_loss: list,
    eval_epochs: list,
    test_metrics: dict,
    uuid: str,
    fold: int,
    save_dir: Path,
) -> None:
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss, "b-", linewidth=2)
    plt.title("Training Loss", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(eval_epochs, test_metrics["auc"], "r-o", linewidth=2)
    plt.title("Test AUC", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(eval_epochs, test_metrics["acc"], "g-s", linewidth=2)
    plt.title("Test Accuracy", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(eval_epochs, test_metrics["sens"], "b-o", linewidth=2, label="Sensitivity")
    plt.plot(eval_epochs, test_metrics["spec"], "m-s", linewidth=2, label="Specificity")
    plt.title("Sensitivity & Specificity", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_summary.png", dpi=300)
    plt.close()


def plot_roc_curve(
    labels: torch.Tensor,
    preds: torch.Tensor,
    auc_value: float,
    fold: int,
    save_dir: Path,
) -> None:
    """Plot and save ROC curve."""
    roc = ROC(task="binary")
    labels = labels.to(torch.long)
    fpr, tpr, thresholds = roc(preds, labels)

    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr.numpy(),
        tpr.numpy(),
        color="darkorange",
        lw=2,
        label=f"ROC (AUC = {auc_value:.4f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - Fold {fold}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_roc.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    labels: torch.Tensor,
    preds: torch.Tensor,
    fold: int,
    save_dir: Path,
    class_names: list[str] = ["Normal", "Abnormal"],
) -> None:
    """Plot and save confusion matrix."""
    cm_metric = ConfusionMatrix(task="binary")
    labels = labels.to(torch.long)
    cm = cm_metric(preds, labels).numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {fold}", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)

    # Normalize and annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_cm.png", dpi=300)
    plt.close()


def plot_pr_curve(
    labels: torch.Tensor,
    preds: torch.Tensor,
    fold: int,
    save_dir: Path,
) -> None:
    """Plot and save Precision-Recall curve."""
    pr_curve = PrecisionRecallCurve(task="binary")
    labels = labels.to(torch.long)
    precision, recall, thresholds = pr_curve(preds, labels)

    plt.figure(figsize=(8, 8))
    plt.plot(
        recall.numpy(), precision.numpy(), color="darkgreen", lw=2, label="PR Curve"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve - Fold {fold}", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_pr.png", dpi=300)
    plt.close()


def plot_paper_results(
    test_metrics,
    fold: int,
    save_dir: Path,
    class_names: list[str] = ["Normal", "Abnormal"],
) -> None:
    """Generate all advanced paper plots for a specific state."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if test_metrics.labels is not None and test_metrics.preds is not None:
        plot_roc_curve(
            test_metrics.labels, test_metrics.preds, test_metrics.auc, fold, save_dir
        )
        plot_confusion_matrix(
            test_metrics.labels, test_metrics.preds, fold, save_dir, class_names
        )
        plot_pr_curve(test_metrics.labels, test_metrics.preds, fold, save_dir)


def plot_combined_roc(
    all_labels: list[torch.Tensor],
    all_preds: list[torch.Tensor],
    save_dir: Path,
) -> None:
    """Plot combined ROC curve for all folds with mean and std shade."""
    plt.figure(figsize=(10, 10))

    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    aucs = []

    for i, (labels, preds) in enumerate(zip(all_labels, all_preds)):
        labels = labels.to(torch.long)
        roc = ROC(task="binary")
        fpr, tpr, _ = roc(preds, labels)
        auc_val = torch.trapz(tpr, fpr).item()
        aucs.append(auc_val)

        # Interpolate TPR to fixed FPR scale for averaging
        tpr_interp = np.interp(base_fpr, fpr.numpy(), tpr.numpy())
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        plt.plot(
            fpr.numpy(),
            tpr.numpy(),
            lw=1,
            alpha=0.3,
            label=f"Fold {i + 1} (AUC = {auc_val:.4f})",
        )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(
        base_fpr,
        mean_tpr,
        color="b",
        label=f"Mean ROC (AUC = {mean_auc:.4f} \u00b1 {std_auc:.4f})",
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        base_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label="\u00b1 1 std. dev.",
    )

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", alpha=0.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Combined ROC - All Folds", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "total_roc.png", dpi=300)
    plt.close()


def plot_combined_pr(
    all_labels: list[torch.Tensor],
    all_preds: list[torch.Tensor],
    save_dir: Path,
) -> None:
    """Plot combined Precision-Recall curve for all folds with mean and std shade."""
    plt.figure(figsize=(10, 10))

    precisions = []
    base_recall = np.linspace(0, 1, 101)

    for i, (labels, preds) in enumerate(zip(all_labels, all_preds)):
        labels = labels.to(torch.long)
        pr_curve = PrecisionRecallCurve(task="binary")
        precision, recall, _ = pr_curve(preds, labels)

        # Flip to make recall increasing for interpolation
        # Note: PrecisionRecallCurve returns recall in descending order usually
        r_sorted_idx = np.argsort(recall.numpy())
        r_sorted = recall.numpy()[r_sorted_idx]
        p_sorted = precision.numpy()[r_sorted_idx]

        p_interp = np.interp(base_recall, r_sorted, p_sorted)
        precisions.append(p_interp)

        plt.plot(
            recall.numpy(), precision.numpy(), lw=1, alpha=0.3, label=f"Fold {i + 1}"
        )

    mean_precision = np.mean(precisions, axis=0)
    plt.plot(
        base_recall, mean_precision, color="g", label="Mean PR Curve", lw=2, alpha=0.8
    )

    std_precision = np.std(precisions, axis=0)
    p_upper = np.minimum(mean_precision + std_precision, 1)
    p_lower = np.maximum(mean_precision - std_precision, 0)
    plt.fill_between(
        base_recall,
        p_lower,
        p_upper,
        color="grey",
        alpha=0.2,
        label="\u00b1 1 std. dev.",
    )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Combined PR Curve - All Folds", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "total_pr.png", dpi=300)
    plt.close()


def plot_global_summary(
    all_results: list,
    save_dir: Path,
) -> None:
    """Generate aggregate plots for all folds."""
    save_dir.mkdir(parents=True, exist_ok=True)

    all_labels = [r.labels for r in all_results if r.labels is not None]
    all_preds = [r.preds for r in all_results if r.preds is not None]

    if len(all_labels) > 0:
        plot_combined_roc(all_labels, all_preds, save_dir)
        plot_combined_pr(all_labels, all_preds, save_dir)
