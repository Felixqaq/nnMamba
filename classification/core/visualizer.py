"""Training visualization utilities."""

from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


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
        save_dir / f"{uuid}_fold{fold}_loss.png",
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
        save_dir / f"{uuid}_fold{fold}_auc.png",
    )

    # Accuracy curve
    _plot_comparison(
        eval_epochs,
        train_metrics["acc"],
        test_metrics["acc"],
        "Train Accuracy",
        "Test Accuracy",
        "Accuracy Curve (Train vs Test)",
        save_dir / f"{uuid}_fold{fold}_accuracy.png",
    )

    # Sensitivity & Specificity
    _plot_comparison(
        eval_epochs,
        test_metrics["sens"],
        test_metrics["spec"],
        "Sensitivity",
        "Specificity",
        "Sensitivity & Specificity",
        save_dir / f"{uuid}_fold{fold}_sens_spec.png",
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
    plt.savefig(save_dir / f"{uuid}_fold{fold}_summary.png", dpi=300)
    plt.close()


def plot_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc_value: float,
    fold: int,
    save_path: Path,
) -> None:
    """Plot and save ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_value: Area under curve value
        fold: Current fold number
        save_path: Path to save figure
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "darkorange", lw=2, label=f"ROC curve (area = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend(loc="lower right")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(save_path / f"auc-fold{fold}-{timestamp}.png")
    plt.close()
