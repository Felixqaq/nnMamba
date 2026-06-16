"""Training visualization utilities for nnMamba regression/classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLASSIFICATION_TASK_TYPES = {
    "gold",
    "gold_severity4",
    "angle_3class",
    "angle_binary_extreme",
    "oi_emphysema",
}


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _with_method(title: str, method_label: str | None) -> str:
    return f"{title}\nMethod: {method_label}" if method_label else title


def _method_note(method_label: str | None) -> str:
    return f"Method: {method_label}\n" if method_label else ""


def _regression_axis_label(task_type: str) -> str:
    """Return the user-facing label for regression target plots."""
    return "OI" if task_type == "oi" else "Angle (degrees)"


def _confusion_tick_labels(class_names: list[str], num_classes: int) -> list[str]:
    """Return compact tick labels for known classification tasks."""
    tick_labels = class_names if class_names else [str(i) for i in range(num_classes)]
    compact_labels = []
    for label in tick_labels:
        if label.startswith("Emphysema/Abnormal"):
            compact_labels.append("Abnormal")
        elif label.startswith("Intermediate"):
            compact_labels.append("Intermediate")
        elif label.startswith("Normal"):
            compact_labels.append("Normal")
        else:
            compact_labels.append(label)
    return compact_labels


def plot_training_curves(
    train_loss: list[float],
    test_metrics: dict[str, list[float]],
    train_metrics: dict[str, list[float]],
    eval_epochs: list[int],
    save_dir: Path,
    uuid: str,
    fold: int,
    task_type: str = "angle",
    method_label: str | None = None,
) -> None:
    """Generate and save task-aware training plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)

    _plot_single(
        epochs,
        train_loss,
        "Training Loss",
        "Epoch",
        "Loss",
        save_dir / f"fold{fold}_loss.png",
        color="#1f77b4",
        method_label=method_label,
    )

    if not eval_epochs:
        return

    if task_type in CLASSIFICATION_TASK_TYPES:
        metric_specs = [
            ("accuracy", "Accuracy", "Accuracy", "#2ca02c"),
            ("macro_f1", "Macro F1", "Macro F1", "#d62728"),
            ("balanced_accuracy", "Balanced Accuracy", "Balanced Accuracy", "#9467bd"),
            ("macro_recall", "Macro Recall", "Macro Recall", "#ff7f0e"),
            ("sensitivity", "Sensitivity", "Sensitivity", "#17becf"),
            ("specificity", "Specificity", "Specificity", "#bcbd22"),
        ]
        for key, title, ylabel, color in metric_specs:
            _plot_comparison(
                eval_epochs,
                train_metrics.get(key, []),
                test_metrics.get(key, []),
                f"Train {title}",
                f"Val {title}",
                f"{title} Curve (Train vs Val)",
                save_dir / f"fold{fold}_{key}.png",
                ylabel=ylabel,
                colors=(color, "#8c564b"),
                method_label=method_label,
            )
        _plot_classification_summary(
            epochs,
            train_loss,
            eval_epochs,
            test_metrics,
            uuid,
            fold,
            save_dir,
            method_label=method_label,
        )
        return

    metric_specs = [
        ("mae", "MAE", "Mean Absolute Error", "#2ca02c"),
        ("rmse", "RMSE", "Root Mean Squared Error", "#d62728"),
        ("r2", "R2", "R2 Score", "#9467bd"),
        ("pearson", "Pearson", "Pearson Correlation", "#ff7f0e"),
        ("mean_error", "Mean Error", "Prediction Bias", "#17becf"),
    ]
    for key, title, ylabel, color in metric_specs:
        _plot_comparison(
            eval_epochs,
            train_metrics.get(key, []),
            test_metrics.get(key, []),
            f"Train {title}",
            f"Test {title}",
            f"{title} Curve (Train vs Test)",
            save_dir / f"fold{fold}_{key}.png",
            ylabel=ylabel,
            colors=(color, "#8c564b"),
            method_label=method_label,
        )
    _plot_regression_summary(
        epochs,
        train_loss,
        eval_epochs,
        test_metrics,
        uuid,
        fold,
        save_dir,
        method_label=method_label,
    )


def _plot_single(
    x: range,
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str = "#1f77b4",
    method_label: str | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(list(x), y, linewidth=2.2, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(_with_method(title, method_label), fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_comparison(
    x: list[int],
    y1: list[float],
    y2: list[float],
    label1: str,
    label2: str,
    title: str,
    path: Path,
    ylabel: str = "Score",
    colors: tuple[str, str] = ("#2ca02c", "#d62728"),
    method_label: str | None = None,
) -> None:
    if not y1 or not y2:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker="o", linewidth=2, label=label1, color=colors[0])
    plt.plot(x, y2, marker="s", linewidth=2, label=label2, color=colors[1])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(_with_method(title, method_label), fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_regression_summary(
    epochs: range,
    train_loss: list[float],
    eval_epochs: list[int],
    test_metrics: dict[str, list[float]],
    uuid: str,
    fold: int,
    save_dir: Path,
    method_label: str | None = None,
) -> None:
    metrics = [
        ("mae", "Test MAE", "#2ca02c"),
        ("rmse", "Test RMSE", "#d62728"),
        ("r2", "Test R2", "#9467bd"),
        ("pearson", "Test Pearson", "#ff7f0e"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].plot(list(epochs), train_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    for ax, (key, title, color) in zip(axes[1:], metrics):
        values = test_metrics.get(key, [])
        if values:
            ax.plot(eval_epochs, values, marker="o", linewidth=2, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)

    axes[4].set_ylabel("Score")
    axes[4].set_ylim(-1.05, 1.05)
    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.88,
        f"{_method_note(method_label)}UUID: {uuid}\nFold: {fold}\n\n"
        "Tracked:\nMAE, RMSE, R2,\nPearson, Mean Error",
        transform=axes[5].transAxes,
        fontsize=11,
        va="top",
        family="serif",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_summary.png", dpi=300)
    plt.close(fig)


def _plot_classification_summary(
    epochs: range,
    train_loss: list[float],
    eval_epochs: list[int],
    test_metrics: dict[str, list[float]],
    uuid: str,
    fold: int,
    save_dir: Path,
    method_label: str | None = None,
) -> None:
    metrics = [
        ("accuracy", "Val Accuracy", "#2ca02c"),
        ("macro_f1", "Val Macro F1", "#d62728"),
        ("balanced_accuracy", "Val Balanced Accuracy", "#9467bd"),
        ("macro_recall", "Val Macro Recall", "#ff7f0e"),
        ("sensitivity", "Val Sensitivity", "#17becf"),
        ("specificity", "Val Specificity", "#bcbd22"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    axes = axes.ravel()
    axes[0].plot(list(epochs), train_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    for ax, (key, title, color) in zip(axes[1:], metrics):
        values = test_metrics.get(key, [])
        if values:
            ax.plot(eval_epochs, values, marker="o", linewidth=2, color=color)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.25)

    axes[7].axis("off")
    axes[7].text(
        0.02,
        0.88,
        f"{_method_note(method_label)}UUID: {uuid}\nFold: {fold}\n\n"
        "Tracked:\nAccuracy,\nMacro F1,\nBalanced Accuracy,\nMacro Recall"
        "\nSensitivity,\nSpecificity",
        transform=axes[7].transAxes,
        fontsize=11,
        va="top",
        family="serif",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_summary.png", dpi=300)
    plt.close(fig)


def _to_numpy(values: torch.Tensor | np.ndarray | list[float]) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().cpu().numpy().astype(float)
    return np.asarray(values, dtype=float)


def _regression_stats(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    errors = preds - labels
    return {
        "mae": float(np.mean(np.abs(errors))) if errors.size else 0.0,
        "rmse": float(np.sqrt(np.mean(errors**2))) if errors.size else 0.0,
        "mean_error": float(np.mean(errors)) if errors.size else 0.0,
        "r2": _r2_score(labels, preds),
        "pearson": _pearson_score(labels, preds),
    }


def _r2_score(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    ss_res = float(np.sum((preds - labels) ** 2))
    ss_tot = float(np.sum((labels - labels.mean()) ** 2))
    return 0.0 if np.isclose(ss_tot, 0.0) else 1.0 - ss_res / ss_tot


def _pearson_score(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size < 2:
        return 0.0
    if np.isclose(np.std(labels), 0.0) or np.isclose(np.std(preds), 0.0):
        return 0.0
    return float(np.corrcoef(labels, preds)[0, 1])


def plot_prediction_scatter(
    labels: torch.Tensor | np.ndarray | list[float],
    preds: torch.Tensor | np.ndarray | list[float],
    fold: int,
    save_dir: Path,
    title_suffix: str = "",
    task_type: str = "angle",
) -> None:
    """Plot predicted vs. true regression targets."""
    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    stats = _regression_stats(labels_np, preds_np)
    axis_label = _regression_axis_label(task_type)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        labels_np,
        preds_np,
        s=38,
        alpha=0.8,
        color="#1f77b4",
        edgecolors="white",
        linewidths=0.5,
    )
    if labels_np.size:
        lo = float(min(labels_np.min(), preds_np.min()))
        hi = float(max(labels_np.max(), preds_np.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5)
        slope, intercept = (
            np.polyfit(labels_np, preds_np, 1) if labels_np.size >= 2 else (1.0, 0.0)
        )
        xs = np.linspace(lo, hi, 100)
        plt.plot(xs, slope * xs + intercept, color="#d62728", linewidth=2)

    plt.xlabel(f"True {axis_label}")
    plt.ylabel(f"Predicted {axis_label}")
    plt.title(f"Prediction Scatter - Fold {fold}{title_suffix}", fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.text(
        0.03,
        0.97,
        f"MAE={stats['mae']:.3f}\nRMSE={stats['rmse']:.3f}\nR2={stats['r2']:.3f}\n"
        f"Pearson={stats['pearson']:.3f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_scatter.png", dpi=300)
    plt.close()


def plot_residuals(
    labels: torch.Tensor | np.ndarray | list[float],
    preds: torch.Tensor | np.ndarray | list[float],
    fold: int,
    save_dir: Path,
    title_suffix: str = "",
    task_type: str = "angle",
) -> None:
    """Plot residuals against true values."""
    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    residuals = preds_np - labels_np
    axis_label = _regression_axis_label(task_type)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        labels_np,
        residuals,
        s=38,
        alpha=0.8,
        color="#9467bd",
        edgecolors="white",
        linewidths=0.5,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel(f"True {axis_label}")
    plt.ylabel("Residual (Pred - True)")
    plt.title(f"Residual Plot - Fold {fold}{title_suffix}", fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_residuals.png", dpi=300)
    plt.close()


def plot_error_histogram(
    labels: torch.Tensor | np.ndarray | list[float],
    preds: torch.Tensor | np.ndarray | list[float],
    fold: int,
    save_dir: Path,
    title_suffix: str = "",
) -> None:
    """Plot histogram of prediction errors."""
    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    errors = preds_np - labels_np
    mean_error = float(np.mean(errors)) if errors.size else 0.0
    std_error = float(np.std(errors)) if errors.size else 0.0

    plt.figure(figsize=(8, 6))
    plt.hist(
        errors,
        bins=min(12, max(5, len(errors) // 2 if len(errors) > 0 else 5)),
        color="#17becf",
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(mean_error, color="red", linestyle="-", linewidth=2)
    plt.axvline(mean_error + std_error, color="gray", linestyle="--", linewidth=1.5)
    plt.axvline(mean_error - std_error, color="gray", linestyle="--", linewidth=1.5)
    plt.xlabel("Prediction Error (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"Error Histogram - Fold {fold}{title_suffix}", fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_error_hist.png", dpi=300)
    plt.close()


def plot_bland_altman(
    labels: torch.Tensor | np.ndarray | list[float],
    preds: torch.Tensor | np.ndarray | list[float],
    fold: int,
    save_dir: Path,
    title_suffix: str = "",
    task_type: str = "angle",
) -> None:
    """Plot Bland-Altman diagram."""
    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    axis_label = _regression_axis_label(task_type)
    means = (labels_np + preds_np) / 2.0
    diffs = preds_np - labels_np
    bias = float(np.mean(diffs)) if diffs.size else 0.0
    sd = float(np.std(diffs)) if diffs.size else 0.0
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    plt.figure(figsize=(8, 6))
    plt.scatter(
        means,
        diffs,
        s=38,
        alpha=0.8,
        color="#8c564b",
        edgecolors="white",
        linewidths=0.5,
    )
    plt.axhline(bias, color="red", linestyle="-", linewidth=2)
    plt.axhline(upper, color="black", linestyle="--", linewidth=1.5)
    plt.axhline(lower, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel(f"Mean of True and Predicted {axis_label}")
    plt.ylabel("Prediction Difference (Pred - True)")
    plt.title(f"Bland-Altman - Fold {fold}{title_suffix}", fontweight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_bland_altman.png", dpi=300)
    plt.close()


def plot_confusion_matrix(
    labels: torch.Tensor | np.ndarray | list[int],
    preds: torch.Tensor | np.ndarray | list[int],
    fold: int,
    save_dir: Path,
    class_names: list[str],
    title_suffix: str = "",
    output_name: str | None = None,
    method_label: str | None = None,
) -> None:
    """Plot confusion matrix for multiclass classification."""
    labels_np = np.asarray(_to_numpy(labels), dtype=int)
    preds_np = np.asarray(_to_numpy(preds), dtype=int)
    num_classes = max(len(class_names), 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(labels_np, preds_np):
        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            cm[true_idx, pred_idx] += 1

    tick_labels = _confusion_tick_labels(class_names, num_classes)
    title = _with_method(f"Confusion Matrix - Fold {fold}{title_suffix}", method_label)
    filename = output_name or f"fold{fold}_confusion_matrix.png"

    with plt.rc_context({"font.family": "sans-serif", "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            cbar_kws={"label": "Count"},
            annot_kws={"size": 16},
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")

        fig.tight_layout()
        fig.savefig(save_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_paper_results(
    test_metrics,
    fold: int,
    save_dir: Path,
    task_type: str = "angle",
    class_names: list[str] | None = None,
    method_label: str | None = None,
) -> None:
    """Generate task-specific detailed plots for a fold."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if getattr(test_metrics, "labels", None) is None:
        return

    if task_type in CLASSIFICATION_TASK_TYPES:
        if getattr(test_metrics, "preds", None) is None:
            return
        plot_confusion_matrix(
            test_metrics.labels,
            test_metrics.preds,
            fold,
            save_dir,
            class_names or [],
            method_label=method_label,
        )
        return

    if getattr(test_metrics, "preds", None) is None:
        return
    plot_prediction_scatter(
        test_metrics.labels,
        test_metrics.preds,
        fold,
        save_dir,
        task_type=task_type,
    )
    plot_residuals(
        test_metrics.labels,
        test_metrics.preds,
        fold,
        save_dir,
        task_type=task_type,
    )
    plot_error_histogram(test_metrics.labels, test_metrics.preds, fold, save_dir)
    plot_bland_altman(
        test_metrics.labels,
        test_metrics.preds,
        fold,
        save_dir,
        task_type=task_type,
    )


def _aggregate_regression_results(all_results: list) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels_list = []
    preds_list = []
    metric_map: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "r2": [],
        "pearson": [],
        "mean_error": [],
    }
    for res in all_results:
        if getattr(res, "labels", None) is not None and getattr(res, "preds", None) is not None:
            labels_list.append(_to_numpy(res.labels))
            preds_list.append(_to_numpy(res.preds))
        for key in metric_map:
            value = getattr(res, key, None)
            if value is not None:
                metric_map[key].append(float(value))

    labels = np.concatenate(labels_list) if labels_list else np.array([], dtype=float)
    preds = np.concatenate(preds_list) if preds_list else np.array([], dtype=float)
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metric_map.items()}
    return labels, preds, metric_arrays


def _aggregate_classification_results(
    all_results: list,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels_list = []
    preds_list = []
    metric_map: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "balanced_accuracy": [],
        "sensitivity": [],
        "specificity": [],
    }
    for res in all_results:
        if getattr(res, "labels", None) is not None and getattr(res, "preds", None) is not None:
            labels_list.append(np.asarray(_to_numpy(res.labels), dtype=int))
            preds_list.append(np.asarray(_to_numpy(res.preds), dtype=int))
        for key in metric_map:
            value = getattr(res, key, None)
            if value is not None:
                metric_map[key].append(float(value))

    labels = np.concatenate(labels_list) if labels_list else np.array([], dtype=int)
    preds = np.concatenate(preds_list) if preds_list else np.array([], dtype=int)
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metric_map.items()}
    return labels, preds, metric_arrays


def plot_metric_boxplot(
    all_results: list,
    save_dir: Path,
    method_label: str | None = None,
) -> None:
    """Plot fold-wise regression metric distributions."""
    _, _, metric_arrays = _aggregate_regression_results(all_results)
    metrics = ["mae", "rmse", "r2", "pearson", "mean_error"]
    metric_labels = ["MAE", "RMSE", "R2", "Pearson", "Mean Error"]
    data = [metric_arrays[m] for m in metrics if metric_arrays[m].size > 0]
    labels = [
        label
        for label, metric in zip(metric_labels, metrics)
        if metric_arrays[metric].size > 0
    ]
    if not data:
        return

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, widths=0.55)
    palette = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, values in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(x, values, color="black", s=12, alpha=0.55, zorder=3)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Score")
    plt.title(
        _with_method("Fold-wise Regression Metrics", method_label),
        fontweight="bold",
    )
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "metric_boxplot.png", dpi=300)
    plt.close()


def plot_classification_metric_boxplot(
    all_results: list,
    save_dir: Path,
    method_label: str | None = None,
) -> None:
    """Plot fold-wise classification metric distributions."""
    _, _, metric_arrays = _aggregate_classification_results(all_results)
    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "sensitivity",
        "specificity",
    ]
    metric_labels = [
        "Accuracy",
        "Macro F1",
        "Balanced Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Sensitivity",
        "Specificity",
    ]
    data = [metric_arrays[m] for m in metrics if metric_arrays[m].size > 0]
    labels = [
        label
        for label, metric in zip(metric_labels, metrics)
        if metric_arrays[metric].size > 0
    ]
    if not data:
        return

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True, widths=0.55)
    palette = [
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#8c564b",
        "#bcbd22",
    ]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, values in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(values))
        plt.scatter(x, values, color="black", s=12, alpha=0.55, zorder=3)

    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Score")
    plt.title(
        _with_method("Fold-wise Classification Metrics", method_label),
        fontweight="bold",
    )
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "metric_boxplot.png", dpi=300)
    plt.close()


def _plot_metric_barplot(
    metric_arrays: dict[str, np.ndarray],
    metric_specs: list[tuple[str, str, str]],
    save_dir: Path,
    title: str,
    method_label: str | None = None,
) -> None:
    """Plot mean and cross-fold std bars for available metrics."""
    labels = []
    means = []
    stds = []
    colors = []
    for key, label, color in metric_specs:
        values = metric_arrays.get(key, np.array([], dtype=float))
        if values.size == 0:
            continue
        labels.append(label)
        means.append(float(values.mean()))
        stds.append(float(values.std()))
        colors.append(color)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(labels)), 5.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.78,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(_with_method(title, method_label), fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean,
            f"{mean:.3f}\n+/-{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_dir / "metric_barplot.png", dpi=300)
    plt.close(fig)


def plot_global_summary(
    all_results: list,
    save_dir: Path,
    task_type: str = "angle",
    class_names: list[str] | None = None,
    method_label: str | None = None,
) -> None:
    """Generate aggregate plots for all folds."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if task_type in CLASSIFICATION_TASK_TYPES:
        labels, preds, metric_arrays = _aggregate_classification_results(all_results)
        if labels.size > 0 and preds.size > 0:
            plot_confusion_matrix(
                labels,
                preds,
                fold=0,
                save_dir=save_dir,
                class_names=class_names or [],
                title_suffix=" (All Folds)",
                output_name="total_confusion_matrix.png",
                method_label=method_label,
            )
        plot_classification_metric_boxplot(
            all_results,
            save_dir,
            method_label=method_label,
        )
        _plot_metric_barplot(
            metric_arrays,
            [
                ("accuracy", "Accuracy", "#2ca02c"),
                ("macro_f1", "Macro F1", "#d62728"),
                ("balanced_accuracy", "Balanced Acc", "#9467bd"),
                ("macro_precision", "Macro Precision", "#ff7f0e"),
                ("macro_recall", "Macro Recall", "#17becf"),
                ("sensitivity", "Sensitivity", "#8c564b"),
                ("specificity", "Specificity", "#bcbd22"),
            ],
            save_dir,
            "Classification Metrics (Mean +/- Std)",
            method_label=method_label,
        )
        return

    labels, preds, metric_arrays = _aggregate_regression_results(all_results)
    if labels.size > 0 and preds.size > 0:
        plot_prediction_scatter(
            labels,
            preds,
            fold=0,
            save_dir=save_dir,
            title_suffix=" (All Folds)",
            task_type=task_type,
        )
        plot_residuals(
            labels,
            preds,
            fold=0,
            save_dir=save_dir,
            title_suffix=" (All Folds)",
            task_type=task_type,
        )
        plot_error_histogram(labels, preds, fold=0, save_dir=save_dir, title_suffix=" (All Folds)")
        plot_bland_altman(
            labels,
            preds,
            fold=0,
            save_dir=save_dir,
            title_suffix=" (All Folds)",
            task_type=task_type,
        )
        for src_name, dst_name in [
            ("fold0_scatter.png", "total_scatter.png"),
            ("fold0_residuals.png", "total_residuals.png"),
            ("fold0_error_hist.png", "total_error_hist.png"),
            ("fold0_bland_altman.png", "total_bland_altman.png"),
        ]:
            src = save_dir / src_name
            dst = save_dir / dst_name
            if src.exists():
                src.replace(dst)

    plot_metric_boxplot(all_results, save_dir, method_label=method_label)
    _plot_metric_barplot(
        metric_arrays,
        [
            ("mae", "MAE", "#2ca02c"),
            ("rmse", "RMSE", "#d62728"),
            ("r2", "R2", "#9467bd"),
            ("pearson", "Pearson", "#ff7f0e"),
            ("mean_error", "Mean Error", "#17becf"),
        ],
        save_dir,
        "Regression Metrics (Mean +/- Std)",
        method_label=method_label,
    )
