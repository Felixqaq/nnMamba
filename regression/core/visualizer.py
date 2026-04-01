"""Training visualization utilities for nnMamba regression."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def plot_training_curves(
    train_loss: list[float],
    test_metrics: dict[str, list[float]],
    train_metrics: dict[str, list[float]],
    eval_epochs: list[int],
    save_dir: Path,
    uuid: str,
    fold: int,
) -> None:
    """Generate and save regression training plots."""

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
    )

    if not eval_epochs:
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
        )

    _plot_summary(epochs, train_loss, eval_epochs, test_metrics, uuid, fold, save_dir)


def _plot_single(
    x: range,
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str = "#1f77b4",
) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(list(x), y, linewidth=2.2, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
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
) -> None:
    if not y1 or not y2:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker="o", linewidth=2, label=label1, color=colors[0])
    plt.plot(x, y2, marker="s", linewidth=2, label=label2, color=colors[1])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_summary(
    epochs: range,
    train_loss: list[float],
    eval_epochs: list[int],
    test_metrics: dict[str, list[float]],
    uuid: str,
    fold: int,
    save_dir: Path,
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
        f"UUID: {uuid}\nFold: {fold}\n\n"
        f"Key metrics tracked:\n"
        f"MAE, RMSE, R2, Pearson, Mean Error",
        transform=axes[5].transAxes,
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
) -> None:
    """Plot predicted vs. true angles."""

    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    stats = _regression_stats(labels_np, preds_np)

    plt.figure(figsize=(8, 8))
    plt.scatter(labels_np, preds_np, s=38, alpha=0.8, color="#1f77b4", edgecolors="white", linewidths=0.5)

    if labels_np.size:
        lo = float(min(labels_np.min(), preds_np.min()))
        hi = float(max(labels_np.max(), preds_np.max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.5, label="Identity")
        slope, intercept = np.polyfit(labels_np, preds_np, 1) if labels_np.size >= 2 else (1.0, 0.0)
        xs = np.linspace(lo, hi, 100)
        plt.plot(xs, slope * xs + intercept, color="#d62728", linewidth=2, label="Fit line")

    plt.xlabel("True Angle (degrees)")
    plt.ylabel("Predicted Angle (degrees)")
    plt.title(f"Prediction Scatter - Fold {fold}{title_suffix}", fontweight="bold")
    plt.legend()
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
) -> None:
    """Plot residuals against true values."""

    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    residuals = preds_np - labels_np

    plt.figure(figsize=(8, 6))
    plt.scatter(labels_np, residuals, s=38, alpha=0.8, color="#9467bd", edgecolors="white", linewidths=0.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("True Angle (degrees)")
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
    plt.hist(errors, bins=min(12, max(5, len(errors) // 2 if len(errors) > 0 else 5)), color="#17becf", edgecolor="black", alpha=0.85)
    plt.axvline(mean_error, color="red", linestyle="-", linewidth=2, label=f"Mean Error = {mean_error:.3f}")
    plt.axvline(mean_error + std_error, color="gray", linestyle="--", linewidth=1.5, label=f"±1 SD = {std_error:.3f}")
    plt.axvline(mean_error - std_error, color="gray", linestyle="--", linewidth=1.5)
    plt.xlabel("Prediction Error (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"Error Histogram - Fold {fold}{title_suffix}", fontweight="bold")
    plt.legend()
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
) -> None:
    """Plot Bland-Altman diagram."""

    labels_np = _to_numpy(labels)
    preds_np = _to_numpy(preds)
    means = (labels_np + preds_np) / 2.0
    diffs = preds_np - labels_np
    bias = float(np.mean(diffs)) if diffs.size else 0.0
    sd = float(np.std(diffs)) if diffs.size else 0.0
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, s=38, alpha=0.8, color="#8c564b", edgecolors="white", linewidths=0.5)
    plt.axhline(bias, color="red", linestyle="-", linewidth=2, label=f"Bias = {bias:.3f}")
    plt.axhline(upper, color="black", linestyle="--", linewidth=1.5, label=f"+1.96 SD = {upper:.3f}")
    plt.axhline(lower, color="black", linestyle="--", linewidth=1.5, label=f"-1.96 SD = {lower:.3f}")
    plt.xlabel("Mean of True and Predicted Angle")
    plt.ylabel("Prediction Difference (Pred - True)")
    plt.title(f"Bland-Altman - Fold {fold}{title_suffix}", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / f"fold{fold}_bland_altman.png", dpi=300)
    plt.close()


def plot_paper_results(
    test_metrics,
    fold: int,
    save_dir: Path,
) -> None:
    """Generate all advanced regression plots for a fold."""

    save_dir.mkdir(parents=True, exist_ok=True)
    if test_metrics.labels is None or test_metrics.preds is None:
        return

    plot_prediction_scatter(test_metrics.labels, test_metrics.preds, fold, save_dir)
    plot_residuals(test_metrics.labels, test_metrics.preds, fold, save_dir)
    plot_error_histogram(test_metrics.labels, test_metrics.preds, fold, save_dir)
    plot_bland_altman(test_metrics.labels, test_metrics.preds, fold, save_dir)


def _aggregate_results(all_results: list) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
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


def plot_metric_boxplot(all_results: list, save_dir: Path) -> None:
    """Plot fold-wise metric distributions."""

    _, _, metric_arrays = _aggregate_results(all_results)
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
    plt.title("Fold-wise Regression Metrics", fontweight="bold")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "metric_boxplot.png", dpi=300)
    plt.close()


def plot_global_summary(
    all_results: list,
    save_dir: Path,
) -> None:
    """Generate aggregate plots for all folds."""

    save_dir.mkdir(parents=True, exist_ok=True)
    labels, preds, _ = _aggregate_results(all_results)

    if labels.size > 0 and preds.size > 0:
        plot_prediction_scatter(labels, preds, fold=0, save_dir=save_dir, title_suffix=" (All Folds)")
        plot_residuals(labels, preds, fold=0, save_dir=save_dir, title_suffix=" (All Folds)")
        plot_error_histogram(labels, preds, fold=0, save_dir=save_dir, title_suffix=" (All Folds)")
        plot_bland_altman(labels, preds, fold=0, save_dir=save_dir, title_suffix=" (All Folds)")

        # Rename the combined outputs to explicit global filenames.
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

    plot_metric_boxplot(all_results, save_dir)
