"""Training loop for nnMamba CT regression and GOLD classification."""

from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .checkpoints import generate_uuid, save_checkpoint
from .config import Config
from .evaluator import ClassificationMetrics, RegressionMetrics, evaluate, save_predictions
from .visualizer import plot_global_summary, plot_paper_results, plot_training_curves

ATTENTION_HEAVY_MODELS = {
    "hybrid",
    "hybrid_mamba_attention",
    "hybrid_mamba_attention_regressor",
    "mamba_hybrid",
    "swinunetr",
}


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_loss(name: str, is_classification: bool) -> nn.Module:
    """Build task-specific loss by name."""
    resolved = name
    if resolved == "auto":
        resolved = "cross_entropy" if is_classification else "smooth_l1"

    registry = {
        "smooth_l1": nn.SmoothL1Loss,
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "cross_entropy": nn.CrossEntropyLoss,
    }
    if resolved not in registry:
        raise ValueError(f"Unknown loss: {name}")
    return registry[resolved]()


class Trainer:
    """Handles model training with k-fold cross-validation."""

    def __init__(self, config: Config, model_factory, loader_helper):
        self.config = config
        self.model_factory = model_factory
        self.loader_helper = loader_helper
        self.best_results: list[RegressionMetrics | ClassificationMetrics | None] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.training.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer_label = "AdamW"
        self.run_started_at: datetime | None = None
        self.run_finished_at: datetime | None = None
        self.total_training_seconds: float | None = None
        self.task_type = str(config.data.target_mode)
        self.is_classification = config.is_classification_task()
        self.class_names = (
            loader_helper.get_class_names()
            if hasattr(loader_helper, "get_class_names")
            else []
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.device_id
        setup_seed(config.training.seed)

        self.uuid = (
            config.resume.uuid
            if config.resume.enabled and config.resume.uuid
            else generate_uuid(str(config.model.name))
        )

    def train(self) -> str:
        """Run full k-fold training and save per-fold/global summaries."""
        cfg = self.config
        start_fold = cfg.resume.start_fold if cfg.resume.enabled else 0
        self.run_started_at = datetime.now()
        started_perf = perf_counter()

        print(f"\n{'=' * 72}")
        print(f"Training: {self.uuid}")
        print(
            f"Model: {cfg.model.name} | Task: {cfg.task} | "
            f"Target mode: {self.task_type} | {cfg.training.k_folds} folds"
        )
        if self.is_classification:
            print(
                f"Classes: {cfg.model.num_classes} | "
                f"Labels: {', '.join(self.class_names) if self.class_names else 'n/a'}"
            )
        if hasattr(self.loader_helper, "batch_size") and hasattr(
            self.loader_helper, "val_batch_size"
        ):
            print(
                f"Train batch: {self.loader_helper.batch_size} | "
                f"Eval batch: {self.loader_helper.val_batch_size} | "
                f"AMP: {'on' if self.use_amp else 'off'}"
            )
        model_key = str(cfg.model.name).lower()
        if model_key == "swinunetr":
            print(
                f"Swin window: {cfg.model.window_size} | "
                f"Checkpointing: {'on' if cfg.model.use_checkpoint else 'off'}"
            )
        elif model_key in ATTENTION_HEAVY_MODELS:
            print(
                f"Hybrid attention: layers={cfg.model.attn_layers}, "
                f"heads={cfg.model.attn_heads}, "
                f"dropout={cfg.model.attn_dropout}"
            )
        print(
            f"Optimizer: {self.optimizer_label} | "
            f"Train metrics during eval: {'on' if cfg.training.track_train_metrics else 'off'}"
        )
        print(f"{'=' * 72}\n")

        for fold in range(start_fold, cfg.training.k_folds):
            print(f"\nFold {fold + 1}/{cfg.training.k_folds}")
            best_res = self._train_fold(fold)
            self.best_results.append(best_res)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"Fold {fold + 1} complete\n")

        fig_dir = self.config.paths.figures / self.config.task / self.uuid
        self.run_finished_at = datetime.now()
        self.total_training_seconds = perf_counter() - started_perf
        plot_global_summary(
            self.best_results,
            fig_dir,
            task_type=self.task_type,
            class_names=self.class_names,
        )
        self._save_results_json(fig_dir)
        return self.uuid

    def _train_fold(self, fold: int):
        """Train a single fold and return the best validation metrics."""
        cfg = self.config.training
        model = self.model_factory().to(self.device)

        train_dl = self.loader_helper.get_train_dl(fold)
        test_dl = self.loader_helper.get_test_dl(fold)
        target_mean, target_std = self.loader_helper.get_target_stats()
        eval_target_mean, eval_target_std = self._get_eval_target_stats(
            target_mean, target_std
        )

        optimizer = self._build_optimizer(model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if self.is_classification else "min",
            factor=0.5,
            patience=2,
        )
        loss_fn = build_loss(cfg.loss, self.is_classification)

        train_loss_history: list[float] = []
        eval_epochs: list[int] = []
        train_metrics = self._empty_metric_history()
        val_metrics = self._empty_metric_history()
        best_score = float("-inf")
        best_fold_result = None
        best_epoch = 0

        log_dir = self.config.paths.logs / self.config.task / self.uuid
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"fold{fold + 1}.log"

        with open(log_path, "a", encoding="utf-8") as log_file:
            for epoch in range(1, cfg.epochs + 1):
                epoch_loss = self._train_epoch(
                    model=model,
                    dataloader=train_dl,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    target_mean=target_mean,
                    target_std=target_std,
                )
                train_loss_history.append(epoch_loss)

                msg = f"Epoch {epoch}/{cfg.epochs}, loss={epoch_loss:.5f}"
                tqdm.write(msg)
                log_file.write(msg + "\n")

                if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                    eval_epochs.append(epoch)
                    train_result = None
                    if cfg.track_train_metrics:
                        train_result = evaluate(
                            model=model,
                            dataloader=train_dl,
                            device=self.device,
                            task_type=self.task_type,
                            target_mean=eval_target_mean,
                            target_std=eval_target_std,
                            use_amp=self.use_amp,
                            num_classes=int(self.config.model.num_classes),
                        )
                    val_result = evaluate(
                        model=model,
                        dataloader=test_dl,
                        device=self.device,
                        task_type=self.task_type,
                        target_mean=eval_target_mean,
                        target_std=eval_target_std,
                        use_amp=self.use_amp,
                        num_classes=int(self.config.model.num_classes),
                    )

                    invalid_train = (
                        train_result.num_invalid_samples
                        if train_result is not None
                        else 0
                    )
                    invalid_val = getattr(val_result, "num_invalid_samples", 0)
                    if invalid_train > 0 or invalid_val > 0:
                        warn_msg = (
                            f"Epoch {epoch}: invalid predictions detected "
                            f"(train={invalid_train}, val={invalid_val})"
                        )
                        tqdm.write(warn_msg)
                        log_file.write(warn_msg + "\n")

                    if train_result is not None:
                        self._append_metric_snapshot(train_metrics, train_result)
                    self._append_metric_snapshot(val_metrics, val_result)

                    current_score = self._selection_score(val_result)
                    scheduler_value = self._scheduler_score(val_result)
                    if math.isfinite(scheduler_value):
                        scheduler.step(scheduler_value)

                    metric_msg = self._format_metric_message(
                        epoch=epoch,
                        train_result=train_result,
                        val_result=val_result,
                    )
                    tqdm.write(metric_msg)
                    log_file.write(metric_msg + "\n")

                    if self._is_better(current_score, best_score):
                        best_score = current_score
                        best_epoch = epoch
                        best_fold_result = val_result

                        weight_path = self.config.paths.weights / self.config.task / self.uuid
                        save_checkpoint(
                            model=model,
                            path=weight_path,
                            fold=fold + 1,
                            epoch=epoch,
                            is_best=True,
                            extra=self._checkpoint_payload(
                                fold=fold + 1,
                                epoch=epoch,
                                eval_target_mean=eval_target_mean,
                                eval_target_std=eval_target_std,
                                metrics=val_result,
                            ),
                        )

                        fig_dir = self.config.paths.figures / self.config.task / self.uuid
                        plot_paper_results(
                            val_result,
                            fold + 1,
                            fig_dir,
                            task_type=self.task_type,
                            class_names=self.class_names,
                        )
                        save_predictions(
                            metrics=val_result,
                            dataset=self.loader_helper.dataset,
                            fold_indices=self.loader_helper.fold_indices[fold][1],
                            save_path=fig_dir,
                            fold=fold + 1,
                            task_type=self.task_type,
                            class_names=self.class_names,
                        )
                        tqdm.write(
                            self._best_metric_message(
                                fold=fold + 1,
                                epoch=epoch,
                                result=val_result,
                            )
                        )

                if epoch % cfg.save_interval == 0:
                    weight_path = self.config.paths.weights / self.config.task / self.uuid
                    save_checkpoint(
                        model=model,
                        path=weight_path,
                        fold=fold + 1,
                        epoch=epoch,
                        is_best=False,
                        extra=self._checkpoint_payload(
                            fold=fold + 1,
                            epoch=epoch,
                            eval_target_mean=eval_target_mean,
                            eval_target_std=eval_target_std,
                            metrics=None,
                        ),
                    )
                    plot_training_curves(
                        train_loss=train_loss_history,
                        test_metrics=val_metrics,
                        train_metrics=train_metrics,
                        eval_epochs=eval_epochs,
                        save_dir=self.config.paths.figures / self.config.task / self.uuid,
                        uuid=self.uuid,
                        fold=fold + 1,
                        task_type=self.task_type,
                    )

        if best_fold_result is not None:
            best_fold_result.fold = fold + 1
            best_fold_result.best_epoch = best_epoch
        return best_fold_result

    def _build_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Build AdamW and use the fused CUDA kernel when available."""
        kwargs = {
            "lr": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
        }

        if self.device.type == "cuda":
            try:
                optimizer = optim.AdamW(model.parameters(), fused=True, **kwargs)
                self.optimizer_label = "Fused AdamW"
                return optimizer
            except (TypeError, RuntimeError):
                pass

        self.optimizer_label = "AdamW"
        return optim.AdamW(model.parameters(), **kwargs)

    def _train_epoch(
        self,
        model: nn.Module,
        dataloader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        target_mean: float,
        target_std: float,
    ) -> float:
        """Run a single epoch."""
        model.train()
        total_loss = 0.0
        num_batches = max(len(dataloader), 1)

        for batch in tqdm(dataloader, leave=False):
            x = (
                batch["mri"].to(self.device, non_blocking=True)
                if "mri" in batch
                else batch["ct"].to(self.device, non_blocking=True)
            )
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                out = model(x)
                if self.is_classification:
                    y = batch["label"].to(self.device, non_blocking=True).long().view(-1)
                    loss = loss_fn(out, y)
                else:
                    y = batch["angle"].to(self.device, non_blocking=True).float().view(-1)
                    y_target = self._normalize_targets(y, target_mean, target_std)
                    loss = loss_fn(out.view(-1), y_target)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during training.")

            self.scaler.scale(loss).backward()

            if self.config.training.clip_grad_norm > 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.config.training.clip_grad_norm
                )

            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / num_batches

    def _normalize_targets(
        self, targets: torch.Tensor, target_mean: float, target_std: float
    ) -> torch.Tensor:
        """Normalize regression targets if configured."""
        if self.config.data.target_normalization == "zscore":
            std = target_std if abs(target_std) > 1e-8 else 1.0
            return (targets - target_mean) / std
        return targets

    def _get_eval_target_stats(
        self, target_mean: float, target_std: float
    ) -> tuple[float, float]:
        """Return the scale needed to map regression outputs back to degree units."""
        if self.is_classification:
            return 0.0, 1.0
        if self.config.data.target_normalization == "zscore":
            std = target_std if abs(target_std) > 1e-8 else 1.0
            return target_mean, std
        return 0.0, 1.0

    def _empty_metric_history(self) -> dict[str, list[float]]:
        """Create the per-epoch metric history container for the active task."""
        if self.is_classification:
            return {
                "accuracy": [],
                "macro_f1": [],
                "macro_precision": [],
                "macro_recall": [],
                "balanced_accuracy": [],
            }
        return {"mae": [], "rmse": [], "r2": [], "pearson": [], "mean_error": []}

    def _append_metric_snapshot(
        self,
        history: dict[str, list[float]],
        metrics: RegressionMetrics | ClassificationMetrics,
    ) -> None:
        """Append one evaluation snapshot into a metric history dict."""
        for key in history:
            history[key].append(self._finite_or_nan(getattr(metrics, key)))

    def _selection_score(
        self, metrics: RegressionMetrics | ClassificationMetrics
    ) -> float:
        """Return the score used for best-checkpoint selection."""
        return (
            float(metrics.macro_f1)
            if self.is_classification
            else -float(metrics.mae)
        )

    def _is_better(self, score: float, best_score: float) -> bool:
        """Compare the current score against the best score so far."""
        return score >= best_score

    def _scheduler_score(
        self, metrics: RegressionMetrics | ClassificationMetrics
    ) -> float:
        """Return the score passed into ReduceLROnPlateau."""
        return (
            float(metrics.macro_f1)
            if self.is_classification
            else float(metrics.mae)
        )

    def _format_metric_message(
        self,
        epoch: int,
        train_result: RegressionMetrics | ClassificationMetrics | None,
        val_result: RegressionMetrics | ClassificationMetrics,
    ) -> str:
        """Format the per-eval log line."""
        if self.is_classification:
            message = (
                f"Epoch {epoch}: "
                f"Val Acc={val_result.accuracy:.4f}, "
                f"Macro-F1={val_result.macro_f1:.4f}, "
                f"Bal Acc={val_result.balanced_accuracy:.4f}"
            )
            if train_result is not None:
                message += (
                    f" | Train Acc={train_result.accuracy:.4f}, "
                    f"Macro-F1={train_result.macro_f1:.4f}, "
                    f"Bal Acc={train_result.balanced_accuracy:.4f}"
                )
            return message

        message = (
            f"Epoch {epoch}: "
            f"Val MAE={val_result.mae:.4f}, "
            f"RMSE={val_result.rmse:.4f}, "
            f"R2={val_result.r2:.4f}"
        )
        if train_result is not None:
            message += (
                f" | Train MAE={train_result.mae:.4f}, "
                f"RMSE={train_result.rmse:.4f}, "
                f"R2={train_result.r2:.4f}"
            )
        return message

    def _best_metric_message(
        self,
        fold: int,
        epoch: int,
        result: RegressionMetrics | ClassificationMetrics,
    ) -> str:
        """Format the new-best checkpoint message."""
        if self.is_classification:
            return (
                f"New best fold {fold} Macro-F1: "
                f"{result.macro_f1:.4f} at epoch {epoch}"
            )
        return f"New best fold {fold} MAE: {result.mae:.4f} at epoch {epoch}"

    def _checkpoint_payload(
        self,
        fold: int,
        epoch: int,
        eval_target_mean: float,
        eval_target_std: float,
        metrics: RegressionMetrics | ClassificationMetrics | None,
    ) -> dict:
        """Assemble checkpoint metadata."""
        payload = {
            "fold": fold,
            "epoch": epoch,
            "task_type": self.task_type,
            "class_names": self.class_names,
            "target_stats": {
                "mean": eval_target_mean,
                "std": eval_target_std,
                "scope": "dataset",
            },
        }
        if metrics is None:
            return payload

        if self.is_classification:
            payload["metrics"] = {
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "macro_precision": metrics.macro_precision,
                "macro_recall": metrics.macro_recall,
                "balanced_accuracy": metrics.balanced_accuracy,
            }
        else:
            payload["metrics"] = {
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "r2": metrics.r2,
                "pearson": metrics.pearson,
                "mean_error": metrics.mean_error,
            }
        return payload

    def _finite_or_nan(self, value: float) -> float:
        """Keep plots numeric without letting inf blow up the axes."""
        return float(value) if math.isfinite(value) else float("nan")

    def _save_results_json(self, save_dir) -> None:
        """Save all fold results to a single JSON file."""
        save_dir.mkdir(parents=True, exist_ok=True)

        fold_entries = []
        summary_values: dict[str, list[float]] = {
            key: [] for key in self._empty_metric_history().keys()
        }

        for idx, res in enumerate(self.best_results, start=1):
            if res is None:
                continue

            if self.is_classification:
                fold_entry = {
                    "fold": idx,
                    "best_epoch": getattr(res, "best_epoch", None),
                    "accuracy": res.accuracy,
                    "macro_f1": res.macro_f1,
                    "macro_precision": res.macro_precision,
                    "macro_recall": res.macro_recall,
                    "balanced_accuracy": res.balanced_accuracy,
                }
            else:
                fold_entry = {
                    "fold": idx,
                    "best_epoch": getattr(res, "best_epoch", None),
                    "mae": res.mae,
                    "rmse": res.rmse,
                    "r2": res.r2,
                    "pearson": res.pearson,
                    "mean_error": res.mean_error,
                }

            fold_entries.append(fold_entry)
            for key in summary_values:
                value = fold_entry.get(key)
                if value is not None:
                    summary_values[key].append(value)

        summary = {}
        for key, values in summary_values.items():
            if values:
                summary[f"mean_{key}"] = round(float(np.mean(values)), 5)
                summary[f"std_{key}"] = round(float(np.std(values)), 5)
            else:
                summary[f"mean_{key}"] = None
                summary[f"std_{key}"] = None

        results = {
            "meta": {
                "uuid": self.uuid,
                "model": self.config.model.name,
                "task": self.config.task,
                "task_type": self.task_type,
                "timestamp": datetime.now().isoformat(),
                "training_started_at": (
                    self.run_started_at.isoformat() if self.run_started_at else None
                ),
                "training_finished_at": (
                    self.run_finished_at.isoformat() if self.run_finished_at else None
                ),
                "training_duration_seconds": round(self.total_training_seconds, 3)
                if self.total_training_seconds is not None
                else None,
                "training_duration_hours": round(
                    self.total_training_seconds / 3600.0, 4
                )
                if self.total_training_seconds is not None
                else None,
                "class_names": self.class_names,
                "config": {
                    "epochs": self.config.training.epochs,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                    "k_folds": self.config.training.k_folds,
                    "seed": self.config.training.seed,
                    "loss": self.config.training.loss,
                    "num_classes": self.config.model.num_classes,
                },
            },
            "folds": fold_entries,
            "summary": summary,
        }

        with open(save_dir / "results.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
