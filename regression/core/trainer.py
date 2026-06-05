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
from .evaluator import (
    ClassificationMetrics,
    RegressionMetrics,
    encode_ordinal_targets,
    evaluate,
    save_predictions,
)
from .visualizer import plot_global_summary, plot_paper_results, plot_training_curves

ATTENTION_HEAVY_MODELS = {
    "hybrid",
    "hybrid_mamba_attention",
    "hybrid_mamba_attention_regressor",
    "hybrid_mamba_tapct_fusion",
    "hybrid_tapct_fusion",
    "tapct_late_fusion",
    "hybrid_mamba_tapct_abmil_fusion",
    "hybrid_tapct_abmil_fusion",
    "tapct_abmil_fusion",
    "hybrid_mamba_tapct_attention_fusion",
    "hybrid_tapct_attention_fusion",
    "tapct_attention_fusion",
    "mamba_hybrid",
    "swinunetr",
}
TAPCT_ONLY_MODELS = {"tapct_abmil", "tapct_abmil_classifier"}
TAPCT_FUSION_MODELS = {
    "hybrid_mamba_tapct_fusion",
    "hybrid_mamba_tapct_abmil_fusion",
    "hybrid_tapct_fusion",
    "hybrid_tapct_abmil_fusion",
    "tapct_late_fusion",
    "tapct_abmil_fusion",
    "hybrid_mamba_tapct_attention_fusion",
    "hybrid_tapct_attention_fusion",
    "tapct_attention_fusion",
}
TAPCT_ABMIL_FUSION_MODELS = {
    "hybrid_mamba_tapct_abmil_fusion",
    "hybrid_tapct_abmil_fusion",
    "tapct_abmil_fusion",
}
TAPCT_ATTENTION_FUSION_MODELS = {
    "hybrid_mamba_tapct_attention_fusion",
    "hybrid_tapct_attention_fusion",
    "tapct_attention_fusion",
}


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_loss(
    name: str,
    is_classification: bool,
    class_weights: torch.Tensor | None = None,
    classification_mode: str = "multiclass",
) -> nn.Module:
    """Build task-specific loss by name."""
    resolved = name
    if resolved == "auto":
        if is_classification and classification_mode == "ordinal":
            resolved = "ordinal_bce"
        else:
            resolved = "cross_entropy" if is_classification else "smooth_l1"

    if resolved not in {"smooth_l1", "mse", "mae", "cross_entropy", "ordinal_bce"}:
        raise ValueError(f"Unknown loss: {name}")
    if resolved == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if resolved == "ordinal_bce":
        return nn.BCEWithLogitsLoss()
    if resolved == "smooth_l1":
        return nn.SmoothL1Loss()
    if resolved == "mse":
        return nn.MSELoss()
    return nn.L1Loss()


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
        self.is_ordinal_classification = config.is_ordinal_classification()
        self.class_names = (
            loader_helper.get_class_names()
            if hasattr(loader_helper, "get_class_names")
            else []
        )
        self.method_label = config.experiment.name

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.device_id
        setup_seed(config.training.seed)

        self.uuid = (
            config.resume.uuid
            if config.resume.enabled and config.resume.uuid
            else generate_uuid(str(config.model.name), config.experiment.name)
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
        if cfg.experiment.name:
            print(f"Method: {cfg.experiment.name}")
        if self.is_classification:
            print(
                f"Classes: {cfg.model.num_classes} | "
                f"Labels: {', '.join(self.class_names) if self.class_names else 'n/a'}"
            )
            print(
                f"Classification mode: {cfg.training.classification_mode} | "
                f"Model logits: {cfg.model_output_dim()}"
            )
            if cfg.training.class_weight_mode != "none":
                example_weights = self._resolve_class_weights(0)
                if example_weights is not None:
                    weight_text = ", ".join(
                        f"{value:.3f}" for value in example_weights.detach().cpu().tolist()
                    )
                    print(
                        f"Class weighting: {cfg.training.class_weight_mode} | "
                        f"Example weights: [{weight_text}]"
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
            if model_key in TAPCT_FUSION_MODELS:
                print(
                    f"Late fusion: TAP-CT dim={cfg.model.tapct_embedding_dim}, "
                    f"projection={cfg.model.fusion_projection_dim}, "
                    f"features={cfg.data.tapct_features}"
                )
                if model_key in TAPCT_ABMIL_FUSION_MODELS:
                    print(
                        f"ABMIL fusion head: attention={cfg.model.tapct_attention_dim}, "
                        f"gated={'on' if cfg.model.tapct_gated_attention else 'off'}"
                    )
                if model_key in TAPCT_ATTENTION_FUSION_MODELS:
                    print(
                        f"Attention concat fusion: attention={cfg.model.tapct_attention_dim}, "
                        f"gated={'on' if cfg.model.tapct_gated_attention else 'off'}"
                    )
        elif model_key in TAPCT_ONLY_MODELS:
            print(
                f"TAP-CT ABMIL: dim={cfg.model.tapct_embedding_dim}, "
                f"hidden={cfg.model.hidden_dim}, "
                f"attention={cfg.model.tapct_attention_dim}, "
                f"key={cfg.data.tapct_feature_key}, "
                f"features={cfg.data.tapct_features}"
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
            method_label=self.method_label,
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
        class_weights = self._resolve_class_weights(fold)

        optimizer = self._build_optimizer(model)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if self.is_classification else "min",
            factor=0.5,
            patience=2,
        )
        loss_fn = build_loss(
            cfg.loss,
            self.is_classification,
            class_weights=class_weights,
            classification_mode=cfg.classification_mode,
        )

        train_loss_history: list[float] = []
        eval_epochs: list[int] = []
        train_metrics = self._empty_metric_history()
        val_metrics = self._empty_metric_history()
        best_score = float("-inf")
        best_fold_result = None
        best_epoch = 0
        stale_eval_count = 0

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
                            classification_mode=self.config.training.classification_mode,
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
                        classification_mode=self.config.training.classification_mode,
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

                    improved = self._is_better(current_score, best_score)
                    if improved:
                        best_score = current_score
                        best_epoch = epoch
                        best_fold_result = val_result
                        stale_eval_count = 0

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
                            method_label=self.method_label,
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
                    else:
                        stale_eval_count += 1

                    if self._should_stop_early(stale_eval_count):
                        stop_msg = (
                            f"Early stopping fold {fold + 1} at epoch {epoch}: "
                            f"no improvement for {stale_eval_count} evals "
                            f"(best epoch {best_epoch}, best score {best_score:.4f})"
                        )
                        tqdm.write(stop_msg)
                        log_file.write(stop_msg + "\n")
                        break

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
                        method_label=self.method_label,
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
            x = self._extract_model_input(batch)
            optimizer.zero_grad(set_to_none=True)

            y = (
                batch["label"].to(self.device, non_blocking=True).long().view(-1)
                if self.is_classification
                else batch["target"].to(self.device, non_blocking=True).float().view(-1)
            )

            def compute_loss(amp_enabled: bool) -> torch.Tensor:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    out = model(x)
                    if self.is_classification:
                        if self.is_ordinal_classification:
                            ordinal_y = encode_ordinal_targets(
                                y,
                                num_classes=int(self.config.model.num_classes),
                            )
                            return loss_fn(out, ordinal_y)
                        return loss_fn(out, y)
                    y_target = self._normalize_targets(y, target_mean, target_std)
                    return loss_fn(out.view(-1), y_target)

            loss = compute_loss(self.use_amp)
            if not torch.isfinite(loss) and self.use_amp:
                tqdm.write(
                    "Non-finite loss encountered under AMP; retrying the batch in full precision."
                )
                loss = compute_loss(False)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss encountered during training "
                    "even after retrying in full precision."
                )

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

    def _extract_model_input(self, batch: dict):
        """Move model inputs to the active device, including optional fusion features."""
        ct = (
            batch["mri"].to(self.device, non_blocking=True)
            if "mri" in batch
            else batch["ct"].to(self.device, non_blocking=True)
        )
        embedding = batch.get("tapct_embedding")
        if embedding is None:
            return ct
        model_input = {
            "ct": ct,
            "tapct_embedding": embedding.to(self.device, non_blocking=True),
        }
        if batch.get("tapct_mask") is not None:
            model_input["tapct_mask"] = batch["tapct_mask"].to(
                self.device,
                non_blocking=True,
            )
        return model_input

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
                "sensitivity": [],
                "specificity": [],
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
        if not math.isfinite(score):
            return False
        if best_score == float("-inf"):
            return True
        if not self.config.early_stopping.enabled:
            return score >= best_score
        return score > best_score + float(self.config.early_stopping.min_delta)

    def _should_stop_early(self, stale_eval_count: int) -> bool:
        """Return whether validation has stalled long enough to stop this fold."""
        cfg = self.config.early_stopping
        return bool(cfg.enabled and stale_eval_count >= max(1, int(cfg.patience)))

    def _scheduler_score(
        self, metrics: RegressionMetrics | ClassificationMetrics
    ) -> float:
        """Return the score passed into ReduceLROnPlateau."""
        return (
            float(metrics.macro_f1)
            if self.is_classification
            else float(metrics.mae)
        )

    def _resolve_class_weights(self, fold: int) -> torch.Tensor | None:
        """Build class weights from the fold's training split when enabled."""
        if not self.is_classification:
            return None
        if self.config.training.class_weight_mode != "balanced":
            return None
        if self.is_ordinal_classification:
            return None
        weights = self.loader_helper.get_fold_class_weights(fold)
        return weights.to(self.device)

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
            "experiment": {
                "name": self.config.experiment.name,
                "description": self.config.experiment.description,
            },
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
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
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
                    "sensitivity": res.sensitivity,
                    "specificity": res.specificity,
                    "confusion_matrix": res.confusion_matrix,
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

        total_confusion_matrix = None
        if self.is_classification:
            matrices = [
                np.asarray(entry["confusion_matrix"], dtype=int)
                for entry in fold_entries
                if entry.get("confusion_matrix") is not None
            ]
            if matrices:
                total_confusion_matrix = np.sum(matrices, axis=0).astype(int).tolist()

        config_meta = {
            "epochs": self.config.training.epochs,
            "batch_size": self.config.training.batch_size,
            "learning_rate": self.config.training.learning_rate,
            "weight_decay": self.config.training.weight_decay,
            "k_folds": self.config.training.k_folds,
            "seed": self.config.training.seed,
            "loss": self.config.training.loss,
            "num_classes": self.config.model.num_classes,
            "model_output_dim": self.config.model_output_dim(),
            "classification_mode": self.config.training.classification_mode,
            "target_mode": self.config.data.target_mode,
        }
        if self.config.data.tapct_features is not None:
            config_meta["tapct_features"] = str(self.config.data.tapct_features)
            config_meta["tapct_embedding_dim"] = self.config.model.tapct_embedding_dim
            config_meta["fusion_projection_dim"] = self.config.model.fusion_projection_dim
            if str(self.config.model.name).lower() in (
                TAPCT_ABMIL_FUSION_MODELS | TAPCT_ATTENTION_FUSION_MODELS
            ):
                config_meta["tapct_attention_dim"] = (
                    self.config.model.tapct_attention_dim
                )
                config_meta["tapct_gated_attention"] = (
                    self.config.model.tapct_gated_attention
                )
        if self.config.data.target_mode == "angle_binary_extreme":
            config_meta.update(
                {
                    "abnormal_rule": "AC <= 131 deg",
                    "excluded_gray_zone": "132 <= AC < 152 deg",
                    "normal_rule": "AC >= 152 deg",
                }
            )

        results = {
            "meta": {
                "uuid": self.uuid,
                "model": self.config.model.name,
                "experiment": {
                    "name": self.config.experiment.name,
                    "description": self.config.experiment.description,
                },
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
                "config": config_meta,
            },
            "folds": fold_entries,
            "summary": summary,
        }
        if total_confusion_matrix is not None:
            results["total_confusion_matrix"] = total_confusion_matrix

        with open(save_dir / "results.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
