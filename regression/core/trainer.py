"""Training loop for nnMamba CT angle regression."""

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
from .evaluator import evaluate, save_predictions
from .visualizer import plot_global_summary, plot_paper_results, plot_training_curves


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_loss(name: str) -> nn.Module:
    """Build regression loss by name."""
    registry = {
        "smooth_l1": nn.SmoothL1Loss,
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss: {name}")
    return registry[name]()


class Trainer:
    """Handles model training with k-fold cross-validation."""

    def __init__(self, config: Config, model_factory, loader_helper):
        self.config = config
        self.model_factory = model_factory
        self.loader_helper = loader_helper
        self.best_results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(config.training.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.optimizer_label = "AdamW"
        self.run_started_at: datetime | None = None
        self.run_finished_at: datetime | None = None
        self.total_training_seconds: float | None = None

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
            f"Model: {cfg.model.name} | Task: {cfg.task} | {cfg.training.k_folds} folds"
        )
        if hasattr(self.loader_helper, "batch_size") and hasattr(
            self.loader_helper, "val_batch_size"
        ):
            print(
                f"Train batch: {self.loader_helper.batch_size} | "
                f"Eval batch: {self.loader_helper.val_batch_size} | "
                f"AMP: {'on' if self.use_amp else 'off'}"
            )
        if str(cfg.model.name).lower() == "swinunetr":
            print(
                f"Swin window: {cfg.model.window_size} | "
                f"Checkpointing: {'on' if cfg.model.use_checkpoint else 'off'}"
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
        plot_global_summary(self.best_results, fig_dir)
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
            optimizer, mode="min", factor=0.5, patience=2
        )
        loss_fn = build_loss(cfg.loss)

        train_loss_history: list[float] = []
        eval_epochs: list[int] = []
        train_metrics = {"mae": [], "rmse": [], "r2": [], "pearson": [], "mean_error": []}
        val_metrics = {"mae": [], "rmse": [], "r2": [], "pearson": [], "mean_error": []}
        best_mae = float("inf")
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
                            target_mean=eval_target_mean,
                            target_std=eval_target_std,
                            use_amp=self.use_amp,
                        )
                    val_result = evaluate(
                        model=model,
                        dataloader=test_dl,
                        device=self.device,
                        target_mean=eval_target_mean,
                        target_std=eval_target_std,
                        use_amp=self.use_amp,
                    )

                    if (
                        train_result is not None
                        and train_result.num_invalid_samples > 0
                    ) or val_result.num_invalid_samples > 0:
                        warn_msg = (
                            f"Epoch {epoch}: invalid predictions detected "
                            f"(train={train_result.num_invalid_samples if train_result is not None else 0}, "
                            f"val={val_result.num_invalid_samples})"
                        )
                        tqdm.write(warn_msg)
                        log_file.write(warn_msg + "\n")

                    if train_result is not None:
                        train_metrics["mae"].append(self._finite_or_nan(train_result.mae))
                        train_metrics["rmse"].append(self._finite_or_nan(train_result.rmse))
                        train_metrics["r2"].append(self._finite_or_nan(train_result.r2))
                        train_metrics["pearson"].append(self._finite_or_nan(train_result.pearson))
                        train_metrics["mean_error"].append(
                            self._finite_or_nan(train_result.mean_error)
                        )
                    val_metrics["mae"].append(self._finite_or_nan(val_result.mae))
                    val_metrics["rmse"].append(self._finite_or_nan(val_result.rmse))
                    val_metrics["r2"].append(self._finite_or_nan(val_result.r2))
                    val_metrics["pearson"].append(self._finite_or_nan(val_result.pearson))
                    val_metrics["mean_error"].append(
                        self._finite_or_nan(val_result.mean_error)
                    )
                    if math.isfinite(val_result.mae):
                        scheduler.step(val_result.mae)

                    metric_msg = (
                        f"Epoch {epoch}: "
                        f"Val MAE={val_result.mae:.4f}, RMSE={val_result.rmse:.4f}, R2={val_result.r2:.4f}"
                    )
                    if train_result is not None:
                        metric_msg += (
                            f" | Train MAE={train_result.mae:.4f}, "
                            f"RMSE={train_result.rmse:.4f}, R2={train_result.r2:.4f}"
                        )
                    tqdm.write(metric_msg)
                    log_file.write(metric_msg + "\n")

                    if val_result.mae <= best_mae:
                        best_mae = val_result.mae
                        best_epoch = epoch
                        best_fold_result = val_result

                        weight_path = self.config.paths.weights / self.config.task / self.uuid
                        save_checkpoint(
                            model=model,
                            path=weight_path,
                            fold=fold + 1,
                            epoch=epoch,
                            is_best=True,
                            extra={
                                "fold": fold + 1,
                                "epoch": epoch,
                                "target_stats": {
                                    "mean": eval_target_mean,
                                    "std": eval_target_std,
                                    "scope": "dataset",
                                },
                                "metrics": {
                                    "mae": val_result.mae,
                                    "rmse": val_result.rmse,
                                    "r2": val_result.r2,
                                    "pearson": val_result.pearson,
                                    "mean_error": val_result.mean_error,
                                },
                            },
                        )

                        fig_dir = self.config.paths.figures / self.config.task / self.uuid
                        plot_paper_results(val_result, fold + 1, fig_dir)
                        save_predictions(
                            metrics=val_result,
                            dataset=self.loader_helper.dataset,
                            fold_indices=self.loader_helper.fold_indices[fold][1],
                            save_path=fig_dir,
                            fold=fold + 1,
                        )
                        tqdm.write(
                            f"New best fold {fold + 1} MAE: {val_result.mae:.4f} at epoch {epoch}"
                        )

                if epoch % cfg.save_interval == 0:
                    weight_path = self.config.paths.weights / self.config.task / self.uuid
                    save_checkpoint(
                        model=model,
                        path=weight_path,
                        fold=fold + 1,
                        epoch=epoch,
                        is_best=False,
                        extra={
                            "fold": fold + 1,
                            "epoch": epoch,
                            "target_stats": {
                                "mean": eval_target_mean,
                                "std": eval_target_std,
                                "scope": "dataset",
                            },
                        },
                    )
                    plot_training_curves(
                        train_loss=train_loss_history,
                        test_metrics=val_metrics,
                        train_metrics=train_metrics,
                        eval_epochs=eval_epochs,
                        save_dir=self.config.paths.figures / self.config.task / self.uuid,
                        uuid=self.uuid,
                        fold=fold + 1,
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
            y = batch["angle"].to(self.device, non_blocking=True).float().view(-1)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):
                out = model(x).view(-1)
                y_target = self._normalize_targets(y, target_mean, target_std)
                loss = loss_fn(out, y_target)
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
        """Normalize targets if configured."""
        if self.config.data.target_normalization == "zscore":
            std = target_std if abs(target_std) > 1e-8 else 1.0
            return (targets - target_mean) / std
        return targets

    def _finite_or_nan(self, value: float) -> float:
        """Keep plots numeric without letting inf blow up the axes."""
        return float(value) if math.isfinite(value) else float("nan")

    def _get_eval_target_stats(
        self, target_mean: float, target_std: float
    ) -> tuple[float, float]:
        """Return the scale needed to map model outputs back to degree units."""
        if self.config.data.target_normalization == "zscore":
            std = target_std if abs(target_std) > 1e-8 else 1.0
            return target_mean, std
        return 0.0, 1.0

    def _save_results_json(self, save_dir) -> None:
        """Save all fold results to a single JSON file."""
        save_dir.mkdir(parents=True, exist_ok=True)

        fold_entries = []
        mae_vals, rmse_vals, r2_vals, pearson_vals = [], [], [], []

        for idx, res in enumerate(self.best_results, start=1):
            if res is None:
                continue
            fold_entries.append(
                {
                    "fold": idx,
                    "best_epoch": getattr(res, "best_epoch", None),
                    "mae": res.mae,
                    "rmse": res.rmse,
                    "r2": res.r2,
                    "pearson": res.pearson,
                    "mean_error": res.mean_error,
                }
            )
            mae_vals.append(res.mae)
            rmse_vals.append(res.rmse)
            r2_vals.append(res.r2)
            pearson_vals.append(res.pearson)

        results = {
            "meta": {
                "uuid": self.uuid,
                "model": self.config.model.name,
                "task": self.config.task,
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
                "config": {
                    "epochs": self.config.training.epochs,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                    "k_folds": self.config.training.k_folds,
                    "seed": self.config.training.seed,
                    "loss": self.config.training.loss,
                },
            },
            "folds": fold_entries,
            "summary": {
                "mean_mae": round(float(np.mean(mae_vals)), 5) if mae_vals else None,
                "std_mae": round(float(np.std(mae_vals)), 5) if mae_vals else None,
                "mean_rmse": round(float(np.mean(rmse_vals)), 5) if rmse_vals else None,
                "std_rmse": round(float(np.std(rmse_vals)), 5) if rmse_vals else None,
                "mean_r2": round(float(np.mean(r2_vals)), 5) if r2_vals else None,
                "std_r2": round(float(np.std(r2_vals)), 5) if r2_vals else None,
                "mean_pearson": round(float(np.mean(pearson_vals)), 5)
                if pearson_vals
                else None,
                "std_pearson": round(float(np.std(pearson_vals)), 5)
                if pearson_vals
                else None,
            },
        }

        with open(save_dir / "results.json", "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
