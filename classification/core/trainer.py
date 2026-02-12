"""Training loop for nnMamba classification."""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from .config import Config
from .checkpoints import save_checkpoint, generate_uuid
from .evaluator import (
    get_predictions,
    find_optimal_threshold,
    compute_metrics,
    save_misclassified,
)
from .visualizer import plot_training_curves, plot_paper_results, plot_global_summary


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    """Handles model training with k-fold cross-validation."""

    def __init__(self, config: Config, model: nn.Module, loader_helper):
        self.config = config
        self.model = model
        self.loader_helper = loader_helper
        self.best_results = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device_id
        setup_seed(config.training.seed)

        self.uuid = (
            config.resume.uuid if config.resume.enabled else generate_uuid("nnMamba")
        )

    def train(self) -> str:
        """Run full training with k-fold cross-validation.

        Returns:
            Model UUID for identification
        """
        cfg = self.config
        start_fold = cfg.resume.start_fold if cfg.resume.enabled else 0

        print(f"\n{'=' * 60}")
        print(f"Training: {self.uuid}")
        print(
            f"Model: {cfg.model.name} | Task: {cfg.task} | {cfg.training.k_folds} folds"
        )
        print(f"{'=' * 60}\n")

        for fold in range(start_fold, cfg.training.k_folds):
            print(f"\n📊 Fold {fold + 1}/{cfg.training.k_folds}")
            best_res = self._train_fold(fold)
            self.best_results.append(best_res)
            print(f"✅ Fold {fold + 1} complete\n")

        # Generate global summary for all folds
        fig_dir = self.config.paths.figures / self.config.task / self.uuid
        plot_global_summary(
            self.best_results, fig_dir, class_names=self.config.get_labels()
        )
        self._save_results_json(fig_dir)

        return self.uuid

    def _train_fold(self, fold: int):
        """Train a single fold. Returns best metrics object."""
        cfg = self.config.training
        self.model.to(self.device)

        train_dl = self.loader_helper.get_train_dl(fold)
        test_dl = self.loader_helper.get_test_dl(fold)
        test_fold_indices = self.loader_helper.fold_indices[fold][1]

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1)
        loss_fn = nn.BCEWithLogitsLoss()

        # Metrics tracking
        train_loss_history = []
        test_metrics = {"acc": [], "auc": [], "sens": [], "spec": []}
        train_metrics = {"acc": [], "auc": []}
        eval_epochs = []
        best_auc = 0.0
        best_fold_result = None

        log_dir = self.config.paths.logs / self.config.task / self.uuid
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.uuid}.txt"

        with open(log_path, "a") as log_file:
            for epoch in range(1, cfg.epochs + 1):
                loss = self._train_epoch(train_dl, optimizer, loss_fn)
                train_loss_history.append(loss)

                log_file.write(f"Epoch: {epoch}/{cfg.epochs}, loss: {loss:.5f}\n")
                tqdm.write(f"Epoch: {epoch}/{cfg.epochs}, loss: {loss:.5f}")

                # Periodic evaluation
                if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                    eval_epochs.append(epoch)
                    auc_res, res_obj = self._evaluate_and_log(
                        train_dl,
                        test_dl,
                        epoch,
                        fold,
                        log_file,
                        test_metrics,
                        train_metrics,
                        best_auc,
                        test_fold_indices,
                    )
                    if auc_res > best_auc:
                        best_auc = auc_res
                        best_fold_result = res_obj

                # Periodic checkpoint
                if epoch % cfg.save_interval == 0:
                    self._save_and_plot(
                        epoch,
                        fold,
                        train_loss_history,
                        test_metrics,
                        train_metrics,
                        eval_epochs,
                    )

        return best_fold_result

    def _train_epoch(
        self, dataloader, optimizer: optim.Optimizer, loss_fn: nn.Module
    ) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in tqdm(dataloader, leave=False):
            x = batch["mri"].to(self.device)
            y = batch["label"].to(self.device)

            optimizer.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def _evaluate_and_log(
        self,
        train_dl,
        test_dl,
        epoch: int,
        fold: int,
        log_file,
        test_metrics: dict,
        train_metrics: dict,
        best_auc: float,
        test_fold_indices: list,
    ) -> tuple[float, any]:
        """Evaluate model and log results. Returns (new best AUC, current metrics object)."""
        # Dynamic thresholding: Find best threshold on Train, apply to Test
        train_labels, train_preds, _ = get_predictions(
            self.model, train_dl, self.device
        )
        best_thresh = find_optimal_threshold(train_labels, train_preds)

        train_result = compute_metrics(
            train_labels, train_preds, best_thresh, self.device
        )

        test_labels, test_preds, _ = get_predictions(self.model, test_dl, self.device)
        test_result = compute_metrics(test_labels, test_preds, best_thresh, self.device)

        test_metrics["acc"].append(test_result.accuracy)
        test_metrics["auc"].append(test_result.auc)
        test_metrics["sens"].append(test_result.sensitivity)
        test_metrics["spec"].append(test_result.specificity)
        train_metrics["acc"].append(train_result.accuracy)
        train_metrics["auc"].append(train_result.auc)

        msg = (
            f"Epoch {epoch}: Test AUC={test_result.auc}, Acc={test_result.accuracy} | "
            f"Train AUC={train_result.auc}, Acc={train_result.accuracy} | "
            f"Thresh={best_thresh:.3f}"
        )
        tqdm.write(msg)
        log_file.write(msg + "\n")

        # Overfitting warning
        if train_result.auc - test_result.auc > 0.1:
            tqdm.write(
                f"⚠️  Overfitting: Train-Test AUC gap = {train_result.auc - test_result.auc:.3f}"
            )

        # Save best model
        if test_result.auc >= best_auc:
            weight_path = self.config.paths.weights / self.config.task / self.uuid
            save_checkpoint(self.model, weight_path, is_best=True)

            # Generate advanced paper plots for the best model
            fig_dir = self.config.paths.figures / self.config.task / self.uuid
            plot_paper_results(
                test_result, fold + 1, fig_dir, class_names=self.config.get_labels()
            )

            # Save misclassified samples
            errors = save_misclassified(
                test_result,
                self.loader_helper.train_ds,
                test_fold_indices,
                self.config.get_labels(),
                fig_dir,
                fold + 1,
            )
            n_fn = len(errors.get("false_negatives", []))
            n_fp = len(errors.get("false_positives", []))
            tqdm.write(f"🎯 New best AUC: {test_result.auc} (FN={n_fn}, FP={n_fp})")
            return test_result.auc, test_result

        return best_auc, test_result

    def _save_and_plot(
        self,
        epoch: int,
        fold: int,
        train_loss: list,
        test_metrics: dict,
        train_metrics: dict,
        eval_epochs: list,
    ) -> None:
        """Save checkpoint and generate plots."""
        weight_path = self.config.paths.weights / self.config.task / self.uuid
        save_checkpoint(self.model, weight_path, epoch=epoch, fold=fold + 1)

        plot_training_curves(
            train_loss,
            test_metrics,
            train_metrics,
            eval_epochs,
            self.config.paths.figures / self.config.task / self.uuid,
            self.uuid,
            fold + 1,
        )

    def _save_results_json(self, save_dir) -> None:
        """Save all training results to a single JSON file."""
        cfg = self.config
        fold_entries = []
        auc_vals, acc_vals, sens_vals, spec_vals = [], [], [], []

        for i, res in enumerate(self.best_results):
            if res is None:
                continue
            fold_entries.append(
                {
                    "fold": i + 1,
                    "best_auc": res.auc,
                    "best_accuracy": res.accuracy,
                    "best_sensitivity": res.sensitivity,
                    "best_specificity": res.specificity,
                    "threshold": res.threshold,
                }
            )
            auc_vals.append(res.auc)
            acc_vals.append(res.accuracy)
            sens_vals.append(res.sensitivity)
            spec_vals.append(res.specificity)

        results = {
            "meta": {
                "uuid": self.uuid,
                "model": cfg.model.name,
                "task": cfg.task,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "epochs": cfg.training.epochs,
                    "batch_size": cfg.training.batch_size,
                    "learning_rate": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                    "k_folds": cfg.training.k_folds,
                    "seed": cfg.training.seed,
                },
            },
            "folds": fold_entries,
            "summary": {
                "mean_auc": round(float(np.mean(auc_vals)), 5),
                "std_auc": round(float(np.std(auc_vals)), 5),
                "mean_accuracy": round(float(np.mean(acc_vals)), 5),
                "std_accuracy": round(float(np.std(acc_vals)), 5),
                "mean_sensitivity": round(float(np.mean(sens_vals)), 5),
                "std_sensitivity": round(float(np.std(sens_vals)), 5),
                "mean_specificity": round(float(np.mean(spec_vals)), 5),
                "std_specificity": round(float(np.std(spec_vals)), 5),
            },
        }

        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        tqdm.write(f"📄 Results saved to {save_dir / 'results.json'}")
