"""Training loop for nnMamba classification."""

import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from .config import Config
from .checkpoints import save_checkpoint, generate_uuid
from .evaluator import evaluate, Metrics
from .visualizer import plot_training_curves


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
            self._train_fold(fold)
            print(f"✅ Fold {fold + 1} complete\n")

        return self.uuid

    def _train_fold(self, fold: int) -> None:
        """Train a single fold."""
        cfg = self.config.training
        self.model.to(self.device)

        train_dl = self.loader_helper.get_train_dl(fold)
        test_dl = self.loader_helper.get_test_dl(fold)

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1)
        loss_fn = nn.BCEWithLogitsLoss()

        # Metrics tracking
        train_loss_history = []
        test_metrics = {"acc": [], "auc": [], "sens": [], "spec": []}
        train_metrics = {"acc": [], "auc": []}
        eval_epochs = []
        best_auc = 0.0

        log_path = self.config.paths.logs / f"{self.uuid}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as log_file:
            for epoch in range(1, cfg.epochs + 1):
                loss = self._train_epoch(train_dl, optimizer, loss_fn)
                train_loss_history.append(loss)

                log_file.write(f"Epoch: {epoch}/{cfg.epochs}, loss: {loss:.5f}\n")
                tqdm.write(f"Epoch: {epoch}/{cfg.epochs}, loss: {loss:.5f}")

                # Periodic evaluation
                if epoch % cfg.eval_interval == 0 or epoch == cfg.epochs:
                    eval_epochs.append(epoch)
                    best_auc = self._evaluate_and_log(
                        train_dl,
                        test_dl,
                        epoch,
                        fold,
                        log_file,
                        test_metrics,
                        train_metrics,
                        best_auc,
                    )

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
    ) -> float:
        """Evaluate model and log results. Returns new best AUC."""
        test_result = evaluate(self.model, test_dl, self.device)
        train_result = evaluate(self.model, train_dl, self.device)

        test_metrics["acc"].append(test_result.accuracy)
        test_metrics["auc"].append(test_result.auc)
        test_metrics["sens"].append(test_result.sensitivity)
        test_metrics["spec"].append(test_result.specificity)
        train_metrics["acc"].append(train_result.accuracy)
        train_metrics["auc"].append(train_result.auc)

        msg = (
            f"Epoch {epoch}: Test AUC={test_result.auc}, Acc={test_result.accuracy} | "
            f"Train AUC={train_result.auc}, Acc={train_result.accuracy}"
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
            tqdm.write(f"🎯 New best AUC: {test_result.auc}")
            return test_result.auc

        return best_auc

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
            self.config.paths.figures,
            self.uuid,
            fold + 1,
        )
