#!/usr/bin/env python
"""Evaluate trained nnMamba regression/classification checkpoints."""

import argparse
from pathlib import Path

import torch

from core.checkpoints import load_checkpoint
from core.config import Config
from core.evaluator import evaluate
from core.runtime import configure_torch_runtime
from data.loader import LoaderHelper
from models import build_model


def _load_fold_model(config: Config, uuid: str, fold: int, device: torch.device):
    model = build_model(config.model, device=device)
    checkpoint_path = config.paths.weights / config.task / uuid / f"fold{fold}_best_weight.pth"
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    return model, checkpoint, checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nnMamba CT angle regressor / GOLD classifier"
    )
    parser.add_argument("--uuid", required=True, help="Run UUID to evaluate")
    parser.add_argument("--fold", type=int, default=None, help="Specific fold to evaluate")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    configure_torch_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    loader_helper = LoaderHelper(config)
    fold_ids = [args.fold] if args.fold else list(range(1, config.training.k_folds + 1))

    for fold in fold_ids:
        model, checkpoint, checkpoint_path = _load_fold_model(config, args.uuid, fold, device)
        stats = checkpoint.get("target_stats", {})
        target_mean = float(stats.get("mean", 0.0))
        target_std = float(stats.get("std", 1.0))
        test_dl = loader_helper.get_test_dl(fold - 1)
        metrics = evaluate(
            model=model,
            dataloader=test_dl,
            device=device,
            task_type=config.data.target_mode,
            target_mean=target_mean,
            target_std=target_std,
            use_amp=bool(config.training.amp and device.type == "cuda"),
            num_classes=int(config.model.num_classes),
        )

        print(f"\nFold {fold}")
        print(f"Checkpoint: {checkpoint_path}")
        if config.is_classification_task():
            print(f"Accuracy:          {metrics.accuracy:.4f}")
            print(f"Macro F1:          {metrics.macro_f1:.4f}")
            print(f"Macro Precision:   {metrics.macro_precision:.4f}")
            print(f"Macro Recall:      {metrics.macro_recall:.4f}")
            print(f"Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
        else:
            print(f"MAE:       {metrics.mae:.4f}")
            print(f"RMSE:      {metrics.rmse:.4f}")
            print(f"R2:        {metrics.r2:.4f}")
            print(f"Pearson r: {metrics.pearson_r:.4f}")
            print(f"Bias:      {metrics.mean_error:.4f}")


if __name__ == "__main__":
    main()
