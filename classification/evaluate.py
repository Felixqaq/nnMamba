#!/usr/bin/env python
"""Evaluate trained nnMamba model.

Usage:
    python evaluate.py --uuid nnMamba_2026-01-21_14:30:00
    python evaluate.py --uuid nnMamba_2026-01-21_14:30:00 --config my.yaml
"""

import argparse
from pathlib import Path

import torch

from core.config import Config
from core.evaluator import evaluate
from core.checkpoints import load_checkpoint
from data.dataset import Task
from data.loader import LoaderHelper
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnMamba classifier")
    parser.add_argument("--uuid", required=True, help="Model UUID to evaluate")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Evaluating on: {device}")

    # Build components
    task_enum = Task[config.task]
    loader_helper = LoaderHelper(task=task_enum)
    model = build_model(config.model.name, device)

    # Load weights
    weight_path = config.paths.weights / config.task / args.uuid / "best_weight.pth"
    model = load_checkpoint(weight_path, model, device)
    model.eval()

    # Evaluate
    test_dl = loader_helper.get_test_dl(0)
    metrics = evaluate(model, test_dl, device)

    print(f"\n📊 Evaluation Results for {args.uuid}")
    print(f"{'=' * 40}")
    print(f"  Accuracy:    {metrics.accuracy:.4f}")
    print(f"  Sensitivity: {metrics.sensitivity:.4f}")
    print(f"  Specificity: {metrics.specificity:.4f}")
    print(f"  AUC:         {metrics.auc:.4f}")


if __name__ == "__main__":
    main()
