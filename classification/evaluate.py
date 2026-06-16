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
from core.gradcam import generate_gradcam
from data.dataset import Task
from data.loader import LoaderHelper
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnMamba classifier")
    parser.add_argument("--uuid", required=True, help="Model UUID to evaluate")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM generation after evaluation",
    )
    parser.add_argument(
        "--gradcam-layer",
        default=None,
        help="Optional model.named_modules() layer to use for Grad-CAM",
    )
    parser.add_argument(
        "--gradcam-max-samples",
        type=int,
        default=8,
        help="Maximum number of validation samples to render",
    )
    parser.add_argument(
        "--gradcam-class",
        type=int,
        default=1,
        help="Class index to explain; binary models use 1 for abnormal/positive",
    )
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
    fold = 0
    test_dl = loader_helper.get_test_dl(fold)
    metrics = evaluate(model, test_dl, device)

    print(f"\n📊 Evaluation Results for {args.uuid}")
    print(f"{'=' * 40}")
    print(f"  Accuracy:    {metrics.accuracy:.4f}")
    print(f"  Sensitivity: {metrics.sensitivity:.4f}")
    print(f"  Specificity: {metrics.specificity:.4f}")
    print(f"  AUC:         {metrics.auc:.4f}")

    if not args.no_gradcam:
        gradcam_dir = config.paths.figures / config.task / args.uuid / "gradcam"
        try:
            samples = generate_gradcam(
                model=model,
                dataloader=test_dl,
                device=device,
                save_dir=gradcam_dir,
                model_name=config.model.name,
                dataset=loader_helper.train_ds,
                fold_indices=loader_helper.fold_indices[fold][1],
                labels=config.get_labels(),
                max_samples=args.gradcam_max_samples,
                target_layer_name=args.gradcam_layer,
                target_layer_names=(
                    None if args.gradcam_layer else config.gradcam.target_layers
                ),
                target_class=args.gradcam_class,
                threshold=metrics.threshold,
                per_outcome=config.gradcam.per_outcome,
            )
            print(f"\n🧠 Grad-CAM: saved {len(samples)} samples to {gradcam_dir}")
        except (RuntimeError, ValueError) as exc:
            print(f"\n⚠️  Grad-CAM skipped: {exc}")


if __name__ == "__main__":
    main()
