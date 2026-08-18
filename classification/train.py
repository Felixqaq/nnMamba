#!/usr/bin/env python
"""Train nnMamba classification model.

Usage:
    python train.py                     # Use config.yaml defaults
    python train.py --config my.yaml    # Use custom config
"""

import argparse
import torch

from core.config import Config
from core.trainer import Trainer, setup_seed
from data.dataset import Task
from data.loader import LoaderHelper
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Train nnMamba classifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip automatic Grad-CAM generation during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the random seed from the config",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.no_gradcam:
        config.gradcam.enabled = False
    if args.seed is not None:
        config.training.seed = args.seed

    setup_seed(config.training.seed)

    print(f"🚀 Running on: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        print("⚠️  Warning: GPU not detected. nnMamba requires CUDA.")

    # Build components
    task_enum = Task[config.task]
    loader_helper = LoaderHelper(
        task=task_enum, k_folds=config.training.k_folds, data_root=config.data.root
    )
    model = build_model(config.model.name)

    # Train
    trainer = Trainer(config, model, loader_helper)
    uuid = trainer.train()

    print(f"\n✅ Training complete! Model UUID: {uuid}")
    print(f"📁 Weights saved to: {config.paths.weights}/{config.task}/{uuid}/")


if __name__ == "__main__":
    main()
