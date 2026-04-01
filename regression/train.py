#!/usr/bin/env python
"""Train nnMamba regression model for CT-to-PFT angle prediction."""

import argparse
import os

import torch

from core.config import Config
from core.trainer import Trainer
from data.loader import LoaderHelper
from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nnMamba CT angle regressor")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.device_id

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device}")
    if not torch.cuda.is_available():
        print("Warning: GPU not detected. nnMamba is designed for CUDA training.")

    loader_helper = LoaderHelper(config)
    model_factory = lambda: build_model(config.model)

    trainer = Trainer(config=config, model_factory=model_factory, loader_helper=loader_helper)
    uuid = trainer.train()

    print(f"\nTraining complete. Run UUID: {uuid}")
    print(f"Weights saved to: {config.paths.weights / config.task / uuid}")


if __name__ == "__main__":
    main()
