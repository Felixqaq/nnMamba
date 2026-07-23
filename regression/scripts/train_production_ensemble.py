"""Retrain the 5-member production ensemble on 100% of the dataset.

No held-out set exists by design, so there is no early stopping, no best-epoch
selection, and no ReduceLROnPlateau (it needs a monitored validation score).
Training runs a fixed epoch budget with CosineAnnealingLR and takes the final
epoch. Members differ only by seed.

Data pipeline (dataset, augmentation, balanced sampling) is reused verbatim from
RegressionLoaderHelper by overriding fold_indices to a single all-data fold, so
the training distribution matches the cross-validated runs.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402

# Reference numbers from prior nested cross-validation. NOT measured on this run.
NESTED_CV_REFERENCE = {"accuracy": 0.726, "auc": 0.803, "source": "nested CV, 66-case, RQ1 image-only"}


def all_data_train_loader(loader_helper):
    """Return a training DataLoader covering every case.

    Overrides fold_indices with a single fold whose train split is all indices
    and whose val split is empty, then reuses get_train_dl so augmentation and
    balanced sampling behave exactly as in cross-validated training.
    """
    n = len(loader_helper.dataset)
    loader_helper.fold_indices = [(list(range(n)), [])]
    return loader_helper.get_train_dl(0, shuffle=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_member(config, loader_helper, seed: int, epochs: int, device: str = "cuda") -> nn.Module:
    """Train a single ensemble member on all data for a fixed epoch budget."""
    _set_seed(seed)
    model = build_model(config.model, output_dim=2).to(device)
    model.train()

    train_dl = all_data_train_loader(loader_helper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    loss_fn = nn.CrossEntropyLoss()
    clip = float(getattr(config.training, "clip_grad_norm", 0.0) or 0.0)

    for epoch in range(1, int(epochs) + 1):
        running = 0.0
        nb = 0
        for batch in train_dl:
            ct = batch["ct"].to(device).float()
            target = batch["target"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(ct)
            loss = loss_fn(logits, target)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += float(loss.item())
            nb += 1
        scheduler.step()
        print(f"  seed {seed} | epoch {epoch}/{epochs} | loss {running / max(nb, 1):.4f}")

    model.eval()
    return model


def write_metrics_json(out_dir: Path, n_cases: int, epochs: int, seeds: list[int]) -> None:
    """Record what this release is, and be explicit that it has no held-out score."""
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_training_cases": int(n_cases),
        "epochs": int(epochs),
        "seeds": list(seeds),
        "held_out": False,
        "note": (
            "Trained on 100% of the dataset: no held-out set, no early stopping, "
            "no best-epoch selection. This release has NO measured performance. "
            "Quote the nested-CV reference below, labelled as a reference."
        ),
        "nested_cv_reference": NESTED_CV_REFERENCE,
    }
    (Path(out_dir) / "metrics.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production ensemble on all data")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.normal_v_abnormal.imageonly.aug5.ensemble.yaml"),
    )
    parser.add_argument("--out", required=True, help="Output release dir, e.g. release/2026-07-23")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_yaml(args.config)
    loader_helper = LoaderHelper(config)
    n_cases = len(loader_helper.dataset)
    seeds = [args.base_seed + i for i in range(args.members)]

    print(f"Production ensemble: {args.members} members x {args.epochs} epochs on {n_cases} cases (no held-out)")
    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== member {i}/{args.members} (seed {seed}) ===")
        model = train_one_member(config, loader_helper, seed=seed, epochs=args.epochs, device="cuda")
        torch.save({"state_dict": model.state_dict(), "seed": seed}, out_dir / f"member_{i}.pth")
        print(f"  saved {out_dir / f'member_{i}.pth'}")

    write_metrics_json(out_dir, n_cases=n_cases, epochs=args.epochs, seeds=seeds)
    print(f"\nDone. Release dir: {out_dir}")


if __name__ == "__main__":
    main()
