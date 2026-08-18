#!/usr/bin/env python
"""Pilot: train image-only model on the CT-RATE pilot subset, TEST on the 54-case cohort.

Additive & standalone — does NOT touch the k-fold pipeline. It reuses the pipeline's
exact preprocessing (data.dataset.load_ct), model (models.build_model) and metrics
(core.evaluator.compute_classification_metrics) so results are directly comparable to
the existing image-only runs.

Design (agreed):
  - Train ONLY on CT-RATE pilot volumes (datasets/ctrate_pilot/{Abnormal,Normal}).
  - Test on ALL 54-case volumes as an external held-out set (no overlap, no CV).
  - Report at a FIXED epoch budget (no test-fold epoch selection -> no optimistic bias).
  - Label map matches prior runs: Abnormal=0, Normal=1.

Run in the `nnMamba` env:
    python scripts/train_ctrate_pilot.py --epochs 40
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Make the regression package root importable when run as scripts/train_ctrate_pilot.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.evaluator import compute_classification_metrics
from data.dataset import load_ct
from models import build_model

CLASS_TO_IDX = {"Abnormal": 0, "Normal": 1}  # matches existing normal_v_abnormal runs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CT-RATE -> 54-case transfer pilot")
    ap.add_argument("--config", default="config.cmp.normal_v_abnormal.imageonly.yaml")
    ap.add_argument("--ctrate-dir", default="datasets/ctrate_pilot")
    ap.add_argument("--test-dir", default="../classification/datasets/normal_v_abnormal_54")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="figures/ctrate_pilot")
    return ap.parse_args()


def collect(root: str) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for group, idx in CLASS_TO_IDX.items():
        for f in sorted(glob.glob(os.path.join(root, group, "*.nii.gz"))):
            pairs.append((f, idx))
    return pairs


class CTSet(Dataset):
    def __init__(self, pairs, image_size, window, norm):
        self.pairs = pairs
        self.cache = [
            (load_ct(p, image_size, intensity_window=window, input_normalization=norm), y)
            for p, y in pairs
        ]

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        ct, y = self.cache[i]
        return torch.from_numpy(ct), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def evaluate_on(model, loader, device) -> tuple:
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        out = model(x.to(device))
        all_logits.append(out.cpu())
        all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    m = compute_classification_metrics(y, preds, probs, num_classes=2)
    # AUC with Abnormal(0) as positive
    y_pos = (y == 0).long().numpy()
    score = probs[:, 0].numpy()
    auc = roc_auc_score(y_pos, score) if len(set(y_pos.tolist())) > 1 else float("nan")
    return m, auc


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = tuple(cfg.data.image_size)
    window = cfg.data.intensity_window
    norm = cfg.data.input_normalization

    train_pairs = collect(args.ctrate_dir)
    test_pairs = collect(args.test_dir)
    if not train_pairs:
        raise SystemExit(f"No CT-RATE volumes under {args.ctrate_dir}")
    print(f"Train (CT-RATE): {len(train_pairs)} | Test (54-case): {len(test_pairs)}")
    print("  train class counts:", {g: sum(1 for _, i in train_pairs if i == idx)
                                     for g, idx in CLASS_TO_IDX.items()})

    train_dl = DataLoader(CTSet(train_pairs, image_size, window, norm),
                          batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dl = DataLoader(CTSet(test_pairs, image_size, window, norm),
                         batch_size=args.batch_size, shuffle=False)

    model = build_model(cfg.model, output_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_dl:
            opt.zero_grad(set_to_none=True)
            out = model(x.to(device))
            loss = loss_fn(out, y.to(device))
            loss.backward()
            opt.step()
            tot += float(loss)
        if epoch % 5 == 0 or epoch == args.epochs:
            m, auc = evaluate_on(model, test_dl, device)
            print(f"epoch {epoch:>3} loss={tot/len(train_dl):.4f} | 54-case "
                  f"Acc={m.accuracy:.3f} BalAcc={m.balanced_accuracy:.3f} "
                  f"Sens={m.sensitivity:.3f} Spec={m.specificity:.3f} AUC={auc:.3f} CM={m.confusion_matrix}")

    m, auc = evaluate_on(model, test_dl, device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "train_source": "CT-RATE pilot", "n_train": len(train_pairs),
        "test_source": "54-case held-out", "n_test": len(test_pairs),
        "epochs": args.epochs, "note": "fixed-epoch budget, no test-fold selection",
        "final": {"accuracy": m.accuracy, "balanced_accuracy": m.balanced_accuracy,
                  "sensitivity": m.sensitivity, "specificity": m.specificity,
                  "auc_abnormal_pos": round(float(auc), 5),
                  "confusion_matrix_[Abn;Nor]": m.confusion_matrix},
    }
    (out_dir / "ctrate_pilot_result.json").write_text(json.dumps(result, indent=2))
    print("\n=== FINAL (train CT-RATE -> test 54-case, fixed epoch) ===")
    print(json.dumps(result["final"], indent=2))
    print(f"saved -> {out_dir/'ctrate_pilot_result.json'}")


if __name__ == "__main__":
    main()
