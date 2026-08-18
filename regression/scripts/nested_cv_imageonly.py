#!/usr/bin/env python
"""Nested cross-validation for the image-only Normal-vs-Abnormal model.

Produces an UNBIASED estimate: the outer test fold is NEVER used for model
selection. The epoch budget (the thing the normal pipeline selects on the test
fold) is chosen by an INNER CV on the outer-training split only.

  outer 5-fold (StratifiedKFold, seed) -> honest test
    inner 4-fold on outer_train -> pick E* = epoch with best mean inner-val macro-F1
    retrain on full outer_train for E* epochs -> evaluate ONCE on outer_test

Additive & standalone. Reuses data.dataset.load_ct, data.transforms.RandomCTAugmentation,
models.build_model, core.evaluator.compute_classification_metrics. Train-only 5x
augmentation + class balancing mirror the headline config so the number is comparable.
Label map Abnormal=0, Normal=1.

Run in the `nnMamba` env (long; use tmux):
    python scripts/nested_cv_imageonly.py --config config.normal_v_abnormal.imageonly.augmentationX5.yaml
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
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.evaluator import compute_classification_metrics
from data.dataset import load_ct
from data.transforms import RandomCTAugmentation
from models import build_model

CLASS_TO_IDX = {"Abnormal": 0, "Normal": 1}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.normal_v_abnormal.imageonly.augmentationX5.yaml")
    ap.add_argument("--data-dir", default="../classification/datasets/normal_v_abnormal_54")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figures/nested_cv_imageonly")
    return ap.parse_args()


def collect(root):
    pairs = []
    for group, idx in CLASS_TO_IDX.items():
        for f in sorted(glob.glob(os.path.join(root, group, "*.nii.gz"))):
            pairs.append((f, idx))
    return pairs


class EvalSet(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ct, y = self.items[i]
        return torch.from_numpy(ct).float(), torch.tensor(y, dtype=torch.long)


class TrainAugSet(Dataset):
    def __init__(self, items, aug, views):
        self.items = items
        self.aug = aug
        self.views = views

    def __len__(self):
        return len(self.items) * self.views

    def __getitem__(self, i):
        ct, y = self.items[i % len(self.items)]
        t = torch.from_numpy(ct).float()
        s = self.aug({"ct": t, "label": torch.tensor(y)})
        return s["ct"].float(), torch.tensor(y, dtype=torch.long)


def make_train_loader(items, aug, views, bs):
    ds = TrainAugSet(items, aug, views)
    labels = [y for _, y in items] * views
    counts = np.bincount(labels, minlength=2).astype(float)
    w = np.array([1.0 / counts[y] for y in labels])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)
    # Augmentation is a CPU-bound 3D affine resample; with the default num_workers=0 it
    # runs inline and starves the GPU (~40% idle). persistent_workers keeps them alive
    # across the epoch loop, which re-iterates this loader up to max_epochs times.
    return DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )


@torch.no_grad()
def eval_macro_f1(model, items, device, bs):
    model.eval()
    dl = DataLoader(EvalSet(items), batch_size=bs, shuffle=False)
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x.to(device)).cpu()); ys.append(y)
    logits = torch.cat(logits); y = torch.cat(ys)
    probs = torch.softmax(logits, 1); preds = probs.argmax(1)
    m = compute_classification_metrics(y, preds, probs, num_classes=2)
    return m, y, preds, probs


def train_record_curve(cfg, aug, views, train_items, val_items, device, max_epochs, ev, bs, lr, wd):
    """Train to max_epochs; return {epoch: val_macro_f1}."""
    model = build_model(cfg.model, output_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_train_loader(train_items, aug, views, bs)
    curve = {}
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if epoch % ev == 0:
            m, *_ = eval_macro_f1(model, val_items, device, bs)
            curve[epoch] = m.macro_f1
    del model; torch.cuda.empty_cache()
    return curve


def train_fixed(cfg, aug, views, train_items, device, epochs, bs, lr, wd):
    model = build_model(cfg.model, output_dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    loader = make_train_loader(train_items, aug, views, bs)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
    return model


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isize = tuple(cfg.data.image_size); win = cfg.data.intensity_window; norm = cfg.data.input_normalization
    lr = cfg.training.learning_rate; wd = cfg.training.weight_decay
    a = cfg.data.augmentation
    views = int(a.views_per_sample)
    aug = RandomCTAugmentation(
        enabled=True, probability=a.probability,
        class_indices=tuple(a.class_indices) if a.class_indices is not None else (0, 1),
        rotation_degrees=a.rotation_degrees, translation_fraction=a.translation_fraction,
        scale_range=tuple(a.scale_range), intensity_scale_range=tuple(a.intensity_scale_range),
        intensity_shift_range=tuple(a.intensity_shift_range), noise_std=a.noise_std,
    )

    pairs = collect(args.data_dir)
    labels = np.array([y for _, y in pairs])
    print(f"54-case: {len(pairs)} | views={views} | max_epochs={args.max_epochs} | "
          f"outer={args.outer_folds} inner={args.inner_folds}")
    print("caching volumes...")
    items = [(load_ct(p, isize, intensity_window=win, input_normalization=norm), y) for p, y in pairs]

    outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    fold_metrics, pooled_y, pooled_p, pooled_s, chosen_epochs = [], [], [], [], []

    for ofold, (otr, ote) in enumerate(outer.split(np.arange(len(items)), labels), 1):
        outer_train = [items[i] for i in otr]
        outer_test = [items[i] for i in ote]
        otr_labels = labels[otr]

        # INNER: average val macro-F1 curve to pick E*
        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)
        agg = {}
        for itr, ival in inner.split(np.arange(len(outer_train)), otr_labels):
            tr_items = [outer_train[i] for i in itr]
            val_items = [outer_train[i] for i in ival]
            curve = train_record_curve(cfg, aug, views, tr_items, val_items, device,
                                        args.max_epochs, args.eval_interval, args.batch_size, lr, wd)
            for e, f1 in curve.items():
                agg.setdefault(e, []).append(f1)
        mean_curve = {e: float(np.mean(v)) for e, v in agg.items()}
        e_star = max(mean_curve, key=mean_curve.get)
        chosen_epochs.append(e_star)
        print(f"[outer {ofold}] inner-selected E*={e_star} (mean inner-val F1={mean_curve[e_star]:.3f})")

        # RETRAIN on full outer_train at E*, evaluate outer_test ONCE
        model = train_fixed(cfg, aug, views, outer_train, device, e_star, args.batch_size, lr, wd)
        m, y, preds, probs = eval_macro_f1(model, outer_test, device, args.batch_size)
        fold_metrics.append(m)
        pooled_y += y.tolist(); pooled_p += preds.tolist(); pooled_s += probs[:, 0].tolist()
        print(f"[outer {ofold}] OUTER-TEST Acc={m.accuracy:.3f} BalAcc={m.balanced_accuracy:.3f} "
              f"Sens={m.sensitivity:.3f} Spec={m.specificity:.3f} (E*={e_star})")
        del model; torch.cuda.empty_cache()
        # crash-resilient partial save after every outer fold
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "nested_cv_partial.json").write_text(json.dumps({
            "folds_done": ofold,
            "running_mean_accuracy": round(float(np.mean([fm.accuracy for fm in fold_metrics])), 4),
            "running_mean_balanced_accuracy": round(float(np.mean([fm.balanced_accuracy for fm in fold_metrics])), 4),
            "per_fold": [{"fold": i + 1, "E_star": chosen_epochs[i],
                          "accuracy": fm.accuracy, "balanced_accuracy": fm.balanced_accuracy,
                          "sensitivity": fm.sensitivity, "specificity": fm.specificity}
                         for i, fm in enumerate(fold_metrics)],
        }, indent=2))

    y = np.array(pooled_y); p = np.array(pooled_p); s = np.array(pooled_s)
    cm = confusion_matrix(y, p, labels=[0, 1]).tolist()
    auc = roc_auc_score((y == 0).astype(int), s) if len(set(y.tolist())) > 1 else float("nan")
    result = {
        "method": "nested CV (unbiased) image-only 5x aug",
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "selected_epochs_per_outer_fold": chosen_epochs,
        "mean_accuracy": round(float(np.mean([m.accuracy for m in fold_metrics])), 4),
        "std_accuracy": round(float(np.std([m.accuracy for m in fold_metrics])), 4),
        "mean_balanced_accuracy": round(float(np.mean([m.balanced_accuracy for m in fold_metrics])), 4),
        "mean_sensitivity": round(float(np.mean([m.sensitivity for m in fold_metrics])), 4),
        "mean_specificity": round(float(np.mean([m.specificity for m in fold_metrics])), 4),
        "pooled_auc": round(float(auc), 4),
        "pooled_cm_[Abn;Nor]": cm,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "nested_cv_result.json").write_text(json.dumps(result, indent=2))
    print("\n=== NESTED CV (UNBIASED) ===")
    print(json.dumps(result, indent=2))
    print(f"saved -> {out/'nested_cv_result.json'}")


if __name__ == "__main__":
    main()
