#!/usr/bin/env python
"""Honest, non-cheating training/eval of the pure nnMamba classifier (Normal vs Abnormal).

Fixes the pipeline's optimistic bias (best epoch chosen on the TEST fold) by choosing
the epoch via an INNER CV on the outer-training split only; the outer TEST fold is
evaluated exactly once and never influences any choice. Adds train-only augmentation
(the classification dataset currently has none) as a legitimate accuracy lever.

  outer 5-fold (same StratifiedKFold seed as the pipeline) -> honest test
    inner k-fold on outer_train -> E* = epoch with best mean inner-val AUC
    retrain on full outer_train for E* -> evaluate outer_test once
  decision threshold is fit on TRAIN only (Youden's J), applied to test.

Additive & standalone; does NOT modify train.py or core/trainer.py.
Label map: Normal=0, Abnormal=1 (Abnormal is the positive class).

Run in the `nnMamba` env (long; use tmux):
    python scripts/train_nnmamba_honest.py --inner-folds 3 --max-epochs 40 --views 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import Task
from data.loader import LoaderHelper
from models import build_model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/normal_v_abnormal_54/")
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--views", type=int, default=3, help="train-only augmentation multiplier")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="../figures/nnmamba_honest")
    return ap.parse_args()


def augment(t: torch.Tensor) -> torch.Tensor:
    """Light train-only augmentation on a (1, D, H, W) CT tensor."""
    if torch.rand(()) < 0.5:
        t = torch.flip(t, dims=[3])  # left-right flip
    t = t * float(torch.empty(()).uniform_(0.95, 1.05))          # intensity scale
    t = t + torch.randn_like(t) * (0.05 * float(t.std()) + 1e-6)  # gaussian noise
    return t


def batches(items, idx, bs, aug=False, shuffle=False):
    idx = list(idx)
    if shuffle:
        np.random.shuffle(idx)
    for k in range(0, len(idx), bs):
        chunk = idx[k:k + bs]
        xs = []
        for i in chunk:
            t = items[i][0].clone().float()
            xs.append(augment(t) if aug else t)
        x = torch.stack(xs)
        y = torch.tensor([[items[i][1]] for i in chunk], dtype=torch.float32)
        yield x, y


def train_epochs(model, items, train_idx, device, epochs, bs, lr, wd, views):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    view_idx = list(train_idx) * views
    for _ in range(epochs):
        model.train()
        for x, y in batches(items, view_idx, bs, aug=True, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
    return model


@torch.no_grad()
def scores(model, items, idx, device, bs):
    model.eval()
    probs, ys = [], []
    for x, y in batches(items, idx, bs, aug=False):
        p = torch.sigmoid(model(x.to(device))).cpu().view(-1)
        probs.append(p); ys.append(y.view(-1))
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def train_to_curve(items, train_idx, val_idx, device, args):
    """Train to max_epochs, evaluating a fresh model at each checkpoint; return {epoch: val_auc}."""
    model = build_model("nnmamba").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.BCEWithLogitsLoss()
    view_idx = list(train_idx) * args.views
    curve = {}
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for x, y in batches(items, view_idx, args.batch_size, aug=True, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x.to(device)), y.to(device)).backward()
            opt.step()
        if epoch % args.eval_interval == 0:
            p, yv = scores(model, items, val_idx, device, args.batch_size)
            curve[epoch] = roc_auc_score(yv, p) if len(set(yv.tolist())) > 1 else 0.5
    del model; torch.cuda.empty_cache()
    return curve


def youden_threshold(p, y):
    """Threshold maximizing sensitivity+specificity-1, fit on TRAIN predictions."""
    best_t, best_j = 0.5, -1.0
    for t in np.unique(np.concatenate([[0.0], p, [1.0]])):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec - 1 > best_j:
            best_j, best_t = sens + spec - 1, float(t)
    return best_t


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    lh = LoaderHelper(task=Task.Normal_v_Abnormal, k_folds=args.outer_folds,
                      seed=args.seed, data_root=args.data_root)
    ds = lh.train_ds
    items = [(ds[i]["mri"], int(ds[i]["label"].item())) for i in range(len(ds))]
    labels = np.array([y for _, y in items])
    print(f"nnMamba honest: {len(items)} cases | views={args.views} | inner={args.inner_folds} "
          f"| max_epochs={args.max_epochs} | Normal=0 Abnormal=1")

    fold_metrics, chosen, pooled_y, pooled_p, pooled_pred = [], [], [], [], []
    for ofold, (otr, ote) in enumerate(lh.fold_indices, 1):
        otr = np.array(otr); ote = np.array(ote)
        otr_labels = labels[otr]

        # INNER: pick E* on inner-val AUC only
        inner = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)
        agg = {}
        for itr, ival in inner.split(otr, otr_labels):
            curve = train_to_curve(items, otr[itr], otr[ival], device, args)
            for e, v in curve.items():
                agg.setdefault(e, []).append(v)
        mean_curve = {e: float(np.mean(v)) for e, v in agg.items()}
        e_star = max(mean_curve, key=mean_curve.get)
        chosen.append(e_star)

        # RETRAIN on full outer_train at E*, threshold on train, eval test once
        model = build_model("nnmamba").to(device)
        train_epochs(model, items, otr, device, e_star, args.batch_size, args.lr, args.wd, args.views)
        ptr, ytr = scores(model, items, otr, device, args.batch_size)
        thr = youden_threshold(ptr, ytr)
        pte, yte = scores(model, items, ote, device, args.batch_size)
        pred = (pte >= thr).astype(int)
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = (pred == yte).mean()
        sens = tp / (tp + fn) if tp + fn else 0.0   # Abnormal recall
        spec = tn / (tn + fp) if tn + fp else 0.0   # Normal recall
        bal = 0.5 * (sens + spec)
        auc = roc_auc_score(yte, pte) if len(set(yte.tolist())) > 1 else float("nan")
        fold_metrics.append(dict(acc=acc, bal=bal, sens=sens, spec=spec, auc=auc))
        pooled_y += yte.tolist(); pooled_p += pte.tolist(); pooled_pred += pred.tolist()
        print(f"[outer {ofold}] E*={e_star} thr={thr:.2f} OUTER-TEST "
              f"Acc={acc:.3f} BalAcc={bal:.3f} Sens={sens:.3f} Spec={spec:.3f} AUC={auc:.3f}")
        del model; torch.cuda.empty_cache()

        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "nnmamba_honest_partial.json").write_text(json.dumps({
            "folds_done": ofold, "chosen_epochs": chosen,
            "running_mean_acc": round(float(np.mean([m["acc"] for m in fold_metrics])), 4),
            "per_fold": fold_metrics}, indent=2))

    y = np.array(pooled_y); pr = np.array(pooled_p); pd = np.array(pooled_pred)
    pooled_cm = confusion_matrix(y, pd, labels=[0, 1]).tolist()
    result = {
        "method": "pure nnMamba, honest nested-CV epoch selection + train-only aug",
        "label_map": "Normal=0, Abnormal=1 (positive=Abnormal)",
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "selected_epochs": chosen,
        "mean_accuracy": round(float(np.mean([m["acc"] for m in fold_metrics])), 4),
        "std_accuracy": round(float(np.std([m["acc"] for m in fold_metrics])), 4),
        "mean_balanced_accuracy": round(float(np.mean([m["bal"] for m in fold_metrics])), 4),
        "mean_sensitivity": round(float(np.mean([m["sens"] for m in fold_metrics])), 4),
        "mean_specificity": round(float(np.mean([m["spec"] for m in fold_metrics])), 4),
        "pooled_auc": round(float(roc_auc_score(y, pr)), 4),
        "pooled_cm_[Nor;Abn]": pooled_cm,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "nnmamba_honest_result.json").write_text(json.dumps(result, indent=2))
    print("\n=== PURE nnMAMBA — HONEST (no test peeking) ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
