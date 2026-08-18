#!/usr/bin/env python
"""Zero-bandwidth A/B: does CT-RATE pretraining help the 54-case model?

Controlled comparison on the SAME 5 folds and SAME fine-tune protocol; the ONLY
difference is weight initialisation:
  - scratch     : fresh model  -> 5-fold CV train/eval on 54-case
  - pretrained  : model pretrained on the 20 CT-RATE pilot volumes, then the SAME
                  5-fold CV fine-tune/eval on 54-case (weights reloaded each fold)

Additive & standalone. Reuses data.dataset.load_ct, models.build_model,
core.evaluator.compute_classification_metrics. Label map Abnormal=0, Normal=1.

Run in the `nnMamba` env:
    python scripts/pretrain_finetune_pilot.py --pretrain-epochs 30 --finetune-epochs 40
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.evaluator import compute_classification_metrics
from data.dataset import load_ct
from models import build_model

CLASS_TO_IDX = {"Abnormal": 0, "Normal": 1}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.cmp.normal_v_abnormal.imageonly.yaml")
    ap.add_argument("--ctrate-dir", default="datasets/ctrate_pilot")
    ap.add_argument("--test-dir", default="../classification/datasets/normal_v_abnormal_54")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="figures/ctrate_pilot")
    return ap.parse_args()


def collect(root: str) -> list[tuple[str, int]]:
    pairs = []
    for group, idx in CLASS_TO_IDX.items():
        for f in sorted(glob.glob(os.path.join(root, group, "*.nii.gz"))):
            pairs.append((f, idx))
    return pairs


class ArrSet(Dataset):
    def __init__(self, items):  # items: list[(np.ndarray ct, int y)]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ct, y = self.items[i]
        return torch.from_numpy(ct), torch.tensor(y, dtype=torch.long)


def train_model(model, loader, device, epochs, lr, wd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, items, device, bs):
    model.eval()
    dl = DataLoader(ArrSet(items), batch_size=bs, shuffle=False)
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
    logits = torch.cat(logits); y = torch.cat(ys)
    probs = torch.softmax(logits, 1); preds = probs.argmax(1)
    return y, preds, probs


def summarize(name, fold_metrics, pooled_y, pooled_pred, pooled_score):
    import numpy as np
    accs = [m.accuracy for m in fold_metrics]
    bals = [m.balanced_accuracy for m in fold_metrics]
    sens = [m.sensitivity for m in fold_metrics]
    spec = [m.specificity for m in fold_metrics]
    from sklearn.metrics import confusion_matrix
    y = np.array(pooled_y); p = np.array(pooled_pred); s = np.array(pooled_score)
    cm = confusion_matrix(y, p, labels=[0, 1]).tolist()
    auc = roc_auc_score((y == 0).astype(int), s) if len(set(y.tolist())) > 1 else float("nan")
    return {
        "arm": name,
        "mean_accuracy": round(float(np.mean(accs)), 4), "std_accuracy": round(float(np.std(accs)), 4),
        "mean_balanced_accuracy": round(float(np.mean(bals)), 4),
        "mean_sensitivity": round(float(np.mean(sens)), 4),
        "mean_specificity": round(float(np.mean(spec)), 4),
        "pooled_auc": round(float(auc), 4),
        "pooled_cm_[Abn;Nor]": cm,
    }


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isize = tuple(cfg.data.image_size); win = cfg.data.intensity_window; norm = cfg.data.input_normalization
    lr = cfg.training.learning_rate; wd = cfg.training.weight_decay

    ct_pairs = collect(args.ctrate_dir)
    te_pairs = collect(args.test_dir)
    print(f"CT-RATE pretrain vols: {len(ct_pairs)} | 54-case: {len(te_pairs)}")

    print("caching volumes (once)...")
    ct_items = [(load_ct(p, isize, intensity_window=win, input_normalization=norm), y) for p, y in ct_pairs]
    te_items = [(load_ct(p, isize, intensity_window=win, input_normalization=norm), y) for p, y in te_pairs]
    te_labels = np.array([y for _, y in te_pairs])

    # 1) pretrain on all CT-RATE
    print(f"pretraining on CT-RATE ({args.pretrain_epochs} ep)...")
    pre = build_model(cfg.model, output_dim=2).to(device)
    ct_loader = DataLoader(ArrSet(ct_items), batch_size=args.batch_size, shuffle=True)
    train_model(pre, ct_loader, device, args.pretrain_epochs, lr, wd)
    pretrained_state = copy.deepcopy(pre.state_dict())
    del pre; torch.cuda.empty_cache()

    # 2) 5-fold CV on 54-case, two arms
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    arms = {"scratch": {"fm": [], "y": [], "p": [], "s": []},
            "pretrained": {"fm": [], "y": [], "p": [], "s": []}}
    for fold, (tr, te) in enumerate(skf.split(np.arange(len(te_items)), te_labels), 1):
        tr_items = [te_items[i] for i in tr]; ev_items = [te_items[i] for i in te]
        tr_loader = DataLoader(ArrSet(tr_items), batch_size=args.batch_size, shuffle=True)
        for arm in ("scratch", "pretrained"):
            model = build_model(cfg.model, output_dim=2).to(device)
            if arm == "pretrained":
                model.load_state_dict(pretrained_state)
            train_model(model, tr_loader, device, args.finetune_epochs, lr, wd)
            y, preds, probs = predict(model, ev_items, device, args.batch_size)
            m = compute_classification_metrics(y, preds, probs, num_classes=2)
            arms[arm]["fm"].append(m)
            arms[arm]["y"] += y.tolist(); arms[arm]["p"] += preds.tolist()
            arms[arm]["s"] += probs[:, 0].tolist()
            print(f"  fold {fold} [{arm:>10}] Acc={m.accuracy:.3f} BalAcc={m.balanced_accuracy:.3f} "
                  f"Sens={m.sensitivity:.3f} Spec={m.specificity:.3f}")
            del model; torch.cuda.empty_cache()

    results = [summarize(a, arms[a]["fm"], arms[a]["y"], arms[a]["p"], arms[a]["s"]) for a in ("scratch", "pretrained")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pretrain_finetune_result.json").write_text(json.dumps({
        "n_ctrate_pretrain": len(ct_pairs), "folds": args.folds,
        "pretrain_epochs": args.pretrain_epochs, "finetune_epochs": args.finetune_epochs,
        "arms": results}, indent=2))
    print("\n=== A/B: scratch vs CT-RATE-pretrained (same folds/protocol) ===")
    for r in results:
        print(f"{r['arm']:>10}: Acc={r['mean_accuracy']:.3f}±{r['std_accuracy']:.3f} "
              f"BalAcc={r['mean_balanced_accuracy']:.3f} Sens={r['mean_sensitivity']:.3f} "
              f"Spec={r['mean_specificity']:.3f} AUC={r['pooled_auc']:.3f} CM={r['pooled_cm_[Abn;Nor]']}")
    print(f"saved -> {out/'pretrain_finetune_result.json'}")


if __name__ == "__main__":
    main()
