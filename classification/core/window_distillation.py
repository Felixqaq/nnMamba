"""Feature-level cross-window distillation and budget-matched training helpers."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def classification_logit(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Support original nnMamba and the project's hybrid two-class head."""
    head = model.mlp if hasattr(model, "mlp") else model.head
    output = head(features)
    if output.ndim == 2 and output.shape[1] == 2:
        return output[:, 1] - output[:, 0]
    return output.flatten()


def feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Raw-feature MSE as in the paper; teacher cannot receive gradients."""
    if student.shape != teacher.shape:
        raise ValueError("Teacher and student feature shapes must match")
    return F.mse_loss(student.float(), teacher.detach().float())


@torch.no_grad()
def predict(model, loader, window: str, device: torch.device, check_deadline) -> tuple:
    model.eval()
    labels, probabilities, ids = [], [], []
    for batch in loader:
        check_deadline()
        logits = classification_logit(model, model.forward_features(batch[window].to(device)))
        probabilities.extend(logits.sigmoid().cpu().tolist())
        labels.extend(batch["label"].tolist())
        ids.extend(batch["patient_id"])
    return np.asarray(labels), np.asarray(probabilities), ids


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
            "threshold": float(threshold), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    return float(thresholds[valid][np.argmax((tpr - fpr)[valid])])


def train_stage(model, train_loader, val_loader, window: str, device: torch.device,
                epochs: int, lr: float, weight_decay: float, checkpoint: Path,
                check_deadline, teacher=None, teacher_window: str | None = None,
                seed: int = 42) -> dict:
    """Select checkpoints on validation AUC only; fixed epoch counts for controls."""
    seed_everything(seed)
    model.to(device)
    if teacher is not None:
        freeze_teacher(teacher.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_auc, history, started = -1.0, [], time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total_bce = total_kd = 0.0
        count = 0
        for batch in train_loader:
            check_deadline()
            x = batch[window].to(device)
            labels = batch["label"].to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            # mamba_ssm's custom kernels vary in autocast support; use fp32 backbone.
            features = model.forward_features(x)
            logits = classification_logit(model, features)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            kd = bce.new_zeros(())
            if teacher is not None:
                with torch.no_grad():
                    target = teacher.forward_features(batch[teacher_window].to(device))
                kd = feature_loss(features, target)
            loss = bce if teacher is None else 0.5 * bce + 0.5 * kd
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_bce += bce.item() * len(labels)
            total_kd += kd.item() * len(labels)
            count += len(labels)
        y, p, _ = predict(model, val_loader, window, device, check_deadline)
        auc = float(roc_auc_score(y, p))
        history.append({"epoch": epoch, "bce": total_bce / count,
                        "feature_mse": total_kd / count, "val_auc": auc})
        print(f"{checkpoint.stem}: {history[-1]}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({"state_dict": model.state_dict(), "window": window,
                        "epoch": epoch, "val_auc": auc}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    return {"best_val_auc": best_auc, "history": history,
            "seconds": time.monotonic() - started}


def paired_auc_interval(labels: np.ndarray, control: np.ndarray,
                        distilled: np.ndarray, seed: int = 42) -> list[float]:
    """Patient-paired bootstrap for KD minus control; not evidence of equivalence."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(1000):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        deltas.append(roc_auc_score(labels[index], distilled[index])
                      - roc_auc_score(labels[index], control[index]))
    return np.quantile(deltas, [0.025, 0.975]).tolist()
