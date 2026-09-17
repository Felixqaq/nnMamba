#!/usr/bin/env python3
"""Run a sub-hour TAP-CT LoRA pilot on the fixed 512/200 COPD split.

The pilot adapts qkv/projection layers in the final transformer blocks and uses
a patient-level multi-task head for airflow-obstruction classification and
continuous FEV1/FVC regression.  Four representative windows are cached per
patient; two are sampled per patient per epoch.  The fixed holdout is never
used for fitting or threshold selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch import nn
from transformers import AutoImageProcessor, AutoModel

from extract_tapct_embeddings import (
    depth_starts,
    output_to_embedding,
    pad_to_window,
    preprocess_volume,
)
from tapct_multitask_ratio import (
    bootstrap_auc_ci,
    choose_threshold,
    classification_metrics,
    load_ratios,
)


RATIO_CUTOFF = 70.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--pft-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="fomofo/tap-ct-s-3d")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--cached-windows", type=int, default=4)
    parser.add_argument("--sampled-windows", type=int, default=2)
    parser.add_argument("--depth-window", type=int, default=12)
    parser.add_argument("--depth-stride", type=int, default=6)
    parser.add_argument("--last-blocks", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--regression-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--classification-loss-weight", type=float, default=1.0,
        help="set to 0 for a pure two-stage run: the trunk then learns only to "
             "predict the ratio and the cutoff is applied afterwards, matching "
             "what the Mamba runs do. The classification head still exists but "
             "receives no gradient, so its reported metrics are meaningless -- "
             "read ratio_head_at_70 and ratio_head_at_training_calibrated_cutoff.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--margin", type=float, default=7.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_status(path: Path, message: str) -> None:
    """Append a timestamped status line and print it immediately."""
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_paths(path: Path) -> dict[str, str]:
    """Map TAP-CT metadata patient IDs to NIfTI paths."""
    paths: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            paths[str(row["patient_id"])] = str(row["path"])
    if not paths:
        raise SystemExit(f"{path}: parsed no patient paths")
    return paths


def representative_starts(starts: list[int], count: int) -> list[int]:
    """Select fixed, lung-spanning window starts without using labels."""
    if count <= 0:
        raise ValueError("--cached-windows must be positive")
    if len(starts) == 1:
        return [starts[0]] * count
    fractions = np.linspace(0.12, 0.88, count)
    return [starts[int(round(value * (len(starts) - 1)))] for value in fractions]


def cache_is_valid(
    array_path: Path,
    manifest_path: Path,
    patient_ids: list[str],
    args: argparse.Namespace,
) -> bool:
    """Check whether a completed cache exactly matches this run."""
    if not array_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("patient_ids") != patient_ids:
            return False
        expected = [
            len(patient_ids),
            args.cached_windows,
            1,
            args.depth_window,
            224,
            224,
        ]
        if manifest.get("shape") != expected or manifest.get("model_id") != args.model_id:
            return False
        array = np.load(array_path, mmap_mode="r")
        return list(array.shape) == expected and array.dtype == np.float16
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def build_window_cache(
    patient_ids: list[str],
    paths: dict[str, str],
    args: argparse.Namespace,
    status_path: Path,
) -> tuple[Path, Path]:
    """Build or reuse a label-agnostic four-window patient cache."""
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = args.cache_dir / "windows.float16.npy"
    manifest_path = args.cache_dir / "cache_manifest.json"
    if cache_is_valid(array_path, manifest_path, patient_ids, args):
        append_status(status_path, f"CACHE_REUSED patients={len(patient_ids)}")
        return array_path, manifest_path

    append_status(
        status_path,
        f"CACHE_START patients={len(patient_ids)} windows_per_patient={args.cached_windows}",
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    resize_dims = tuple(int(value) for value in processor.resize_dims)
    if resize_dims != (224, 224):
        raise SystemExit(f"unexpected TAP-CT resize_dims={resize_dims}")
    shape = (
        len(patient_ids),
        args.cached_windows,
        1,
        args.depth_window,
        resize_dims[0],
        resize_dims[1],
    )
    temporary_path = args.cache_dir / "windows.float16.tmp.npy"
    cache = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.float16,
        shape=shape,
    )
    selected_starts: dict[str, list[int]] = {}
    started = time.perf_counter()
    for index, patient_id in enumerate(patient_ids, start=1):
        volume = preprocess_volume(processor, Path(paths[patient_id]))
        volume = pad_to_window(volume, args.depth_window)
        starts = depth_starts(
            int(volume.shape[2]), args.depth_window, args.depth_stride
        )
        chosen_starts = representative_starts(starts, args.cached_windows)
        windows = torch.cat(
            [
                volume[:, :, start : start + args.depth_window, :, :]
                for start in chosen_starts
            ],
            dim=0,
        )
        cache[index - 1] = windows.numpy().astype(np.float16, copy=False)
        selected_starts[patient_id] = chosen_starts
        if index % 25 == 0 or index == len(patient_ids):
            elapsed = time.perf_counter() - started
            append_status(
                status_path,
                f"CACHE_PROGRESS {index}/{len(patient_ids)} elapsed_min={elapsed / 60:.1f}",
            )
    cache.flush()
    del cache
    os.replace(temporary_path, array_path)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": args.model_id,
        "patient_ids": patient_ids,
        "paths": [paths[patient_id] for patient_id in patient_ids],
        "shape": list(shape),
        "dtype": "float16",
        "selection_fractions": np.linspace(0.12, 0.88, args.cached_windows).tolist(),
        "selected_starts": selected_starts,
        "depth_window": args.depth_window,
        "depth_stride": args.depth_stride,
        "label_agnostic": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    append_status(
        status_path,
        f"CACHE_COMPLETE size_gib={array_path.stat().st_size / 1024**3:.2f}",
    )
    return array_path, manifest_path


class LoRALinear(nn.Module):
    """Low-rank update around a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha / rank)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.base(features) + self.lora_b(self.lora_a(features)) * self.scale


class MultiTaskHead(nn.Module):
    """Patient-level classification and continuous-ratio head."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.classifier(hidden).squeeze(1), self.regressor(hidden).squeeze(1)


def prepare_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, MultiTaskHead, int, list[str]]:
    """Load TAP-CT and inject LoRA into the requested final blocks."""
    model = AutoModel.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.requires_grad_(False)
    model = model.to(device=device, dtype=torch.float16)
    blocks = model.model.blocks
    if args.last_blocks <= 0 or args.last_blocks > len(blocks):
        raise SystemExit(f"--last-blocks must be in [1, {len(blocks)}]")
    adapted_names: list[str] = []
    start = len(blocks) - args.last_blocks
    for block_index in range(start, len(blocks)):
        block = blocks[block_index]
        block.attn.qkv = LoRALinear(
            block.attn.qkv, args.rank, args.lora_alpha
        ).to(device)
        block.attn.proj = LoRALinear(
            block.attn.proj, args.rank, args.lora_alpha
        ).to(device)
        adapted_names.extend(
            [f"model.blocks.{block_index}.attn.qkv", f"model.blocks.{block_index}.attn.proj"]
        )
    head = MultiTaskHead(384, args.hidden_dim, args.dropout).to(device)
    trainable = sum(
        parameter.numel()
        for parameter in list(model.parameters()) + list(head.parameters())
        if parameter.requires_grad
    )
    return model, head, trainable, adapted_names


def train(
    model: nn.Module,
    head: MultiTaskHead,
    cache: np.ndarray,
    train_cache_indices: np.ndarray,
    train_labels: np.ndarray,
    train_ratios: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    status_path: Path,
) -> list[dict[str, float]]:
    """Train for a fixed number of epochs without consulting the holdout."""
    trainable_parameters = [
        parameter
        for parameter in list(model.parameters()) + list(head.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positives = float(train_labels.sum())
    positive_weight = torch.tensor(
        [(len(train_labels) - positives) / max(positives, 1.0)],
        dtype=torch.float32,
        device=device,
    )
    classification_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    regression_loss = nn.SmoothL1Loss(beta=0.5)
    ratio_mean = float(train_ratios.mean())
    ratio_std = max(float(train_ratios.std()), 1e-6)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        order = rng.permutation(len(train_cache_indices))
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_classification = 0.0
        running_regression = 0.0
        epoch_started = time.perf_counter()
        for step_number, position in enumerate(order, start=1):
            choices = rng.choice(
                args.cached_windows,
                size=args.sampled_windows,
                replace=False,
            )
            windows = np.array(
                cache[train_cache_indices[position], choices],
                dtype=np.float16,
                copy=True,
            )
            inputs = torch.from_numpy(windows).to(device=device, non_blocking=True)
            label = torch.tensor(
                [train_labels[position]], dtype=torch.float32, device=device
            )
            ratio_target = torch.tensor(
                [(train_ratios[position] - ratio_mean) / ratio_std],
                dtype=torch.float32,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(inputs)
                window_embeddings = output_to_embedding(output)
                patient_embedding = window_embeddings.mean(dim=0, keepdim=True)
                logits, predicted_ratio = head(patient_embedding)
                class_loss = classification_loss(logits.float(), label)
                ratio_loss = regression_loss(predicted_ratio.float(), ratio_target)
                loss = (args.classification_loss_weight * class_loss
                        + args.regression_loss_weight * ratio_loss)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            if (
                step_number % args.gradient_accumulation == 0
                or step_number == len(order)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())
            running_classification += float(class_loss.detach())
            running_regression += float(ratio_loss.detach())

        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": float(epoch),
            "loss": running_loss / len(order),
            "classification_loss": running_classification / len(order),
            "regression_loss": running_regression / len(order),
            "seconds": elapsed,
        }
        history.append(row)
        append_status(
            status_path,
            f"EPOCH {epoch}/{args.epochs} loss={row['loss']:.5f} "
            f"cls={row['classification_loss']:.5f} reg={row['regression_loss']:.5f} "
            f"elapsed_min={elapsed / 60:.2f}",
        )
    return history


def predict_patients(
    model: nn.Module,
    head: MultiTaskHead,
    cache: np.ndarray,
    cache_indices: np.ndarray,
    ratio_mean: float,
    ratio_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each patient by averaging all four cached window embeddings."""
    model.eval()
    head.eval()
    probabilities: list[float] = []
    predicted_ratios: list[float] = []
    with torch.inference_mode():
        for cache_index in cache_indices:
            windows = np.array(cache[cache_index], dtype=np.float16, copy=True)
            inputs = torch.from_numpy(windows).to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(inputs)
                embeddings = output_to_embedding(output).mean(dim=0, keepdim=True)
                logits, normalized_ratio = head(embeddings)
            probabilities.append(float(torch.sigmoid(logits.float()).cpu().item()))
            predicted_ratios.append(
                float(normalized_ratio.float().cpu().item() * ratio_std + ratio_mean)
            )
    return (
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(predicted_ratios, dtype=np.float64),
    )


def main() -> None:
    """Build the cache, fit LoRA, and evaluate the fixed holdout once."""
    args = parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.sampled_windows <= 0 or args.sampled_windows > args.cached_windows:
        raise SystemExit("--sampled-windows must be in [1, --cached-windows]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "training_status.txt"
    append_status(status_path, "STARTED TAP-CT LoRA quick pilot")

    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    train_ids = [str(value) for value in split["training_patient_ids"]]
    holdout_ids = [str(value) for value in split["validation_patient_ids"]]
    patient_ids = train_ids + holdout_ids
    if len(set(patient_ids)) != len(patient_ids):
        raise SystemExit("split contains duplicate or overlapping patient IDs")
    paths = load_paths(args.metadata_csv)
    ratios = load_ratios(args.pft_csv)
    missing = [
        patient_id
        for patient_id in patient_ids
        if patient_id not in paths or patient_id not in ratios
    ]
    if missing:
        raise SystemExit(f"missing CT path or FEV1/FVC for {len(missing)}: {missing[:5]}")

    cache_path, cache_manifest_path = build_window_cache(
        patient_ids, paths, args, status_path
    )
    cache = np.load(cache_path, mmap_mode="r")
    index_by_id = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    train_cache_indices = np.asarray(
        [index_by_id[patient_id] for patient_id in train_ids], dtype=int
    )
    holdout_cache_indices = np.asarray(
        [index_by_id[patient_id] for patient_id in holdout_ids], dtype=int
    )
    train_ratios = np.asarray([ratios[patient_id] for patient_id in train_ids])
    holdout_ratios = np.asarray([ratios[patient_id] for patient_id in holdout_ids])
    train_labels = (train_ratios < RATIO_CUTOFF).astype(int)
    holdout_labels = (holdout_ratios < RATIO_CUTOFF).astype(int)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    append_status(
        status_path,
        f"MODEL_LOAD train={len(train_ids)} abnormal={train_labels.sum()} "
        f"holdout={len(holdout_ids)} abnormal={holdout_labels.sum()}",
    )
    model, head, trainable_count, adapted_names = prepare_model(args, device)
    append_status(
        status_path,
        f"TRAIN_START trainable={trainable_count} blocks={args.last_blocks} "
        f"rank={args.rank} sampled_windows={args.sampled_windows}",
    )
    history = train(
        model,
        head,
        cache,
        train_cache_indices,
        train_labels,
        train_ratios,
        args,
        device,
        status_path,
    )

    append_status(status_path, "CALIBRATE_THRESHOLD_ON_TRAINING")
    ratio_mean = float(train_ratios.mean())
    ratio_std = max(float(train_ratios.std()), 1e-6)
    training_probabilities, predicted_training_ratios = predict_patients(
        model,
        head,
        cache,
        train_cache_indices,
        ratio_mean,
        ratio_std,
        device,
    )
    threshold = choose_threshold(train_labels, training_probabilities)
    regression_score_threshold = choose_threshold(
        train_labels, -predicted_training_ratios
    )

    append_status(status_path, "EVALUATE_FIXED_HOLDOUT_200")
    holdout_probabilities, predicted_holdout_ratios = predict_patients(
        model,
        head,
        cache,
        holdout_cache_indices,
        ratio_mean,
        ratio_std,
        device,
    )
    classification_holdout = classification_metrics(
        holdout_labels, holdout_probabilities, threshold
    )
    ratio_at_70 = classification_metrics(
        holdout_labels, -predicted_holdout_ratios, -RATIO_CUTOFF
    )
    ratio_calibrated = classification_metrics(
        holdout_labels, -predicted_holdout_ratios, regression_score_threshold
    )
    distance = np.abs(holdout_ratios - RATIO_CUTOFF)
    difficulty: dict[str, object] = {}
    for name, mask in (
        (f"borderline_abs_distance_lt_{args.margin:g}", distance < args.margin),
        (f"clear_abs_distance_ge_{args.margin:g}", distance >= args.margin),
    ):
        difficulty[name] = {
            "n": int(mask.sum()),
            "n_abnormal": int(holdout_labels[mask].sum()),
            "classification_head_auc": round(
                float(roc_auc_score(holdout_labels[mask], holdout_probabilities[mask])),
                5,
            ),
            "ratio_head_auc": round(
                float(
                    roc_auc_score(
                        holdout_labels[mask], -predicted_holdout_ratios[mask]
                    )
                ),
                5,
            ),
        }

    peak_memory = (
        float(torch.cuda.max_memory_allocated(device) / 1024**3)
        if device.type == "cuda"
        else None
    )
    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "pilot": True,
            "interpretation": (
                "Sub-hour directional LoRA pilot; confirm a promising result with "
                "longer training or repeated seeds before paper reporting."
            ),
            "model_id": args.model_id,
            "metadata_csv": str(args.metadata_csv),
            "split_json": str(args.split_json),
            "pft_csv": str(args.pft_csv),
            "cache_manifest": str(cache_manifest_path),
            "n_training": len(train_ids),
            "n_holdout": len(holdout_ids),
            "holdout_used_for_fitting_or_threshold_selection": False,
            "threshold_selection": "maximum training in-sample balanced accuracy",
            "epochs": args.epochs,
            "cached_windows_per_patient": args.cached_windows,
            "sampled_windows_per_patient_per_epoch": args.sampled_windows,
            "last_blocks": args.last_blocks,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_patients": args.gradient_accumulation,
            "regression_loss_weight": args.regression_loss_weight,
            "classification_loss_weight": args.classification_loss_weight,
            "objective": ("pure two-stage: ratio only" if args.classification_loss_weight == 0
                          else "multi-task: classification + ratio"),
            "seed": args.seed,
            "trainable_parameters": trainable_count,
            "adapted_modules": adapted_names,
            "peak_torch_allocated_gib": peak_memory,
        },
        "training_history": history,
        "training_in_sample": {
            "classification_head": classification_metrics(
                train_labels, training_probabilities, threshold
            ),
            "ratio_mae": round(
                float(mean_absolute_error(train_ratios, predicted_training_ratios)), 5
            ),
            "ratio_calibrated_cutoff": -float(regression_score_threshold),
        },
        "holdout": {
            "classification_head": classification_holdout,
            "classification_head_auc_95ci": bootstrap_auc_ci(
                holdout_labels, holdout_probabilities
            ),
            "ratio_head_at_70": ratio_at_70,
            "ratio_head_at_training_calibrated_cutoff": ratio_calibrated,
            "ratio_head_auc_95ci": bootstrap_auc_ci(
                holdout_labels, -predicted_holdout_ratios
            ),
            "ratio_mae": round(
                float(mean_absolute_error(holdout_ratios, predicted_holdout_ratios)), 5
            ),
            "by_difficulty": difficulty,
        },
        "patients": {
            patient_id: {
                "true_ratio": float(true_ratio),
                "true_label": "Abnormal" if true_label else "Normal",
                "prob_abnormal": float(probability),
                "predicted_ratio": float(predicted_ratio),
                "pred_label": (
                    "Abnormal" if probability >= threshold else "Normal"
                ),
            }
            for patient_id, true_ratio, true_label, probability, predicted_ratio in zip(
                holdout_ids,
                holdout_ratios,
                holdout_labels,
                holdout_probabilities,
                predicted_holdout_ratios,
                strict=True,
            )
        },
    }
    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    checkpoint = {
        "meta": payload["meta"],
        "lora_state": {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        },
        "head_state": {name: value.detach().cpu() for name, value in head.state_dict().items()},
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "classification_threshold": threshold,
        "ratio_score_threshold": regression_score_threshold,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    append_status(
        status_path,
        f"COMPLETE auc={classification_holdout['auc']:.5f} "
        f"accuracy={classification_holdout['accuracy']:.5f} "
        f"balanced_accuracy={classification_holdout['balanced_accuracy']:.5f}",
    )
    print(json.dumps(payload["holdout"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
