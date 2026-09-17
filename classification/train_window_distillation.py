"""Run a two-window nnMamba pilot with a matched continued-training control."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import sys

import torch
from torch.utils.data import DataLoader

from core.window_distillation import (freeze_teacher, metrics, paired_auc_interval,
    predict, seed_everything, train_stage, validation_threshold)
from data.window_distillation import WINDOWS, WindowDataset, prepare_cache, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hu-confirmed", action="store_true",
                        help="Assert input is original HU, not normalized or windowed data")
    parser.add_argument("--shape", type=int, nargs=3, default=[112, 136, 112])
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per stage (four stages)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", choices=["nnmamba", "hybrid_mamba_attention"], default="nnmamba")
    parser.add_argument("--minutes", type=float, default=180)
    args = parser.parse_args()
    if not args.hu_confirmed:
        parser.error("Verify the source retains HU and pass --hu-confirmed")
    if min(args.shape) < 32 or args.epochs < 1 or args.batch_size < 2 or args.minutes <= 0:
        parser.error("Require shape >=32, epochs >=1, batch-size >=2, minutes >0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this nnMamba experiment")
    # Lazy import allows --help and manifest tooling without custom CUDA kernels.
    if args.backbone == "nnmamba":
        from networks.window_mamba_adapter import WindowMambaAdapter
        factory = WindowMambaAdapter
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from regression.networks.hybrid_mamba_attention_regressor import HybridMambaAttentionRegressor
        factory = lambda: HybridMambaAttentionRegressor(
            in_channels=1, num_classes=2, base_channels=32, depths=(3, 3, 3),
            head_hidden_dim=256, dropout=0.3, attn_heads=8, attn_layers=1,
            attn_mlp_ratio=2.0, attn_dropout=0.1)

    records = read_manifest(args.manifest.resolve())
    if sum(r["split"] == "train" for r in records) < args.batch_size:
        raise ValueError("Training cohort must contain at least one complete batch")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + args.minutes * 60

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("Experiment time budget exhausted; checkpoints retained")

    report = {"status": "running", "protocol": "two-window nnMamba pilot, not paper reproduction",
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "cohort": {s: {str(y): sum(r["split"] == s and r["label"] == y for r in records)
                             for y in (0, 1)} for s in ("train", "val", "test")}}

    def save_report() -> None:
        report["elapsed_seconds"] = time.monotonic() - start
        (args.output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    save_report()
    try:
        stats = prepare_cache(records, args.output / "cache", tuple(args.shape), check_deadline)
        report["normalization"] = stats
        loaders = {split: DataLoader(WindowDataset(records, args.output / "cache", stats, split),
                   batch_size=args.batch_size, shuffle=split == "train",
                   drop_last=split == "train", num_workers=0)
                   for split in ("train", "val", "test")}
        device = torch.device("cuda")
        seed_everything(args.seed)
        initial = factory().cpu()
        trained, stages = {}, {}

        def run(model, window, name, teacher=None, teacher_window=None):
            stage = train_stage(model, loaders["train"], loaders["val"], window, device,
                args.epochs, args.learning_rate, args.weight_decay, args.output / f"{name}.pt",
                check_deadline, teacher=teacher, teacher_window=teacher_window, seed=args.seed)
            stages[name] = stage
            report["stages"] = stages
            save_report()
            return model

        for window in WINDOWS:
            trained[window] = run(copy.deepcopy(initial), window, f"baseline_{window}").cpu()
        teacher_window = max(WINDOWS, key=lambda w: stages[f"baseline_{w}"]["best_val_auc"])
        student_window = next(w for w in WINDOWS if w != teacher_window)
        report.update(teacher_window=teacher_window, student_window=student_window)
        # Both arms start from the same selected student checkpoint, reset optimizer
        # and RNG, and consume exactly the same number/order of training batches.
        control = run(copy.deepcopy(trained[student_window]), student_window, "control").cpu()
        teacher = freeze_teacher(trained[teacher_window].to(device))
        distilled = run(copy.deepcopy(trained[student_window]), student_window, "distilled",
                        teacher, teacher_window).cpu()
        teacher.cpu()
        predictions, scores = {}, {}
        for name, model, window in [("teacher", teacher, teacher_window),
                                   ("student_initial", trained[student_window], student_window),
                                   ("control", control, student_window),
                                   ("distilled", distilled, student_window)]:
            model.to(device)
            y_val, p_val, _ = predict(model, loaders["val"], window, device, check_deadline)
            threshold = validation_threshold(y_val, p_val)
            y, p, ids = predict(model, loaders["test"], window, device, check_deadline)
            scores[name] = metrics(y, p, threshold)
            predictions[name] = p
            model.cpu()
        with (args.output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "label", *predictions])
            writer.writerows([patient, int(label), *[float(p[i]) for p in predictions.values()]]
                             for i, (patient, label) in enumerate(zip(ids, y)))
        report["test"] = scores
        report["distilled_minus_control"] = {
            key: scores["distilled"][key] - scores["control"][key]
            for key in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity")}
        report["delta_auc_bootstrap_95ci"] = paired_auc_interval(
            y, predictions["control"], predictions["distilled"], args.seed)
        report["status"] = "complete"
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_report()


if __name__ == "__main__":
    main()
