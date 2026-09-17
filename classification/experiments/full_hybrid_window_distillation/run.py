"""Resumable five-window Hybrid-Mamba training, distillation and stacking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from .model import build, forward
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data.window_distillation import read_manifest
from core.window_distillation import metrics, paired_auc_interval, validation_threshold
from experiments.full_window_distillation.data import WINDOWS, RawSlices, atomic_json, transform_window
from .data import prepare


def atomic_save(path: Path, payload: dict) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


@torch.no_grad()
def predict(model, loader, window, stats, deadline):
    model.eval()
    labels, probs, ids, features = [], [], [], []
    for batch in loader:
        deadline()
        x = transform_window(batch["hu"].cuda(), window, stats)
        with torch.autocast("cuda", enabled=False):
            logits, h = forward(model, x)
        labels.extend(batch["label"].tolist())
        probs.extend(logits.float().sigmoid().cpu().tolist())
        features.append(h.float().cpu())
        ids.extend(batch["patient_id"])
    return np.asarray(labels), np.asarray(probs), ids, torch.cat(features)


def train_stage(name, window, records, cache, stats, output, args, deadline,
                initialization: Path, teacher_features=None, matched_epochs: int | None = None):
    stage = output / name
    stage.mkdir(exist_ok=True)
    done = stage / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train = DataLoader(RawSlices(records, cache, "train", teacher_features), batch_size=1,
                       shuffle=True, num_workers=2, pin_memory=True)
    val = DataLoader(RawSlices(records, cache, "val"), batch_size=1, num_workers=2, pin_memory=True)
    model = build().cuda()
    model.load_state_dict(torch.load(initialization, map_location="cpu", weights_only=False)["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_epoch, best_loss, patience, best_epoch, history = 0, float("inf"), 0, 0, []
    latest = stage / "last.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_loss, patience = state["epoch"], state["best_loss"], state["patience"]
        best_epoch, history = state["best_epoch"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(start_epoch + 1, (matched_epochs or args.epochs) + 1):
        if matched_epochs is None and patience >= args.patience:
            break
        deadline()
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_bce = total_kd = 0.
        for index, batch in enumerate(train):
            deadline()
            x = transform_window(batch["hu"].cuda(non_blocking=True), window, stats)
            labels = batch["label"].cuda().float()
            with torch.autocast("cuda", enabled=False):
                logits, h = forward(model, x)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                kd = torch.zeros_like(bce)
                if teacher_features is not None:
                    kd = torch.nn.functional.mse_loss(h.float(), batch["teacher"].cuda().float())
                loss = bce if teacher_features is None else .5*bce + .5*kd
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {name}")
            group_start = (index // 4) * 4
            group_size = min(4, len(train) - group_start)
            scaler.scale(loss / group_size).backward()
            if (index+1) % 4 == 0 or index+1 == len(train):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_bce += bce.item()
            total_kd += kd.item()
        y, p, _, _ = predict(model, val, window, stats, deadline)
        clipped = np.clip(p, 1e-7, 1-1e-7)
        val_loss = float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean())
        auc = float(roc_auc_score(y, p))
        if val_loss < best_loss:
            best_loss, patience, best_epoch = val_loss, 0, epoch
            atomic_save(stage / "best.pt", {"model": model.state_dict(), "epoch": epoch,
                                             "window": window, "val_auc": auc})
        else:
            patience += 1
        scheduler.step()
        history.append({"epoch": epoch, "train_bce": total_bce/len(train),
                        "train_kd": total_kd/len(train), "val_loss": val_loss,
                        "val_auc": auc, "seconds": time.monotonic()-started})
        atomic_save(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
            "best_loss": best_loss, "best_epoch": best_epoch, "patience": patience, "history": history,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()})
        atomic_json(stage / "progress.json", {"history": history, "best_epoch": best_epoch})
        print(name, history[-1], flush=True)
    best = torch.load(stage / "best.pt", map_location="cpu", weights_only=False)
    result = {"best_epoch": best["epoch"], "val_auc": best["val_auc"],
              "epochs_run": len(history), "history": history}
    atomic_json(done, result)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=23)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.hours <= 0 or args.epochs < 1 or args.patience < 1:
        parser.error("hours, epochs and patience must be positive")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records = read_manifest(args.manifest.resolve())
    source = json.loads(args.source_summary.read_text())
    source_records = {r["patient_id"]: r for r in source["records"] if r["ok"]}
    for row in records:
        original = source_records[row["patient_id"]]
        if row["label"] != int(float(original["fev1_fvc_pct"]) < 70):
            raise ValueError("Manifest label differs from source spirometry")
        if not Path(original["dicom_dir"]).is_dir():
            raise FileNotFoundError("Original DICOM directory unavailable")
    config = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(args.source_summary.read_bytes()).hexdigest(),
              "windows": WINDOWS, "epochs": args.epochs, "patience": args.patience,
              "seed": args.seed, "microbatch": 1, "accumulate": 4, "amp": "disabled_fp32",
              "shape": [32, 512, 512], "architecture": "HybridMambaAttention 3D depths333 head256 feature352",
              "sampling": "32 uniform unique slices at 40%-90% original apex-to-base; expand if needed",
              "limitations": ["hybrid uses GroupNorm; differs from paper SE-ResNet50 BatchNorm",
                "sampling range is explicit interpretation of underspecified paper",
                "local 777-patient cohort; mixed pre/post bronchodilator labels",
                "no promise of reproducing published metrics"]}
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg_path = output / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != json.loads(json.dumps(config)):
        raise ValueError("Resume configuration differs")
    atomic_json(cfg_path, config)
    report_path = output / "status.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"stages": {}}
    if report.get("status") == "complete":
        print("Already complete", flush=True)
        return
    start = time.monotonic()
    deadline_time = start + args.hours * 3600

    def deadline():
        if time.monotonic() >= deadline_time:
            raise TimeoutError("Overnight budget reached; resume from saved epochs")

    def status(stage):
        report.update(status="running", active_stage=stage, invocation_seconds=time.monotonic()-start)
        atomic_json(report_path, report)

    try:
        status("prepare_original_DICOM")
        cache = output / "cache"
        stats = prepare(records, source_records, cache, fingerprint, deadline)
        initial = output / "initial.pt"
        if not initial.exists():
            torch.manual_seed(args.seed)
            atomic_save(initial, {"model": build().state_dict()})
        for window in WINDOWS:
            name = f"baseline_{window}"
            status(name)
            report["stages"][name] = train_stage(name, window, records, cache, stats, output,
                                                 args, deadline, initial)
        teacher_window = max(WINDOWS, key=lambda w: report["stages"][f"baseline_{w}"]["val_auc"])
        report["teacher_window"] = teacher_window
        teacher_path = output / "teacher_features.pt"
        status("cache_frozen_teacher")
        if not teacher_path.exists():
            teacher = build().cuda().eval().requires_grad_(False)
            teacher.load_state_dict(torch.load(output/f"baseline_{teacher_window}/best.pt",
                                               map_location="cpu", weights_only=False)["model"])
            loader = DataLoader(RawSlices(records, cache, "train"), batch_size=1, num_workers=2)
            _, _, ids, h = predict(teacher, loader, teacher_window, stats, deadline)
            atomic_save(teacher_path, {"teacher_window": teacher_window,
                                      "features": dict(zip(ids, h))})
            del teacher
            torch.cuda.empty_cache()
        targets = torch.load(teacher_path, map_location="cpu", weights_only=False)
        if targets["teacher_window"] != teacher_window:
            raise ValueError("Teacher cache mismatch")
        for window in WINDOWS:
            if window == teacher_window:
                continue
            name = f"distilled_{window}"
            status(name)
            report["stages"][name] = train_stage(name, window, records, cache, stats, output,
                args, deadline, output/f"baseline_{window}/best.pt", targets["features"])
        for window in WINDOWS:
            if window == teacher_window:
                continue
            name = f"control_{window}"
            status(name)
            report["stages"][name] = train_stage(name, window, records, cache, stats, output,
                args, deadline, output/f"baseline_{window}/best.pt", matched_epochs=
                report["stages"][f"distilled_{window}"]["epochs_run"])
        # Only after every base/distilled model is fixed do we touch the test set.
        status("evaluate_and_stack")
        val_matrices = {p: [] for p in ("baseline", "distilled", "control")}
        test_matrices = {p: [] for p in val_matrices}
        individual = {}
        for window in WINDOWS:
            for prefix in val_matrices:
                actual = "baseline" if window == teacher_window else prefix
                model = build().cuda().eval()
                model.load_state_dict(torch.load(output/f"{actual}_{window}/best.pt",
                                                 map_location="cpu", weights_only=False)["model"])
                predictions = {}
                for split in ("val", "test"):
                    loader = DataLoader(RawSlices(records, cache, split), batch_size=1, num_workers=2)
                    y, p, ids, _ = predict(model, loader, window, stats, deadline)
                    predictions[split] = p
                    if split == "val":
                        val_y = y
                        threshold = validation_threshold(y, p)
                    else:
                        test_y, test_ids = y, ids
                        individual[f"{prefix}_{window}"] = metrics(y, p, threshold)
                val_matrices[prefix].append(predictions["val"])
                test_matrices[prefix].append(predictions["test"])
                del model
                torch.cuda.empty_cache()
        results, ensembles, meta = {}, {}, {}
        for name in val_matrices:
            v, t = val_matrices[name], test_matrices[name]
            lr = LogisticRegression(max_iter=1000, random_state=args.seed)
            lr.fit(np.asarray(v).T, val_y)
            threshold = validation_threshold(val_y, lr.predict_proba(np.asarray(v).T)[:, 1])
            probabilities = lr.predict_proba(np.asarray(t).T)[:, 1]
            results[name] = metrics(test_y, probabilities, threshold)
            ensembles[name] = probabilities
            meta[name] = {"coefficients": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
                          "classes": lr.classes_.tolist(), "windows": list(WINDOWS)}
        delta = results["distilled"]["auc"] - results["baseline"]["auc"]
        final = {"status": "complete", "teacher_window": teacher_window, "individual": individual,
                 "ensemble": results, "delta_auc": delta,
                 "delta_auc_95ci": paired_auc_interval(test_y, ensembles["baseline"], ensembles["distilled"]),
                 "distilled_minus_matched_control_auc": results["distilled"]["auc"]-results["control"]["auc"],
                 "matched_delta_auc_95ci": paired_auc_interval(test_y, ensembles["control"], ensembles["distilled"]),
                 "meta_learners": meta, "limitations": config["limitations"] +
                 ["single seed; use matched control for distillation attribution, not only initial baseline"],
                 "invocation_seconds": time.monotonic()-start}
        with (output/"test_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["patient_id", "label", "baseline", "distilled", "control"])
            writer.writerows(zip(test_ids, test_y, ensembles["baseline"], ensembles["distilled"], ensembles["control"]))
        atomic_json(output/"results.json", final)
        report.update(status="complete", active_stage="done", invocation_seconds=time.monotonic()-start)
        atomic_json(report_path, report)
        print(json.dumps(final, indent=2), flush=True)
    except Exception as error:
        report.update(status="incomplete", error=f"{type(error).__name__}: {error}",
                      invocation_seconds=time.monotonic()-start)
        atomic_json(report_path, report)
        raise


if __name__ == "__main__":
    main()



