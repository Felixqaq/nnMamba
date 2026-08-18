#!/usr/bin/env python
"""Export one out-of-fold Grad-CAM figure per patient for a trained k-fold run.

For every patient the figure is produced with the fold checkpoint that did *not*
train on that patient, so the attribution matches the reported cross-validation
predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.checkpoints import load_checkpoint  # noqa: E402
from core.config import Config  # noqa: E402
from core.gradcam import GradCAM, resolve_target_layer  # noqa: E402
from data.loader import LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.rq1.normal_v_abnormal.image.yaml")
    parser.add_argument(
        "--uuid",
        default="hybrid_mamba_attention_rq1_normal_v_abnormal_image_2026-07-14_15:15:09",
        help="Run directory under weights/<task>/ holding fold*_best_weight.pth",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: figures/<task>/<uuid>/gradcam_per_patient)",
    )
    parser.add_argument(
        "--target-layer",
        default=None,
        help="Module name to hook (default: model default, e.g. attention_layers)",
    )
    parser.add_argument(
        "--target-class",
        default="Abnormal",
        help="Class name or index the CAM explains ('predicted' for per-case argmax)",
    )
    parser.add_argument(
        "--cam-threshold",
        type=float,
        default=0.40,
        help="CAM value below which the overlay stays transparent",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def resolve_target_class(value: str, class_names: list[str]) -> int | None:
    if value.lower() == "predicted":
        return None
    if value.isdigit():
        return int(value)
    lowered = [name.lower() for name in class_names]
    if value.lower() not in lowered:
        raise SystemExit(f"Unknown class '{value}'. Available: {class_names}")
    return lowered.index(value.lower())


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    lower, upper = np.percentile(volume, [1, 99])
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def cam_alpha(cam_slice: np.ndarray, alpha: float, threshold: float = 0.40) -> np.ndarray:
    """Per-pixel opacity so only high-attention voxels are tinted."""
    ramp = np.clip((cam_slice - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return ramp * alpha


def axial(array: np.ndarray, index: int) -> np.ndarray:
    """Axial slice in radiological display orientation (anterior up)."""
    return array[:, :, index]


def coronal(array: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(array[index])


def sagittal(array: np.ndarray, index: int) -> np.ndarray:
    return np.rot90(array[:, index])


def save_patient_figure(
    volume: np.ndarray,
    cam: np.ndarray,
    path: Path,
    title: str,
    subtitle: str,
    alpha: float = 0.60,
    threshold: float = 0.40,
    dpi: int = 160,
) -> tuple[int, int, int]:
    """Save orthogonal peak views plus an axial montage; return the CAM peak voxel."""
    volume = normalize_volume(volume)
    peak_c, peak_s, peak_a = (
        int(value) for value in np.unravel_index(int(np.argmax(cam)), cam.shape)
    )
    slices = np.linspace(0.15 * volume.shape[2], 0.85 * volume.shape[2], 5).astype(int)

    fig, axes = plt.subplots(2, 5, figsize=(17, 7.5))
    for axis in axes.ravel():
        axis.axis("off")

    ortho = [
        (f"Axial (slice {peak_a})", axial(volume, peak_a), axial(cam, peak_a)),
        (f"Coronal ({peak_c})", coronal(volume, peak_c), coronal(cam, peak_c)),
        (f"Sagittal ({peak_s})", sagittal(volume, peak_s), sagittal(cam, peak_s)),
    ]
    for axis, (view_name, image_slice, cam_slice) in zip(axes[0, :3], ortho):
        axis.imshow(image_slice, cmap="gray")
        axis.imshow(
            cam_slice, cmap="jet", alpha=cam_alpha(cam_slice, alpha, threshold), vmin=0, vmax=1
        )
        axis.set_title(view_name, fontsize=10)

    axes[0, 3].imshow(axial(volume, peak_a), cmap="gray")
    axes[0, 3].set_title("Axial peak slice (CT only)", fontsize=10)

    mappable = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(mappable, ax=axes[0, 4], fraction=0.5, pad=0.05)
    cbar.set_label("Grad-CAM (normalized)", fontsize=9)

    for axis, slice_index in zip(axes[1], slices):
        cam_slice = axial(cam, slice_index)
        axis.imshow(axial(volume, slice_index), cmap="gray")
        axis.imshow(
            cam_slice, cmap="jet", alpha=cam_alpha(cam_slice, alpha, threshold), vmin=0, vmax=1
        )
        axis.set_title(f"axial {slice_index}", fontsize=9)

    fig.suptitle(title, fontsize=13, y=0.99)
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return peak_c, peak_s, peak_a


def save_contact_sheet(
    entries: list[dict],
    volumes: dict[str, np.ndarray],
    path: Path,
    threshold: float = 0.40,
) -> None:
    """Save a thumbnail grid of the axial peak slice for quick scanning."""
    if not entries:
        return
    columns = 6
    rows = int(np.ceil(len(entries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.7 * rows))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, entry in zip(axes.ravel(), entries):
        volume, cam = volumes[entry["patient_id"]]
        peak_a = entry["cam_peak"][2]
        cam_slice = axial(cam, peak_a)
        axis.imshow(axial(normalize_volume(volume), peak_a), cmap="gray")
        axis.imshow(
            cam_slice, cmap="jet", alpha=cam_alpha(cam_slice, 0.55, threshold), vmin=0, vmax=1
        )
        mark = "✓" if entry["correct"] else "✗"
        axis.set_title(
            f"{entry['patient_id']} {mark}\npred={entry['pred_label']} "
            f"p={entry['pred_probability']:.2f}",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_index(entries: list[dict], meta: dict, path: Path) -> None:
    lines = [
        "# Grad-CAM per patient — 醫師確認用",
        "",
        f"- Run: `{meta['uuid']}`",
        f"- Config: `{meta['config']}`",
        f"- Target layer: `{meta['target_layer']}`",
        f"- CAM explains class: **{meta['target_class_label']}**",
        f"- Attribution scope: out-of-fold (每位病人用沒有訓練過他的那一折 best checkpoint)",
        f"- Patients: {len(entries)}",
        "",
        "熱區顏色 = 該區域對「Abnormal」這個判斷的貢獻度(紅=高,藍=低);"
        "影像已重取樣為 112×136×112 並做 per-volume z-score,故與原始 DICOM 的層厚/視野不同。",
        "",
        "| # | Patient | True | GOLD | Pred | p | Correct | Fold | Figure |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, entry in enumerate(entries, start=1):
        lines.append(
            f"| {idx} | {entry['patient_id']} | {entry['true_label']} | "
            f"{entry['gold_stage_label'] or '-'} | {entry['pred_label']} | "
            f"{entry['pred_probability']:.3f} | {'yes' if entry['correct'] else 'NO'} | "
            f"{entry['fold']} | [{entry['image']}]({entry['image']}) |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_helper = LoaderHelper(config)
    class_names = loader_helper.get_class_names()
    target_class = resolve_target_class(args.target_class, class_names)

    weights_dir = config.paths.weights / config.task / args.uuid
    if not weights_dir.exists():
        raise SystemExit(f"Weights directory not found: {weights_dir}")

    out_dir = (
        Path(args.out)
        if args.out
        else config.paths.figures / config.task / args.uuid / "gradcam_per_patient"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Weights: {weights_dir}")
    print(f"Output:  {out_dir}")
    print(f"Classes: {class_names} | CAM class: {args.target_class}")

    entries: list[dict] = []
    volumes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    target_layer_name = args.target_layer

    for fold in range(config.training.k_folds):
        checkpoint = weights_dir / f"fold{fold + 1}_best_weight.pth"
        if not checkpoint.exists():
            raise SystemExit(f"Missing checkpoint: {checkpoint}")

        model = build_model(config.model, output_dim=config.model_output_dim())
        load_checkpoint(checkpoint, model, device)
        model.to(device).eval()

        layer_name, target_layer = resolve_target_layer(
            model,
            model_name=str(config.model.name),
            target_layer_name=target_layer_name,
        )
        target_layer_name = layer_name
        cam_extractor = GradCAM(model, target_layer)

        val_indices = loader_helper.fold_indices[fold][1]
        print(f"\nFold {fold + 1}: {len(val_indices)} held-out patients | layer={layer_name}")

        try:
            for dataset_index in val_indices:
                sample = loader_helper.dataset[int(dataset_index)]
                ct = torch.from_numpy(np.asarray(sample["ct"], dtype=np.float32))
                ct = ct.unsqueeze(0).to(device).requires_grad_(True)

                cam_volume, outputs = cam_extractor(ct, target_class=target_class)
                logits = outputs.detach().cpu().float().view(-1)
                probabilities = torch.softmax(logits, dim=0)
                pred_index = int(torch.argmax(probabilities).item())
                true_index = int(np.asarray(sample["label"]).item())

                patient_id = str(sample["patient_id"])
                gold = sample.get("gold_stage_label")
                correct = pred_index == true_index
                title = (
                    f"Patient {patient_id} — true: {class_names[true_index]}"
                    f"{f' ({gold})' if gold else ''}"
                )
                subtitle = (
                    f"model: {class_names[pred_index]} "
                    f"(p={probabilities[pred_index].item():.3f}) | "
                    f"{'correct' if correct else 'INCORRECT'} | "
                    f"fold {fold + 1} held-out | CAM class: "
                    f"{class_names[target_class] if target_class is not None else class_names[pred_index]}"
                )

                volume = ct[0, 0].detach().cpu().numpy()
                cam_np = cam_volume[0].cpu().numpy()
                filename = (
                    f"{class_names[true_index].lower()}_{patient_id}"
                    f"_fold{fold + 1}_{'ok' if correct else 'err'}.png"
                )
                peak = save_patient_figure(
                    volume=volume,
                    cam=cam_np,
                    path=out_dir / filename,
                    title=title,
                    subtitle=subtitle,
                    threshold=args.cam_threshold,
                    dpi=args.dpi,
                )

                volumes[patient_id] = (volume, cam_np)
                entries.append(
                    {
                        "patient_id": patient_id,
                        "fold": fold + 1,
                        "dataset_index": int(dataset_index),
                        "source": sample.get("path"),
                        "true_label": class_names[true_index],
                        "gold_stage_label": gold,
                        "pred_label": class_names[pred_index],
                        "pred_probability": round(
                            float(probabilities[pred_index].item()), 5
                        ),
                        "probabilities": {
                            name: round(float(probabilities[i].item()), 5)
                            for i, name in enumerate(class_names)
                        },
                        "correct": correct,
                        "cam_peak": list(peak),
                        "image": filename,
                    }
                )
                print(
                    f"  {patient_id:>10}  true={class_names[true_index]:<8} "
                    f"pred={class_names[pred_index]:<8} p={probabilities[pred_index]:.3f} "
                    f"{'ok' if correct else 'ERR'}"
                )
        finally:
            cam_extractor.close()
            del model
            torch.cuda.empty_cache()

    entries.sort(key=lambda item: (item["true_label"], item["patient_id"]))
    accuracy = sum(entry["correct"] for entry in entries) / max(len(entries), 1)
    meta = {
        "uuid": args.uuid,
        "config": args.config,
        "target_layer": target_layer_name,
        "cam_display_threshold": args.cam_threshold,
        "target_class": target_class,
        "target_class_label": (
            class_names[target_class] if target_class is not None else "predicted class"
        ),
        "class_names": class_names,
        "attribution_scope": "out_of_fold_best_checkpoint",
        "cam_peak_axes": ["coronal_index", "sagittal_index", "axial_index"],
        "n_patients": len(entries),
        "pooled_accuracy": round(accuracy, 5),
        "samples": entries,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    write_index(entries, meta, out_dir / "index.md")

    for label in class_names:
        subset = [entry for entry in entries if entry["true_label"] == label]
        save_contact_sheet(
            subset,
            volumes,
            out_dir / f"contact_sheet_{label.lower()}.png",
            threshold=args.cam_threshold,
        )

    print(f"\nSaved {len(entries)} patient figures to {out_dir}")
    print(f"Pooled out-of-fold accuracy of these checkpoints: {accuracy:.4f}")


if __name__ == "__main__":
    main()
