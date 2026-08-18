#!/usr/bin/env python
"""Compose a publication figure: N patients x 3 orthogonal views of stage2 Grad-CAM.

Each row is one patient, sliced at that patient's CAM peak. Colour scales are
normalised per patient, so intensities are not comparable across rows.
"""

from __future__ import annotations

import argparse
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
from scripts.export_gradcam_per_patient import (  # noqa: E402
    axial,
    cam_alpha,
    coronal,
    normalize_volume,
    sagittal,
)


def pad_to_canvas(slice_2d: np.ndarray, height: int, width: int) -> np.ndarray:
    """Centre a slice on a fixed canvas, padding with background (0).

    Padding rather than rescaling keeps anatomical proportions intact while giving
    every panel the same shape, so the figure grid aligns.
    """
    pad_h = max(0, height - slice_2d.shape[0])
    pad_w = max(0, width - slice_2d.shape[1])
    if pad_h == 0 and pad_w == 0:
        return slice_2d
    return np.pad(
        slice_2d,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
        mode="constant",
        constant_values=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.rq1.normal_v_abnormal.image.yaml")
    parser.add_argument(
        "--uuid",
        default="hybrid_mamba_attention_rq1_normal_v_abnormal_image_2026-07-14_15:15:09",
    )
    parser.add_argument("--patients", nargs="+", required=True)
    parser.add_argument("--target-layer", default="stage2")
    parser.add_argument("--target-class", default="Abnormal")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cam-threshold", type=float, default=0.40)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--fig-width", type=float, default=9.5)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = LoaderHelper(config)
    class_names = loader.get_class_names()
    lowered = [name.lower() for name in class_names]
    target_class = (
        None
        if args.target_class.lower() == "predicted"
        else lowered.index(args.target_class.lower())
    )
    weights_dir = config.paths.weights / config.task / args.uuid

    wanted = list(args.patients)
    located: dict[str, tuple[int, int]] = {}
    for fold in range(config.training.k_folds):
        for index in loader.fold_indices[fold][1]:
            pid = str(loader.dataset.records[int(index)].patient_id)
            if pid in wanted:
                located[pid] = (fold, int(index))
    missing = [pid for pid in wanted if pid not in located]
    if missing:
        raise SystemExit(f"Patients not found in any held-out fold: {missing}")

    panels = []
    for fold in sorted({fold for fold, _ in located.values()}):
        model = build_model(config.model, output_dim=config.model_output_dim())
        load_checkpoint(weights_dir / f"fold{fold + 1}_best_weight.pth", model, device)
        model.to(device).eval()
        _, layer = resolve_target_layer(
            model, model_name=str(config.model.name), target_layer_name=args.target_layer
        )
        extractor = GradCAM(model, layer)
        try:
            for pid in wanted:
                if located[pid][0] != fold:
                    continue
                sample = loader.dataset[located[pid][1]]
                ct = torch.from_numpy(np.asarray(sample["ct"], np.float32))
                ct = ct.unsqueeze(0).to(device).requires_grad_(True)
                cam_volume, outputs = extractor(ct, target_class=target_class)
                probs = torch.softmax(outputs.detach().cpu().float().view(-1), dim=0)
                panels.append(
                    {
                        "patient_id": pid,
                        "fold": fold + 1,
                        "gold": sample.get("gold_stage_label"),
                        "pred": class_names[int(torch.argmax(probs).item())],
                        "prob": float(probs.max()),
                        "volume": normalize_volume(ct[0, 0].detach().cpu().numpy()),
                        "cam": cam_volume[0].detach().cpu().numpy(),
                    }
                )
        finally:
            extractor.close()
            del model
            torch.cuda.empty_cache()

    panels.sort(key=lambda p: wanted.index(p["patient_id"]))

    rows = len(panels)
    views = ("Axial", "Coronal", "Sagittal")

    extracted = []
    for panel in panels:
        cam, volume = panel["cam"], panel["volume"]
        peak_c, peak_s, peak_a = (
            int(v) for v in np.unravel_index(int(np.argmax(cam)), cam.shape)
        )
        extracted.append(
            [
                (axial(volume, peak_a), axial(cam, peak_a)),
                (coronal(volume, peak_c), coronal(cam, peak_c)),
                (sagittal(volume, peak_s), sagittal(cam, peak_s)),
            ]
        )

    # Pad every panel to one common canvas so the grid is uniform without
    # rescaling anatomy: identical pixel dimensions mean identical aspect ratio.
    canvas_h = max(s.shape[0] for row in extracted for s, _ in row)
    canvas_w = max(s.shape[1] for row in extracted for s, _ in row)
    print(f"Common panel canvas: {canvas_h} x {canvas_w} px (anatomy unscaled)")

    # Size the figure from the panel aspect so equal-aspect images leave no dead
    # space: derive the height the axes grid needs, then add the fixed margins.
    left, right, top, bottom = 0.055, 0.88, 0.94, 0.02
    wspace, hspace = 0.04, 0.04
    grid_width_in = args.fig_width * (right - left)
    panel_width_in = grid_width_in / (3 + 2 * wspace)
    panel_height_in = panel_width_in * canvas_h / canvas_w
    grid_height_in = panel_height_in * (rows + (rows - 1) * hspace)
    fig_height = grid_height_in / (top - bottom)

    fig, axes = plt.subplots(rows, 3, figsize=(args.fig_width, fig_height))
    axes = np.atleast_2d(axes)

    for row, panel in enumerate(panels):
        slices = [
            (pad_to_canvas(s, canvas_h, canvas_w), pad_to_canvas(c, canvas_h, canvas_w))
            for s, c in extracted[row]
        ]
        for col, (image_slice, cam_slice) in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(image_slice, cmap="gray")
            ax.imshow(
                cam_slice,
                cmap="jet",
                alpha=cam_alpha(cam_slice, args.alpha, args.cam_threshold),
                vmin=0,
                vmax=1,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(views[col], fontsize=12, pad=8)
        axes[row, 0].set_ylabel(f"Case {row + 1}", fontsize=11, labelpad=10)

    fig.subplots_adjust(
        left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace
    )

    mappable = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    cbar_ax = fig.add_axes([right + 0.03, bottom + 0.06, 0.018, (top - bottom) * 0.8])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Grad-CAM (normalised per patient)", fontsize=10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved {out_path}")
    for panel in panels:
        print(
            f"  {panel['patient_id']}: fold{panel['fold']} | {panel['gold']} | "
            f"pred={panel['pred']} p={panel['prob']:.3f}"
        )


if __name__ == "__main__":
    main()
