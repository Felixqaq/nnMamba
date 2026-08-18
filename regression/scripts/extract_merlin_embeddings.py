#!/usr/bin/env python
"""Extract frozen Merlin image embeddings for the Normal_v_Abnormal 54-case cohort.

Additive, standalone tool: reads the ORIGINAL full-resolution NIfTIs, applies
Merlin's official preprocessing (resample 1.5/3mm, HU->[0,1], center-crop
224x224x160), runs the frozen Merlin image encoder (I3D-ResNet152, 2048-d), and
writes features.npz in the SAME format as scripts/extract_tapct_embeddings.py so
the existing embedding-probe pipeline (tapct_abmil head) can consume it unchanged.

Run in the dedicated `merlin` conda env:
    conda activate merlin
    python scripts/extract_merlin_embeddings.py \
        --source-dir ../classification/datasets/normal_v_abnormal_54 \
        --output-dir embeddings/merlin
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen Merlin CT embeddings")
    parser.add_argument(
        "--source-dir",
        default="../classification/datasets/normal_v_abnormal_54",
        help="Root with <label>/<patient>.nii.gz (label from parent folder).",
    )
    parser.add_argument("--output-dir", default="embeddings/merlin")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def patient_id_from_path(path: Path) -> str:
    """Match the pipeline convention: id = filename up to the first underscore."""
    stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
    return stem.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )

    # Import here so --help works without the merlin env.
    import merlin.data.monai_transforms as mt
    from merlin import Merlin

    transform = mt.ImageTransforms
    model = Merlin(ImageEmbedding=True).eval().to(device)

    files = sorted(source_dir.glob("*/*.nii.gz"))
    if not files:
        raise SystemExit(f"No NIfTI files under {source_dir}")
    print(f"Found {len(files)} CT volumes under {source_dir}")

    features: list[np.ndarray] = []
    rows: list[dict] = []
    for i, path in enumerate(files, start=1):
        pid = patient_id_from_path(path)
        group = path.parent.name  # Normal / Abnormal
        img = transform({"image": str(path)})["image"]  # (1, 224, 224, 160)
        x = img.unsqueeze(0).to(device).float()  # (1, 1, 224, 224, 160)
        with torch.no_grad():
            out = model(x)  # (1, 1, 2048)
        vec = out.reshape(-1).detach().cpu().numpy().astype(np.float32)
        features.append(vec)
        rows.append({"patient_id": pid, "source_group": group, "path": str(path)})
        print(f"[{i:>2}/{len(files)}] {pid:>10} ({group}) -> emb dim {vec.shape[0]}")

    feature_matrix = np.stack(features).astype(np.float32)
    patient_ids = np.array([r["patient_id"] for r in rows])
    np.savez_compressed(
        output_dir / "features.npz",
        features=feature_matrix,
        patient_ids=patient_ids,
    )
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["patient_id", "source_group", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved {feature_matrix.shape[0]} x {feature_matrix.shape[1]} embeddings to "
        f"{output_dir / 'features.npz'}"
    )


if __name__ == "__main__":
    main()
