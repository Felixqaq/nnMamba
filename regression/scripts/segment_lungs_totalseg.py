#!/usr/bin/env python
"""Segment lungs with TotalSegmentator and derive a binary lung mask per patient.

Writes, for each patient in a manifest:
  <out>/multilabel/<patient_id>.nii.gz  full TotalSegmentator label map
  <out>/lung/<patient_id>.nii.gz        binary lung mask (union of the five lobes)

Masks stay on the source image grid, so downstream loading can apply them directly.
Already-finished patients are skipped, making the run resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

LOBES = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="./datasets/generated/rq1_nva66_manifest.image.json"
    )
    parser.add_argument("--out", default="./masks/totalseg")
    parser.add_argument("--task", default="total")
    parser.add_argument(
        "--fast", action="store_true", help="Use the 3mm model (much faster, coarser)"
    )
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--limit", type=int, default=0, help="Only process N patients")
    return parser.parse_args()


def totalsegmentator_executable() -> str:
    """Resolve the CLI that ships with the interpreter running this script."""
    candidate = Path(sys.executable).parent / "TotalSegmentator"
    return str(candidate) if candidate.exists() else "TotalSegmentator"


def lobe_label_ids(task: str) -> dict[str, int]:
    """Resolve lobe names to label values from TotalSegmentator's own class map."""
    from totalsegmentator.map_to_binary import class_map

    name_to_id = {name: index for index, name in class_map[task].items()}
    missing = [lobe for lobe in LOBES if lobe not in name_to_id]
    if missing:
        raise SystemExit(f"Task '{task}' does not provide lung lobes: {missing}")
    return {lobe: name_to_id[lobe] for lobe in LOBES}


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    multilabel_dir = out_root / "multilabel"
    lung_dir = out_root / "lung"
    multilabel_dir.mkdir(parents=True, exist_ok=True)
    lung_dir.mkdir(parents=True, exist_ok=True)

    records = json.load(open(args.manifest, encoding="utf-8"))["records"]
    if args.limit:
        records = records[: args.limit]
    label_ids = lobe_label_ids(args.task)
    print(f"Lobe label ids: {label_ids}")

    summary = []
    for position, record in enumerate(records, start=1):
        patient_id = str(record["patient_id"])
        source = Path(record["path"])
        multilabel_path = multilabel_dir / f"{patient_id}.nii.gz"
        lung_path = lung_dir / f"{patient_id}.nii.gz"

        if not lung_path.exists():
            if not multilabel_path.exists():
                command = [
                    totalsegmentator_executable(),
                    "-i", str(source),
                    "-o", str(multilabel_path),
                    "--task", args.task,
                    "--ml",
                    "--device", args.device,
                ]
                if args.fast:
                    command.append("--fast")
                print(f"[{position}/{len(records)}] {patient_id}: segmenting", flush=True)
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(result.stdout[-2000:])
                    print(result.stderr[-2000:], file=sys.stderr)
                    raise SystemExit(f"TotalSegmentator failed on {patient_id}")

            labels = nib.load(str(multilabel_path))
            label_data = np.asarray(labels.dataobj)
            lung = np.isin(label_data, list(label_ids.values()))
            nib.save(
                nib.Nifti1Image(lung.astype(np.uint8), labels.affine, labels.header),
                str(lung_path),
            )

        image = nib.load(str(source))
        hu = np.asarray(image.dataobj, dtype=np.float32)
        lung = np.asarray(nib.load(str(lung_path)).dataobj).astype(bool)
        if lung.shape != hu.shape:
            raise SystemExit(
                f"{patient_id}: mask shape {lung.shape} != image shape {hu.shape}"
            )
        spacing = image.header.get_zooms()[:3]
        volume_ml = float(lung.sum()) * float(np.prod(spacing)) / 1000.0
        mean_hu = float(hu[lung].mean()) if lung.any() else float("nan")
        summary.append(
            {
                "patient_id": patient_id,
                "class_label": record.get("class_label"),
                "gold_stage_label": record.get("gold_stage_label"),
                "lung_volume_ml": round(volume_ml, 1),
                "lung_mean_hu": round(mean_hu, 1),
                "lung_voxels": int(lung.sum()),
            }
        )
        print(
            f"[{position}/{len(records)}] {patient_id}: "
            f"{volume_ml:7.0f} mL | mean HU {mean_hu:7.1f}",
            flush=True,
        )

    with open(out_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    volumes = np.array([row["lung_volume_ml"] for row in summary])
    hus = np.array([row["lung_mean_hu"] for row in summary])
    print(f"\n{len(summary)} patients")
    print(f"  lung volume mL: min {volumes.min():.0f} | median {np.median(volumes):.0f} "
          f"| max {volumes.max():.0f}   (成人約 2000-7000 mL)")
    print(f"  mean HU:        min {hus.min():.0f} | median {np.median(hus):.0f} "
          f"| max {hus.max():.0f}   (充氣肺約 -700 ~ -900)")


if __name__ == "__main__":
    main()
