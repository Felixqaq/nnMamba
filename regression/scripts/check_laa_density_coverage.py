#!/usr/bin/env python3
"""Fail unless every patient in a manifest has a valid LAA density array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, nargs=3, default=(112, 136, 112))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = payload["records"]
    expected = (2, *tuple(int(value) for value in args.image_size))
    missing: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for record in records:
        patient_id = str(record["patient_id"])
        if patient_id in seen:
            invalid.append(f"{patient_id}: duplicate manifest row")
            continue
        seen.add(patient_id)
        path = args.density_dir / f"{patient_id}.npy"
        if not path.exists():
            missing.append(patient_id)
            continue
        try:
            array = np.load(path, allow_pickle=False, mmap_mode="r")
        except Exception as exc:
            invalid.append(f"{patient_id}: {type(exc).__name__}: {exc}")
            continue
        if array.shape != expected or array.dtype != np.uint8:
            invalid.append(
                f"{patient_id}: shape={array.shape}, dtype={array.dtype}, expected={expected}/uint8"
            )

    print(
        f"manifest={len(records)} valid={len(records) - len(missing) - len(invalid)} "
        f"missing={len(missing)} invalid={len(invalid)}"
    )
    if missing:
        print("missing patient IDs:", ", ".join(missing[:30]))
    if invalid:
        print("invalid rows:")
        for message in invalid[:30]:
            print("  ", message)
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
