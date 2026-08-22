#!/usr/bin/env python3
"""Concatenate two frozen-feature sets into one npz the probe scripts can read.

Used to ask whether learned embeddings and the classical density indices carry
*different* information: if the combination beats both, they are complementary,
which is a stronger claim than either winning outright.

Patients are matched by id and intersected, so a case missing from either side
(e.g. no lung mask) is dropped from both rather than silently misaligned.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    data = np.load(path, allow_pickle=True)
    pids = [str(p) for p in data["patient_ids"]]
    X = np.asarray(data["features"], dtype=np.float64)
    names = (
        [str(n) for n in data["feature_names"]]
        if "feature_names" in data.files
        else [f"{path.parent.name}_{i}" for i in range(X.shape[1])]
    )
    return {p: X[i] for i, p in enumerate(pids)}, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    a, a_names = load(args.a)
    b, b_names = load(args.b)
    shared = sorted(set(a) & set(b))
    dropped = sorted((set(a) | set(b)) - set(shared))
    if not shared:
        raise SystemExit("No patients in common")
    print(f"A={len(a)}  B={len(b)}  shared={len(shared)}  dropped={len(dropped)}")
    if dropped:
        print(f"  dropped: {dropped}")

    X = np.hstack([
        np.array([a[p] for p in shared]),
        np.array([b[p] for p in shared]),
    ]).astype(np.float32)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "features.npz",
        features=X,
        patient_ids=np.array(shared),
        feature_names=np.array(a_names + b_names),
    )

    # The probe reads labels from metadata.csv next to the features.
    src_meta = args.a.parent / "metadata.csv"
    groups: dict[str, str] = {}
    with src_meta.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            groups[str(row["patient_id"])] = str(row["source_group"]).strip()
    with (args.out / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient_id", "source_group"])
        w.writeheader()
        for p in shared:
            w.writerow({"patient_id": p, "source_group": groups[p]})

    print(f"wrote {args.out / 'features.npz'}  {X.shape}")


if __name__ == "__main__":
    main()
