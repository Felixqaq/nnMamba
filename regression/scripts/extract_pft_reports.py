#!/usr/bin/env python3
"""Pull the spirometry report out of each patient's DICOM export and save it as JPG.

The PFT is not a separate file in these exports — the pulmonary function lab's
report is filed into the same study as a secondary-capture image, so it arrives
inside the DICOM folder alongside the CT series and has to be found by its tags:

    (0008,0060) Modality               OT
    (0008,0016) SOPClassUID            1.2.840.10008.5.1.4.1.1.7  (Secondary Capture)
    (0028,0004) PhotometricInterp      RGB, 3 samples  -- a colour page, not a slice
    (0008,1030) StudyDescription       contains "PFT"

The description is what separates a report from the other OT objects a PACS export
carries (key-image snapshots, screen captures). Objects that are OT but whose study
is not a PFT are written to a review folder rather than dropped, because that
judgement is the radiographer's and not this script's.

The saved page still carries the patient's name, printed on the report itself, and
the DICOM header carries name and birth date. Treat the output as identifiable.

FEV1/FVC still has to be read off the page and entered into fev1_fvc.csv; this
script only gets the page out of the archive and records where it came from.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

DEFAULT_ROOT = Path("/mnt/d/Felix/Hospital/copd_dataset")
BATCH_DIR = re.compile(r"^\d{8}$")
SECONDARY_CAPTURE = "1.2.840.10008.5.1.4.1.1.7"
PFT_IN_DESC = re.compile(r"\bPFT\b", re.I)


def tag(ds, name: str, default: str = "") -> str:
    return str(getattr(ds, name, default) or default).strip()


def find_ot_objects(dicom_dir: Path) -> list[tuple[Path, object]]:
    """Every OT report page under one patient folder.

    Modality must be OT. Matching on the Secondary Capture SOP class instead — or
    as well — is too loose: the scanner files its own "Patient Protocol" screenshot
    under that class with Modality CT, which pulled 20 non-reports per batch into
    an earlier version of this filter. A report page is additionally RGB, where
    those screenshots are MONOCHROME2.
    """
    hits: list[tuple[Path, object]] = []
    for path in sorted(dicom_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if tag(ds, "Modality").upper() != "OT":
            continue
        if tag(ds, "SOPClassUID") and tag(ds, "SOPClassUID") != SECONDARY_CAPTURE:
            continue
        hits.append((path, ds))
    return hits


def save_jpg(path: Path, out: Path, quality: int) -> tuple[int, int, int]:
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array
    if arr.ndim == 4:            # multi-frame colour: keep the first page
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    if arr.dtype != np.uint8:    # scanned pages are 8-bit; anything else needs scaling
        lo, hi = float(arr.min()), float(arr.max())
        arr = np.zeros_like(arr, dtype=np.uint8) if hi <= lo else \
            ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=quality, optimize=True)
    return img.width, img.height, out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--batches", default="",
                    help="comma-separated batch folders; default = every one present")
    ap.add_argument("--out", type=Path, default=None,
                    help="default <root>/PFT_JPG")
    ap.add_argument("--index", type=Path, default=None,
                    help="index CSV to append to; default <out>/pft_index_new.csv")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_root = args.out or (args.root / "PFT_JPG")
    index_path = args.index or (out_root / "pft_index_new.csv")

    wanted = {b.strip() for b in args.batches.split(",") if b.strip()}
    batches = [d for d in sorted(args.root.iterdir())
               if d.is_dir() and BATCH_DIR.match(d.name)
               and (not wanted or d.name in wanted)]
    if not batches:
        raise SystemExit(f"no batches matched under {args.root}")
    print(f"batches: {[b.name for b in batches]}\n")

    rows: list[dict] = []
    no_pft: list[str] = []
    extra_ot = 0

    for batch in batches:
        for patient in sorted(p for p in batch.iterdir() if p.is_dir()):
            pid = patient.name
            dicom_dir = patient / "DICOM"
            if not dicom_dir.is_dir():
                dicom_dir = patient
            objects = find_ot_objects(dicom_dir)
            # Split on the path, not on the dataset: pydicom compares Datasets by
            # value, so `(p, ds) not in pfts` silently misclassifies.
            pfts = [(p, ds) for p, ds in objects
                    if PFT_IN_DESC.search(tag(ds, "StudyDescription"))]
            pft_paths = {p for p, _ in pfts}
            others = [(p, ds) for p, ds in objects if p not in pft_paths]

            if not pfts:
                no_pft.append(f"{batch.name}/{pid}")
            for n, (src, ds) in enumerate(pfts, 1):
                stem = pid if len(pfts) == 1 else f"{pid}_{n}"
                dst = out_root / batch.name / f"{stem}.jpg"
                if dst.exists() and not args.overwrite:
                    print(f"  {batch.name}/{pid}: exists, skipped")
                    continue
                if args.dry_run:
                    print(f"  {batch.name}/{pid}: would write {dst.name}  "
                          f"[{tag(ds, 'StudyDescription')}]")
                    continue
                w, h, size = save_jpg(src, dst, args.quality)
                rows.append({
                    "Date": batch.name,
                    "PatientID": pid,
                    "SourceDicom": str(src.relative_to(args.root)),
                    "JpgPath": str(dst.relative_to(args.root)),
                    "StudyDescription": tag(ds, "StudyDescription"),
                    "StudyDate": tag(ds, "StudyDate"),
                    "Width": w, "Height": h, "JpgBytes": size,
                })
                print(f"  {batch.name}/{pid}: {w}x{h}  {tag(ds, 'StudyDescription')}")

            for n, (src, ds) in enumerate(others, 1):
                extra_ot += 1
                if args.dry_run:
                    continue
                dst = out_root / "_non_PFT_OT" / f"{batch.name}_{pid}_{n}.jpg"
                if dst.exists() and not args.overwrite:
                    continue
                try:
                    save_jpg(src, dst, args.quality)
                except Exception as exc:                      # noqa: BLE001
                    print(f"    non-PFT OT {pid} #{n} unreadable: "
                          f"{type(exc).__name__}")

    print(f"\nPFT pages written : {len(rows)}")
    print(f"other OT objects  : {extra_ot} -> {out_root / '_non_PFT_OT'} (for review)")
    print(f"patients with none: {len(no_pft)}")
    for p in no_pft:
        print(f"    {p}")

    if rows and not args.dry_run:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        exists = index_path.exists()
        with index_path.open("a", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\nindex appended: {index_path}")
        print("next: read FEV1/FVC off these pages into fev1_fvc.csv, then rebuild "
              "the cohort with build_fev1fvc70_dataset.py")


if __name__ == "__main__":
    main()
