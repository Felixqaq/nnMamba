#!/usr/bin/env python3
"""Audit NIfTI geometry and DICOM-series provenance for a frozen cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-inplane-mm", type=float, default=220.0)
    parser.add_argument("--min-axial-mm", type=float, default=190.0)
    parser.add_argument("--min-axial-cosine", type=float, default=0.85)
    parser.add_argument(
        "--reuse-valid-output",
        action="store_true",
        help="Reuse an existing successful audit when cohort and thresholds match.",
    )
    return parser.parse_args()


def find_cts(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(root.glob("*/*.nii.gz")):
        patient_id = path.name[:-7].split("_", 1)[0]
        if patient_id in found:
            duplicates.setdefault(patient_id, [str(found[patient_id])]).append(str(path))
        else:
            found[patient_id] = path
    if duplicates:
        raise SystemExit(f"duplicate NIfTI patients: {duplicates}")
    return found


def read_dicom_anatomy(provenance: dict[str, object]) -> dict[str, str]:
    """Read anatomy tags from the exact DICOM series selected for conversion."""
    dicom_dir = str(provenance.get("dicom_dir", ""))
    series_uid = str(provenance.get("series_uid", ""))
    if not dicom_dir or not series_uid:
        return {}
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_uid)
    if not files:
        return {}
    reader = sitk.ImageFileReader()
    reader.SetFileName(files[0])
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    def tag(key: str) -> str:
        return reader.GetMetaData(key).strip() if reader.HasMetaDataKey(key) else ""

    return {
        "body_part_examined": tag("0018|0015"),
        "study_description": tag("0008|1030"),
        "series_description": tag("0008|103e"),
    }


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_json.read_text(encoding="utf-8-sig"))
    required_ids = set(split["training_patient_ids"]) | set(
        split["validation_patient_ids"]
    )
    new_ids = set(split.get("new_training_patient_ids", []))
    expected_thresholds = {
        "min_inplane_mm": args.min_inplane_mm,
        "min_axial_mm": args.min_axial_mm,
        "min_axial_cosine": args.min_axial_cosine,
    }
    if args.reuse_valid_output and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8-sig"))
        if (
            existing.get("n_cohort") == len(required_ids)
            and existing.get("n_training") == len(split["training_patient_ids"])
            and existing.get("n_validation") == len(split["validation_patient_ids"])
            and existing.get("n_new_training") == len(new_ids)
            and existing.get("n_failures") == 0
            and existing.get("thresholds") == expected_thresholds
        ):
            print(
                f"reusing successful audit: cohort={len(required_ids)} "
                f"new={len(new_ids)} output={args.output}"
            )
            return
    cts = find_cts(args.source_dir)
    missing = sorted(required_ids - set(cts))
    if missing:
        raise SystemExit(f"missing NIfTI for {len(missing)} frozen patients: {missing}")

    summary = json.loads(args.build_summary.read_text(encoding="utf-8-sig"))
    records = {
        str(record["patient_id"]): record for record in summary.get("records", [])
    }
    missing_provenance = sorted(new_ids - set(records))
    if missing_provenance:
        raise SystemExit(
            f"new patients missing build-summary provenance: {missing_provenance}"
        )

    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    lung_tokens = ("lung", "lw", "b60", "br60", "b70", "sharp")
    for patient_id in sorted(required_ids):
        path = cts[patient_id]
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        extent = tuple(shape[index] * spacing[index] for index in range(3))
        provenance = records.get(patient_id, {})
        cosine = provenance.get("axial_cosine")
        description = str(provenance.get("series_description", ""))
        anatomy = read_dicom_anatomy(provenance) if patient_id in new_ids else {}
        patient_failures: list[str] = []
        patient_warnings: list[str] = []
        if len(shape) != 3 or min(shape) < 2:
            patient_failures.append(f"invalid shape {shape}")
        geometry_issues: list[str] = []
        if min(extent[:2]) < args.min_inplane_mm:
            geometry_issues.append(
                f"in-plane FOV {extent[0]:.1f}x{extent[1]:.1f} mm"
            )
        if extent[2] < args.min_axial_mm:
            geometry_issues.append(f"axial coverage {extent[2]:.1f} mm")
        if patient_id in new_ids:
            patient_failures.extend(geometry_issues)
        else:
            patient_warnings.extend(
                f"legacy cohort geometry: {issue}" for issue in geometry_issues
            )
        if patient_id in new_ids:
            if not provenance.get("ok", False):
                patient_failures.append("conversion provenance is not successful")
            if cosine is None or float(cosine) < args.min_axial_cosine:
                patient_failures.append(f"axial cosine {cosine}")
            normalized_description = description.lower()
            if not any(token in normalized_description for token in lung_tokens):
                patient_warnings.append(
                    f"series description lacks a known lung-kernel token: {description!r}"
                )
            dicom_patient_id = str(provenance.get("dicom_patient_id", "")).strip()
            if dicom_patient_id and dicom_patient_id != patient_id:
                patient_failures.append(
                    f"DICOM patient ID {dicom_patient_id!r} does not match"
                )
            anatomy_text = " ".join(anatomy.values()).lower()
            non_thoracic_tokens = ("head", "neck", "brain", "c-spine", "c spine")
            if any(token in anatomy_text for token in non_thoracic_tokens):
                patient_failures.append(
                    f"DICOM anatomy tags indicate non-thoracic imaging: {anatomy}"
                )
            body_part = anatomy.get("body_part_examined", "").lower()
            if body_part and not any(
                token in body_part for token in ("chest", "thorax", "lung")
            ):
                patient_warnings.append(
                    f"unexpected BodyPartExamined={anatomy['body_part_examined']!r}"
                )
        row = {
            "patient_id": patient_id,
            "is_new_training_patient": patient_id in new_ids,
            "path": str(path),
            "shape": shape,
            "spacing_mm": spacing,
            "extent_mm": extent,
            "series_description": description,
            "axial_cosine": cosine,
            "dicom_anatomy": anatomy,
            "failures": patient_failures,
            "warnings": patient_warnings,
        }
        audit_rows.append(row)
        if patient_failures:
            failures.append(row)
        if patient_warnings:
            warnings.append(row)

    payload = {
        "n_cohort": len(required_ids),
        "n_training": len(split["training_patient_ids"]),
        "n_validation": len(split["validation_patient_ids"]),
        "n_new_training": len(new_ids),
        "n_failures": len(failures),
        "n_warnings": len(warnings),
        "thresholds": expected_thresholds,
        "failures": failures,
        "warnings": warnings,
        "patients": audit_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"audited={len(required_ids)} new={len(new_ids)} "
        f"failures={len(failures)} warnings={len(warnings)}"
    )
    if failures:
        for row in failures:
            print(f"FAIL {row['patient_id']}: {row['failures']}")
        raise SystemExit(1)
    for row in warnings:
        print(f"WARN {row['patient_id']}: {row['warnings']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
