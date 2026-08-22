#!/usr/bin/env python3
"""Read patient demographics from the DICOM headers of each cohort CT.

Age/sex/height are known before spirometry is performed, so they are legitimate
predictors of airflow obstruction. Everything else in fev1_fvc.csv is derived
from the spirometry itself and would leak the label — FEV1FVC_pct *is* the label,
and the pctpred/severity/obstruction columns are functions of it. Only the
reference (predicted) FEV1/FVC is safe, and that is itself just a function of
age, sex and height.

DICOM is used rather than the PFT table because the 66 hospital patients have no
demographic columns anywhere, while every patient necessarily has a header. Using
one source for all 182 also avoids two cohorts carrying subtly different fields.
The extraction is cross-checked against the CSV for the patients that have both.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REGRESSION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REGRESSION_ROOT.parent
DEFAULT_COHORT = REPO_ROOT / "classification/datasets/normal_v_abnormal_fev1fvc70"
DEFAULT_CSV = Path("/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv")

AGE_TAG, SEX_TAG, SIZE_TAG, WEIGHT_TAG = "0010|1010", "0010|0040", "0010|1020", "0010|1030"


def parse_age(raw: str) -> float | None:
    """DICOM ages look like '065Y'; months/weeks/days are not expected in adults."""
    raw = (raw or "").strip()
    m = re.fullmatch(r"(\d{1,3})\s*([YMWD])?", raw, re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "Y").upper()
    return {"Y": value, "M": value / 12, "W": value / 52.1, "D": value / 365.25}[unit]


def header_of(dicom_dir: str, series_uid: str) -> dict:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid)
    if not files:
        return {}
    meta = sitk.ImageFileReader()
    meta.SetFileName(files[0])
    meta.LoadPrivateTagsOn()
    meta.ReadImageInformation()

    def tag(key: str) -> str:
        return meta.GetMetaData(key).strip() if meta.HasMetaDataKey(key) else ""

    size = tag(SIZE_TAG)
    weight = tag(WEIGHT_TAG)
    return {
        "age": parse_age(tag(AGE_TAG)),
        "sex": tag(SEX_TAG).upper()[:1],
        "height_m": float(size) if size and size.replace(".", "", 1).isdigit() else None,
        "weight_kg": float(weight) if weight and weight.replace(".", "", 1).isdigit() else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="for cross-checking")
    ap.add_argument("--output", type=Path, default=REGRESSION_ROOT / "embeddings/demographics")
    args = ap.parse_args()

    recs = json.loads((args.cohort / "build_summary.json").read_text("utf-8"))["records"]
    rows: list[dict] = []
    for r in recs:
        if not r.get("ok"):
            continue
        head = header_of(r["dicom_dir"], r["series_uid"])
        rows.append({
            "patient_id": r["patient_id"],
            "label": r["label"],
            "cohort": r.get("cohort", ""),
            **head,
        })
        print(f"{r['patient_id']}  age={head.get('age')}  sex={head.get('sex')!r}  "
              f"height={head.get('height_m')}  weight={head.get('weight_kg')}", flush=True)

    # --- cross-check against the PFT table where both exist -----------------
    csv_ref: dict[str, dict] = {}
    if args.csv.exists():
        with io.open(args.csv, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                csv_ref[row["PatientID"]] = row
    checked = age_bad = sex_bad = 0
    for r in rows:
        ref = csv_ref.get(r["patient_id"])
        if not ref:
            continue
        checked += 1
        try:
            if r["age"] is not None and abs(float(ref["Age"]) - r["age"]) > 2.0:
                age_bad += 1
        except ValueError:
            pass
        if r["sex"] and ref.get("Sex") and r["sex"] != ref["Sex"].upper()[:1]:
            sex_bad += 1
    print(f"\ncross-check against CSV: {checked} patients")
    print(f"  age differs by >2y : {age_bad}")
    print(f"  sex mismatch       : {sex_bad}")

    missing = {k: sum(1 for r in rows if r.get(k) is None) for k in
               ("age", "height_m", "weight_kg")}
    missing["sex"] = sum(1 for r in rows if not r.get("sex"))
    print(f"missing values: {missing}  (of {len(rows)})")

    # --- build the feature matrix ------------------------------------------
    ages = [r["age"] for r in rows if r["age"] is not None]
    median_age = float(np.median(ages)) if ages else 65.0
    feature_names = ["age", "sex_male", "height_m", "bmi"]
    X = []
    for r in rows:
        age = r["age"] if r["age"] is not None else median_age
        sex_male = 1.0 if r["sex"] == "M" else 0.0
        h = r["height_m"]
        w = r["weight_kg"]
        # DICOM PatientSize is in metres; guard against files that store cm.
        if h is not None and h > 3:
            h = h / 100.0
        bmi = (w / (h * h)) if (h and w) else np.nan
        X.append([age, sex_male, h if h is not None else np.nan, bmi])
    X = np.array(X, dtype=np.float32)

    # Columns that are entirely missing carry no information and would only add
    # a constant after imputation, so drop them and say so.
    keep = [i for i in range(X.shape[1]) if not np.all(np.isnan(X[:, i]))]
    dropped = [feature_names[i] for i in range(X.shape[1]) if i not in keep]
    if dropped:
        print(f"dropping all-missing features: {dropped}")
    X = X[:, keep]
    feature_names = [feature_names[i] for i in keep]
    # Median-impute what remains.
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            col[np.isnan(col)] = np.nanmedian(col)
            X[:, j] = col

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "features.npz",
        features=X,
        patient_ids=np.array([r["patient_id"] for r in rows]),
        feature_names=np.array(feature_names),
    )
    with (args.output / "metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["patient_id", "source_group"])
        w.writeheader()
        for r in rows:
            w.writerow({"patient_id": r["patient_id"], "source_group": r["label"]})
    with (args.output / "demographics.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {args.output / 'features.npz'}  {X.shape}  {feature_names}")
    lab = np.array([r["label"] for r in rows])
    if "age" in feature_names:
        a = X[:, feature_names.index("age")]
        print(f"  age  Abnormal median {np.median(a[lab == 'Abnormal']):.1f} | "
              f"Normal {np.median(a[lab == 'Normal']):.1f}")
    if "sex_male" in feature_names:
        s = X[:, feature_names.index("sex_male")]
        print(f"  male Abnormal {s[lab == 'Abnormal'].mean():.0%} | "
              f"Normal {s[lab == 'Normal'].mean():.0%}")


if __name__ == "__main__":
    main()
