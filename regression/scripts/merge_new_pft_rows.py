#!/usr/bin/env python3
"""Append newly-read PFT rows to fev1_fvc.csv, in that file's existing schema.

The derived columns follow the definitions already used in the file — ATS/ERS 2005
severity and GOLD grade on FEV1 %pred, the 2005 and 2021 bronchodilator-response
criteria, GOLD fixed-ratio obstruction — and both GLI blocks are computed with the
same ports the existing rows were built from, checked against 25 of those rows to
within rounding before being used here.

Age comes from the DICOM birth date and the study date, not the integer printed on
the report, because the existing rows carry fractional ages and GLI is sensitive to
the difference.

Two columns stay empty for the new rows: Check_FVCpct_diff and Check_FEV1pct_diff
compare the *printed* %Ref against a recomputed one, and the printed percentages
were not transcribed. Writing 0 there would claim a check that was not performed.
Two columns are added for everyone: Weight_kg, and the analyser's FVL ECode, which
flags an unclean expiratory effort and is the reason two patients are already out
of the cohort.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import date
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "gli"))
sys.path.insert(0, str(SCRIPTS))

import gli2012                                    # noqa: E402
import gli_global                                 # noqa: E402
from pft_new_readings import D as READINGS        # noqa: E402

CSV_DIR = Path("/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG")
NEW_COLS = ["Weight_kg", "FVL_ECode_pre", "FVL_ECode_post"]
ETHNICITY = 3  # NE Asian


def severity_ats(p: float) -> str:
    """ATS/ERS 2005 severity, graded on FEV1 %pred."""
    if p > 70: return "Mild"
    if p >= 60: return "Moderate"
    if p >= 50: return "Moderately severe"
    if p >= 35: return "Severe"
    return "Very severe"


def gold(p: float) -> str:
    """GOLD 1-4 airflow limitation; only meaningful once obstruction is present."""
    if p >= 80: return "GOLD 1"
    if p >= 50: return "GOLD 2"
    if p >= 30: return "GOLD 3"
    return "GOLD 4"


def severity_gli(z: float) -> str:
    """GLI 2022 severity, graded on the FEV1 z-score."""
    if z >= -2.5: return "Mild"
    if z >= -4.0: return "Moderate"
    return "Severe"


def fractional_age(birth: str, study: str) -> float | None:
    try:
        b = date(int(birth[:4]), int(birth[4:6]), int(birth[6:8]))
        s = date(int(study[:4]), int(study[4:6]), int(study[6:8]))
    except (ValueError, TypeError, IndexError):
        return None
    return round((s - b).days / 365.25, 2)


def dicom_demographics(index_csv: Path, root: Path) -> dict[str, dict]:
    """patient_id -> {birth, study, desc, jpg} from the new-batch PFT index."""
    import pydicom
    out: dict[str, dict] = {}
    with index_csv.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            src = root / row["SourceDicom"].replace("\\", "/")
            try:
                ds = pydicom.dcmread(str(src), stop_before_pixels=True, force=True)
                birth = str(getattr(ds, "PatientBirthDate", "") or "")
            except Exception:                                  # noqa: BLE001
                birth = ""
            out[row["PatientID"]] = {
                "birth": birth,
                "study": row.get("StudyDate", ""),
                "desc": row.get("StudyDescription", ""),
                "jpg": row.get("JpgPath", ""),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_DIR / "fev1_fvc.csv")
    ap.add_argument("--index", type=Path, default=CSV_DIR / "pft_index_new.csv")
    ap.add_argument("--root", type=Path, default=CSV_DIR.parent)
    ap.add_argument("--out", type=Path, default=None, help="default: overwrite --csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with args.csv.open(encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        existing = [r for r in reader]
    print(f"existing: {len(existing)} rows, {len(header)} columns")

    have = {r[1].strip() for r in existing}
    meta = dicom_demographics(args.index, args.root)

    out_header = header + [c for c in NEW_COLS if c not in header]
    col = {name: i for i, name in enumerate(out_header)}
    padded = [r + [""] * (len(out_header) - len(r)) for r in existing]

    added, skipped, no_age = [], [], []
    for (d, pid, sex_s, age_int, h_cm, wt,
         fvc_ref, fev1_ref, ratio_ref,
         fvc_pre, fev1_pre, ratio_pre,
         fvc_post, fev1_post, ratio_post, ec_pre, ec_post) in READINGS:
        if pid in have:
            skipped.append(pid)
            continue

        m = meta.get(pid, {})
        age = fractional_age(m.get("birth", ""), m.get("study", "")) or float(age_int)
        if not m.get("birth"):
            no_age.append(pid)
        sex = 1 if sex_s == "M" else 2

        post = ratio_post is not None
        ratio = ratio_post if post else ratio_pre
        fvc = fvc_post if post else fvc_pre
        fev1 = fev1_post if post else fev1_pre
        fvc_pct = round(fvc / fvc_ref * 100)
        fev1_pct = round(fev1 / fev1_ref * 100)

        row = [""] * len(out_header)
        def put(name, value):
            row[col[name]] = "" if value is None else value

        put("Date", d); put("PatientID", pid)
        put("FEV1FVC_pct", ratio); put("Source", "Post" if post else "Pre")
        put("FEV1FVC_pre", ratio_pre); put("FEV1FVC_post", ratio_post)
        put("FEV1FVC_ref", ratio_ref)
        put("FVC_pre", fvc_pre); put("FEV1_pre", fev1_pre)
        put("FVC_post", fvc_post); put("FEV1_post", fev1_post)
        put("FVC_ref", fvc_ref); put("FEV1_ref", fev1_ref)
        put("FVC_pctpred", fvc_pct); put("FEV1_pctpred", fev1_pct)
        put("FVC_pctpred_pre", round(fvc_pre / fvc_ref * 100))
        put("FEV1_pctpred_pre", round(fev1_pre / fev1_ref * 100))
        if post:
            put("FVC_pctpred_post", round(fvc_post / fvc_ref * 100))
            put("FEV1_pctpred_post", round(fev1_post / fev1_ref * 100))
            put("dFEV1_mL", round((fev1_post - fev1_pre) * 1000))
            put("dFEV1_pct", round((fev1_post - fev1_pre) / fev1_pre * 100, 1))
            put("dFVC_mL", round((fvc_post - fvc_pre) * 1000))
            put("dFVC_pct", round((fvc_post - fvc_pre) / fvc_pre * 100, 1))
            d_fev1_ml = (fev1_post - fev1_pre) * 1000
            d_fev1_pc = (fev1_post - fev1_pre) / fev1_pre * 100
            d_fvc_ml = (fvc_post - fvc_pre) * 1000
            d_fvc_pc = (fvc_post - fvc_pre) / fvc_pre * 100
            put("BD_response_2005",
                "Y" if ((d_fev1_pc >= 12 and d_fev1_ml >= 200) or
                        (d_fvc_pc >= 12 and d_fvc_ml >= 200)) else "N")
            put("BD_response_2021",
                "Y" if ((fev1_post - fev1_pre) / fev1_ref * 100 > 10 or
                        (fvc_post - fvc_pre) / fvc_ref * 100 > 10) else "N")
        else:
            put("BD_response_2005", "NA"); put("BD_response_2021", "NA")

        obstructed = ratio < 70
        put("Obstruction_fixed70", "Y" if obstructed else "N")
        put("Severity_ATS", severity_ats(fev1_pct))
        put("Severity_GOLD", gold(fev1_pct) if obstructed else "")

        calc = round(fev1 / fvc * 100, 1)
        put("Check_calc", calc); put("Check_diff", round(calc - ratio, 1))
        # Check_FVCpct_diff / Check_FEV1pct_diff intentionally left blank.

        put("Age", age); put("Sex", sex_s); put("Height_cm", h_cm)
        put("Weight_kg", wt)
        put("FVL_ECode_pre", ec_pre); put("FVL_ECode_post", ec_post)

        for prefix, mod, suffix in (("", gli2012, "_GLI"), ("gl", gli_global, "_GLIgl")):
            try:
                kw = {"ethnicity": ETHNICITY} if mod is gli2012 else {}
                _, m_fev1, _ = mod.lms(age, h_cm, sex, "FEV1", **kw)
                _, m_fvc, _ = mod.lms(age, h_cm, sex, "FVC", **kw)
                _, m_rat, _ = mod.lms(age, h_cm, sex, "FEV1FVC", **kw)
                z_fev1 = mod.zscore(fev1, age, h_cm, sex, "FEV1", **kw)
                z_fvc = mod.zscore(fvc, age, h_cm, sex, "FVC", **kw)
                z_rat = mod.zscore(ratio / 100, age, h_cm, sex, "FEV1FVC", **kw)
                l_fev1 = mod.lln(age, h_cm, sex, "FEV1", **kw)
                l_fvc = mod.lln(age, h_cm, sex, "FVC", **kw)
                l_rat = mod.lln(age, h_cm, sex, "FEV1FVC", **kw) * 100
            except Exception as exc:                            # noqa: BLE001
                print(f"  {pid}: GLI{suffix} failed: {type(exc).__name__}: {exc}")
                continue
            put(f"FEV1_pred{suffix}", round(m_fev1, 2))
            put(f"FEV1_LLN{suffix}", round(l_fev1, 2))
            put("FEV1_z" if not prefix else "FEV1_z_GLIgl", round(z_fev1, 2))
            put(f"FVC_pred{suffix}", round(m_fvc, 2))
            put(f"FVC_LLN{suffix}", round(l_fvc, 2))
            put("FVC_z" if not prefix else "FVC_z_GLIgl", round(z_fvc, 2))
            put(f"FEV1FVC_pred{suffix}", round(m_rat * 100, 1))
            put(f"FEV1FVC_LLN{suffix}", round(l_rat, 1))
            put("FEV1FVC_z" if not prefix else "FEV1FVC_z_GLIgl", round(z_rat, 2))
            obs_gli = ratio < l_rat
            put(f"Obstruction{suffix}", "Y" if obs_gli else "N")
            sev_col = "Severity_GLI_2022" if not prefix else "Severity_GLIgl_2022"
            put(sev_col, severity_gli(z_fev1) if obs_gli else "")
            if not prefix:
                put("FEV1_pctpred_GLI", round(fev1 / m_fev1 * 100))
                put("FVC_pctpred_GLI", round(fvc / m_fvc * 100))

        put("StudyDate", m.get("study", ""))
        put("StudyDescription", m.get("desc", ""))
        # match the backslash form the existing rows use
        put("JpgPath", m.get("jpg", "").replace("/", "\\"))
        added.append(row)

    print(f"to append : {len(added)}")
    print(f"already in: {len(skipped)} {skipped}")
    if no_age:
        print(f"no DICOM birth date, used the printed integer age: {no_age}")

    if args.dry_run:
        for r in added[:2]:
            print("  sample:", dict(zip(out_header, r)))
        return

    out = args.out or args.csv
    backup = args.csv.with_suffix(".before_merge.csv")
    if out == args.csv and not backup.exists():
        backup.write_bytes(args.csv.read_bytes())
        print(f"backup   : {backup}")
    with io.open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(out_header)
        w.writerows(padded + added)
    print(f"wrote {out}: {len(padded) + len(added)} rows, {len(out_header)} columns")


if __name__ == "__main__":
    main()
