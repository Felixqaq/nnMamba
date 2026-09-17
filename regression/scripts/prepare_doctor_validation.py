#!/usr/bin/env python3
"""Freeze the doctor-validation split and export its audited Excel workbook.

The curated PFT CSV is the cohort source of truth. Patients added since the
previous 383-case manifest are placed in the holdout first; older patients are
sampled only to reach the requested size. The split is written once and reused
on later runs so model results cannot influence membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


REPO_ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = REPO_ROOT / "regression"
DEFAULT_PFT_CSV = Path("/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv")
DEFAULT_CT_ROOT = REPO_ROOT / "classification/datasets/normal_v_abnormal_fev1fvc70"
DEFAULT_COHORT_ROOT = (
    REPO_ROOT / "classification/datasets/doctor_validation_fev1fvc70_pft430"
)
DEFAULT_OUTPUT_DIR = REGRESSION_ROOT / "outputs/doctor_validation_20260827"
DEFAULT_PRIOR_MANIFEST = (
    REGRESSION_ROOT / "datasets/generated/rq1_fev1fvc70_manifest.image.json"
)
DEFAULT_DECISIONS = REGRESSION_ROOT / "cohort_decisions.local.json"


def _default_excluded() -> dict[str, str]:
    """Excluded patients and their reasons, from the gitignored decisions file.

    The IDs sat here as a literal dict until 2026-09-17. They name patients
    alongside a clinical reason, and this repository is public, so they moved to
    a file git does not track. A missing file raises rather than returning an
    empty mapping: an empty mapping would readmit patients whose spirometry is
    known to be invalid, and the workbook this script writes would carry them as
    if they were sound.
    """
    if not DEFAULT_DECISIONS.is_file():
        raise SystemExit(
            f"{DEFAULT_DECISIONS} is missing. It holds the patient exclusions and is "
            "deliberately untracked; copy it from the machine that has it. Refusing "
            "to build a validation workbook without the exclusion list."
        )
    payload = json.loads(DEFAULT_DECISIONS.read_text(encoding="utf-8"))
    if "excluded" not in payload:
        raise SystemExit(f"{DEFAULT_DECISIONS}: no 'excluded' section")
    return dict(payload["excluded"])


DEFAULT_EXCLUDED = _default_excluded()
REQUIRED_COLUMNS = {
    "Date",
    "PatientID",
    "FEV1FVC_pct",
    "Sex",
    "Age",
    "FEV1_pctpred",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pft-csv", type=Path, default=DEFAULT_PFT_CSV)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--prior-manifest", type=Path, default=DEFAULT_PRIOR_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--excel-name", default="doctor_validation_200.xlsx")
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--exclude-imaging-id",
        action="append",
        default=[],
        help="exclude a PFT-valid patient whose available CT is not usable thoracic imaging",
    )
    parser.add_argument(
        "--exclude-imaging",
        action="append",
        default=[],
        metavar="PATIENT_ID=REASON",
        help="exclude an imaging-invalid patient and preserve the exact reason",
    )
    parser.add_argument(
        "--use-frozen-split-subset",
        action="store_true",
        help=(
            "when split.json already exists, ignore later-added PFT rows while requiring "
            "every patient in the frozen split to remain present"
        ),
    )
    parser.add_argument("--mamba-json", type=Path)
    parser.add_argument("--tapct-json", type=Path)
    parser.add_argument("--qct-csv", type=Path)
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="materialize cohort/manifests only; do not create a workbook",
    )
    parser.add_argument(
        "--require-results",
        action="store_true",
        help="fail unless all validation patients have Mamba, TAPCT and QCT results",
    )
    return parser.parse_args()


def read_pft_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read, normalize and validate the curated PFT rows."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = [
        {(key or "").strip(): (value or "").strip() for key, value in row.items()}
        for row in raw_rows
    ]
    columns = set(rows[0]) if rows else set()
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        raise SystemExit(f"PFT CSV is missing columns: {missing_columns}")

    ids = [row["PatientID"] for row in rows]
    duplicates = sorted(pid for pid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"PFT CSV has duplicate PatientID values: {duplicates[:10]}")

    excluded: list[dict[str, str]] = []
    eligible: list[dict[str, str]] = []
    for row in rows:
        pid = row["PatientID"]
        if pid in DEFAULT_EXCLUDED:
            excluded.append(
                {
                    "PatientID": pid,
                    "Reason": DEFAULT_EXCLUDED[pid],
                    "Batch": row.get("Date", ""),
                }
            )
            continue
        ratio = float(row["FEV1FVC_pct"])
        row["Fixed70_Label"] = "Abnormal" if ratio < 70.0 else "Normal"
        if row["Sex"] not in {"F", "M"}:
            raise SystemExit(f"{pid}: unexpected Sex={row['Sex']!r}")
        float(row["Age"])
        float(row["FEV1_pctpred"])
        eligible.append(row)
    return eligible, excluded


def load_manifest_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {str(row["patient_id"]) for row in payload.get("records", [])}


def proportional_targets(rows: list[dict[str, str]], total: int) -> dict[str, int]:
    """Largest-remainder quotas for label-by-sex strata."""
    counts = Counter(f"{row['Fixed70_Label']}|{row['Sex']}" for row in rows)
    exact = {key: total * count / len(rows) for key, count in counts.items()}
    targets = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(targets.values())
    order = sorted(counts, key=lambda key: (exact[key] - targets[key], key), reverse=True)
    for key in order[:remainder]:
        targets[key] += 1
    return targets


def create_split(
    rows: list[dict[str, str]],
    prior_ids: set[str],
    validation_size: int,
    seed: int,
) -> dict:
    """Put every never-before-modelled patient in holdout, then fill by strata."""
    import random

    if not 1 <= validation_size < len(rows):
        raise SystemExit(
            f"validation_size must be between 1 and {len(rows) - 1}, got {validation_size}"
        )
    by_id = {row["PatientID"]: row for row in rows}
    priority_ids = sorted(set(by_id) - prior_ids)
    if len(priority_ids) > validation_size:
        raise SystemExit(
            f"{len(priority_ids)} new patients exceed validation_size={validation_size}"
        )

    targets = proportional_targets(rows, validation_size)
    selected = set(priority_ids)
    selected_counts = Counter(
        f"{by_id[pid]['Fixed70_Label']}|{by_id[pid]['Sex']}" for pid in selected
    )
    pool: dict[str, list[str]] = defaultdict(list)
    for pid, row in by_id.items():
        if pid not in selected:
            pool[f"{row['Fixed70_Label']}|{row['Sex']}"] .append(pid)

    rng = random.Random(seed)
    for key in sorted(targets):
        needed = targets[key] - selected_counts[key]
        if needed < 0:
            raise SystemExit(
                f"priority holdout already exceeds the proportional quota for {key}"
            )
        candidates = sorted(pool[key])
        rng.shuffle(candidates)
        if needed > len(candidates):
            raise SystemExit(f"not enough {key} patients to fill validation quota")
        selected.update(candidates[:needed])

    if len(selected) != validation_size:
        raise AssertionError(f"selected {len(selected)} validation patients, expected {validation_size}")
    training = sorted(set(by_id) - selected)
    validation = sorted(selected)
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "strategy": (
            "all patients absent from the frozen prior manifest enter holdout first; "
            "older patients fill label-by-sex proportional quotas"
        ),
        "validation_size": len(validation),
        "training_size": len(training),
        "priority_new_patient_ids": priority_ids,
        "validation_patient_ids": validation,
        "training_patient_ids": training,
        "validation_stratum_targets": targets,
    }


def cohort_digest(rows: list[dict[str, str]]) -> str:
    fields = ("PatientID", "Date", "Fixed70_Label", "Sex", "Age", "FEV1FVC_pct", "FEV1_pctpred")
    canonical = "\n".join(
        "|".join(row.get(field, "") for field in fields)
        for row in sorted(rows, key=lambda item: item["PatientID"])
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_cts(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.nii.gz")):
        pid = path.name[:-7].partition("_")[0]
        if pid in found:
            raise SystemExit(f"multiple CT files for patient {pid}: {found[pid]} and {path}")
        found[pid] = path.resolve()
    return found


def materialize_cohort(
    rows: list[dict[str, str]],
    split: dict,
    ct_root: Path,
    cohort_root: Path,
    output_dir: Path,
) -> None:
    """Create a PFT-only symlink view and manifests for training and QCT."""
    ct_by_id = find_cts(ct_root)
    missing = sorted(row["PatientID"] for row in rows if row["PatientID"] not in ct_by_id)
    if missing:
        raise SystemExit(f"{len(missing)} eligible PFT patients have no converted CT: {missing}")

    by_id = {row["PatientID"]: row for row in rows}
    records = []
    for pid in sorted(by_id):
        label = by_id[pid]["Fixed70_Label"]
        source = ct_by_id[pid]
        if source.parent.name != label:
            raise SystemExit(
                f"{pid}: CT folder label {source.parent.name} disagrees with PFT label {label}"
            )
        target_dir = cohort_root / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise SystemExit(f"refusing to replace mismatched cohort link: {target}")
        else:
            os.symlink(source, target)
        records.append(
            {
                "patient_id": pid,
                "path": str(target.resolve()),
                "source_group": label,
                "class_label": label,
                "class_index": 0 if label == "Abnormal" else 1,
            }
        )

    validation_ids = set(split["validation_patient_ids"])
    payload = {"class_names": ["Abnormal", "Normal"], "records": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cohort_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_payload = {
        "class_names": payload["class_names"],
        "records": [row for row in records if row["patient_id"] in validation_ids],
    }
    (output_dir / "validation_manifest.json").write_text(
        json.dumps(validation_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_prediction_rows(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    patients = payload.get("patients", payload)
    if not isinstance(patients, dict):
        raise SystemExit(f"prediction JSON must contain a patient mapping: {path}")
    return {str(pid): dict(value) for pid, value in patients.items()}


def printed_age(value: str) -> int:
    """Age as a spirometry report prints it: completed years, not a fraction.

    fev1_fvc.csv stores age computed from the DICOM birth date, so it carries two
    decimals -- about four days of resolution. Beside the exam date, which is also
    in these sheets, that is enough to reconstruct a date of birth, and it adds
    nothing a clinician reads. Floor, because a report shows completed years.
    """
    return int(float(value))


def load_qct_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["patient_id"]): row for row in csv.DictReader(handle)}


def style_sheet(sheet, table_name: str) -> None:
    sheet.freeze_panes = "A2"
    # Deliberately NOT sheet.auto_filter.ref: the Table added below carries its
    # own <autoFilter> for the same range, and setting both also emits an
    # _xlnm._FilterDatabase defined name. Excel sees three filter declarations
    # over one range, refuses to open the file cleanly and offers to "repair" it.
    # The table's own filter dropdowns give the same behaviour.
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in sheet[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    if sheet.max_row >= 2 and sheet.max_column >= 1:
        table = Table(displayName=table_name, ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False
        )
        sheet.add_table(table)
    for column in sheet.columns:
        width = min(42, max(11, max(len(str(cell.value or "")) for cell in column) + 2))
        sheet.column_dimensions[column[0].column_letter].width = width


def write_workbook(
    path: Path,
    rows: list[dict[str, str]],
    excluded: list[dict[str, str]],
    split: dict,
    mamba: dict[str, dict],
    tapct: dict[str, dict],
    qct: dict[str, dict[str, str]],
    require_results: bool,
) -> None:
    by_id = {row["PatientID"]: row for row in rows}
    validation_ids = split["validation_patient_ids"]
    if require_results:
        for name, mapping in (("Mamba", mamba), ("TAPCT", tapct), ("QCT", qct)):
            missing = sorted(set(validation_ids) - set(mapping))
            if missing:
                raise SystemExit(f"{name} is missing {len(missing)} validation patients: {missing}")

    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"

    confusion_counts: dict[str, tuple[int, int, int, int]] = {}

    def metric_rows(name: str, predictions: dict[str, dict]) -> list[tuple[str, str]]:
        """Accuracy alone is misleading on a 140/60 split -- calling everyone
        Normal already scores 70%. Report the prevalence-independent measures
        beside it so the number can be compared with published work at all."""
        evaluated = [
            pid
            for pid in validation_ids
            if predictions.get(pid, {}).get("pred_label") is not None
        ]
        if not evaluated:
            return [(f"{name} accuracy", "Not available")]

        tp = fp = tn = fn = 0
        for pid in evaluated:
            truth = by_id[pid]["Fixed70_Label"] == "Abnormal"
            pred = predictions[pid]["pred_label"] == "Abnormal"
            if truth and pred:
                tp += 1
            elif truth:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        confusion_counts[name] = (tp, fn, fp, tn)
        n = len(evaluated)
        correct = tp + tn
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")

        # AUC over the continuous score, so it does not depend on the threshold.
        scored = [
            (
                predictions[pid].get(
                    "mean_prob_abnormal", predictions[pid].get("prob_abnormal")
                ),
                by_id[pid]["Fixed70_Label"] == "Abnormal",
            )
            for pid in evaluated
        ]
        scored = [(s, t) for s, t in scored if s is not None]
        auc_text = "Not available"
        if scored and 0 < sum(t for _, t in scored) < len(scored):
            order = sorted(range(len(scored)), key=lambda i: scored[i][0])
            ranks = [0.0] * len(scored)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and scored[order[j + 1]][0] == scored[order[i]][0]:
                    j += 1
                shared = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    ranks[order[k]] = shared
                i = j + 1
            pos = sum(t for _, t in scored)
            neg = len(scored) - pos
            rank_sum = sum(r for r, (_, t) in zip(ranks, scored) if t)
            auc_text = f"{(rank_sum - pos * (pos + 1) / 2) / (pos * neg):.4f}"

        majority = max(tp + fn, tn + fp) / n
        return [
            (f"{name} accuracy", f"{100.0 * correct / n:.2f}% ({correct}/{n})"),
            (f"{name} balanced accuracy", f"{100.0 * (sens + spec) / 2:.2f}%"),
            (f"{name} AUC", auc_text),
            (f"{name} sensitivity (abnormal caught)",
             f"{100.0 * sens:.2f}% ({tp}/{tp + fn})"),
            (f"{name} specificity (normal correct)",
             f"{100.0 * spec:.2f}% ({tn}/{tn + fp})"),
            (f"{name} vs always-Normal baseline",
             f"{100.0 * (correct / n - majority):+.2f} pt "
             f"(baseline {100.0 * majority:.2f}%)"),
        ]

    summary_rows = [
        ("Item", "Value"),
        ("Eligible PFT patients", len(rows)),
        ("Doctor validation patients", len(validation_ids)),
        ("Training patients (no valid)", len(split["training_patient_ids"])),
        *metric_rows("Mamba 5-voting", mamba),
        *metric_rows("TAPCT", tapct),
        # The cohort-composition counts, the split seed, the cohort hash, the
        # split strategy, the label rule and the FEV1 REF source were all dropped
        # from this sheet at the reader's request. Every one of them is still in
        # split.json, which stays the authoritative provenance record.
        ("EMPHYSEMA definition", "%LAA-950: percent of lung-mask voxels below -950 HU"),
    ]
    for row in summary_rows:
        summary.append(row)
    style_sheet(summary, "SummaryTable")

    # A 2x2 grid rather than four scattered rows: a reader checks the diagonal
    # against the off-diagonal in one glance, and the two error types stay
    # visibly distinct instead of being averaged into one accuracy figure.
    matrix = workbook.create_sheet("混淆矩陣")
    matrix.append(["模型", "", "預測 Abnormal", "預測 Normal", "合計"])
    for model_name in ("Mamba 5-voting", "TAPCT"):
        if model_name not in confusion_counts:
            continue
        tp, fn, fp, tn = confusion_counts[model_name]
        matrix.append([model_name, "真實 Abnormal", tp, fn, tp + fn])
        matrix.append(["", "真實 Normal", fp, tn, fp + tn])
        matrix.append(["", "合計", tp + fp, fn + tn, tp + fn + fp + tn])
        matrix.append(["", "", "", "", ""])
    matrix.append(["對角線 = 判對", "", "", "", ""])
    matrix.append(["左下 FP = 誤報:實際 Normal 被判 Abnormal,代價是多做一次肺功能",
                   "", "", "", ""])
    matrix.append(["右上 FN = 漏診:實際 Abnormal 被判 Normal,代價是病人被放走",
                   "", "", "", ""])
    matrix.freeze_panes = "A2"
    head_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in matrix[1]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for column in matrix.columns:
        width = min(66, max(13, max(len(str(c.value or "")) for c in column) + 2))
        matrix.column_dimensions[column[0].column_letter].width = width

    validation = workbook.create_sheet("Validation_200")
    validation_headers = [
        "Patient_ID",
        "Mamba5_Votes_Abnormal",
        "Mamba5_Result",
        "TAPCT_Result",
        "TAPCT_Abnormal_Probability",
        "Mean_Lung_HU",
        "EMPHYSEMA_LAA950_pct",
        "Sex",
        "Age_years",
        "FEV1_FVC_pct",
        "FEV1_REF_pct",
        "GroundTruth_Fixed70",
        "Batch",
    ]
    validation.append(validation_headers)
    for pid in validation_ids:
        row = by_id[pid]
        m = mamba.get(pid, {})
        t = tapct.get(pid, {})
        q = qct.get(pid, {})
        votes = m.get("votes_for_abnormal")
        validation.append(
            [
                pid,
                m.get("vote_text", f"{votes}/5" if votes is not None else None),
                m.get("pred_label"),
                t.get("pred_label"),
                t.get("prob_abnormal"),
                float(q["mean_hu"]) if q.get("mean_hu") not in (None, "") else None,
                float(q["laa950"]) if q.get("laa950") not in (None, "") else None,
                row["Sex"],
                printed_age(row["Age"]),
                float(row["FEV1FVC_pct"]),
                float(row["FEV1_pctpred"]),
                row["Fixed70_Label"],
                row["Date"],
            ]
        )
    style_sheet(validation, "ValidationPatients")
    for cell in validation["A"]:
        cell.number_format = "@"
    for cell in validation["E"][1:]:
        cell.number_format = "0.0000"
    for cell in validation["F"][1:]:
        cell.number_format = "0.0"
    # %LAA-950 is a genuinely fractional measurement, so it keeps two decimals.
    # Age, the FEV1/FVC ratio and FEV1 %predicted are whole numbers in every
    # record -- formatting them as 0.00 only printed a misleading ".00".
    for cell in validation["G"][1:]:
        cell.number_format = "0.00"
    for column in ("I", "J", "K"):
        for cell in validation[column][1:]:
            cell.number_format = "0"

    training = workbook.create_sheet(
        f"Training_{len(split['training_patient_ids'])}_NoValid"
    )
    training.append(
        ["Patient_ID", "Sex", "Age_years", "FEV1_FVC_pct", "FEV1_REF_pct", "GroundTruth_Fixed70", "Batch"]
    )
    for pid in split["training_patient_ids"]:
        row = by_id[pid]
        training.append(
            [pid, row["Sex"], printed_age(row["Age"]), float(row["FEV1FVC_pct"]), float(row["FEV1_pctpred"]), row["Fixed70_Label"], row["Date"]]
        )
    style_sheet(training, "TrainingPatients")
    for cell in training["A"]:
        cell.number_format = "@"

    excluded_sheet = workbook.create_sheet("Excluded")
    excluded_sheet.append(["Patient_ID", "Batch", "Reason"])
    for row in excluded:
        excluded_sheet.append([row["PatientID"], row["Batch"], row["Reason"]])
    style_sheet(excluded_sheet, "ExcludedPatients")
    for cell in excluded_sheet["A"]:
        cell.number_format = "@"

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def main() -> None:
    args = parse_args()
    rows, excluded = read_pft_rows(args.pft_csv)
    imaging_exclusions = {
        patient_id: "invalid imaging — available CT is non-thoracic head/neck imaging"
        for patient_id in args.exclude_imaging_id
    }
    for value in args.exclude_imaging:
        patient_id, separator, reason = value.partition("=")
        patient_id = patient_id.strip()
        reason = reason.strip()
        if not separator or not patient_id or not reason:
            raise SystemExit(
                "--exclude-imaging must use PATIENT_ID=REASON with both values present"
            )
        if patient_id in imaging_exclusions and imaging_exclusions[patient_id] != reason:
            raise SystemExit(f"conflicting imaging exclusion reasons for {patient_id}")
        imaging_exclusions[patient_id] = reason
    for patient_id, reason in imaging_exclusions.items():
        matches = [row for row in rows if row["PatientID"] == patient_id]
        if len(matches) != 1:
            raise SystemExit(
                f"--exclude-imaging-id {patient_id}: expected one eligible PFT row, "
                f"found {len(matches)}"
            )
        row = matches[0]
        rows.remove(row)
        excluded.append(
            {
                "PatientID": patient_id,
                "Batch": row.get("Date", ""),
                "Reason": reason,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_path = args.output_dir / "split.json"
    split = None
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if args.use_frozen_split_subset:
            frozen_ids = set(split["training_patient_ids"]) | set(
                split["validation_patient_ids"]
            )
            available_ids = {row["PatientID"] for row in rows}
            missing_ids = sorted(frozen_ids - available_ids)
            if missing_ids:
                raise SystemExit(
                    "patients from the frozen split are missing from the current PFT cohort: "
                    f"{missing_ids}"
                )
            later_ids = sorted(available_ids - frozen_ids)
            rows = [row for row in rows if row["PatientID"] in frozen_ids]
            print(
                f"using frozen split cohort ({len(rows)} patients); "
                f"ignoring {len(later_ids)} later-added PFT rows"
            )
    digest = cohort_digest(rows)
    if split is not None:
        if split.get("cohort_sha256") != digest:
            raise SystemExit(
                "PFT cohort changed after the split was frozen; refusing to silently re-split"
            )
        if (
            "priority_new_patient_ids" not in split
            and split.get("base_split")
        ):
            base_split_path = Path(split["base_split"])
            base_split = json.loads(base_split_path.read_text(encoding="utf-8-sig"))
            split["priority_new_patient_ids"] = base_split.get(
                "priority_new_patient_ids", []
            )
            split.setdefault("seed", base_split.get("seed"))
    else:
        split = create_split(
            rows,
            prior_ids=load_manifest_ids(args.prior_manifest),
            validation_size=args.validation_size,
            seed=args.seed,
        )
        split["pft_csv"] = str(args.pft_csv)
        split["cohort_sha256"] = digest
        split["excluded"] = excluded
        split_path.write_text(
            json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    materialize_cohort(rows, split, args.ct_root, args.cohort_root, args.output_dir)
    excel_path: Path | None = None
    if not args.skip_excel:
        mamba = load_prediction_rows(args.mamba_json)
        tapct = load_prediction_rows(args.tapct_json)
        qct = load_qct_rows(args.qct_csv)
        if Path(args.excel_name).name != args.excel_name:
            raise SystemExit("--excel-name must be a file name, not a path")
        excel_path = args.output_dir / args.excel_name
        write_workbook(
            excel_path, rows, excluded, split, mamba, tapct, qct, args.require_results
        )
    print(f"eligible={len(rows)} validation={len(split['validation_patient_ids'])} training={len(split['training_patient_ids'])}")
    print(
        "current_new_in_training="
        f"{len(split.get('new_training_patient_ids', []))}"
    )
    print(f"split={split_path}")
    print(f"excel={excel_path if excel_path is not None else 'skipped'}")
    print(f"cohort_root={args.cohort_root}")


if __name__ == "__main__":
    main()
