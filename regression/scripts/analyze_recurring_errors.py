#!/usr/bin/env python3
"""Find patients that are misclassified repeatedly across prediction runs.

The analysis keeps exact-cohort comparisons separate from partial-overlap
comparisons.  This matters because two runs can only be said to fail on the
"same patients" when they evaluated the same people; otherwise a small overlap
may simply reflect different holdout construction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REGRESSION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REGRESSION_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "error_recurrence_analysis"


@dataclass(frozen=True)
class PatientPrediction:
    """One held-out patient's final decision in one run."""

    true_label: str
    pred_label: str
    prob_abnormal: float | None

    @property
    def is_error(self) -> bool:
        return self.true_label != self.pred_label


@dataclass(frozen=True)
class PredictionRun:
    """A named prediction file and its patient-level decisions."""

    name: str
    path: Path
    patients: dict[str, PatientPrediction]

    @property
    def patient_ids(self) -> frozenset[str]:
        return frozenset(self.patients)

    @property
    def error_ids(self) -> frozenset[str]:
        return frozenset(
            patient_id
            for patient_id, prediction in self.patients.items()
            if prediction.is_error
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory searched recursively for *predictions.json",
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        action="append",
        default=[],
        help="additional prediction JSON (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--pft-csv",
        type=Path,
        default=None,
        help="optional PatientID/FEV1FVC_pct CSV used to annotate patient rows",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=1,
        help="minimum shared patients required for a pairwise row",
    )
    return parser.parse_args()


def _run_name(path: Path, input_dir: Path) -> str:
    try:
        relative = path.resolve().relative_to(input_dir.resolve())
    except ValueError:
        relative = path.resolve()
    return relative.with_suffix("").as_posix()


def load_prediction_run(path: Path, input_dir: Path) -> PredictionRun:
    """Load the common doctor-validation ``patients`` JSON schema."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    patients_payload = payload.get("patients")
    if not isinstance(patients_payload, dict) or not patients_payload:
        raise ValueError(f"{path}: expected a non-empty 'patients' object")

    patients: dict[str, PatientPrediction] = {}
    for raw_patient_id, raw_row in patients_payload.items():
        patient_id = str(raw_patient_id).strip()
        if not patient_id:
            raise ValueError(f"{path}: empty patient ID")
        if not isinstance(raw_row, dict):
            raise ValueError(f"{path}: patient {patient_id} is not an object")
        true_label = raw_row.get("true_label")
        pred_label = raw_row.get("pred_label")
        if not isinstance(true_label, str) or not isinstance(pred_label, str):
            raise ValueError(
                f"{path}: patient {patient_id} lacks true_label/pred_label"
            )
        probability = raw_row.get(
            "mean_prob_abnormal", raw_row.get("prob_abnormal")
        )
        patients[patient_id] = PatientPrediction(
            true_label=true_label,
            pred_label=pred_label,
            prob_abnormal=None if probability is None else float(probability),
        )

    return PredictionRun(
        name=_run_name(path, input_dir),
        path=path.resolve(),
        patients=patients,
    )


def discover_runs(input_dir: Path, extra_paths: Iterable[Path]) -> list[PredictionRun]:
    """Discover ensemble/final prediction files while rejecting duplicate paths."""
    discovered = list(input_dir.rglob("*predictions.json")) if input_dir.exists() else []
    paths = sorted(
        {path.resolve() for path in [*discovered, *extra_paths]},
        key=lambda path: path.as_posix(),
    )
    if not paths:
        raise ValueError(f"no *predictions.json files found under {input_dir}")
    return [load_prediction_run(path, input_dir) for path in paths]


def cohort_fingerprint(patient_ids: Iterable[str]) -> str:
    """Return a stable, non-identifying fingerprint of an exact patient set."""
    joined = "\0".join(sorted(patient_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def assign_cohort_ids(runs: list[PredictionRun]) -> dict[str, str]:
    """Map patient-set fingerprints to compact IDs in first-seen order."""
    cohort_ids: dict[str, str] = {}
    for run in runs:
        fingerprint = cohort_fingerprint(run.patient_ids)
        cohort_ids.setdefault(fingerprint, f"cohort_{len(cohort_ids) + 1:02d}")
    return cohort_ids


def _hypergeometric_overlap_p(
    population: int,
    errors_a: int,
    errors_b: int,
    observed_overlap: int,
) -> float:
    """One-sided P(X >= observed) for overlap of two fixed-size error sets."""
    if population <= 0:
        return float("nan")
    denominator = math.comb(population, errors_b)
    upper = min(errors_a, errors_b)
    lower = max(observed_overlap, errors_b - (population - errors_a), 0)
    numerator = sum(
        math.comb(errors_a, overlap)
        * math.comb(population - errors_a, errors_b - overlap)
        for overlap in range(lower, upper + 1)
    )
    return numerator / denominator


def _phi_coefficient(n11: int, n10: int, n01: int, n00: int) -> float | None:
    denominator = math.sqrt(
        (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    )
    if denominator == 0:
        return None
    return (n11 * n00 - n10 * n01) / denominator


def pairwise_rows(runs: list[PredictionRun], min_shared: int = 1) -> list[dict]:
    """Compare every pair on the patients actually shared by that pair."""
    rows: list[dict] = []
    for index, run_a in enumerate(runs):
        for run_b in runs[index + 1 :]:
            shared = run_a.patient_ids & run_b.patient_ids
            if len(shared) < min_shared:
                continue

            label_mismatches = sorted(
                patient_id
                for patient_id in shared
                if run_a.patients[patient_id].true_label
                != run_b.patients[patient_id].true_label
            )
            if label_mismatches:
                sample = ", ".join(label_mismatches[:5])
                raise ValueError(
                    f"truth labels differ between {run_a.name} and {run_b.name}: {sample}"
                )

            errors_a = run_a.error_ids & shared
            errors_b = run_b.error_ids & shared
            both_wrong = errors_a & errors_b
            either_wrong = errors_a | errors_b
            expected = len(errors_a) * len(errors_b) / len(shared)
            enrichment = len(both_wrong) / expected if expected else None
            jaccard = len(both_wrong) / len(either_wrong) if either_wrong else 1.0
            n11 = len(both_wrong)
            n10 = len(errors_a - errors_b)
            n01 = len(errors_b - errors_a)
            n00 = len(shared) - n11 - n10 - n01
            rows.append(
                {
                    "run_a": run_a.name,
                    "run_b": run_b.name,
                    "same_exact_cohort": run_a.patient_ids == run_b.patient_ids,
                    "shared_patients": len(shared),
                    "errors_a_in_shared": len(errors_a),
                    "errors_b_in_shared": len(errors_b),
                    "shared_errors": len(both_wrong),
                    "expected_shared_errors_if_independent": round(expected, 4),
                    "overlap_enrichment": None
                    if enrichment is None
                    else round(enrichment, 4),
                    "error_jaccard": round(jaccard, 4),
                    "a_errors_repeated_in_b": None
                    if not errors_a
                    else round(len(both_wrong) / len(errors_a), 4),
                    "b_errors_repeated_in_a": None
                    if not errors_b
                    else round(len(both_wrong) / len(errors_b), 4),
                    "error_phi": None
                    if (phi := _phi_coefficient(n11, n10, n01, n00)) is None
                    else round(phi, 4),
                    "overlap_p_one_sided": _hypergeometric_overlap_p(
                        len(shared), len(errors_a), len(errors_b), len(both_wrong)
                    ),
                }
            )
    return rows


def run_rows(
    runs: list[PredictionRun], cohort_ids: dict[str, str]
) -> list[dict]:
    rows = []
    for run in runs:
        errors = len(run.error_ids)
        false_normal_as_abnormal = sum(
            prediction.true_label == "Normal"
            and prediction.pred_label == "Abnormal"
            for prediction in run.patients.values()
        )
        false_abnormal_as_normal = sum(
            prediction.true_label == "Abnormal"
            and prediction.pred_label == "Normal"
            for prediction in run.patients.values()
        )
        rows.append(
            {
                "run": run.name,
                "path": str(run.path),
                "cohort": cohort_ids[cohort_fingerprint(run.patient_ids)],
                "patients": len(run.patients),
                "errors": errors,
                "error_rate": round(errors / len(run.patients), 4),
                "normal_predicted_abnormal": false_normal_as_abnormal,
                "abnormal_predicted_normal": false_abnormal_as_normal,
            }
        )
    return rows


def cohort_rows(
    runs: list[PredictionRun], cohort_ids: dict[str, str]
) -> list[dict]:
    grouped: dict[str, list[PredictionRun]] = defaultdict(list)
    for run in runs:
        grouped[cohort_fingerprint(run.patient_ids)].append(run)

    rows = []
    for fingerprint, cohort_runs in grouped.items():
        patient_ids = cohort_runs[0].patient_ids
        counts = Counter(
            sum(run.patients[patient_id].is_error for run in cohort_runs)
            for patient_id in patient_ids
        )
        any_error = sum(value for errors, value in counts.items() if errors >= 1)
        recurrent = sum(value for errors, value in counts.items() if errors >= 2)
        recurrent_ids = {
            patient_id
            for patient_id in patient_ids
            if sum(run.patients[patient_id].is_error for run in cohort_runs) >= 2
        }
        always_wrong_ids = {
            patient_id
            for patient_id in patient_ids
            if all(run.patients[patient_id].is_error for run in cohort_runs)
        }
        evaluable = len(cohort_runs) >= 2
        truth_counts = Counter(
            cohort_runs[0].patients[patient_id].true_label
            for patient_id in patient_ids
        )
        recurrent_label_counts = Counter(
            cohort_runs[0].patients[patient_id].true_label
            for patient_id in recurrent_ids
        )
        always_wrong_label_counts = Counter(
            cohort_runs[0].patients[patient_id].true_label
            for patient_id in always_wrong_ids
        )
        rows.append(
            {
                "cohort": cohort_ids[fingerprint],
                "patients": len(patient_ids),
                "runs": len(cohort_runs),
                "recurrence_evaluable": evaluable,
                "run_names": [run.name for run in cohort_runs],
                "never_wrong": counts[0],
                "wrong_in_any_run": any_error,
                "wrong_in_at_least_two_runs": recurrent if evaluable else None,
                "recurrent_fraction_of_any_error": None
                if not evaluable or not any_error
                else round(recurrent / any_error, 4),
                "wrong_in_every_run": counts[len(cohort_runs)] if evaluable else None,
                "true_label_counts": dict(sorted(truth_counts.items())),
                "recurrent_by_true_label": dict(
                    sorted(recurrent_label_counts.items())
                )
                if evaluable
                else {},
                "recurrent_rate_by_true_label": {
                    label: round(recurrent_label_counts[label] / count, 4)
                    for label, count in sorted(truth_counts.items())
                }
                if evaluable
                else {},
                "wrong_in_every_run_by_true_label": dict(
                    sorted(always_wrong_label_counts.items())
                )
                if evaluable
                else {},
                "error_count_distribution": {
                    str(error_count): counts[error_count]
                    for error_count in range(len(cohort_runs) + 1)
                },
            }
        )
    return rows


def load_fev1_fvc(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    ratios: dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {(field or "").strip() for field in (reader.fieldnames or [])}
        required = {"PatientID", "FEV1FVC_pct"}
        if not required <= fields:
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        for raw_row in reader:
            row = {(key or "").strip(): (value or "").strip() for key, value in raw_row.items()}
            if row["PatientID"] and row["FEV1FVC_pct"]:
                try:
                    ratios[row["PatientID"]] = float(row["FEV1FVC_pct"])
                except ValueError:
                    continue
    return ratios


def patient_rows(
    runs: list[PredictionRun], ratios: dict[str, float] | None = None
) -> list[dict]:
    """Rank patients across all runs, retaining the number of opportunities."""
    ratios = ratios or {}
    all_patient_ids = sorted(set().union(*(run.patient_ids for run in runs)))
    rows = []
    for patient_id in all_patient_ids:
        seen = [run for run in runs if patient_id in run.patients]
        labels = {run.patients[patient_id].true_label for run in seen}
        if len(labels) != 1:
            raise ValueError(
                f"patient {patient_id} has inconsistent truth labels: {sorted(labels)}"
            )
        wrong_runs = [run.name for run in seen if run.patients[patient_id].is_error]
        correct_runs = [run.name for run in seen if not run.patients[patient_id].is_error]
        ratio = ratios.get(patient_id)
        rows.append(
            {
                "patient_id": patient_id,
                "true_label": next(iter(labels)),
                "opportunities": len(seen),
                "errors": len(wrong_runs),
                "error_rate": round(len(wrong_runs) / len(seen), 4),
                "recurrent_error": len(wrong_runs) >= 2,
                "wrong_runs": "; ".join(wrong_runs),
                "correct_runs": "; ".join(correct_runs),
                "fev1_fvc": ratio,
                "distance_from_70": None
                if ratio is None
                else round(abs(ratio - 70.0), 4),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["errors"],
            -row["error_rate"],
            -row["opportunities"],
            row["patient_id"],
        )
    )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    serializable_rows = []
    for row in rows:
        serializable_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable_rows[0]))
        writer.writeheader()
        writer.writerows(serializable_rows)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_markdown(
    path: Path,
    runs: list[dict],
    cohorts: list[dict],
    pairs: list[dict],
    patients: list[dict],
) -> None:
    lines = [
        "# Recurrent patient-level error analysis",
        "",
        "Exact-cohort comparisons are the primary evidence. Partial-overlap pairs are "
        "kept in the CSV/JSON but are not treated as directly comparable runs.",
        "",
        "## Run summary",
        "",
        "| Cohort | Run | Patients | Errors | Error rate | Normal→Abnormal | Abnormal→Normal |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in runs:
        lines.append(
            f"| {row['cohort']} | `{row['run']}` | {row['patients']} | "
            f"{row['errors']} | {_pct(row['error_rate'])} | "
            f"{row['normal_predicted_abnormal']} | "
            f"{row['abnormal_predicted_normal']} |"
        )

    lines.extend(["", "## Exact-cohort recurrence", ""])
    for row in cohorts:
        lines.extend(
            [
                f"### {row['cohort']}: {row['patients']} patients, {row['runs']} run(s)",
                "",
            ]
        )
        if not row["recurrence_evaluable"]:
            lines.extend(
                [
                    "Recurrence is not evaluable because only one run currently exists.",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"- Wrong at least once: {row['wrong_in_any_run']}",
                f"- Wrong in at least two runs: {row['wrong_in_at_least_two_runs']}",
                "- Recurrent among ever-wrong patients: "
                f"{_pct(row['recurrent_fraction_of_any_error'])}",
                f"- Wrong in every run: {row['wrong_in_every_run']}",
                "- Recurrent truth labels: "
                + ", ".join(
                    f"{label}={count}/{row['true_label_counts'][label]} "
                    f"({_pct(row['recurrent_rate_by_true_label'][label])})"
                    for label, count in row["recurrent_by_true_label"].items()
                ),
                "- Wrong-every-run truth labels: "
                + ", ".join(
                    f"{label}={count}"
                    for label, count in row["wrong_in_every_run_by_true_label"].items()
                ),
                "- Error-count distribution: "
                + ", ".join(
                    f"{count} error(s)={patients_count}"
                    for count, patients_count in row["error_count_distribution"].items()
                ),
                "",
            ]
        )

    exact_pairs = [row for row in pairs if row["same_exact_cohort"]]
    lines.extend(
        [
            "## Pairwise error overlap on identical cohorts",
            "",
            "| Run A | Run B | Both wrong | Expected if independent | Enrichment | "
            "Jaccard | Phi | p (one-sided) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in exact_pairs:
        enrichment = row["overlap_enrichment"]
        phi = row["error_phi"]
        lines.append(
            f"| `{row['run_a']}` | `{row['run_b']}` | {row['shared_errors']} | "
            f"{row['expected_shared_errors_if_independent']:.2f} | "
            f"{'—' if enrichment is None else f'{enrichment:.2f}x'} | "
            f"{row['error_jaccard']:.3f} | "
            f"{'—' if phi is None else f'{phi:.3f}'} | "
            f"{row['overlap_p_one_sided']:.3g} |"
        )

    recurrent = [row for row in patients if row["recurrent_error"]]
    lines.extend(
        [
            "",
            "## Most recurrent patients across all available runs",
            "",
            "The denominator is shown because patients were not present in every cohort.",
            "",
            "| Patient | Truth | Errors / opportunities | Error rate | Distance from 70 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in recurrent[:30]:
        distance = row["distance_from_70"]
        lines.append(
            f"| {row['patient_id']} | {row['true_label']} | "
            f"{row['errors']} / {row['opportunities']} | {_pct(row['error_rate'])} | "
            f"{'—' if distance is None else f'{distance:g}'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis(
    runs: list[PredictionRun], ratios: dict[str, float] | None = None
) -> dict:
    cohort_ids = assign_cohort_ids(runs)
    return {
        "runs": run_rows(runs, cohort_ids),
        "cohorts": cohort_rows(runs, cohort_ids),
        "pairwise": pairwise_rows(runs),
        "patients": patient_rows(runs, ratios),
    }


def main() -> None:
    args = parse_args()
    if args.min_shared < 1:
        raise SystemExit("--min-shared must be at least 1")
    runs = discover_runs(args.input_dir, args.prediction)
    ratios = load_fev1_fvc(args.pft_csv)
    analysis = build_analysis(runs, ratios)
    analysis["pairwise"] = pairwise_rows(runs, args.min_shared)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "recurring_error_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(args.output_dir / "run_summary.csv", analysis["runs"])
    _write_csv(args.output_dir / "cohort_summary.csv", analysis["cohorts"])
    _write_csv(args.output_dir / "pairwise_error_overlap.csv", analysis["pairwise"])
    _write_csv(args.output_dir / "patient_error_frequency.csv", analysis["patients"])
    write_markdown(
        args.output_dir / "recurring_error_report.md",
        analysis["runs"],
        analysis["cohorts"],
        analysis["pairwise"],
        analysis["patients"],
    )

    print(f"loaded {len(runs)} prediction runs")
    for cohort in analysis["cohorts"]:
        recurrent = cohort["wrong_in_at_least_two_runs"]
        print(
            f"{cohort['cohort']}: {cohort['patients']} patients, "
            f"{cohort['runs']} runs, recurrent errors="
            f"{'not evaluable' if recurrent is None else recurrent}"
        )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
