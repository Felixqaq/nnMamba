"""Tests for cross-run patient-level error recurrence analysis."""

from __future__ import annotations

import json

import pytest

from scripts.analyze_recurring_errors import (
    build_analysis,
    load_prediction_run,
    pairwise_rows,
)


def _write_run(tmp_path, name: str, decisions: dict[str, tuple[str, str]]):
    path = tmp_path / f"{name}_predictions.json"
    path.write_text(
        json.dumps(
            {
                "patients": {
                    patient_id: {
                        "true_label": truth,
                        "pred_label": prediction,
                        "prob_abnormal": 0.5,
                    }
                    for patient_id, (truth, prediction) in decisions.items()
                }
            }
        ),
        encoding="utf-8",
    )
    return load_prediction_run(path, tmp_path)


def test_exact_cohort_recurring_errors_are_counted(tmp_path) -> None:
    first = _write_run(
        tmp_path,
        "first",
        {
            "P1": ("Normal", "Abnormal"),
            "P2": ("Abnormal", "Normal"),
            "P3": ("Normal", "Normal"),
            "P4": ("Abnormal", "Abnormal"),
        },
    )
    second = _write_run(
        tmp_path,
        "second",
        {
            "P1": ("Normal", "Abnormal"),
            "P2": ("Abnormal", "Abnormal"),
            "P3": ("Normal", "Abnormal"),
            "P4": ("Abnormal", "Abnormal"),
        },
    )

    analysis = build_analysis([first, second])
    cohort = analysis["cohorts"][0]

    assert cohort["runs"] == 2
    assert cohort["wrong_in_any_run"] == 3
    assert cohort["wrong_in_at_least_two_runs"] == 1
    assert cohort["wrong_in_every_run"] == 1
    assert cohort["true_label_counts"] == {"Abnormal": 2, "Normal": 2}
    assert cohort["recurrent_rate_by_true_label"] == {
        "Abnormal": 0.0,
        "Normal": 0.5,
    }
    assert cohort["error_count_distribution"] == {"0": 1, "1": 2, "2": 1}


def test_pairwise_overlap_uses_only_shared_patients(tmp_path) -> None:
    first = _write_run(
        tmp_path,
        "first",
        {
            "P1": ("Normal", "Abnormal"),
            "P2": ("Normal", "Abnormal"),
            "P3": ("Normal", "Normal"),
        },
    )
    second = _write_run(
        tmp_path,
        "second",
        {
            "P1": ("Normal", "Abnormal"),
            "P2": ("Normal", "Normal"),
            "P4": ("Abnormal", "Normal"),
        },
    )

    row = pairwise_rows([first, second])[0]

    assert row["same_exact_cohort"] is False
    assert row["shared_patients"] == 2
    assert row["errors_a_in_shared"] == 2
    assert row["errors_b_in_shared"] == 1
    assert row["shared_errors"] == 1
    assert row["expected_shared_errors_if_independent"] == 1.0


def test_truth_label_disagreement_is_rejected(tmp_path) -> None:
    first = _write_run(tmp_path, "first", {"P1": ("Normal", "Normal")})
    second = _write_run(tmp_path, "second", {"P1": ("Abnormal", "Abnormal")})

    with pytest.raises(ValueError, match="truth labels differ"):
        pairwise_rows([first, second])


def test_single_run_does_not_claim_recurrence(tmp_path) -> None:
    only = _write_run(
        tmp_path,
        "only",
        {
            "P1": ("Normal", "Abnormal"),
            "P2": ("Abnormal", "Abnormal"),
        },
    )

    cohort = build_analysis([only])["cohorts"][0]

    assert cohort["recurrence_evaluable"] is False
    assert cohort["wrong_in_at_least_two_runs"] is None
    assert cohort["recurrent_fraction_of_any_error"] is None
    assert cohort["wrong_in_every_run"] is None
