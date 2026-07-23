"""Inline-runner tests for scripts/label_backfill.py."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.label_backfill import (
    load_gold_labels,
    gold_class_to_folder,
    plan_backfill,
    apply_backfill,
)


def _write_nifti(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), str(path))


def _make_staging(root: Path, entries: list[tuple[str, str]]) -> Path:
    """entries: list of (patient_id, nifti_filename)."""
    staging = root / "staging"
    (staging / "incoming").mkdir(parents=True, exist_ok=True)
    lines = []
    for pid, fname in entries:
        _write_nifti(staging / "incoming" / fname)
        lines.append(
            json.dumps(
                {
                    "patient_id": pid,
                    "nifti": f"incoming/{fname}",
                    "captured_at": "20260723_120000_000000",
                    "series_uid": "1.2.3",
                    "prediction": {"prob_abnormal": 0.5, "pred": "Normal"},
                    "label": None,
                }
            )
        )
    (staging / "capture_log.jsonl").write_text("\n".join(lines) + "\n")
    return staging


def _make_gold_json(root: Path, mapping: dict[str, int]) -> Path:
    p = root / "gold.json"
    records = [
        {
            "order": i + 1,
            "patient_id": pid,
            "fev1_fvc_measured_percent": 89,
            "fev1_fvc_reference_percent": 82,
            "post_fev1_percent_predicted": 105,
            "class": cls,
            "gold_stage": f"Class {cls}",
            "severity": "x",
        }
        for i, (pid, cls) in enumerate(mapping.items())
    ]
    p.write_text(json.dumps({"records": records}), encoding="utf-8")
    return p


def test_gold_class_to_folder():
    assert gold_class_to_folder(0) == "Normal"
    for c in (1, 2, 3, 4):
        assert gold_class_to_folder(c) == "Abnormal"
    print("test_gold_class_to_folder PASS")


def test_load_gold_labels():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        g = _make_gold_json(d, {"P1": 0, "P2": 3})
        m = load_gold_labels(g)
        assert m == {"P1": 0, "P2": 3}
    print("test_load_gold_labels PASS")


def test_plan_splits_ingest_and_waiting():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        staging = _make_staging(d, [("P1", "P1_a.nii.gz"), ("PX", "PX_a.nii.gz")])
        gold = _make_gold_json(d, {"P1": 3})  # PX has no PFT yet
        dataset_root = d / "dataset"
        (dataset_root / "Normal").mkdir(parents=True)
        (dataset_root / "Abnormal").mkdir(parents=True)

        plan = plan_backfill(staging, gold, dataset_root)
        assert len(plan.to_ingest) == 1
        assert plan.to_ingest[0]["patient_id"] == "P1"
        assert plan.to_ingest[0]["folder"] == "Abnormal"
        assert plan.waiting_for_pft == ["PX"]
        assert plan.conflicts == []
    print("test_plan_splits_ingest_and_waiting PASS")


def test_dry_run_moves_nothing_and_commit_moves():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        staging = _make_staging(d, [("P1", "P1_a.nii.gz")])
        gold = _make_gold_json(d, {"P1": 0})
        dataset_root = d / "dataset"
        (dataset_root / "Normal").mkdir(parents=True)
        (dataset_root / "Abnormal").mkdir(parents=True)
        ledger = d / "ledger.jsonl"

        plan = plan_backfill(staging, gold, dataset_root)
        # dry run: planning alone must not move anything
        assert list((dataset_root / "Normal").glob("*.nii.gz")) == []

        n = apply_backfill(plan, staging, dataset_root, ledger)
        assert n == 1
        moved = list((dataset_root / "Normal").glob("*.nii.gz"))
        assert len(moved) == 1
        assert moved[0].name.startswith("P1_")
        assert ledger.exists()
        rec = json.loads(ledger.read_text().strip())
        assert rec["patient_id"] == "P1"
        assert rec["folder"] == "Normal"
        assert rec["gold_class"] == 0
    print("test_dry_run_moves_nothing_and_commit_moves PASS")


def test_conflict_when_patient_already_in_dataset():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        staging = _make_staging(d, [("P1", "P1_a.nii.gz")])
        gold = _make_gold_json(d, {"P1": 0})
        dataset_root = d / "dataset"
        (dataset_root / "Normal").mkdir(parents=True)
        (dataset_root / "Abnormal").mkdir(parents=True)
        _write_nifti(dataset_root / "Normal" / "P1_existing.nii.gz")

        plan = plan_backfill(staging, gold, dataset_root)
        assert plan.to_ingest == []
        assert len(plan.conflicts) == 1
        assert plan.conflicts[0]["patient_id"] == "P1"
    print("test_conflict_when_patient_already_in_dataset PASS")


def test_idempotent_second_run_ingests_nothing():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        staging = _make_staging(d, [("P1", "P1_a.nii.gz")])
        gold = _make_gold_json(d, {"P1": 0})
        dataset_root = d / "dataset"
        (dataset_root / "Normal").mkdir(parents=True)
        (dataset_root / "Abnormal").mkdir(parents=True)
        ledger = d / "ledger.jsonl"

        apply_backfill(plan_backfill(staging, gold, dataset_root), staging, dataset_root, ledger)
        plan2 = plan_backfill(staging, gold, dataset_root)
        assert plan2.to_ingest == []
        n2 = apply_backfill(plan2, staging, dataset_root, ledger)
        assert n2 == 0
    print("test_idempotent_second_run_ingests_nothing PASS")


if __name__ == "__main__":
    test_gold_class_to_folder()
    test_load_gold_labels()
    test_plan_splits_ingest_and_waiting()
    test_dry_run_moves_nothing_and_commit_moves()
    test_conflict_when_patient_already_in_dataset()
    test_idempotent_second_run_ingests_nothing()
