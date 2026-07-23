# Research-Side Retrain / Backfill / Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three research-side scripts that close the data-growth loop for `copd-ct-app`: ingest captured CTs into the dataset (`label_backfill.py`), retrain the 5-member production ensemble on all data (`train_production_ensemble.py`), and bundle a drift-checked release for the hospital (`package_release.py`).

**Architecture:** Three standalone CLI scripts in `regression/scripts/`, each reusing the existing nnMamba pipeline components rather than reimplementing them. Backfill only moves NIfTI files into the correct class folder (the `normal_v_abnormal` manifest takes its label from the parent folder, so no JSON writing is needed). Production training reuses `RegressionLoaderHelper` by overriding `fold_indices` to a single all-data fold, then runs a fixed-epoch loop. Packaging asserts the hospital repo's frozen preprocessing is bit-for-bit identical to training before bundling.

**Tech Stack:** Python 3.10+, PyTorch 2.5.1+cu124, mamba-ssm (CUDA-only), nibabel, scikit-image, numpy.

## Global Constraints

- **Scope is Normal-vs-Abnormal only.** The user only needs the COPD normal/abnormal binary task. Do NOT touch, write to, or validate against `patient_angle_classification_by_group.json`, and do not implement anything for the angle tasks (RQ2a/2b/2c) or GOLD multiclass.
- **Backfill writes no JSON into the training pipeline.** Verified in `regression/data/manifest.py`: for `target_mode == "normal_v_abnormal"`, `is_folder_label_target` is True and the code comment states *"Label comes from the parent folder name, not the angle JSON, so do not skip patients that lack an angle annotation."* Putting the file in `Normal/` or `Abnormal/` is sufficient and complete.
- **Class-index convention:** `class_index 0 = Abnormal`, `1 = Normal` (`NORMAL_V_ABNORMAL_NAMES = ["Abnormal", "Normal"]` in `regression/data/manifest.py`).
- **PFT ground truth source:** `regression/GOLD_2026_classification.json`. It has a `records` list; each record is `{"order": int, "patient_id": str, "fev1_fvc_measured_percent": num, "fev1_fvc_reference_percent": num, "post_fev1_percent_predicted": num, "class": int, "gold_stage": str, "severity": str}`. **`class == 0` → Normal** (FEV1/FVC ≥ 70%, no COPD); **`class` in 1..4 → Abnormal** (GOLD 1-4). The file is UTF-8-BOM encoded — open with `encoding="utf-8-sig"`.
- **Production training uses 100% of the data with a fixed epoch budget.** No held-out set exists, therefore: no early stopping, no best-epoch selection, and no `ReduceLROnPlateau` (it requires a monitored validation score). Use `CosineAnnealingLR` instead, and take the final epoch. This is a deliberate decision — holding out ~6-7 cases from n=66 would give an early-stopping signal dominated by noise while costing training data.
- **Production checkpoints have no clean held-out metric.** `metrics.json` must record the nested-CV reference values (Acc 0.726, AUC 0.803) and label them as a reference from prior nested CV, never as this run's measured performance.
- **Checkpoint payload format:** `{"state_dict": model.state_dict(), ...}` — must stay compatible with `copd-ct-app`'s `Ensemble.from_dir`, which does `torch.load(..., weights_only=True)` and reads `payload["state_dict"]`. Only plain tensors/primitives may go into the payload.
- **Model hyperparameters (locked):** built via `build_model(config.model, output_dim=2)` from `regression/config.normal_v_abnormal.imageonly.aug5.ensemble.yaml`.
- **Tests use the project's inline runner convention, not pytest.** Each test file ends with `if __name__ == "__main__":` calling its test functions. Run with `conda activate nnMamba && python <test_file>`.
- **Scripts run from `regression/`** (its modules are imported as top-level: `from data.loader import ...`, `from models import build_model`).
- **Hospital repo path:** `~/Research/copd-ct-app/` (separate git repo; `package_release.py` reads its `core/preprocess.py` and writes releases for it).

---

## File Structure

```
regression/
├── scripts/
│   ├── label_backfill.py             # Task 1: staging -> dataset folders + ledger
│   ├── train_production_ensemble.py  # Task 2: all-data 5-member retrain -> release/<date>/
│   └── package_release.py            # Task 3: drift check + bundle
└── tests/                            # (inline-runner tests, created by these tasks)
    ├── test_label_backfill.py
    ├── test_train_production_ensemble.py
    └── test_package_release.py
```

Note: `regression/` currently keeps its test files at the top level (`regression/test_*.py`). Follow that existing convention — create the test files as `regression/test_label_backfill.py`, `regression/test_train_production_ensemble.py`, `regression/test_package_release.py`.

---

### Task 1: `label_backfill.py` — ingest captured CTs into the dataset ✅ DONE

**Files:**
- Create: `/home/felix/Research/nnMamba/regression/scripts/label_backfill.py`
- Test: `/home/felix/Research/nnMamba/regression/test_label_backfill.py`

**Interfaces:**
- Produces:
  - `load_gold_labels(gold_json: Path) -> dict[str, int]` — patient_id → GOLD class (0-4).
  - `gold_class_to_folder(gold_class: int) -> str` — returns `"Normal"` for 0, `"Abnormal"` for 1-4.
  - `plan_backfill(staging_dir: Path, gold_json: Path, dataset_root: Path) -> BackfillPlan` — pure, no side effects.
  - `BackfillPlan` dataclass: `to_ingest: list[dict]`, `waiting_for_pft: list[str]`, `conflicts: list[dict]`, `unreadable: list[dict]`.
  - `apply_backfill(plan: BackfillPlan, staging_dir: Path, dataset_root: Path, ledger_path: Path) -> int` — performs moves, appends ledger, returns count ingested.
  - CLI: `python scripts/label_backfill.py --staging <dir> --dataset-root <dir> [--gold-json <path>] [--commit]`. Default is dry-run.

- [ ] **Step 1: Write the failing test** — `regression/test_label_backfill.py`

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/felix/Research/nnMamba/regression && conda activate nnMamba && python test_label_backfill.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.label_backfill'`

- [ ] **Step 3: Write `regression/scripts/label_backfill.py`**

```python
"""Ingest captured CTs from a collected hospital staging dir into the dataset.

Normal-vs-Abnormal only. The manifest takes the class label from the parent
folder (see regression/data/manifest.py, is_folder_label_target), so putting the
NIfTI in Normal/ or Abnormal/ is all that is required — no JSON is written into
the training pipeline. Provenance goes to a separate ledger.

Idempotent: re-running skips anything already ingested. Dry-run by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import nibabel as nib

DEFAULT_GOLD_JSON = Path(__file__).resolve().parents[1] / "GOLD_2026_classification.json"


@dataclass
class BackfillPlan:
    to_ingest: list[dict] = field(default_factory=list)
    waiting_for_pft: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    unreadable: list[dict] = field(default_factory=list)


def load_gold_labels(gold_json: Path) -> dict[str, int]:
    """Map patient_id -> GOLD class (0-4) from GOLD_2026_classification.json."""
    data = json.loads(Path(gold_json).read_text(encoding="utf-8-sig"))
    return {
        str(rec["patient_id"]).strip(): int(rec["class"])
        for rec in data["records"]
        if rec.get("patient_id") is not None and rec.get("class") is not None
    }


def gold_class_to_folder(gold_class: int) -> str:
    """GOLD class 0 = no COPD -> Normal; classes 1-4 = GOLD 1-4 -> Abnormal."""
    return "Normal" if int(gold_class) == 0 else "Abnormal"


def _patient_ids_in_dataset(dataset_root: Path) -> set[str]:
    ids: set[str] = set()
    for folder in ("Normal", "Abnormal"):
        d = dataset_root / folder
        if not d.is_dir():
            continue
        for p in d.glob("*.nii.gz"):
            ids.add(p.name.split("_")[0])
    return ids


def _read_capture_log(staging_dir: Path) -> list[dict]:
    log = staging_dir / "capture_log.jsonl"
    if not log.exists():
        return []
    records = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in rec or rec.get("nifti") is None:
            continue
        records.append(rec)
    return records


def plan_backfill(staging_dir: Path, gold_json: Path, dataset_root: Path) -> BackfillPlan:
    """Decide what to ingest. Pure — performs no filesystem mutation."""
    staging_dir = Path(staging_dir)
    dataset_root = Path(dataset_root)
    gold = load_gold_labels(gold_json)
    existing = _patient_ids_in_dataset(dataset_root)
    plan = BackfillPlan()

    for rec in _read_capture_log(staging_dir):
        pid = str(rec["patient_id"]).strip()
        src = staging_dir / rec["nifti"]

        if pid not in gold:
            if pid not in plan.waiting_for_pft:
                plan.waiting_for_pft.append(pid)
            continue
        if pid in existing:
            plan.conflicts.append({"patient_id": pid, "reason": "already in dataset"})
            continue
        if not src.exists():
            plan.unreadable.append({"patient_id": pid, "reason": f"missing file {src}"})
            continue
        try:
            vol = nib.load(str(src)).get_fdata()
            if vol.ndim < 3:
                raise ValueError(f"expected 3D volume, got shape {vol.shape}")
        except Exception as exc:
            plan.unreadable.append({"patient_id": pid, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        gold_class = gold[pid]
        plan.to_ingest.append(
            {
                "patient_id": pid,
                "src": str(src),
                "folder": gold_class_to_folder(gold_class),
                "gold_class": gold_class,
            }
        )
        existing.add(pid)  # a second capture of the same patient in one run is a conflict

    return plan


def apply_backfill(
    plan: BackfillPlan, staging_dir: Path, dataset_root: Path, ledger_path: Path
) -> int:
    """Copy planned files into the dataset and append ledger rows. Returns count."""
    dataset_root = Path(dataset_root)
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ingested = 0

    for item in plan.to_ingest:
        src = Path(item["src"])
        dest_dir = dataset_root / item["folder"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copyfile(src, dest)
        with ledger_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "patient_id": item["patient_id"],
                        "folder": item["folder"],
                        "gold_class": item["gold_class"],
                        "source": str(src),
                        "dest": str(dest),
                        "ingested_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                + "\n"
            )
        ingested += 1
    return ingested


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill captured CTs into the dataset")
    parser.add_argument("--staging", required=True, help="Collected staging dir from the hospital")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing Normal/ and Abnormal/")
    parser.add_argument("--gold-json", default=str(DEFAULT_GOLD_JSON))
    parser.add_argument("--ledger", default=None, help="Ledger path (default: <dataset-root>/backfill_ledger.jsonl)")
    parser.add_argument("--commit", action="store_true", help="Actually move files (default: dry-run)")
    args = parser.parse_args()

    staging = Path(args.staging)
    dataset_root = Path(args.dataset_root)
    ledger = Path(args.ledger) if args.ledger else dataset_root / "backfill_ledger.jsonl"

    plan = plan_backfill(staging, Path(args.gold_json), dataset_root)

    print(f"to ingest       : {len(plan.to_ingest)}")
    for item in plan.to_ingest:
        print(f"  {item['patient_id']} -> {item['folder']}/ (GOLD class {item['gold_class']})")
    print(f"waiting for PFT : {len(plan.waiting_for_pft)} {plan.waiting_for_pft}")
    print(f"conflicts       : {len(plan.conflicts)}")
    for c in plan.conflicts:
        print(f"  {c['patient_id']}: {c['reason']}")
    print(f"unreadable      : {len(plan.unreadable)}")
    for u in plan.unreadable:
        print(f"  {u['patient_id']}: {u['reason']}")

    if not args.commit:
        print("\nDRY RUN — nothing moved. Re-run with --commit to apply.")
        return

    n = apply_backfill(plan, staging, dataset_root, ledger)
    print(f"\nIngested {n} case(s). Ledger: {ledger}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/felix/Research/nnMamba/regression && python test_label_backfill.py`
Expected: six `PASS` lines.

- [ ] **Step 5: Commit**

```bash
cd /home/felix/Research/nnMamba
git add regression/scripts/label_backfill.py regression/test_label_backfill.py
git commit -m "feat: label_backfill script for ingesting captured CTs"
```

---

### Task 2: `train_production_ensemble.py` — all-data 5-member retrain ✅ DONE

**Files:**
- Create: `/home/felix/Research/nnMamba/regression/scripts/train_production_ensemble.py`
- Test: `/home/felix/Research/nnMamba/regression/test_train_production_ensemble.py`

**Interfaces:**
- Consumes: `Config.from_yaml` (`core/config.py`), `RegressionLoaderHelper` (`data/loader.py`, exported as `LoaderHelper`), `build_model` (`models.py`).
- Produces:
  - `all_data_train_loader(loader_helper) -> torch.utils.data.DataLoader` — overrides `loader_helper.fold_indices` to a single all-data fold and returns `get_train_dl(0)`.
  - `train_one_member(config, loader_helper, seed: int, epochs: int, device: str) -> torch.nn.Module`
  - `write_metrics_json(out_dir: Path, n_cases: int, epochs: int, seeds: list[int]) -> None`
  - CLI: `python scripts/train_production_ensemble.py --config <yaml> --out release/<date> [--members 5] [--epochs 160] [--limit N]`

- [ ] **Step 1: Write the failing test** — `regression/test_train_production_ensemble.py`

This test is a smoke test: it trains 2 members for 1 epoch on a small subset and asserts the checkpoints are written in a format `copd-ct-app`'s `Ensemble` can load.

```python
"""Inline-runner smoke test for scripts/train_production_ensemble.py."""

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import Config
from data.loader import RegressionLoaderHelper as LoaderHelper
from scripts.train_production_ensemble import (
    all_data_train_loader,
    train_one_member,
    write_metrics_json,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.normal_v_abnormal.imageonly.aug5.ensemble.yaml"


def test_all_data_loader_uses_every_case():
    config = Config.from_yaml(str(CONFIG_PATH))
    helper = LoaderHelper(config)
    n_cases = len(helper.dataset)
    dl = all_data_train_loader(helper)
    assert helper.fold_indices[0][1] == [], "val split must be empty for production training"
    assert len(helper.fold_indices[0][0]) == n_cases, "train split must cover every case"
    assert len(dl) > 0
    print(f"test_all_data_loader_uses_every_case PASS (n={n_cases})")


def test_smoke_train_writes_loadable_checkpoints():
    config = Config.from_yaml(str(CONFIG_PATH))
    helper = LoaderHelper(config)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "release"
        out.mkdir(parents=True)
        seeds = [42, 43]
        for i, seed in enumerate(seeds, start=1):
            model = train_one_member(config, helper, seed=seed, epochs=1, device="cuda")
            torch.save({"state_dict": model.state_dict(), "seed": seed}, out / f"member_{i}.pth")
        write_metrics_json(out, n_cases=len(helper.dataset), epochs=1, seeds=seeds)

        assert (out / "metrics.json").exists()
        meta = json.loads((out / "metrics.json").read_text())
        assert meta["n_training_cases"] == len(helper.dataset)
        assert meta["held_out"] is False

        # must be loadable the way copd-ct-app loads it
        for i in range(1, 3):
            payload = torch.load(out / f"member_{i}.pth", map_location="cpu", weights_only=True)
            assert "state_dict" in payload
    print("test_smoke_train_writes_loadable_checkpoints PASS")


if __name__ == "__main__":
    test_all_data_loader_uses_every_case()
    test_smoke_train_writes_loadable_checkpoints()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/felix/Research/nnMamba/regression && python test_train_production_ensemble.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.train_production_ensemble'`

- [ ] **Step 3: Write `regression/scripts/train_production_ensemble.py`**

```python
"""Retrain the 5-member production ensemble on 100% of the dataset.

No held-out set exists by design, so there is no early stopping, no best-epoch
selection, and no ReduceLROnPlateau (it needs a monitored validation score).
Training runs a fixed epoch budget with CosineAnnealingLR and takes the final
epoch. Members differ only by seed.

Data pipeline (dataset, augmentation, balanced sampling) is reused verbatim from
RegressionLoaderHelper by overriding fold_indices to a single all-data fold, so
the training distribution matches the cross-validated runs.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import Config  # noqa: E402
from data.loader import RegressionLoaderHelper as LoaderHelper  # noqa: E402
from models import build_model  # noqa: E402

# Reference numbers from prior nested cross-validation. NOT measured on this run.
NESTED_CV_REFERENCE = {"accuracy": 0.726, "auc": 0.803, "source": "nested CV, 66-case, RQ1 image-only"}


def all_data_train_loader(loader_helper):
    """Return a training DataLoader covering every case.

    Overrides fold_indices with a single fold whose train split is all indices
    and whose val split is empty, then reuses get_train_dl so augmentation and
    balanced sampling behave exactly as in cross-validated training.
    """
    n = len(loader_helper.dataset)
    loader_helper.fold_indices = [(list(range(n)), [])]
    return loader_helper.get_train_dl(0, shuffle=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_member(config, loader_helper, seed: int, epochs: int, device: str = "cuda") -> nn.Module:
    """Train a single ensemble member on all data for a fixed epoch budget."""
    _set_seed(seed)
    model = build_model(config.model, output_dim=2).to(device)
    model.train()

    train_dl = all_data_train_loader(loader_helper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    loss_fn = nn.CrossEntropyLoss()
    clip = float(getattr(config.training, "clip_grad_norm", 0.0) or 0.0)

    for epoch in range(1, int(epochs) + 1):
        running = 0.0
        nb = 0
        for batch in train_dl:
            ct = batch["ct"].to(device).float()
            target = batch["target"].to(device).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(ct)
            loss = loss_fn(logits, target)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            running += float(loss.item())
            nb += 1
        scheduler.step()
        print(f"  seed {seed} | epoch {epoch}/{epochs} | loss {running / max(nb, 1):.4f}")

    model.eval()
    return model


def write_metrics_json(out_dir: Path, n_cases: int, epochs: int, seeds: list[int]) -> None:
    """Record what this release is, and be explicit that it has no held-out score."""
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_training_cases": int(n_cases),
        "epochs": int(epochs),
        "seeds": list(seeds),
        "held_out": False,
        "note": (
            "Trained on 100% of the dataset: no held-out set, no early stopping, "
            "no best-epoch selection. This release has NO measured performance. "
            "Quote the nested-CV reference below, labelled as a reference."
        ),
        "nested_cv_reference": NESTED_CV_REFERENCE,
    }
    (Path(out_dir) / "metrics.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the production ensemble on all data")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.normal_v_abnormal.imageonly.aug5.ensemble.yaml"),
    )
    parser.add_argument("--out", required=True, help="Output release dir, e.g. release/2026-07-23")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_yaml(args.config)
    loader_helper = LoaderHelper(config)
    n_cases = len(loader_helper.dataset)
    seeds = [args.base_seed + i for i in range(args.members)]

    print(f"Production ensemble: {args.members} members x {args.epochs} epochs on {n_cases} cases (no held-out)")
    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== member {i}/{args.members} (seed {seed}) ===")
        model = train_one_member(config, loader_helper, seed=seed, epochs=args.epochs, device="cuda")
        torch.save({"state_dict": model.state_dict(), "seed": seed}, out_dir / f"member_{i}.pth")
        print(f"  saved {out_dir / f'member_{i}.pth'}")

    write_metrics_json(out_dir, n_cases=n_cases, epochs=args.epochs, seeds=seeds)
    print(f"\nDone. Release dir: {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/felix/Research/nnMamba/regression && python test_train_production_ensemble.py`
Expected: two `PASS` lines. Note this loads and caches the real dataset, so it takes several minutes.

- [ ] **Step 5: Commit**

```bash
cd /home/felix/Research/nnMamba
git add regression/scripts/train_production_ensemble.py regression/test_train_production_ensemble.py
git commit -m "feat: all-data production ensemble training script"
```

---

### Task 3: `package_release.py` — drift check + bundle ✅ DONE

**Files:**
- Create: `/home/felix/Research/nnMamba/regression/scripts/package_release.py`
- Test: `/home/felix/Research/nnMamba/regression/test_package_release.py`

**Interfaces:**
- Consumes: `load_ct` from `data/dataset.py` (training reference), and the hospital repo's frozen `core/preprocess.py`.
- Produces:
  - `check_preprocess_matches(app_repo: Path) -> tuple[bool, str]` — loads the hospital repo's frozen `load_ct`, runs both it and the training `load_ct` on a fixed random volume, returns `(True, "")` on bit-for-bit equality, else `(False, reason)`.
  - `bundle_release(release_dir: Path, app_repo: Path, dest: Path) -> Path` — copies `member_*.pth` + `metrics.json` + the frozen `core/preprocess.py` into `dest`, writes `PREPROCESS_HASH`.
  - CLI: `python scripts/package_release.py --release <dir> --app-repo ~/Research/copd-ct-app --dest <dir>`. Exits non-zero (blocking the release) if the drift check fails.

- [ ] **Step 1: Write the failing test** — `regression/test_package_release.py`

```python
"""Inline-runner tests for scripts/package_release.py."""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.package_release import check_preprocess_matches, bundle_release

APP_REPO = Path.home() / "Research" / "copd-ct-app"


def test_check_passes_on_current_frozen_preprocess():
    ok, reason = check_preprocess_matches(APP_REPO)
    assert ok, f"expected frozen preprocess to match training, got: {reason}"
    print("test_check_passes_on_current_frozen_preprocess PASS")


def test_check_fails_on_tampered_preprocess():
    with tempfile.TemporaryDirectory() as d:
        fake_repo = Path(d) / "copd-ct-app"
        (fake_repo / "core").mkdir(parents=True)
        src = (APP_REPO / "core" / "preprocess.py").read_text()
        # Tamper: change the resize interpolation order, which alters output values.
        tampered = src.replace("order=1,", "order=0,")
        assert tampered != src, "tamper failed to modify the source"
        (fake_repo / "core" / "preprocess.py").write_text(tampered)

        ok, reason = check_preprocess_matches(fake_repo)
        assert not ok, "tampered preprocess must be rejected"
        assert reason
    print("test_check_fails_on_tampered_preprocess PASS")


def test_bundle_copies_members_metrics_and_preprocess():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        release = d / "release"
        release.mkdir()
        for i in (1, 2):
            torch.save({"state_dict": {"w": torch.zeros(2)}, "seed": i}, release / f"member_{i}.pth")
        (release / "metrics.json").write_text(json.dumps({"held_out": False}))

        dest = d / "bundle"
        out = bundle_release(release, APP_REPO, dest)
        assert (out / "member_1.pth").exists()
        assert (out / "member_2.pth").exists()
        assert (out / "metrics.json").exists()
        assert (out / "preprocess.py").exists()
        assert (out / "PREPROCESS_HASH").exists()
        assert len((out / "PREPROCESS_HASH").read_text().strip()) == 64
    print("test_bundle_copies_members_metrics_and_preprocess PASS")


if __name__ == "__main__":
    test_check_passes_on_current_frozen_preprocess()
    test_check_fails_on_tampered_preprocess()
    test_bundle_copies_members_metrics_and_preprocess()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/felix/Research/nnMamba/regression && python test_package_release.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.package_release'`

- [ ] **Step 3: Write `regression/scripts/package_release.py`**

```python
"""Bundle a production release for copd-ct-app, blocking on preprocessing drift.

The hospital repo ships a FROZEN copy of the CT preprocessing. This script is
the gate that proves that frozen copy still produces bit-for-bit identical
output to the training preprocessing before any release leaves the research
machine. A failed check blocks the release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import load_ct as training_load_ct  # noqa: E402

IMAGE_SIZE = (112, 136, 112)
INTENSITY_WINDOW = (-1000.0, 400.0)
INPUT_NORMALIZATION = "zscore"


def _load_frozen_preprocess(app_repo: Path):
    """Import the hospital repo's frozen preprocess module from its file path."""
    path = Path(app_repo) / "core" / "preprocess.py"
    if not path.exists():
        raise FileNotFoundError(f"Frozen preprocess not found: {path}")
    spec = importlib.util.spec_from_file_location("frozen_preprocess", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_preprocess_matches(app_repo: Path) -> tuple[bool, str]:
    """Return (True, "") iff the frozen preprocess matches training bit-for-bit."""
    try:
        frozen = _load_frozen_preprocess(app_repo)
    except Exception as exc:
        return False, f"could not load frozen preprocess: {type(exc).__name__}: {exc}"

    rng = np.random.default_rng(0)
    volume = (rng.random((90, 100, 80)).astype(np.float32) * 1400.0) - 1000.0
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "vol.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(p))
        kwargs = dict(intensity_window=INTENSITY_WINDOW, input_normalization=INPUT_NORMALIZATION)
        try:
            a = frozen.load_ct(p, IMAGE_SIZE, **kwargs)
        except Exception as exc:
            return False, f"frozen load_ct raised: {type(exc).__name__}: {exc}"
        b = training_load_ct(p, IMAGE_SIZE, **kwargs)

    if a.shape != b.shape:
        return False, f"shape mismatch: frozen {a.shape} vs training {b.shape}"
    if not np.array_equal(a, b):
        diff = float(np.abs(a - b).max())
        return False, f"value mismatch: max abs diff {diff}"
    return True, ""


def bundle_release(release_dir: Path, app_repo: Path, dest: Path) -> Path:
    """Copy checkpoints, metrics, and the frozen preprocess into a release bundle."""
    release_dir = Path(release_dir)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    members = sorted(release_dir.glob("member_*.pth"))
    if not members:
        raise FileNotFoundError(f"No member_*.pth in {release_dir}")
    for m in members:
        shutil.copyfile(m, dest / m.name)

    metrics = release_dir / "metrics.json"
    if metrics.exists():
        shutil.copyfile(metrics, dest / "metrics.json")

    preprocess_src = Path(app_repo) / "core" / "preprocess.py"
    shutil.copyfile(preprocess_src, dest / "preprocess.py")
    digest = hashlib.sha256(preprocess_src.read_bytes()).hexdigest()
    (dest / "PREPROCESS_HASH").write_text(digest + "\n")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a production release for copd-ct-app")
    parser.add_argument("--release", required=True, help="Dir containing member_*.pth and metrics.json")
    parser.add_argument("--app-repo", default=str(Path.home() / "Research" / "copd-ct-app"))
    parser.add_argument("--dest", required=True, help="Output bundle dir")
    args = parser.parse_args()

    ok, reason = check_preprocess_matches(Path(args.app_repo))
    if not ok:
        print(f"RELEASE BLOCKED — preprocessing drift detected:\n  {reason}")
        raise SystemExit(1)
    print("Preprocessing check: frozen copy matches training bit-for-bit.")

    out = bundle_release(Path(args.release), Path(args.app_repo), Path(args.dest))
    print(f"Release bundled: {out}")
    print("Ship this dir to the hospital and point models/current at it, then restart the app.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/felix/Research/nnMamba/regression && python test_package_release.py`
Expected: three `PASS` lines.

- [ ] **Step 5: Commit**

```bash
cd /home/felix/Research/nnMamba
git add regression/scripts/package_release.py regression/test_package_release.py
git commit -m "feat: package_release with preprocessing drift gate"
```

---

## Self-Review Notes

- **Spec coverage:** spec §6 data-growth loop steps ④ (backfill) → Task 1, ⑤ (all-data retrain) → Task 2, ⑥ (drift check + bundle) → Task 3. Spec §7 backfill details → Task 1. Spec §8 packaging drift gate → Task 3.
- **Deliberate deviations from the spec, both driven by the user's narrowed scope (Normal-vs-Abnormal only):**
  1. Backfill does **not** write to `patient_angle_classification_by_group.json`. Verified unnecessary: `regression/data/manifest.py` takes the `normal_v_abnormal` label from the parent folder and explicitly does not skip patients lacking an angle. Writing angle-less patients into an angle-keyed structure would corrupt its statistics. Provenance goes to `backfill_ledger.jsonl` instead.
  2. Production training uses a fixed epoch budget with `CosineAnnealingLR` and no early stopping — a held-out set cannot exist when training on 100% of the data, and a ~6-case validation split from n=66 would give a noise-dominated signal.
- **GOLD → label mapping** (`class 0` → Normal, `1-4` → Abnormal) is used identically in Task 1's implementation and tests.
- **Checkpoint format** `{"state_dict": ..., "seed": ...}` in Task 2 matches what `copd-ct-app`'s `Ensemble.from_dir` reads with `weights_only=True`; Task 2's test asserts this explicitly.
- **No pytest** — all tests use the inline-runner convention, matching existing `regression/test_*.py` files.
