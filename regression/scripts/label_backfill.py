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
