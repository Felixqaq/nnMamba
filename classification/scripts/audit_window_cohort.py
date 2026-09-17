"""Audit on-disk CT files against the authoritative conversion summary."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path("/home/felix/Research/nnMamba/classification/datasets/normal_v_abnormal_fev1fvc70")
    summary = json.loads((root / "build_summary.json").read_text())
    records = {str(r["patient_id"]): r for r in summary["records"] if r["ok"]}
    files = list(root.glob("*/*.nii.gz"))
    ids = [p.name[:-7].split("_", 1)[0] for p in files]
    extras = sorted(set(ids) - set(records))
    print(json.dumps({"files": len(files), "unique_file_ids": len(set(ids)),
        "valid_records": len(records), "extra_file_ids": extras,
        "missing_ids": sorted(set(records) - set(ids)),
        "duplicate_ids": {k: v for k, v in Counter(ids).items() if v > 1},
        "excluded_extra_reasons": {k: summary["excluded_patients"].get(k) for k in extras},
        "record_fields": list(next(iter(records.values())).keys())}, indent=2))
    project = Path("/mnt/d/Felix/Hospital/nnMamba")
    split_path = project / "regression/outputs/doctor_validation_official200_20260831/split.json"
    split = json.loads(split_path.read_text())
    print("split_fields", {k: len(v) if isinstance(v, (list, dict)) else v
                           for k, v in split.items() if k.endswith("ids") or isinstance(v, (list, dict))})
    # Report only coverage counts, not the patient list.
    train = set(split.get("training_patient_ids", []))
    test = set(split.get("validation_patient_ids", []))
    print("split_coverage", {"train": len(train), "test": len(test),
          "valid_not_in_split": len(set(records) - train - test),
          "split_not_valid": len((train | test) - set(records))})


if __name__ == "__main__":
    main()
