"""Build a 66-patient Normal/Abnormal dir (clinical grouping) from by_angle_all."""
from __future__ import annotations
import json
from pathlib import Path

REG = Path(__file__).resolve().parents[1]
LABELS = REG / ".." / "patient_angle_classification_by_group.json"
SRC = REG / ".." / "by_angle_all"
OUT = REG / ".." / "classification" / "datasets" / "normal_v_abnormal_66"


def patient_ids_for(group_block: dict) -> set[str]:
    ids: set[str] = set()
    for angle_side in group_block.get("by_angle", {}).values():
        ids.update(str(pid) for pid in angle_side.keys())
    return ids


def ct_files_by_pid() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in SRC.rglob("*.nii.gz"):
        pid = path.name.split("_", 1)[0].split(" ", 1)[0]
        mapping.setdefault(pid, path)
    return mapping


def main() -> None:
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    abnormal_ids = patient_ids_for(data["abnormal_group_33"])
    normal_ids = patient_ids_for(data["normal_group_21"])
    files = ct_files_by_pid()

    for cls, ids in (("Abnormal", abnormal_ids), ("Normal", normal_ids)):
        dst_dir = OUT / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for pid in sorted(ids):
            src = files.get(pid)
            if src is None:
                raise FileNotFoundError(f"No CT for patient {pid} in {SRC}")
            link = dst_dir / src.name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(src.resolve())
    print(f"Abnormal={len(abnormal_ids)} Normal={len(normal_ids)} -> {OUT}")


if __name__ == "__main__":
    main()
