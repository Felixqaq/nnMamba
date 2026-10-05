#!/usr/bin/env python3
"""Build a Normal/Abnormal CT dataset labelled by the GOLD fixed-ratio criterion.

Label rule (matches copd-ct-app/reports/20260811-batch-results.md):
    FEV1/FVC < 70%  -> Abnormal      (strict <; a ratio of exactly 70 is Normal)
    FEV1/FVC >= 70% -> Normal

Two patient sources are merged, both relabelled by that one rule so the combined
cohort has a single label definition:

  * copd_dataset/<batch>/<pid>/DICOM  -- 117 patients, ratio from the
    FEV1FVC_pct column of PFT_JPG/fev1_fvc.csv (post-bronchodilator where
    available, else pre; see that file's Source column).
  * 醫院資料集DICOM_all/<Normal|Abnormal>/<pid>/DICOM -- the original 66, ratio
    from regression/GOLD_2026_classification.json's fev1_fvc_measured_percent.
    Its own folder names encode the older clinical grouping and are ignored;
    relabelling moves exactly one patient (E797258, ratio exactly 70).

The two sets are disjoint (verified: 0 shared patient IDs, 183 unique).

DICOM -> NIfTI conversion is delegated to copd-ct-app's core.dicom_io, which
owns two details that fail silently if reimplemented: the (1,2,0) axis
permutation restoring the training convention, and the series scoring that
picks the thin-slice axial lung-kernel series out of a full PACS export.

Re-running skips patients whose NIfTI already exists, so an interrupted run
resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DICOM_ROOT = Path("/mnt/d/Felix/Hospital/copd_dataset")
DEFAULT_CSV = DEFAULT_DICOM_ROOT / "PFT_JPG" / "fev1_fvc.csv"
DEFAULT_HOSPITAL66 = Path("/mnt/d/Felix/Hospital/醫院資料集DICOM_all")
DEFAULT_GOLD_JSON = REPO / "regression" / "GOLD_2026_classification.json"
DEFAULT_APP = Path("/mnt/d/Felix/Hospital/copd-ct-app")
DEFAULT_OUT = (
    REPO / "classification" / "datasets" / "normal_v_abnormal_fev1fvc70"
)
BATCH_DIR = re.compile(r"^\d{8}$")

# Deliveries that sit outside the weekly <batch>/<pid> layout, as globs relative to
# the DICOM root whose matches are folders of patient folders. Added 2026-09-15
# after 21 patients turned out to have spirometry read but no CT in any batch:
# their imaging was in these two places all along. `LDCT/*/non_PFT` is deliberately
# not listed -- those 26 patients have no spirometry row, so no label.
EXTRA_DELIVERIES: tuple[str, ...] = (
    "LDCT/*/PFT",                  # low-dose pull, its PFT-matched subset
    "醫生提供(乾淨資料集)",          # curated by the referring physician
)
RATIO_CUTOFF = 70.0

# The batches this cohort is defined over. copd_dataset also holds later pulls that
# are staged for future work and have no spirometry yet; scanning them in would be
# harmless for labelling (no PFT row, so no label) but not for provenance — a
# patient present in two pulls would have their source folder decided by sort order
# rather than by the cohort definition. Set to None to scan every batch present.
COHORT_BATCHES: set[str] | None = {
    # Added 2026-09-29 (evening). Late May and early June 2025; the PFT pass of
    # the same evening added 44 patients and removed or changed no row.
    "20250522", "20250529", "20250605",
    # Added 2026-09-29. The June to early-July 2025 pulls. The PFT pass of
    # 2026-09-29 removed no row; it re-pointed one patient from a spirometry
    # nine months before the CT to one six weeks after it, which the rebuild
    # picks up.
    "20250612", "20250619", "20250626", "20250703",
    # Added 2026-09-24. The four July 2025 pulls; the PFT pass of 2026-09-24 added
    # 58 patients to the export without removing or changing any existing row.
    "20250710", "20250717", "20250724", "20250731",
    # Added 2026-09-22. The four August 2025 pulls; the PFT pass of 2026-09-21
    # added 73 patients to the export without removing or changing any existing
    # row, and 66 of the new patients sit in these four batches.
    "20250807", "20250814", "20250821", "20250828",
    # Added 2026-09-18. The twelve weekly pulls that predate 20251127, curated in
    # the latest PFT pass. 183 of their patients carry a FEV1FVC_pct row; six
    # further stragglers in batches already listed below became labelled in the
    # same pass, so this pull adds 189 patients in total.
    "20250904", "20250911", "20250918", "20250925", "20251002",
    "20251009", "20251016", "20251023",
    "20251030", "20251106", "20251113", "20251120",
    # Added 2026-09-15. The six weekly pulls that predate 20260108, which the
    # 2026-08-31 pass started from and so never reached. All 103 carry both a
    # FEV1FVC_pct row and a DICOM folder; 2 of them are already in the cohort
    # through another pull, leaving 101 new patients (30 abnormal, 73 normal).
    # Spirometry rows dated outside any of these batches exist -- 26 of them --
    # but 6 are hospital_66 patients already present and the other 20 have no
    # imaging at all, so none of those can join.
    "20251127", "20251204", "20251211", "20251218", "20251225", "20260101",
    # Added 2026-08-31. The three earliest pulls; their PFT pages were curated in
    # the latest pass, which also completed 20260129 and filled two stragglers in
    # 20260212 and 20260319 -- 78 patients in total still missing a NIfTI.
    "20260108", "20260115", "20260122",
    # Added 2026-08-29 after the corresponding PFT_JPG reports were curated.
    # These eleven historical pulls contribute the newly available training
    # patients while the already frozen 200-patient holdout remains unchanged.
    "20260129", "20260205", "20260212", "20260219", "20260226",
    "20260305", "20260312", "20260319", "20260326",
    "20260402", "20260409",
    # Added 2026-08-27. PFT_JPG/fev1_fvc.csv is the curated inclusion source;
    # these five newly received batches contribute 99 labelled patients.
    "20260416", "20260423", "20260430", "20260507", "20260514",
    "20260702", "20260709", "20260716",   # added 2026-08-21, spirometry now read
    "20260723", "20260730", "20260806", "20260813",
    # Added 2026-08-26. These six batches are dated earlier than the seven above
    # but arrived later; their PFT pages were extracted and read in the same pass,
    # so every patient in them carries a FEV1FVC_pct row. 146 patients.
    "20260521", "20260528", "20260604",
    "20260611", "20260618", "20260625",
}
# |cos| between slice normal and patient z. A true axial stack sits at 1.0; tilted
# gantry acquisitions stay well above 0.9, while sagittal/coronal reformats are ~0.
AXIAL_MIN_COSINE = 0.85

# Cohort decisions live in regression/cohort_decisions.local.json, not here.
# Each one names patients by hospital ID next to a clinical reason -- invalid
# spirometry, a non-thoracic scan, two spirometry sessions nineteen points apart --
# and this repository is public, so an ID beside a reason is identifiable to
# anyone with access to the hospital system. The file is gitignored and the build
# refuses to run without it: an absent file must stop the build, never quietly
# produce a cohort with no exclusions and no cross-cohort reconciliation.
#
# See cohort_decisions.example.json for the shape.
_DECISIONS_PATH = REPO / "regression/cohort_decisions.local.json"


def _load_decisions() -> dict:
    if not _DECISIONS_PATH.is_file():
        raise SystemExit(
            f"{_DECISIONS_PATH} is missing. It holds the patient-level exclusions and "
            "cross-cohort reconciliations and is deliberately not in version control; "
            "copy it from the machine that has it, or rebuild it from "
            "cohort_decisions.example.json. Refusing to build a cohort without it."
        )
    payload = json.loads(_DECISIONS_PATH.read_text(encoding="utf-8"))
    for key in ("cross_cohort_keep", "cross_cohort_ratio_override", "excluded"):
        if key not in payload:
            raise SystemExit(f"{_DECISIONS_PATH}: no {key!r} section")
    return payload


_DECISIONS = _load_decisions()
CROSS_COHORT_KEEP: dict[str, str] = _DECISIONS["cross_cohort_keep"]
CROSS_COHORT_RATIO_OVERRIDE: dict[str, dict] = _DECISIONS["cross_cohort_ratio_override"]
EXCLUDED: dict[str, str] = _DECISIONS["excluded"]


def label_for(ratio: float) -> str:
    """GOLD fixed-ratio criterion. Strict <, so exactly 70 is Normal."""
    return "Abnormal" if ratio < RATIO_CUTOFF else "Normal"


def load_labels(csv_path: Path) -> dict[str, dict]:
    """Map patient_id -> {ratio, label, source, batch} from fev1_fvc.csv."""
    with io.open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    labels: dict[str, dict] = {}
    for row in rows:
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        pid = row["PatientID"]
        raw = row["FEV1FVC_pct"]
        if not raw:
            continue
        ratio = float(raw)
        label = label_for(ratio)

        # The CSV ships a precomputed fixed-70 flag; disagreeing with it means
        # one of the two is wrong, and guessing which is not this script's job.
        flag = row.get("Obstruction_fixed70", "")
        if flag in ("Y", "N") and (flag == "Y") != (label == "Abnormal"):
            raise SystemExit(
                f"{pid}: FEV1FVC_pct={ratio} implies {label} but "
                f"Obstruction_fixed70={flag}. Refusing to guess."
            )

        labels[pid] = {
            "ratio": ratio,
            "label": label,
            "source": row.get("Source", ""),
            "batch": row.get("Date", ""),
            "severity_gold": row.get("Severity_GOLD", ""),
            "cohort": "copd_dataset_117",
        }
    return labels


def load_gold_labels(gold_json: Path) -> dict[str, dict]:
    """Map patient_id -> label for the original 66, from the GOLD 2026 records."""
    records = json.loads(gold_json.read_text(encoding="utf-8-sig"))["records"]
    labels: dict[str, dict] = {}
    for rec in records:
        ratio = rec.get("fev1_fvc_measured_percent")
        if ratio is None:
            continue
        ratio = float(ratio)
        labels[str(rec["patient_id"])] = {
            "ratio": ratio,
            "label": label_for(ratio),
            "source": "measured",
            "batch": "hospital66",
            "severity_gold": rec.get("severity", ""),
            "cohort": "hospital_66",
        }
    return labels


def stage_labels(dicom_root: Path, batches: set[str]) -> dict[str, dict]:
    """Placeholder entries for batches that have imaging but no spirometry yet.

    Converting these now means that when the PFT arrives the cohort can be rebuilt
    without touching the DICOM again, and any series that will need a human
    decision — a study with no axial lung reconstruction, a second acquisition on
    a different date — surfaces now rather than on the day the labels land.
    """
    labels: dict[str, dict] = {}
    for batch in sorted(batches):
        folder = dicom_root / batch
        if not folder.is_dir():
            raise SystemExit(f"batch {batch} not found under {dicom_root}")
        for patient in sorted(folder.iterdir()):
            if patient.is_dir():
                labels[patient.name] = {
                    "ratio": float("nan"),
                    "label": "Unlabelled",
                    "source": "",
                    "batch": batch,
                    "severity_gold": "",
                    "cohort": f"staged_{batch}",
                }
    return labels


def find_staged_dirs(dicom_root: Path, batches: set[str]) -> dict[str, list[Path]]:
    """DICOM folders for staged batches, newest batch winning on a repeat."""
    found: dict[str, list[Path]] = {}
    for batch in sorted(batches):
        folder = dicom_root / batch
        for patient in sorted(folder.iterdir()):
            if not patient.is_dir():
                continue
            inner = patient / "DICOM"
            found[patient.name] = [inner if inner.is_dir() else patient]
    return found


def find_dicom_dirs(dicom_root: Path) -> dict[str, Path]:
    """Map patient_id -> DICOM folder, for copd_dataset's <batch>/<pid> layout.

    A patient can appear in more than one pull (one sits in both 20260716 and
    20260723 as byte-identical copies). Batches are visited oldest first and each
    assignment overwrites, so the most recent pull wins — the later export is the
    more complete one, and tying the choice to the date makes it reproducible
    instead of an accident of directory ordering.
    """
    found: dict[str, Path] = {}
    for batch in sorted(dicom_root.iterdir()):  # ascending date; last write wins
        if not batch.is_dir() or not BATCH_DIR.match(batch.name):
            continue
        if COHORT_BATCHES is not None and batch.name not in COHORT_BATCHES:
            continue
        for patient in sorted(batch.iterdir()):
            if not patient.is_dir():
                continue
            inner = patient / "DICOM"
            found[patient.name] = inner if inner.is_dir() else patient
    return found


def _dicom_subdirs(patient_dir: Path) -> list[Path]:
    """Candidate DICOM folders under one patient, across the layouts seen here.

    Three variants exist in the hospital export, and one folder is booby-trapped:
      <pid>/DICOM              (42 patients)
      <pid>/DICOM (1)          (21 patients, re-downloaded copies)
      <pid>/<study date>/DICOM (2 patients with two studies each)
    Plus <pid>/PFT, which holds the spirometry report and must never be scanned
    as if it were the CT. Patient 2291134 also contains 14 stray folders named
    after *other* patients holding 1-17 file fragments; only its own DICOM/ is
    real, so directories whose name is a bare patient id are not recursed into.
    """
    candidates: list[Path] = []
    for sub in sorted(patient_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.upper().startswith("PFT"):
            continue
        if name.upper().startswith("DICOM"):
            candidates.append(sub)
        elif re.fullmatch(r"\d{6}", name):  # a study-date folder
            inner = sub / "DICOM"
            if inner.is_dir():
                candidates.append(inner)
    return candidates or [patient_dir]


def find_extra_delivery_dirs(
    dicom_root: Path,
    already: set[str],
) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    """Map patient_id -> DICOM folder for deliveries outside the weekly layout.

    Two folders arrived as one-off deliveries rather than dated pulls, so
    `find_dicom_dirs` never sees them: the LDCT pull keeps its PFT-matched
    patients under `LDCT/<date>/PFT/`, and the referring physician's curated set
    has no date level at all. Between them they carry the imaging for patients
    whose spirometry was already read but whose CT appeared to be missing.

    A patient who also has a weekly batch folder is returned as a conflict rather
    than overridden. The two folders can hold different studies -- one patient has
    spirometry from two visits nineteen ratio points apart -- and silently
    preferring one is how a label stops describing the image it is attached to.
    """
    found: dict[str, Path] = {}
    conflicts: list[tuple[str, str]] = []
    for pattern in EXTRA_DELIVERIES:
        for parent in sorted(dicom_root.glob(pattern)):
            if not parent.is_dir():
                continue
            for patient in sorted(parent.iterdir()):
                if not patient.is_dir():
                    continue
                if patient.name in already:
                    conflicts.append((patient.name, pattern))
                    continue
                inner = patient / "DICOM"
                found.setdefault(patient.name, inner if inner.is_dir() else patient)
    return found, conflicts


def find_hospital66_dirs(root: Path) -> dict[str, list[Path]]:
    """Map patient_id -> candidate DICOM folders, for the <class>/<pid> layout.

    The class folder is the older clinical grouping and is deliberately not read
    as a label; labels come from the measured FEV1/FVC ratio instead.
    """
    # Only the two clinical grouping folders are patient containers. The root also
    # accumulates backups and tooling (copd_backup_A_raw_ct, nnmamba_backup_*,
    # _deid_tool, DicomToNii_essentials_*), and walking those enumerated 28
    # repo directories as if they were patients. They were harmless only because
    # none happened to be named like a real ID -- and setdefault keeps the first
    # match in sort order, so "DicomToNii_essentials_20260820" would have won
    # against "Normal" for any that did.
    CLASS_DIRS = {"Normal", "Abnormal"}

    found: dict[str, list[Path]] = {}
    for cls in sorted(root.iterdir()):
        if not cls.is_dir() or cls.name not in CLASS_DIRS:
            continue
        for patient in sorted(cls.iterdir()):
            if not patient.is_dir():
                continue
            found.setdefault(patient.name, _dicom_subdirs(patient))
    return found


def normalize_desc(text: str) -> str:
    """Comparison form for a series description, robust to path-safe rewriting."""
    return re.sub(r"\s+", " ", safe_name(text)).strip().lower()


def load_series_hints(manifest_path: Path) -> dict[str, str]:
    """patient_id -> the series description the original 66-case build selected.

    Two patients have two studies on disk and several have re-downloaded copies,
    so scoring alone could pick a different series than the published cohort
    used. The old manifest encodes the choice in each filename ("<pid>_<desc>"),
    which lets the rebuild reproduce it exactly.
    """
    if not manifest_path.exists():
        return {}
    records = json.loads(manifest_path.read_text(encoding="utf-8-sig"))["records"]
    hints: dict[str, str] = {}
    for rec in records:
        name = Path(rec["path"]).name
        if name.endswith(".nii.gz"):
            name = name[:-7]
        pid, _, desc = name.partition("_")
        if desc:
            hints[pid] = desc
    return hints


def safe_name(text: str) -> str:
    """Make a series description usable as a filename component."""
    cleaned = re.sub(r"[^\w\s.-]", "_", text).strip()
    return re.sub(r"\s+", " ", cleaned) or "series"


def _convert_dropping_odd_slices(dicom_dir: str, series_uid: str, out_path: Path):
    """Rebuild one series after removing frames whose matrix size is the minority.

    Mirrors copd-ct-app's dicom_series_to_nifti, reusing its _TRAINING_AXES so the
    axis convention stays single-sourced; only the file list differs.
    """
    import collections

    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from core.dicom_io import DicomError, DicomResult, _TRAINING_AXES, _tag

    reader = sitk.ImageSeriesReader()
    files = list(reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid))

    sizes: dict[str, tuple[int, int]] = {}
    for path in files:
        meta = sitk.ImageFileReader()
        meta.SetFileName(path)
        meta.ReadImageInformation()
        sizes[path] = tuple(meta.GetSize()[:2])
    keep_size, _ = collections.Counter(sizes.values()).most_common(1)[0]
    kept = [p for p in files if sizes[p] == keep_size]
    dropped = len(files) - len(kept)
    if not kept or dropped == 0:
        raise DicomError(f"No consistent-size frames in series {series_uid}")

    reader.SetFileNames(kept)
    image = reader.Execute()

    meta = sitk.ImageFileReader()
    meta.SetFileName(kept[0])
    meta.LoadPrivateTagsOn()
    meta.ReadImageInformation()

    array = np.transpose(sitk.GetArrayFromImage(image), _TRAINING_AXES)
    sx, sy, sz = image.GetSpacing()
    affine = np.diag([sy, sx, sz, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(out_path))

    return (
        DicomResult(
            nifti_path=out_path,
            patient_id=_tag(meta, "0010|0020") or "UNKNOWN",
            series_uid=series_uid,
            num_slices=len(kept),
            series_description=_tag(meta, "0008|103e"),
        ),
        dropped,
    )


def _axial_cosine(dicom_dir: str, series_uid: str) -> float | None:
    """|cos| between the series' slice normal and the patient z-axis, or None.

    The app's non-axial filter reads the series *description*, so a reformat whose
    name never says "sag"/"cor" sails through: patient 2404337's "Aorta 3/3" is a
    sagittal aortic reformat that outscored every other series in its study. The
    geometry cannot lie the way the description can — 1.0 is a true axial stack,
    ~0.0 is a sagittal or coronal one.
    """
    import numpy as np
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_uid)
    if not files:
        return None
    meta = sitk.ImageFileReader()
    meta.SetFileName(files[0])
    try:
        meta.ReadImageInformation()
    except RuntimeError:
        return None
    if not meta.HasMetaDataKey("0020|0037"):
        return None
    try:
        vals = [float(v) for v in meta.GetMetaData("0020|0037").split("\\")]
    except ValueError:
        return None
    if len(vals) != 6:
        return None
    normal = np.cross(vals[:3], vals[3:])
    return float(abs(normal[2]))


def convert_one(args: tuple[str, list[str], str, str, str | None, bool]) -> dict:
    """Convert one patient. Runs in a worker process."""
    pid, dicom_dirs, out_dir, app_root, hint, use_hint = args
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    from core.dicom_io import DicomError, dicom_series_to_nifti, list_dicom_series

    try:
        # Gather every readable series across this patient's folders, then decide
        # once — a per-folder decision could not tell a re-downloaded copy or a
        # second study apart from the study the cohort was built from.
        found: list[tuple[str, object]] = []
        errors: list[str] = []
        for d in dicom_dirs:
            try:
                for s in list_dicom_series(d):
                    found.append((d, s))
            except (DicomError, RuntimeError) as exc:
                errors.append(f"{d}: {exc}")
        if not found:
            raise DicomError("; ".join(errors) or f"no series under {dicom_dirs}")

        # Drop reformats the description-based filter missed. Only series that
        # actually score are checked, to avoid paying the header read on scouts.
        non_axial: list[str] = []
        kept: list[tuple[str, object]] = []
        for d, s in found:
            if s.score < 0:
                kept.append((d, s))
                continue
            cos = _axial_cosine(d, s.series_uid)
            if cos is not None and cos < AXIAL_MIN_COSINE:
                non_axial.append(f"{s.description} (|cos|={cos:.2f})")
                continue
            kept.append((d, s))
        found = kept

        matched_hint = False
        if hint and use_hint:
            wanted = normalize_desc(hint)
            hits = [(d, s) for d, s in found if normalize_desc(s.description) == wanted]
            if hits:
                # Same description can appear in both studies; take the fullest.
                chosen_dir, chosen = max(hits, key=lambda ds: ds[1].num_slices)
                matched_hint = True
            else:
                chosen_dir, chosen = max(found, key=lambda ds: ds[1].score)
        else:
            chosen_dir, chosen = max(found, key=lambda ds: ds[1].score)

        if chosen.score < 0:
            detail = "; ".join(s.label() for _, s in found[:6])
            if non_axial:
                detail += " | rejected as non-axial: " + "; ".join(non_axial)
            raise DicomError(
                "No suitable axial CT series (only scouts/reformats/reports?): " + detail
            )

        out = Path(out_dir) / f"{pid}_{safe_name(chosen.description)}.nii.gz"
        dropped = 0
        try:
            result = dicom_series_to_nifti(chosen_dir, out, series_uid=chosen.series_uid)
        except DicomError as exc:
            # Some exports slip a scanner-generated extra frame (dose report,
            # summary image) into a real series under the same SeriesInstanceUID.
            # It has a different matrix size and a z-position far off the stack,
            # which both breaks the reader and injects a phantom gap. Drop the
            # odd-sized minority and rebuild; anything else re-raises.
            if "does not fully contain the requested region" not in str(exc):
                raise
            result, dropped = _convert_dropping_odd_slices(
                chosen_dir, chosen.series_uid, out
            )

        return {
            "patient_id": pid,
            "ok": True,
            "path": str(result.nifti_path),
            "series_description": result.series_description,
            "num_slices": result.num_slices,
            "series_uid": result.series_uid,
            "dicom_dir": str(chosen_dir),
            "dicom_patient_id": result.patient_id,
            "series_hint": hint,
            "hint_matched": matched_hint,
            "hint_overridden": bool(hint and not use_hint),
            "odd_slices_dropped": dropped,
            "rejected_non_axial": non_axial,
            "axial_cosine": _axial_cosine(chosen_dir, chosen.series_uid),
            "n_dirs_scanned": len(dicom_dirs),
            "candidates": [s.label() for _, s in sorted(
                found, key=lambda ds: ds[1].score, reverse=True)[:4]],
        }
    except (DicomError, RuntimeError, ValueError, OSError) as exc:
        return {
            "patient_id": pid,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--hospital66-root", type=Path, default=DEFAULT_HOSPITAL66)
    ap.add_argument("--gold-json", type=Path, default=DEFAULT_GOLD_JSON)
    ap.add_argument(
        "--series-hints",
        type=Path,
        default=REPO / "regression/datasets/generated/rq1_nva66_manifest.image.json",
        help="old manifest whose filenames record the originally selected series",
    )
    ap.add_argument(
        "--cohorts",
        default="both",
        choices=("both", "copd117", "hospital66"),
        help="which patient sources to include (default: both)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--app", type=Path, default=DEFAULT_APP)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="convert at most N (0=all)")
    ap.add_argument(
        "--only", default="", help="comma-separated patient ids to convert (debugging)"
    )
    ap.add_argument(
        "--ignore-hints",
        action="store_true",
        help="pick purely by score, ignoring what the original 66-case build chose. "
        "Used for the six patients whose original series was not a thin-slice lung "
        "reconstruction (contrast/5mm-soft-kernel/Br40); the hint is still recorded "
        "in the summary so the deviation stays visible.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    hints = load_series_hints(args.series_hints)

    labels: dict[str, dict] = {}
    dicom_dirs: dict[str, list[Path]] = {}
    if args.cohorts in ("both", "copd117"):
        labels.update(load_labels(args.csv))
        batch_dirs = find_dicom_dirs(args.dicom_root)
        dicom_dirs.update({pid: [d] for pid, d in batch_dirs.items()})
        extra, extra_clashes = find_extra_delivery_dirs(
            args.dicom_root, set(batch_dirs)
        )
        for pid, folder in sorted(extra.items()):
            dicom_dirs[pid] = [folder]
            print(f"EXTRA DELIVERY   : {pid} — {folder.parent.parent.name}/"
                  f"{folder.parent.name}")
        for pid, pattern in sorted(extra_clashes):
            print(f"EXTRA SKIPPED    : {pid} — also in a weekly batch, keeping the "
                  f"batch copy rather than {pattern}")
    if args.cohorts in ("both", "hospital66"):
        gold_labels = load_gold_labels(args.gold_json)
        gold_dirs = find_hospital66_dirs(args.hospital66_root)
        # A shared ID would mean the same patient under two ratios; the merge
        # below would silently keep one. Any clash is fatal unless it is named in
        # CROSS_COHORT_KEEP *and* both sources agree on the ratio -- a later
        # copd_dataset pull re-collecting one of the original 66 is benign, two
        # different ratios for one ID never is.
        clash = (set(gold_labels) & set(labels)) | (set(gold_dirs) & set(dicom_dirs))
        for pid in sorted(clash & set(CROSS_COHORT_KEEP)):
            a = labels.get(pid, {}).get("ratio")
            b = gold_labels.get(pid, {}).get("ratio")
            if a is not None and b is not None and float(a) != float(b):
                override = CROSS_COHORT_RATIO_OVERRIDE.get(pid)
                if override is None:
                    raise SystemExit(
                        f"{pid}: copd_dataset ratio {a} but hospital66 ratio {b}. "
                        "Refusing to reconcile a genuine disagreement."
                    )
                # An override still has to describe the disagreement it permits,
                # or it would keep applying after the underlying data changed.
                if (float(override["copd_dataset_ratio"]) != float(a)
                        or float(override["hospital66_ratio"]) != float(b)):
                    raise SystemExit(
                        f"{pid}: recorded override is for {override['copd_dataset_ratio']}"
                        f" vs {override['hospital66_ratio']}, but the data now reads "
                        f"{a} vs {b}. Re-decide before this runs."
                    )
                print(f"OVERRIDE         : {pid} — {a} vs {b}, keeping "
                      f"{CROSS_COHORT_KEEP[pid]} ({override['reason']}; "
                      f"decided by {override['decided_by']})")
            keep = CROSS_COHORT_KEEP[pid]
            drop = "copd_dataset" if keep == "hospital66" else "hospital66"
            if drop == "copd_dataset":
                labels.pop(pid, None)
                dicom_dirs.pop(pid, None)
            else:
                gold_labels.pop(pid, None)
                gold_dirs.pop(pid, None)
            # Say "in both" only when it is true; the override case already
            # printed the two differing ratios above.
            if a is not None and b is not None and float(a) != float(b):
                print(f"RECONCILED       : {pid} — kept {keep} ratio {b}, "
                      f"dropped {drop} ratio {a}")
            else:
                print(f"RECONCILED       : {pid} — ratio {a} in both; keeping {keep}")
        clash -= set(CROSS_COHORT_KEEP)
        if clash:
            raise SystemExit(f"patient IDs present in both cohorts: {sorted(clash)}")
        labels.update(gold_labels)
        dicom_dirs.update(gold_dirs)

    excluded_here = sorted(set(EXCLUDED) & (set(labels) | set(dicom_dirs)))
    for pid in excluded_here:
        labels.pop(pid, None)
        dicom_dirs.pop(pid, None)

    have_both = sorted(set(labels) & set(dicom_dirs))
    pft_only = sorted(set(labels) - set(dicom_dirs))
    ct_only = sorted(set(dicom_dirs) - set(labels))

    counts = {"Normal": 0, "Abnormal": 0}
    per_cohort: dict[str, dict[str, int]] = {}
    for pid in have_both:
        meta = labels[pid]
        counts[meta["label"]] += 1
        per_cohort.setdefault(meta["cohort"], {"Normal": 0, "Abnormal": 0})
        per_cohort[meta["cohort"]][meta["label"]] += 1

    print(f"cohorts           : {args.cohorts}")
    print(f"PFT rows          : {len(labels)}")
    print(f"DICOM folders     : {len(dicom_dirs)}")
    print(f"usable (both)     : {len(have_both)}  -> {counts}")
    for name, c in sorted(per_cohort.items()):
        print(f"    {name:20s}: {c}")
    print(f"PFT without CT    : {len(pft_only)} {pft_only}")
    print(f"CT without PFT    : {len(ct_only)} {ct_only}")
    for pid in excluded_here:
        print(f"EXCLUDED          : {pid} — {EXCLUDED[pid]}")

    if args.dry_run:
        return

    for cls in ("Normal", "Abnormal"):
        (args.out / cls).mkdir(parents=True, exist_ok=True)

    only = {p.strip() for p in args.only.split(",") if p.strip()}
    todo = []
    skipped = 0
    for pid in have_both:
        if only and pid not in only:
            continue
        cls_dir = args.out / labels[pid]["label"]
        if any(cls_dir.glob(f"{pid}_*.nii.gz")):
            skipped += 1
            continue
        todo.append(
            (
                pid,
                [str(d) for d in dicom_dirs[pid]],
                str(cls_dir),
                str(args.app),
                hints.get(pid),
                not args.ignore_hints,
            )
        )
    if args.limit:
        todo = todo[: args.limit]

    print(f"already converted : {skipped}")
    print(f"to convert        : {len(todo)}\n", flush=True)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, t): t[0] for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            pid = res["patient_id"]
            meta = labels[pid]
            res.update(
                fev1_fvc_pct=meta["ratio"],
                label=meta["label"],
                pft_source=meta["source"],
                batch=meta["batch"],
                cohort=meta["cohort"],
            )
            results.append(res)
            if res["ok"]:
                tag = " [hint]" if res.get("hint_matched") else ""
                print(
                    f"[{i}/{len(todo)}] {pid} {meta['label']:8s} "
                    f"ratio={meta['ratio']:5.1f} slices={res['num_slices']:4d} "
                    f"| {res['series_description']}{tag}",
                    flush=True,
                )
            else:
                print(f"[{i}/{len(todo)}] {pid} FAILED  {res['error']}", flush=True)

    # Runs are resumable and can target a subset (--only), so fold this run's
    # records into whatever a previous run wrote instead of replacing them.
    out_json = args.out / "build_summary.json"
    merged: dict[str, dict] = {}
    if out_json.exists():
        try:
            previous = json.loads(out_json.read_text(encoding="utf-8"))
            merged = {r["patient_id"]: r for r in previous.get("records", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            merged = {}
    for rec in results:
        merged[rec["patient_id"]] = rec

    # Merging keeps provenance for patients this run did not touch, but it must not
    # keep patients who have since left the cohort: a stale record made the
    # demographics step emit 182 rows against a 180-patient cohort. Drop anything
    # that is excluded or no longer has a file on disk.
    on_disk = {p.name[:-7].split("_", 1)[0] for p in args.out.glob("*/*.nii.gz")}
    dropped = sorted(pid for pid in merged
                     if pid in EXCLUDED or pid not in on_disk)
    for pid in dropped:
        merged.pop(pid)
    if dropped:
        print(f"dropped stale records: {dropped}")
    results = list(merged.values())

    failed = [r for r in results if not r["ok"]]
    summary = {
        "cutoff": f"FEV1/FVC < {RATIO_CUTOFF}% = Abnormal (strict)",
        "csv": str(args.csv),
        "dicom_root": str(args.dicom_root),
        "out": str(args.out),
        "cohorts_included": args.cohorts,
        "excluded_patients": EXCLUDED,
        "cohort_counts": counts,
        "counts_per_cohort": per_cohort,
        "pft_without_ct": pft_only,
        "ct_without_pft": ct_only,
        "converted": len(results) - len(failed),
        "skipped_existing": skipped,
        "failed": failed,
        "records": sorted(results, key=lambda r: r["patient_id"]),
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")

    on_disk = {
        cls: len(list((args.out / cls).glob("*.nii.gz"))) for cls in ("Normal", "Abnormal")
    }
    print(f"\nconverted={len(results) - len(failed)} failed={len(failed)}")
    print(f"on disk: {on_disk}")
    print(f"summary: {out_json}")


if __name__ == "__main__":
    main()
