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
RATIO_CUTOFF = 70.0

# The batches this cohort is defined over. copd_dataset also holds later pulls that
# are staged for future work and have no spirometry yet; scanning them in would be
# harmless for labelling (no PFT row, so no label) but not for provenance — a
# patient present in two pulls would have their source folder decided by sort order
# rather than by the cohort definition. Set to None to scan every batch present.
COHORT_BATCHES: set[str] | None = {
    "20260702", "20260709", "20260716",   # added 2026-08-21, spirometry now read
    "20260723", "20260730", "20260806", "20260813",
}
# |cos| between slice normal and patient z. A true axial stack sits at 1.0; tilted
# gantry acquisitions stay well above 0.9, while sagittal/coronal reformats are ~0.
AXIAL_MIN_COSINE = 0.85

# Patients dropped from the cohort, with the reason. Excluded before conversion so
# a re-run cannot silently resurrect them.
EXCLUDED: dict[str, str] = {
    # Contrast aortic study, no lung reconstruction anywhere in it. Its only
    # positive-scoring series, "Aorta 3/3", is a SAGITTAL reformat (948x565
    # in-plane, spine running across the frame) that the description-based
    # _NON_AXIAL filter cannot catch because the name never says "sag". Every
    # other series is a 5mm Br40 contrast phase or a coronal reformat.
    "2404337": "no axial lung series; best candidate is a sagittal aortic reformat",
    # Spirometry effort was inadequate, so the measured ratio is not a usable
    # label. Sub-maximal effort truncates FVC more than FEV1 and therefore biases
    # FEV1/FVC upward — an obstructed patient can be recorded as normal. Excluded
    # on the validity of the test, not on how any model scored them: one had been
    # classified correctly by both models, the other by one.
    "2175556": "invalid spirometry — inadequate expiratory effort",
    "2906076": "invalid spirometry — inadequate expiratory effort",
}


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

    A patient can appear in more than one pull (2043242 sits in both 20260716 and
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


def find_hospital66_dirs(root: Path) -> dict[str, list[Path]]:
    """Map patient_id -> candidate DICOM folders, for the <class>/<pid> layout.

    The class folder is the older clinical grouping and is deliberately not read
    as a label; labels come from the measured FEV1/FVC ratio instead.
    """
    found: dict[str, list[Path]] = {}
    for cls in sorted(root.iterdir()):
        if not cls.is_dir():
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
        dicom_dirs.update(
            {pid: [d] for pid, d in find_dicom_dirs(args.dicom_root).items()}
        )
    if args.cohorts in ("both", "hospital66"):
        gold_labels = load_gold_labels(args.gold_json)
        gold_dirs = find_hospital66_dirs(args.hospital66_root)
        # A shared ID would mean the same patient under two ratios; the merge
        # below would silently keep one. Verified disjoint, but check anyway.
        clash = (set(gold_labels) & set(labels)) | (set(gold_dirs) & set(dicom_dirs))
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
