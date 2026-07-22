# Hospital Inference + Data-Capture App — Design Spec

**Date:** 2026-07-22
**Status:** Approved design, pending implementation plan
**Author:** felixchang

## 1. Purpose

A tool to take to the hospital to demo the Normal-vs-Abnormal (COPD-related)
CT classifier with clinicians, and to **grow the training dataset as a side effect
of clinical use**: while a doctor runs inference on a patient's CT, the same CT +
patient number is captured into a staging area for later labeling.

- **Phase now:** Gradio local web app for face-to-face demo (2-month build window).
- **Phase later:** possible migration to a hospital-hosted service / cloud. The
  architecture keeps this cheap without building it now.

### Delivery model — two repos

The code will be **handed to hospital staff**, so it cannot carry the whole nnMamba
research codebase. The system is split across two repos:

- **`~/Research/copd-ct-app/`** (NEW, hospital-facing, self-contained) — the Gradio
  app + inference/capture core. No dependency on nnMamba. This is what the hospital
  runs. Hospital staff only run inference + capture; they never retrain or backfill.
- **nnMamba / research side** (NOT handed over) — production retraining, label
  backfill (touches the dataset), and release packaging. Stays with the researcher.

**Why this resolves the preprocessing-drift risk:** the hospital does not retrain, so
preprocessing is **frozen and shipped alongside each checkpoint release**. Each
release = 5 checkpoints + the exact preprocessing used to train them, bundled
together. Drift cannot occur at the hospital (they never change preprocessing). The
only place drift could occur is between training and packaging on the research side,
and that is caught at **release time** by a bit-for-bit consistency test in
`package_release.py`. Drift is a release-time concern (researcher's), not a runtime
concern (hospital's).

**Data flow back:** captured CTs accumulate in the hospital machine's `staging/`. The
researcher periodically collects `staging/` back to their own machine, runs
`label_backfill` (matches PFT, ingests into the dataset), and after enough
accumulation retrains and packages a new release for the hospital.

## 2. Key Decisions (locked)

| Topic | Decision |
|---|---|
| Repo split | New self-contained hospital repo `~/Research/copd-ct-app/` (app + capture); training + backfill + packaging stay research-side, not handed over |
| Preprocessing consistency | Frozen and shipped with each checkpoint release; drift caught at release time by `package_release.py` consistency test, never at the hospital |
| App form factor | Gradio local web app, runs on an NVIDIA GPU machine (localhost) |
| CT input | DICOM series exported from PACS (a folder / zip) |
| Ground-truth at capture time | Not available. App stores CT + patient ID only; labels added later offline |
| Patient number storage | **Real patient number as filename** (matches existing `{patient_id}_*.nii.gz`). De-identification is a config toggle, default OFF, interface kept in code |
| Deployed model | 5-member soft-vote ensemble, **retrained on ALL data** (no held-out) — production checkpoints, not the CV fold models |
| Retrain cadence | Human-in-the-loop, periodic. Script trains on whatever is in the dataset at run time; the researcher decides when to run it |
| Model version switching | App loads `models/current/` at startup; promoting a new release = repoint `models/current` + restart app |
| Disclaimer | "研究用途、非診斷" line shown in UI, config toggle, **default ON** |
| Cloud readiness | Keep core stateless/config-driven and storage access behind one module. Do NOT build cloud infra now. PHI-in-cloud is a regulatory gate, decided later with IRB/hospital |

### Hard constraint
The model uses `mamba_ssm` (CUDA-only selective-scan kernel). **An NVIDIA GPU is
mandatory** — no CPU / AMD / Intel-GPU deployment is possible. Measured footprint:
1.15M params, ~5.8MB per checkpoint (~29MB for 5), ~0.1GB inference VRAM, ~6ms
GPU forward per volume per member.

### Honest performance note
Production checkpoints are trained on all data, so they have **no clean held-out
metric**. Cite the nested-CV reference (~0.73 Acc / 0.80 AUC), never the
best-epoch 0.833/0.879 numbers, when describing expected performance.

## 3. Architecture

Two repos. The hospital repo is self-contained (a Gradio app + inference/capture
core, plus a frozen preprocessing module and a bundled checkpoint release). The
research side owns retraining, backfill, and release packaging, and does not modify
`regression/` training code.

### Hospital repo — `~/Research/copd-ct-app/` (handed over)
```
copd-ct-app/
├── app.py                 # Gradio UI shell (thin, ~50 lines)
├── core/
│   ├── api.py             # predict_and_capture(dicom_dir, *, capture=True) -> PredictionResult  (ONLY public entry)
│   ├── dicom_io.py        # DICOM series -> NIfTI (SimpleITK)
│   ├── preprocess.py      # FROZEN copy: clip[-1000,400] + resize[112,136,112] + zscore (shipped with the release)
│   ├── ensemble.py        # load 5 checkpoints -> soft-vote -> probability
│   ├── gradcam.py         # Grad-CAM heatmap for one representative member
│   └── staging.py         # ALL staging read/write; storage backend swappable (local now, object store later)
├── models/
│   └── current/           # bundled release: member_1.pth ... member_5.pth, metrics.json, PREPROCESS_HASH
├── staging/
│   ├── incoming/{patient_id}_{YYYYMMDD_HHMMSS}.nii.gz
│   └── capture_log.jsonl
├── config.yaml            # model dir, staging path, image_size, disclaimer + de-id toggles
├── environment.yml / Dockerfile
└── tests/                 # incl. preprocess consistency test against the frozen hash
```

### Research side — nnMamba / private (NOT handed over)
```
scripts/
├── train_production_ensemble.py  # retrain 5 seed-diverse members on ALL data -> release/<date>/
├── label_backfill.py             # collected staging -> match PFT -> label -> move into dataset
└── package_release.py            # bit-for-bit check (frozen preprocess == training preprocess),
                                  #   then bundle 5 checkpoints + frozen preprocess -> a release for copd-ct-app
```

`label_backfill.py` and retraining reuse the real `regression/data/dataset.py`
preprocessing and `regression/data/manifest.py` label rules directly (same repo, no
drift). The hospital repo's `core/preprocess.py` is a frozen copy whose equality to
the training preprocessing is asserted by `package_release.py` at release time.

### Principles
- **core/ fully decoupled from UI.** `app.py` only calls `predict_and_capture`.
  A future FastAPI/CLI/desktop frontend wraps the same entry, core unchanged.
- **`PredictionResult` is pure data** (probability, predicted class, patient_id,
  staging path, optional Grad-CAM image) — no UI concepts.
- **Preprocessing must be bit-for-bit identical to training.** Reuse
  `regression/data/dataset.py` logic; do not fork a version that can drift.
- **Inference and capture are decoupled.** Capture wrapped in try/except; a capture
  failure only logs an error and never blocks the doctor from seeing the prediction.
- **Storage access is centralized in `staging.py`** so the backend (local FS now,
  object storage later) is swappable in one place.

### Data flow
Doctor uploads DICOM → `dicom_io` → NIfTI → `preprocess` → `ensemble` → probability
shown to doctor → same NIfTI handed to `staging` (filename = real patient number) +
one line appended to `capture_log.jsonl`. Later, `label_backfill.py` reads staging +
PFT to assign labels and move data into the dataset.

## 4. Inference Flow (§2 detail)

**① DICOM → NIfTI (`dicom_io.py`)** — SimpleITK reads the folder. Handle:
- HU rescale (`RescaleSlope`/`RescaleIntercept`) — verify applied, else clip range is wrong.
- Slice ordering / spacing / orientation — sort by `ImagePositionPatient`, normalize to training orientation (RAS).
- Multiple series in one folder — group by `SeriesInstanceUID`, select the correct one (or let doctor pick).
- Extract `PatientID` from DICOM header for the filename.

**② Preprocess (`preprocess.py`)** — clip `[-1000, 400]` → resize `[112,136,112]`
(skimage, same params) → z-score. Aligned to `regression/data/dataset.py`. Any
mismatch shifts the distribution and silently degrades predictions.

**③ Ensemble (`ensemble.py`)** — load 5 checkpoints from `models/current/` to
GPU at startup (~0.1GB VRAM, resident). Per case: forward through all 5 → mean of
softmax probabilities (soft-vote) → Abnormal probability.

**④ Presentation** — Abnormal probability (0–100% bar), predicted class, disclaimer
(config toggle, default ON). Grad-CAM heatmap added in §7 below.

## 5. Capture Flow & Staging (§3 detail)

Capture happens in the same request as the prediction. Capture failure must never
affect the doctor seeing the prediction.

```
staging/
├── incoming/{patient_id}_{YYYYMMDD_HHMMSS}.nii.gz   # converted NIfTI, real patient number
└── capture_log.jsonl                                # one line per capture, append-only
```

`capture_log.jsonl` per record: `patient_id`, nii filename, capture timestamp,
model prediction (prob/class), source `SeriesInstanceUID`, `label: null` (pending).

Behavior:
- Capture wrapped in try/except: failure logs an error line, doctor still sees prediction.
- Duplicate patient: distinguish by timestamp, keep both (same patient may have multiple CTs), never overwrite.
- Store the **converted NIfTI at original resolution** (not the resized volume), so
  future retraining at a different `image_size` is not locked out.
- De-identification toggle (default OFF, real number): when enabled, `staging.py`
  swaps filename to a study code, strips DICOM PHI headers, and writes a separate
  mapping table. Interface kept in code now, not activated.
- **All staging read/write goes through `staging.py`**; storage backend swappable.

## 6. Data Growth Loop (§ retraining)

Human-in-the-loop, not automatic, and crosses the two-repo boundary:

```
① Hospital runs copd-ct-app with the current release (models/current/)
② Doctor uses app -> new CT into the hospital machine's staging/incoming/ (label=null)
③ Researcher periodically collects staging/ back to their own machine
④ Research side: label_backfill.py matches PFT -> labels available data -> moves into dataset
⑤ After enough accumulation -> train_production_ensemble.py on the enlarged dataset
   -> release/<date>/  (does NOT touch the hospital)
⑥ package_release.py: bit-for-bit check (frozen preprocess == training preprocess),
   bundle 5 checkpoints + frozen preprocess -> a release
⑦ Ship release to the hospital repo; on the hospital machine, promote models/current
   + restart app
-> back to ① (dataset is larger, model is newer)
```

- `train_production_ensemble.py` (research side) trains on **all data present in the
  dataset folder at run time**, no fold split; 5 members differ only by seed (init +
  augmentation sampling). Reuses model/training settings from
  `config.normal_v_abnormal.imageonly.aug5.ensemble.yaml` (160ep, 5× aug, early-stop).
  Writes to `release/<date>/` with a `metrics.json` (nested-CV reference numbers,
  training set size). **Does not touch the hospital.**
- Not automatic/real-time because: new data needs PFT ground truth first (days–weeks);
  adding 3–5 cases isn't worth a retrain (wait for ~+20–30); and model swaps should
  be reviewed, not unsupervised.
- `label_backfill.py` owns collected-staging → dataset; `train_production_ensemble.py`
  only reads the dataset folder, never staging. Separate responsibilities.

### Version switching (hospital machine)
- App reads only `models/current/`; it doesn't know about dates.
- A new release is bundled by `package_release.py` and shipped as a dated dir; it is
  not live until promoted.
- Promote is an explicit action on the hospital machine: review `metrics.json` diff →
  point `models/current` at the new release → restart app (loads new checkpoints;
  startup-only load, no hot-reload by design).
- Rollback: point `models/current` back to an older release, restart. Old releases kept.

## 7. Label Backfill (§4 detail)

Offline, idempotent, re-runnable script mapping staging captures to PFT ground truth.

**Inputs:** `staging/capture_log.jsonl` + `staging/incoming/*.nii.gz`; PFT source
(`pft.json` / `GOLD_2026_classification.json`, patient_id → PFT metrics).

**Per pending capture:**
1. Look up `patient_id` in PFT source.
2. Not found → PFT not out yet → leave in staging, skip (idempotent, retried next run).
3. Found → derive Normal (FEV1/FVC ≥ 70%) / Abnormal (< 70%) via
   `regression/data/manifest.py` `normal_v_abnormal_label` (reuse rules; also store
   GOLD 1–4 for future multiclass).
4. Move NIfTI to `classification/datasets/normal_v_abnormal_XX/{Normal|Abnormal}/{patient_id}_*.nii.gz`.
5. Write label/angle/GOLD into the labels JSON (existing
   `patient_angle_classification_by_group.json` structure).
6. Mark the `capture_log.jsonl` record as labeled + where it moved.

**Safety:**
- Idempotent + `--dry-run` default (prints what would move + derived labels); `--commit` to act. Re-runs skip already-ingested.
- Conflict check: patient_id already in dataset → report for decision (new CT vs duplicate), no silent overwrite.
- PFT in gray zone / missing metrics → `needs_review` list, no guessing.
- Validate NIfTI opens and shape is sane before ingest; bad files not ingested.
- Output a backfill report (added N, per-class counts, still-waiting-on-PFT, needs-review).

## 8. Explainability / Error Handling / Packaging / Testing (§5–§8)

**Grad-CAM** — reuse `regression/test_gradcam.py` logic on one representative member,
overlay on a CT slice. Config toggle, default ON. One member only (5 is too heavy).

**Error handling / input validation** — guard at the core entry: non-DICOM / empty
folder / unresolved multi-series / abnormal HU / bad shape → clear error to the
frontend, never a raw traceback, and bad data never reaches the model or staging.

**Packaging** — hospital repo ships `environment.yml` (torch cu124 + mamba-ssm +
SimpleITK + gradio) + README; recommended **Dockerfile (CUDA base)** because
mamba-ssm builds are fragile; start with `python app.py`. Research side ships model
updates via `package_release.py`, which asserts the frozen `core/preprocess.py`
matches the training preprocessing (bit-for-bit / hash) before bundling — a failed
check blocks the release.

**Testing** — per-module inline tests (this project has no pytest; use the `nnMamba`
conda env inline runner): DICOM conversion (small fixture series), preprocess
bit-for-bit alignment with training, ensemble soft-vote values, staging idempotency
+ failure decoupling, backfill idempotency / dry-run / conflict. No automated UI
tests (validated by manual demo).

## 9. Out of Scope (YAGNI)

Containerization/K8s/autoscaling, real object-storage SDK wiring, multi-user auth /
API keys, cloud deployment scripts, hot model reload, desktop GUI. Interfaces are
kept clean so these can be added later without a rewrite; implementations are deferred.
