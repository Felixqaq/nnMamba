#!/usr/bin/env bash
set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba

REPO="$HOME/Research/nnMamba"
REG="$REPO/regression"
WINDOWS_OUTPUT="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_20260827"
RUN_OUTPUT="$REG/outputs/doctor_validation_20260827"
PFT_COHORT="$REPO/classification/datasets/doctor_validation_fev1fvc70_pft430"
SPLIT="$WINDOWS_OUTPUT/split.json"
PRIOR_MANIFEST="$WINDOWS_OUTPUT/prior_383_manifest.json"
TAPCT_DIR="$REG/embeddings/tapct_doctor_validation_430"
MAMBA_DIR="$RUN_OUTPUT/mamba5"
QCT_DIR="$RUN_OUTPUT/qct"

mkdir -p "$WINDOWS_OUTPUT" "$RUN_OUTPUT"
cd "$REPO"

echo "[1/8] Convert all curated PFT_JPG patients: $(date)"
python -u regression/scripts/build_fev1fvc70_dataset.py \
    --cohorts copd117 \
    --workers 3

echo "[2/8] Freeze 200/230 split and create PFT-only CT view: $(date)"
python -u regression/scripts/prepare_doctor_validation.py \
    --prior-manifest "$PRIOR_MANIFEST" \
    --output-dir "$WINDOWS_OUTPUT" \
    --cohort-root "$PFT_COHORT"

echo "[3/8] Incremental TAP-CT extraction: $(date)"
mkdir -p "$TAPCT_DIR/cases"
if [[ -d "$REG/embeddings/tapct_fev1fvc70_384/cases" ]]; then
    cp -al "$REG/embeddings/tapct_fev1fvc70_384/cases/." "$TAPCT_DIR/cases/" 2>/dev/null || true
fi
python -u regression/scripts/extract_tapct_embeddings.py \
    --model-id fomofo/tap-ct-s-3d \
    --source-root "$PFT_COHORT" \
    --output-dir "$TAPCT_DIR" \
    --target-mode normal_v_abnormal \
    --device cuda \
    --dtype float16 \
    --depth-window 12 \
    --depth-stride 6 \
    --sw-batch-size 4 \
    --pooling mean_std_max

echo "[4/8] Fixed TAP-CT probe (C=0.01, no holdout tuning): $(date)"
python -u regression/scripts/fit_fixed_holdout_tapct.py \
    --features "$TAPCT_DIR/features.npz" \
    --split-json "$SPLIT" \
    --c 0.01 \
    --output "$RUN_OUTPUT/tapct_holdout_predictions.json"

echo "[5/8] Mamba 5-voting on train=230, holdout=200, no valid: $(date)"
cd "$REG"
python -u scripts/train_fixed_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$SPLIT" \
    --source-dir "$PFT_COHORT" \
    --manifest datasets/generated/doctor_validation_fev1fvc70_manifest.json \
    --out "$MAMBA_DIR" \
    --members 5 \
    --epochs 45 \
    --base-seed 42

echo "[6/8] Segment holdout lungs with the existing 3 mm TotalSegmentator protocol: $(date)"
python -u scripts/segment_lungs_totalseg.py \
    --manifest "$WINDOWS_OUTPUT/validation_manifest.json" \
    --out masks/totalseg \
    --fast \
    --device gpu

echo "[7/8] Compute %LAA-950: $(date)"
python -u scripts/quantitative_ct_features.py \
    --source-dir "$PFT_COHORT" \
    --masks masks/totalseg/lung \
    --build-summary "$REPO/classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json" \
    --output "$QCT_DIR"

echo "[8/8] Write final Excel and enforce complete results: $(date)"
cd "$REPO"
python -u regression/scripts/prepare_doctor_validation.py \
    --prior-manifest "$PRIOR_MANIFEST" \
    --output-dir "$WINDOWS_OUTPUT" \
    --cohort-root "$PFT_COHORT" \
    --mamba-json "$MAMBA_DIR/mamba5_holdout_predictions.json" \
    --tapct-json "$RUN_OUTPUT/tapct_holdout_predictions.json" \
    --qct-csv "$QCT_DIR/qct_features.csv" \
    --require-results

echo "DOCTOR_VALIDATION_EXIT=0"
echo "finished: $(date)"
