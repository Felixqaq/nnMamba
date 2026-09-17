#!/usr/bin/env bash
set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba

REPO="$HOME/Research/nnMamba"
REG="$REPO/regression"
WINDOWS_OUTPUT="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_easy200_20260828"
RUN_OUTPUT="$REG/outputs/doctor_validation_easy200_20260828"
PFT_COHORT="$REPO/classification/datasets/doctor_validation_fev1fvc70_pft430"
SPLIT="$WINDOWS_OUTPUT/split.json"
PRIOR_MANIFEST="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_20260827/prior_383_manifest.json"
MAMBA_DIR="$RUN_OUTPUT/mamba5_easy_holdout_fixed45"
TAPCT_FEATURES="$REG/embeddings/tapct_doctor_validation_430/features.npz"
TAPCT_OUTPUT="$RUN_OUTPUT/tapct_easy_holdout_predictions.json"
QCT_DIR="$RUN_OUTPUT/qct"

mkdir -p "$MAMBA_DIR" "$QCT_DIR"
cd "$REG"

echo "[1/5] Mamba5 fixed 45 epochs: all 183 boundary-hard + 47 easy, no valid: $(date)"
python -u scripts/train_fixed_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$SPLIT" \
    --source-dir "$PFT_COHORT" \
    --manifest datasets/generated/doctor_validation_fev1fvc70_manifest.json \
    --out "$MAMBA_DIR" \
    --members 5 \
    --base-seed 52 \
    --epochs 45

echo "[2/5] TAP-CT fixed encoder + fixed C=0.01 probe, train=230, no valid: $(date)"
python -u scripts/fit_fixed_holdout_tapct.py \
    --features "$TAPCT_FEATURES" \
    --split-json "$SPLIT" \
    --c 0.01 \
    --seed 52 \
    --output "$TAPCT_OUTPUT"

echo "[3/5] Complete resumable 3 mm TotalSegmentator masks for easy200: $(date)"
python -u scripts/segment_lungs_totalseg.py \
    --manifest "$WINDOWS_OUTPUT/validation_manifest.json" \
    --out masks/totalseg \
    --fast \
    --device gpu

echo "[4/5] Rebuild QCT table and require full easy200 coverage: $(date)"
python -u scripts/quantitative_ct_features.py \
    --source-dir "$PFT_COHORT" \
    --masks masks/totalseg/lung \
    --build-summary "$REPO/classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json" \
    --output "$QCT_DIR"

echo "[5/5] Write easy200 Mamba/TAP-CT/HU/clinical Excel: $(date)"
cd "$REPO"
python -u regression/scripts/prepare_doctor_validation.py \
    --prior-manifest "$PRIOR_MANIFEST" \
    --output-dir "$WINDOWS_OUTPUT" \
    --cohort-root "$PFT_COHORT" \
    --mamba-json "$MAMBA_DIR/mamba5_holdout_predictions.json" \
    --tapct-json "$TAPCT_OUTPUT" \
    --qct-csv "$QCT_DIR/qct_features.csv" \
    --excel-name doctor_validation_easy200_results.xlsx \
    --require-results

cp "$MAMBA_DIR/mamba5_holdout_predictions.json" "$WINDOWS_OUTPUT/mamba5_easy200_predictions.json"
cp "$TAPCT_OUTPUT" "$WINDOWS_OUTPUT/tapct_easy200_predictions.json"

echo "EASY_HOLDOUT_EXIT=0"
echo "finished: $(date)"
