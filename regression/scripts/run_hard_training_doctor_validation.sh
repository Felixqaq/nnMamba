#!/usr/bin/env bash
set -eo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba

REPO="$HOME/Research/nnMamba"
REG="$REPO/regression"
WINDOWS_OUTPUT="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_20260827"
RUN_OUTPUT="$REG/outputs/doctor_validation_20260827"
PFT_COHORT="$REPO/classification/datasets/doctor_validation_fev1fvc70_pft430"
PFT_CSV="/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv"
SPLIT="$WINDOWS_OUTPUT/split.json"
PRIOR_MANIFEST="$WINDOWS_OUTPUT/prior_383_manifest.json"
MAMBA_DIR="$RUN_OUTPUT/mamba5_hard_trainonly"

mkdir -p "$MAMBA_DIR"
cd "$REG"

echo "[1/2] Training-only CV, hard-example mining, and final Mamba5: $(date)"
python -u scripts/train_hard_mined_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$SPLIT" \
    --source-dir "$PFT_COHORT" \
    --manifest datasets/generated/doctor_validation_fev1fvc70_manifest.json \
    --pft-csv "$PFT_CSV" \
    --out "$MAMBA_DIR" \
    --members 5 \
    --base-seed 42 \
    --selection-seed 20260827 \
    --selection-folds 5 \
    --selection-epochs 100 \
    --eval-interval 5 \
    --hard-extra-fraction 0.35

echo "[2/2] Write separate Excel with hard-trained Mamba, TAPCT, and HU adjacent: $(date)"
cd "$REPO"
python -u regression/scripts/prepare_doctor_validation.py \
    --prior-manifest "$PRIOR_MANIFEST" \
    --output-dir "$WINDOWS_OUTPUT" \
    --cohort-root "$PFT_COHORT" \
    --mamba-json "$MAMBA_DIR/mamba5_hard_holdout_predictions.json" \
    --tapct-json "$RUN_OUTPUT/tapct_holdout_predictions.json" \
    --qct-csv "$RUN_OUTPUT/qct/qct_features.csv" \
    --excel-name doctor_validation_200_HU_together_hard_mamba.xlsx \
    --require-results

echo "HARD_TRAINING_EXIT=0"
echo "finished: $(date)"
