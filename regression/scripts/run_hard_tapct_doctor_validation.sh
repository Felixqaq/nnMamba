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
MAMBA_DIR="$RUN_OUTPUT/mamba5_hard_trainonly"
TAPCT_DIR="$RUN_OUTPUT/tapct_hard_trainonly"
TAPCT_FEATURES="$REG/embeddings/tapct_doctor_validation_430/features.npz"

mkdir -p "$TAPCT_DIR"
cd "$REG"

echo "[1/2] Training-only CV and Mamba-matched hard-case TAP-CT probe: $(date)"
python -u scripts/train_hard_mined_tapct_probe.py \
    --features "$TAPCT_FEATURES" \
    --split-json "$SPLIT" \
    --hardness-csv "$MAMBA_DIR/training230_hardness.csv" \
    --mamba-selection-summary "$MAMBA_DIR/selection_summary.json" \
    --output-dir "$TAPCT_DIR" \
    --selection-folds 5 \
    --selection-seed 20260827

echo "[2/2] Write separate Excel with hard Mamba, hard TAP-CT, and HU adjacent: $(date)"
cd "$REPO"
python -u regression/scripts/prepare_doctor_validation.py \
    --prior-manifest "$PRIOR_MANIFEST" \
    --output-dir "$WINDOWS_OUTPUT" \
    --cohort-root "$PFT_COHORT" \
    --mamba-json "$MAMBA_DIR/mamba5_hard_holdout_predictions.json" \
    --tapct-json "$TAPCT_DIR/tapct_hard_holdout_predictions.json" \
    --qct-csv "$RUN_OUTPUT/qct/qct_features.csv" \
    --excel-name doctor_validation_200_HU_together_hard_mamba_hard_tapct.xlsx \
    --require-results

echo "HARD_TAPCT_EXIT=0"
echo "finished: $(date)"
