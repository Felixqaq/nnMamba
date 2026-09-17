#!/usr/bin/env bash
set -eo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /home/felix/Research/nnMamba

output=regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/mamba5_training_oof_calibrated_fixed80
log=regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/mamba_calibrated.log
mkdir -p "$output" "$(dirname "$log")"

python -u regression/scripts/train_calibrated_holdout_ensemble.py \
    --config regression/config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json /mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/split.json \
    --source-dir classification/datasets/doctor_validation_augmented635_pft \
    --manifest /mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/cohort_manifest.json \
    --out "$output" \
    --calibration-folds 5 \
    --calibration-seed 20260829 \
    --calibration-base-seed 102 \
    --members 5 \
    --base-seed 72 \
    --epochs 80 \
    2>&1 | tee "$log"
