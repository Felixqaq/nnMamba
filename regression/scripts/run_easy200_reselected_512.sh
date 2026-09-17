#!/usr/bin/env bash
set -eo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /home/felix/Research/nnMamba

name=doctor_validation_easy200_reselected_20260902
windows_output="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/$name"
run_output="regression/outputs/$name"
mamba_dir="$run_output/mamba5_training_oof_calibrated_fixed80"
log="$run_output/easy200_reselected_experiment.log"
status="$windows_output/training_status.txt"
mkdir -p "$run_output" "$mamba_dir"

{
    echo "state=RUNNING_MAMBA"
    echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "split=512 training / 200 deliberately easy validation"
} > "$status"

echo "[1/3] Mamba training-only OOF calibration plus full Mamba5: $(date)" | tee "$log"
python -u regression/scripts/train_calibrated_holdout_ensemble.py \
    --config regression/config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$windows_output/split.json" \
    --source-dir classification/datasets/doctor_validation_easy200_reselected712_pft \
    --manifest "$windows_output/cohort_manifest.json" \
    --out "$mamba_dir" \
    --calibration-folds 5 \
    --calibration-seed 20260902 \
    --calibration-base-seed 102 \
    --members 5 \
    --base-seed 72 \
    --epochs 80 \
    2>&1 | tee -a "$log"

{
    echo "state=RUNNING_QCT"
    echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "split=512 training / 200 deliberately easy validation"
} > "$status"

echo "[2/3] Complete resumable lung masks for the new easy200: $(date)" | tee -a "$log"
python -u regression/scripts/segment_lungs_totalseg.py \
    --manifest "$windows_output/validation_manifest.json" \
    --out regression/masks/totalseg \
    --task total \
    --fast \
    --device gpu \
    2>&1 | tee -a "$log"

echo "[3/3] Rebuild QCT features from available verified masks: $(date)" | tee -a "$log"
python -u regression/scripts/quantitative_ct_features.py \
    --source-dir classification/datasets/doctor_validation_easy200_reselected712_pft \
    --masks regression/masks/totalseg/lung \
    --build-summary classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json \
    --output "$run_output/qct" \
    2>&1 | tee -a "$log"

cp "$mamba_dir/mamba5_holdout_predictions.json" "$windows_output/mamba5_predictions.json"
cp "$mamba_dir/training_oof_calibration.json" "$windows_output/mamba5_training_oof_calibration.json"
cp "$run_output/tapct_calibrated_predictions.json" "$windows_output/tapct_calibrated_predictions.json"
cp "$run_output/qct/qct_features.csv" "$windows_output/qct_features.csv"
cp "$log" "$windows_output/easy200_reselected_experiment.log"

{
    echo "state=READY_FOR_EXCEL"
    echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "split=512 training / 200 deliberately easy validation"
} > "$status"
echo "EASY200_RESELECTED_EXIT=0" | tee -a "$log"
