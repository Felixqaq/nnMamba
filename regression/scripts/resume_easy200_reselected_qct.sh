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

{
    echo "state=RUNNING_QCT"
    echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "split=512 training / 200 deliberately easy validation"
    echo "resume=existing Mamba/TAPCT kept; missing masks only"
} > "$status"

echo "[RESUME] Continue missing lung masks: $(date)" | tee -a "$log"
python -u regression/scripts/segment_lungs_totalseg.py \
    --manifest "$windows_output/validation_manifest.json" \
    --out regression/masks/totalseg \
    --task total \
    --fast \
    --device gpu \
    2>&1 | tee -a "$log"

echo "[RESUME] Rebuild QCT features: $(date)" | tee -a "$log"
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
    echo "resume=completed"
} > "$status"
echo "EASY200_RESELECTED_RESUME_EXIT=0" | tee -a "$log"
