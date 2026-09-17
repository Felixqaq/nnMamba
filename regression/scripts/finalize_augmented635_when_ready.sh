#!/usr/bin/env bash
set -eo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /home/felix/Research/nnMamba

windows_output=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830
run_output=regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830
mamba_dir="$run_output/mamba5_training_oof_calibrated_fixed80"
mamba_json="$mamba_dir/mamba5_holdout_predictions.json"
tapct_json="$run_output/tapct_calibrated_predictions.json"
qct_csv=regression/outputs/doctor_validation_easy200_ctqa_20260828/qct/qct_features.csv
log="$run_output/finalize_excel.log"

while [[ ! -f "$mamba_json" ]]; do
    if ! tmux has-session -t aug635cal 2>/dev/null; then
        echo "Mamba session ended before predictions were written" | tee "$log"
        exit 1
    fi
    sleep 60
done

    # Imaging exclusions come from regression/cohort_decisions.local.json, which the
    # script reads directly; they are no longer passed on the command line.
python -u regression/scripts/prepare_doctor_validation.py \
    --pft-csv "$windows_output/fev1_fvc_frozen.csv" \
    --output-dir "$windows_output" \
    --ct-root classification/datasets/normal_v_abnormal_fev1fvc70 \
    --cohort-root classification/datasets/doctor_validation_augmented635_pft \
    --use-frozen-split-subset \
    --mamba-json "$mamba_json" \
    --tapct-json "$tapct_json" \
    --qct-csv "$qct_csv" \
    --excel-name doctor_validation_augmented206_easy200_calibrated.xlsx \
    --require-results \
    2>&1 | tee "$log"

cp "$mamba_json" "$windows_output/mamba5_calibrated_predictions.json"
cp "$mamba_dir/training_oof_calibration.json" "$windows_output/mamba5_training_oof_calibration.json"
cp "$tapct_json" "$windows_output/tapct_calibrated_predictions.json"
cp "$run_output/mamba_calibrated.log" "$windows_output/mamba_calibrated.log"
echo "FINALIZE_EXIT=0" | tee -a "$log"
