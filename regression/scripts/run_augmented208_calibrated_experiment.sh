#!/usr/bin/env bash
set -eo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

repo=/home/felix/Research/nnMamba
reg="$repo/regression"
windows_output=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830
run_output="$reg/outputs/doctor_validation_augmented_ctqa_easy200_20260830"
split="$windows_output/split.json"
pft_snapshot="$windows_output/fev1_fvc_frozen.csv"
source_ct="$repo/classification/datasets/normal_v_abnormal_fev1fvc70"
cohort="$repo/classification/datasets/doctor_validation_augmented635_pft"
manifest="$windows_output/cohort_manifest.json"
build_summary="$source_ct/build_summary.json"
tapct_old="$reg/embeddings/tapct_doctor_validation_430"
tapct_dir="$reg/embeddings/tapct_doctor_validation_augmented635"
tapct_output="$run_output/tapct_calibrated_predictions.json"
mamba_dir="$run_output/mamba5_training_oof_calibrated_fixed80"
qct_csv="$reg/outputs/doctor_validation_easy200_ctqa_20260828/qct/qct_features.csv"
log="$run_output/augmented206_calibrated_experiment.log"

mkdir -p "$run_output" "$tapct_dir/cases" "$mamba_dir"
cd "$repo"

echo "[1/6] Audit the frozen 635-patient CT cohort: $(date)" | tee "$log"
python -u regression/scripts/audit_fixed_cohort_cts.py \
    --split-json "$split" \
    --source-dir "$source_ct" \
    --build-summary "$build_summary" \
    --output "$windows_output/ct_audit.json" \
    --reuse-valid-output \
    2>&1 | tee -a "$log"

echo "[2/6] Materialize the exact 635-patient cohort: $(date)" | tee -a "$log"
    # Imaging exclusions come from regression/cohort_decisions.local.json, which the
    # script reads directly; they are no longer passed on the command line.
python -u regression/scripts/prepare_doctor_validation.py \
    --pft-csv "$pft_snapshot" \
    --output-dir "$windows_output" \
    --ct-root "$source_ct" \
    --cohort-root "$cohort" \
    --use-frozen-split-subset \
    --excel-name augmented206_split_preview.xlsx \
    2>&1 | tee -a "$log"

echo "[3/6] Complete resumable TAP-CT embeddings for 635 patients: $(date)" | tee -a "$log"
cp -an "$tapct_old/cases/." "$tapct_dir/cases/"
python -u regression/scripts/extract_tapct_embeddings.py \
    --source-root "$cohort" \
    --output-dir "$tapct_dir" \
    --target-mode normal_v_abnormal \
    --device cuda \
    --dtype float16 \
    --depth-window 12 \
    --depth-stride 6 \
    --sw-batch-size 4 \
    --pooling mean_std_max \
    2>&1 | tee -a "$log"

echo "[4/6] TAP-CT training-only 5-fold C/threshold calibration: $(date)" | tee -a "$log"
python -u regression/scripts/fit_calibrated_holdout_tapct.py \
    --features "$tapct_dir/features.npz" \
    --split-json "$split" \
    --output "$tapct_output" \
    --folds 5 \
    --seed 20260829 \
    2>&1 | tee -a "$log"

echo "[5/6] Mamba training-only 5-fold calibration plus full Mamba5 at 80 epochs: $(date)" | tee -a "$log"
python -u regression/scripts/train_calibrated_holdout_ensemble.py \
    --config regression/config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$split" \
    --source-dir "$cohort" \
    --manifest "$manifest" \
    --out "$mamba_dir" \
    --calibration-folds 5 \
    --calibration-seed 20260829 \
    --calibration-base-seed 102 \
    --members 5 \
    --base-seed 72 \
    --epochs 80 \
    2>&1 | tee -a "$log"

echo "[6/6] Write Mamba/TAP-CT/HU/clinical Excel and require complete easy200 results: $(date)" | tee -a "$log"
python -u regression/scripts/prepare_doctor_validation.py \
    --pft-csv "$pft_snapshot" \
    --output-dir "$windows_output" \
    --ct-root "$source_ct" \
    --cohort-root "$cohort" \
    --use-frozen-split-subset \
    --mamba-json "$mamba_dir/mamba5_holdout_predictions.json" \
    --tapct-json "$tapct_output" \
    --qct-csv "$qct_csv" \
    --excel-name doctor_validation_augmented206_easy200_calibrated.xlsx \
    --require-results \
    2>&1 | tee -a "$log"

cp "$mamba_dir/mamba5_holdout_predictions.json" "$windows_output/mamba5_calibrated_predictions.json"
cp "$mamba_dir/training_oof_calibration.json" "$windows_output/mamba5_training_oof_calibration.json"
cp "$tapct_output" "$windows_output/tapct_calibrated_predictions.json"

echo "AUGMENTED_CALIBRATED_EXIT=0" | tee -a "$log"
echo "finished: $(date)" | tee -a "$log"
