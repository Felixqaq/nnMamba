#!/usr/bin/env bash
# Phase A -- the official 200-patient holdout (hard cases included) is frozen and
# every newly eligible patient joins training. Threshold and epoch budget come
# from training-only folds; the holdout is touched exactly once, at scoring.
set -eo pipefail
source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

repo=/home/felix/Research/nnMamba
reg="$repo/regression"
win=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831
run="$reg/outputs/doctor_validation_official200_20260831"
base_split=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_20260827/split.json
live_csv=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
snapshot="$win/fev1_fvc_frozen.csv"
split="$win/split.json"
source_ct="$repo/classification/datasets/normal_v_abnormal_fev1fvc70"
cohort="$repo/classification/datasets/doctor_validation_official200_pft"
manifest="$win/cohort_manifest.json"
tapct_old="$reg/embeddings/tapct_doctor_validation_augmented635"
tapct_dir="$reg/embeddings/tapct_doctor_validation_official200"
tapct_out="$run/tapct_calibrated_predictions.json"
mamba_dir="$run/mamba5_training_oof_calibrated_fixed80"
qct_csv="$reg/outputs/doctor_validation_easy200_ctqa_20260828/qct/qct_features.csv"
log="$run/official200_experiment.log"

mkdir -p "$win" "$run" "$tapct_dir/cases" "$mamba_dir"
cd "$repo"

# create_augmented_fixed_holdout_split.py writes fev1_fvc_frozen.csv itself and
# refuses to run if one already exists, so the LIVE csv goes in and the frozen
# copy comes out. Do not pre-copy it.
echo "[1/5] Extend the frozen 200/230 split with every new eligible patient: $(date)" | tee "$log"
python -u regression/scripts/create_augmented_fixed_holdout_split.py \
    --pft-csv "$live_csv" \
    --base-split "$base_split" \
    # Imaging exclusions come from regression/cohort_decisions.local.json, which the
    # script reads directly; they are no longer passed on the command line.
    --output-dir "$win" \
    2>&1 | tee -a "$log"

echo "[2/5] Materialize the cohort directory: $(date)" | tee -a "$log"
python -u regression/scripts/prepare_doctor_validation.py \
    --pft-csv "$snapshot" \
    --output-dir "$win" \
    --ct-root "$source_ct" \
    --cohort-root "$cohort" \
    --use-frozen-split-subset \
    --excel-name official200_split_preview.xlsx \
    2>&1 | tee -a "$log"

echo "[3/5] TAP-CT embeddings, reusing what is already extracted: $(date)" | tee -a "$log"
cp -an "$tapct_old/cases/." "$tapct_dir/cases/" 2>/dev/null || true
python -u regression/scripts/extract_tapct_embeddings.py \
    --source-root "$cohort" \
    --output-dir "$tapct_dir" \
    --target-mode normal_v_abnormal \
    --device cuda --dtype float16 \
    --depth-window 12 --depth-stride 6 --sw-batch-size 4 \
    --pooling mean_std_max \
    2>&1 | tee -a "$log"

echo "[4/5] TAP-CT training-only calibration: $(date)" | tee -a "$log"
python -u regression/scripts/fit_calibrated_holdout_tapct.py \
    --features "$tapct_dir/features.npz" \
    --split-json "$split" \
    --output "$tapct_out" \
    --folds 5 --seed 20260831 \
    2>&1 | tee -a "$log"

# 80 epochs is carried over from the two prior training-only selections on this
# cohort. It was not chosen against this holdout.
echo "[5/5] Mamba training-only calibration then Mamba5 at 80 epochs: $(date)" | tee -a "$log"
python -u regression/scripts/train_calibrated_holdout_ensemble.py \
    --config regression/config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml \
    --split-json "$split" \
    --source-dir "$cohort" \
    --manifest "$manifest" \
    --out "$mamba_dir" \
    --calibration-folds 5 --calibration-seed 20260831 --calibration-base-seed 102 \
    --members 5 --base-seed 72 --epochs 80 \
    2>&1 | tee -a "$log"

echo "[final] Excel: $(date)" | tee -a "$log"
python -u regression/scripts/prepare_doctor_validation.py \
    --pft-csv "$snapshot" \
    --output-dir "$win" \
    --ct-root "$source_ct" \
    --cohort-root "$cohort" \
    --use-frozen-split-subset \
    --mamba-json "$mamba_dir/mamba5_holdout_predictions.json" \
    --tapct-json "$tapct_out" \
    --qct-csv "$qct_csv" \
    --excel-name doctor_validation_official200_retrained.xlsx \
    --require-results \
    2>&1 | tee -a "$log"

cp "$mamba_dir/mamba5_holdout_predictions.json" "$win/mamba5_predictions.json"
cp "$mamba_dir/training_oof_calibration.json" "$win/mamba5_training_oof_calibration.json"
echo "OFFICIAL200_EXIT=0" | tee -a "$log"
echo "finished: $(date)" | tee -a "$log"
