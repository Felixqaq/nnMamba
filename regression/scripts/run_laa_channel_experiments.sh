#!/usr/bin/env bash
# Resume-safe chain for the frozen official-200 auxiliary-channel probes.
# It waits for the active COPDxNet probe before taking the GPU, fills only the
# missing official-cohort masks, rebuilds density channels, validates coverage,
# then trains one seed-matched member for each 2/3-channel Mamba variant.
set -eEo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

REPO=/home/felix/Research/nnMamba
REGRESSION="$REPO/regression"
WINDOWS_REPO=/mnt/d/Felix/Hospital/nnMamba
OFFICIAL="$WINDOWS_REPO/regression/outputs/doctor_validation_official200_20260831"
REPORT="$WINDOWS_REPO/regression/outputs/laa_channel_experiments_20260903"
STATUS="$REPORT/training_status.txt"
LOG=/home/felix/laa_channel_experiments.log
COPDX_OUT="$REGRESSION/outputs/copdxnet_probe_20260903"
LAA2_OUT="$REGRESSION/outputs/mamba_laa2_probe_20260903"
LAA3_OUT="$REGRESSION/outputs/mamba_laa3_probe_20260903"
DENSITY="$REGRESSION/datasets/generated/laa_density_112x136x112"
MASK_ROOT="$REGRESSION/masks/totalseg"

mkdir -p "$REPORT" "$LAA2_OUT" "$LAA3_OUT"
exec > >(tee -a "$LOG") 2>&1

status() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$STATUS"
}

on_error() {
    code=$?
    status "FAILED exit=$code line=${BASH_LINENO[0]} (see $LOG)"
    exit "$code"
}
trap on_error ERR

status "STARTED; waiting for the current COPDxNet probe to release the GPU"
while pgrep -f 'train_fixed_holdout_ensemble.py.*copdxnet_probe_20260903' >/dev/null; do
    latest=$(tr '\r' '\n' < /home/felix/copdxnet_probe.log 2>/dev/null \
        | grep -E 'seed 72 epoch [0-9]+/80' | tail -n 1 || true)
    status "WAIT_COPDXNET ${latest:-process active}"
    sleep 60
done
if [[ -f "$COPDX_OUT/mamba5_holdout_predictions.json" ]]; then
    status "COPDXNET_COMPLETE"
else
    status "COPDXNET_STOPPED_WITHOUT_RESULT; continuing independent LAA experiments"
fi

status "SYNC_CODE_FROM_WINDOWS"
install -m 644 "$WINDOWS_REPO/regression/core/config.py" "$REGRESSION/core/config.py"
install -m 644 "$WINDOWS_REPO/regression/data/dataset.py" "$REGRESSION/data/dataset.py"
install -m 644 "$WINDOWS_REPO/regression/data/loader.py" "$REGRESSION/data/loader.py"
install -m 644 "$WINDOWS_REPO/regression/data/transforms.py" "$REGRESSION/data/transforms.py"
install -m 644 "$WINDOWS_REPO/regression/test_laa_density_channels.py" "$REGRESSION/test_laa_density_channels.py"
install -m 644 "$WINDOWS_REPO/regression/config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa2.yaml" \
    "$REGRESSION/config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa2.yaml"
install -m 644 "$WINDOWS_REPO/regression/config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa3.yaml" \
    "$REGRESSION/config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa3.yaml"
install -m 755 "$WINDOWS_REPO/regression/scripts/check_laa_density_coverage.py" \
    "$REGRESSION/scripts/check_laa_density_coverage.py"
install -m 755 "$WINDOWS_REPO/regression/scripts/compare_fixed_holdout_probes.py" \
    "$REGRESSION/scripts/compare_fixed_holdout_probes.py"

cd "$REGRESSION"
status "SMOKE_TEST"
python -m py_compile core/config.py data/dataset.py data/loader.py data/transforms.py \
    scripts/check_laa_density_coverage.py scripts/compare_fixed_holdout_probes.py
python test_laa_density_channels.py

status "SEGMENT_MISSING_OFFICIAL_MASKS"
python -u scripts/segment_lungs_totalseg.py \
    --manifest "$OFFICIAL/cohort_manifest.json" \
    --out "$MASK_ROOT" --task total --fast --device gpu

status "REBUILD_LAA_DENSITY"
python -u scripts/precompute_laa_density.py \
    --ct-root "$REPO/classification/datasets/normal_v_abnormal_fev1fvc70" \
    --mask-dir "$MASK_ROOT/lung" --out "$DENSITY" --workers 4

status "CHECK_OFFICIAL_712_DENSITY_COVERAGE"
python scripts/check_laa_density_coverage.py \
    --manifest "$OFFICIAL/cohort_manifest.json" --density-dir "$DENSITY"

status "TRAIN_LAA2 seed=72 epochs=80 train=512 holdout=200"
python -u scripts/train_fixed_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa2.yaml \
    --split-json "$OFFICIAL/split.json" \
    --source-dir "$REPO/classification/datasets/doctor_validation_official200_pft" \
    --manifest "$OFFICIAL/cohort_manifest.json" --out "$LAA2_OUT" \
    --members 1 --epochs 80 --base-seed 72
cp -f "$LAA2_OUT/mamba5_holdout_predictions.json" "$REPORT/mamba_laa2_probe_predictions.json"

status "TRAIN_LAA3 seed=72 epochs=80 train=512 holdout=200"
python -u scripts/train_fixed_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.mamba_laa3.yaml \
    --split-json "$OFFICIAL/split.json" \
    --source-dir "$REPO/classification/datasets/doctor_validation_official200_pft" \
    --manifest "$OFFICIAL/cohort_manifest.json" --out "$LAA3_OUT" \
    --members 1 --epochs 80 --base-seed 72
cp -f "$LAA3_OUT/mamba5_holdout_predictions.json" "$REPORT/mamba_laa3_probe_predictions.json"

baseline="$REGRESSION/outputs/doctor_validation_official200_20260831/mamba5_training_oof_calibrated_fixed80/full_training_members/member_01_predictions.json"
comparison=(
    --result "Mamba_CT_1ch=$baseline"
    --result "Mamba_CT_LAA_2ch=$LAA2_OUT/mamba5_holdout_predictions.json"
    --result "Mamba_CT_LAA_Occupancy_3ch=$LAA3_OUT/mamba5_holdout_predictions.json"
)
if [[ -f "$COPDX_OUT/mamba5_holdout_predictions.json" ]]; then
    comparison+=(--result "COPDxNet_CT_1ch=$COPDX_OUT/mamba5_holdout_predictions.json")
fi

status "COMPARE_FROZEN_OFFICIAL_200"
python scripts/compare_fixed_holdout_probes.py \
    --manifest "$OFFICIAL/cohort_manifest.json" --split-json "$OFFICIAL/split.json" \
    "${comparison[@]}" --out-json "$REPORT/probe_comparison.json" \
    --out-csv "$REPORT/probe_comparison.csv"
status "COMPLETE"
cp -f "$LOG" "$REPORT/laa_channel_experiments.log" 2>/dev/null || true
