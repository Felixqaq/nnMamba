#!/usr/bin/env bash
# Frozen TapCT feature-level ablation on the exact official 512/200 split.
set -eEo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

REPO=/home/felix/Research/nnMamba
REGRESSION="$REPO/regression"
WINDOWS_REPO=/mnt/d/Felix/Hospital/nnMamba
OFFICIAL="$WINDOWS_REPO/regression/outputs/doctor_validation_official200_20260831"
REPORT="$WINDOWS_REPO/regression/outputs/tapct_laa_ablation_official200_20260903"
STATUS="$REPORT/training_status.txt"
LOG=/home/felix/tapct_laa_ablation.log
DENSITY="$REGRESSION/datasets/generated/laa_density_112x136x112"
TAPCT="$REGRESSION/embeddings/tapct_doctor_validation_official200/features.npz"
FEATURE_ROOT="$REGRESSION/outputs/tapct_laa_ablation_official200_20260903/features"

mkdir -p "$REPORT" "$FEATURE_ROOT"
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

cd "$REGRESSION"
status "STARTED; waiting for valid LAA arrays for all official 712 patients"
while ! python scripts/check_laa_density_coverage.py \
    --manifest "$OFFICIAL/cohort_manifest.json" --density-dir "$DENSITY" \
    >/tmp/tapct_laa_coverage.txt 2>&1; do
    status "WAIT_LAA_COVERAGE $(head -n 1 /tmp/tapct_laa_coverage.txt)"
    sleep 60
done
status "LAA_COVERAGE_COMPLETE"

status "EXTRACT_LAA_FEATURES"
python -u scripts/extract_laa_tapct_features.py \
    --tapct-features "$TAPCT" --density-dir "$DENSITY" \
    --out-root "$FEATURE_ROOT/auxiliary" --grid-size 4

status "COMBINE_TAPCT_WITH_LAA_DENSITY"
python scripts/combine_features.py --a "$TAPCT" \
    --b "$FEATURE_ROOT/auxiliary/laa_density/features.npz" \
    --out "$FEATURE_ROOT/tapct_laa_density"

status "COMBINE_TAPCT_WITH_LAA_DENSITY_OCCUPANCY"
python scripts/combine_features.py --a "$TAPCT" \
    --b "$FEATURE_ROOT/auxiliary/laa_density_occupancy/features.npz" \
    --out "$FEATURE_ROOT/tapct_laa_density_occupancy"

status "FIT_CT_ONLY"
python scripts/fit_calibrated_holdout_tapct.py --features "$TAPCT" \
    --split-json "$OFFICIAL/split.json" --output "$REPORT/tapct_ct_only.json"

status "FIT_CT_PLUS_LAA_DENSITY"
python scripts/fit_calibrated_holdout_tapct.py \
    --features "$FEATURE_ROOT/tapct_laa_density/features.npz" \
    --split-json "$OFFICIAL/split.json" --output "$REPORT/tapct_ct_laa_density.json"

status "FIT_CT_PLUS_LAA_DENSITY_OCCUPANCY"
python scripts/fit_calibrated_holdout_tapct.py \
    --features "$FEATURE_ROOT/tapct_laa_density_occupancy/features.npz" \
    --split-json "$OFFICIAL/split.json" \
    --output "$REPORT/tapct_ct_laa_density_occupancy.json"

status "COMPARE_FIXED_OFFICIAL_200"
python scripts/compare_fixed_holdout_probes.py \
    --manifest "$OFFICIAL/cohort_manifest.json" --split-json "$OFFICIAL/split.json" \
    --result "TapCT_CT_only=$REPORT/tapct_ct_only.json" \
    --result "TapCT_CT_LAA_density=$REPORT/tapct_ct_laa_density.json" \
    --result "TapCT_CT_LAA_density_occupancy=$REPORT/tapct_ct_laa_density_occupancy.json" \
    --out-json "$REPORT/tapct_laa_ablation_comparison.json" \
    --out-csv "$REPORT/tapct_laa_ablation_comparison.csv"

status "COMPLETE"
cp -f "$LOG" "$REPORT/tapct_laa_ablation.log" 2>/dev/null || true
