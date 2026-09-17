#!/usr/bin/env bash
# Overnight two-stage FEV1/FVC runs: regress the ratio, then read the cutoff off.
#
# Deliberately NOT `set -u`: conda's activate script references SYS_SYSROOT
# unbound and killed an earlier unattended runner here. Also deliberately not
# `set -e` around the training calls -- one crashed run must not silently cancel
# the five that follow, so each records its own exit code and the loop continues.
#
# Seven runs, about 45 minutes each. Three seeds of the 1-channel regressor and
# three of the 2-channel (CT plus native-resolution LAA-950 density) give a paired
# comparison rather than two isolated numbers, and the seventh is the direct
# classifier at matched settings -- the control that was missing when a
# single-member 2-channel classifier (AUC 0.687) was set against a five-member
# 1-channel ensemble (0.642) and the difference could not be attributed.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
OUT=$REG/outputs/overnight_ratio_$(date +%Y%m%d_%H%M)
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
MANIFEST=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/cohort_manifest.json
SRCDIR=/home/felix/Research/nnMamba/classification/datasets/doctor_validation_official200_pft
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
EPOCHS=80

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG" || { note "FATAL cannot cd to $REG"; exit 1; }

note "START  輸出目錄 $OUT"
note "七個訓練,每個約 45 分鐘,預計 5.3 小時"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
    while read -r line; do note "GPU 起始 $line"; done

run_ratio() {
    local cfg=$1 seed=$2 tag=$3
    note "RATIO_START $tag seed=$seed"
    python -u scripts/train_fixed_holdout_ratio_regression.py \
        --config "$cfg" \
        --split-json "$SPLIT" \
        --source-dir "$SRCDIR" \
        --manifest "$MANIFEST" \
        --pft-csv "$PFT" \
        --out "$OUT/$tag" \
        --epochs "$EPOCHS" \
        --seed "$seed" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        note "RATIO_OK $tag seed=$seed"
    else
        note "RATIO_FAILED $tag seed=$seed exit=$rc  (exit 137 = 被 OOM killer 砍掉)"
    fi
}

run_classifier() {
    local cfg=$1 seed=$2 tag=$3
    note "CLS_START $tag seed=$seed"
    python -u scripts/train_fixed_holdout_ensemble.py \
        --config "$cfg" \
        --split-json "$SPLIT" \
        --source-dir "$SRCDIR" \
        --manifest "$MANIFEST" \
        --out "$OUT/$tag" \
        --members 1 \
        --epochs "$EPOCHS" \
        --base-seed "$seed" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        note "CLS_OK $tag seed=$seed"
    else
        note "CLS_FAILED $tag seed=$seed exit=$rc"
    fi
}

for seed in 72 73 74; do
    run_ratio config.rq1.ratio_regression.1ch.yaml "$seed" "ratio_1ch_seed$seed"
done
for seed in 72 73 74; do
    run_ratio config.rq1.ratio_regression.2ch.yaml "$seed" "ratio_2ch_seed$seed"
done

# The missing control: direct classification, 1 channel, one member, same epochs.
run_classifier config.rq1.normal_v_abnormal.image.fev1fvc70.ensemble5.384.yaml 72 "cls_1ch_seed72"

note "COLLATE"
python -u scripts/collate_overnight_ratio.py --run-dir "$OUT" >> "$STATUS" 2>&1
note "DONE"
