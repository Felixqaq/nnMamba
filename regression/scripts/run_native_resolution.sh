#!/usr/bin/env bash
# Two-stage ratio regression at native in-plane resolution, run unattended.
#
# Not `set -u`: conda's activate script references SYS_SYSROOT unbound and killed
# an earlier unattended runner on this box. Not `set -e` around training either --
# a crash in one seed must not cancel the rest, so each records its own exit code.
#
# Inputs are the volumes already resampled to 0.7 mm in-plane by
# preprocess_to_target_spacing.py; the config still names that spacing, so the
# loader finds the factors equal and skips the resample. Feeding this the
# original dataset instead would silently cost 4x per sample.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
OUT=$REG/outputs/native07_$(date +%Y%m%d_%H%M)
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
MANIFEST=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/cohort_manifest.json
SRCDIR=/home/felix/Research/nnMamba/classification/datasets/official200_sp07
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
CONFIG=config.rq1.ratio_regression.512.2ch.yaml
EPOCHS=80
SEEDS=${1:-72}

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG" || { note "FATAL cannot cd to $REG"; exit 1; }

note "START  輸出 $OUT"
note "設定 $CONFIG  epochs=$EPOCHS  seeds=[$SEEDS]"
note "來源 $SRCDIR (已重採樣到 0.7mm)"
df -h /home | tail -1 | while read -r l; do note "磁碟 $l"; done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
    while read -r l; do note "GPU 起始 $l"; done

for seed in $SEEDS; do
    note "RUN_START seed=$seed"
    python -u scripts/train_fixed_holdout_ratio_regression.py \
        --config "$CONFIG" \
        --split-json "$SPLIT" \
        --source-dir "$SRCDIR" \
        --manifest "$MANIFEST" \
        --pft-csv "$PFT" \
        --out "$OUT/native07_2ch_seed$seed" \
        --epochs "$EPOCHS" \
        --seed "$seed" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        note "RUN_OK seed=$seed"
        python - "$OUT/native07_2ch_seed$seed" <<'PY' 2>&1 | tee -a "$STATUS"
import glob, json, os, sys
hits = sorted(glob.glob(os.path.join(sys.argv[1], "results_seed*.json")))
if not hits:
    print("  (沒有找到結果檔)")
else:
    d = json.load(open(hits[0], encoding="utf-8"))
    h, r = d["holdout"], d["holdout_ratio"]
    print("  凍結 MAE {}  r {}".format(r["mae"], r["pearson_r"]))
    for k in ("fixed70_at_label_rule", "gli_at_label_rule"):
        b = h[k]
        print("  {:28s} auc {:.4f} bal {:.4f} sen {:.4f} 邊界 {:.4f}".format(
            k, b["auc"], b["balanced_accuracy"], b["sensitivity"],
            b["by_band"]["borderline"]["auc"]))
    print("  對照 112 解析度 2 通道: fixed70 0.7519 / GLI 0.7460 (同 seed 72)")
PY
    else
        note "RUN_FAILED seed=$seed exit=$rc  (137 = 被 OOM killer 砍掉)"
        tail -25 "$LOG" | tr '\r' '\n' | grep -viE "it/s|s/it" | tail -12 | \
            while read -r l; do note "    $l"; done
    fi
    df -h /home | tail -1 | while read -r l; do note "磁碟 $l"; done
done

note "DONE"
