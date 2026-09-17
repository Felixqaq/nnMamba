#!/usr/bin/env bash
# Does the two-stage objective help TAP-CT the way it helps the Mamba?
#
# On the Mamba, replacing binary classification with ratio regression was worth
# +0.089 AUC on the frozen 200 -- more than any architecture or channel change
# tried. Whether that transfers to a pretrained backbone is the open half of the
# question: a LoRA head on frozen-ish features may not care what it is asked to
# predict, because the objective cannot reshape a representation it barely moves.
#
# Same seeds on both arms, so this is a paired comparison rather than two
# isolated numbers. Everything else is held at the pilot's settings -- 6 epochs,
# rank 8, last 4 blocks -- so the multi-task arm reproduces the existing run.
#
# Not `set -u`: conda's activate script references SYS_SYSROOT unbound and killed
# an earlier unattended runner here. Not `set -e` around training: one crashed
# seed must not cancel the rest.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
OUT=$REG/outputs/tapct_objective_$(date +%Y%m%d_%H%M)
CACHE=$REG/outputs/tapct_lora_quick_official200_20260903/cache
META=$REG/embeddings/tapct_doctor_validation_official200/metadata.csv
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
EPOCHS=6
SEEDS="72 73 74"

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG/scripts" || { note "FATAL cannot cd to $REG/scripts"; exit 1; }

note "START  輸出 $OUT"
note "六個執行:兩種目標函數 x 三個 seed,每個約 8 分鐘"
note "重用既有窗快取 $CACHE"
[ -f "$CACHE/windows.float16.npy" ] || note "WARNING 快取不存在,每個執行會多花約 5 分鐘重建"

run_arm() {
    local tag=$1 cls_weight=$2 seed=$3
    note "RUN_START $tag seed=$seed classification_loss_weight=$cls_weight"
    python -u tapct_lora_quick.py \
        --metadata-csv "$META" \
        --split-json "$SPLIT" \
        --pft-csv "$PFT" \
        --cache-dir "$CACHE" \
        --output-dir "$OUT/${tag}_seed${seed}" \
        --epochs "$EPOCHS" \
        --seed "$seed" \
        --classification-loss-weight "$cls_weight" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        note "RUN_OK $tag seed=$seed"
    else
        note "RUN_FAILED $tag seed=$seed exit=$rc  (137 = 被 OOM killer 砍掉)"
        tail -20 "$LOG" | tr '\r' '\n' | grep -viE "it/s|s/it" | tail -10 | \
            while read -r l; do note "    $l"; done
    fi
}

for seed in $SEEDS; do
    run_arm "multitask" 1.0 "$seed"
    run_arm "twostage"  0.0 "$seed"
done

note "COLLATE"
python - "$OUT" <<'PY' 2>&1 | tee -a "$STATUS"
import glob, json, os, sys
import numpy as np

root = sys.argv[1]
arms = {}
missing = []
for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
    hits = glob.glob(os.path.join(d, "results.json"))
    if not hits:
        missing.append(os.path.basename(d))
        continue
    r = json.load(open(hits[0], encoding="utf-8"))
    h = r["holdout"]
    arm = os.path.basename(d).rsplit("_seed", 1)[0]
    # The classification head is only trained in the multi-task arm; the ratio
    # head is trained in both, so it is the only column that compares.
    row = {"cls_auc": h.get("classification_head", {}).get("auc")}
    for key in ("ratio_head_at_70", "ratio_head_at_training_calibrated_cutoff"):
        if key in h:
            row[key + "_auc"] = h[key]["auc"]
            row[key + "_bal"] = h[key]["balanced_accuracy"]
            row[key + "_sen"] = h[key]["sensitivity"]
    row["mae"] = r.get("training_in_sample", {}).get("ratio_mae")
    arms.setdefault(arm, []).append(row)

print("")
print("=" * 84)
print("TAP-CT 目標函數消融(凍結 200,配對 seed)")
print("=" * 84)
if missing:
    print("\n沒有結果的執行:")
    for m in missing:
        print("  " + m)

def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "     -"
    if len(vals) == 1:
        return "{:.4f}".format(vals[0])
    return "{:.4f}±{:.4f}".format(np.mean(vals), np.std(vals))

print("\n比值頭,切點 70(未經擬合)")
print("{:14s}{:>4}{:>18}{:>18}{:>18}".format("arm", "n", "AUC", "balacc", "敏感度"))
print("-" * 72)
for arm, rows in sorted(arms.items()):
    print("{:14s}{:>4}{:>18}{:>18}{:>18}".format(
        arm, len(rows), agg(rows, "ratio_head_at_70_auc"),
        agg(rows, "ratio_head_at_70_bal"), agg(rows, "ratio_head_at_70_sen")))

print("\n比值頭,訓練校準切點(in-sample,樂觀)")
print("{:14s}{:>4}{:>18}{:>18}".format("arm", "n", "balacc", "敏感度"))
print("-" * 54)
for arm, rows in sorted(arms.items()):
    k = "ratio_head_at_training_calibrated_cutoff"
    print("{:14s}{:>4}{:>18}{:>18}".format(
        arm, len(rows), agg(rows, k + "_bal"), agg(rows, k + "_sen")))

print("\n分類頭 AUC(只有 multitask 這一臂有訓練,twostage 的無意義)")
for arm, rows in sorted(arms.items()):
    print("  {:14s} {}".format(arm, agg(rows, "cls_auc")))

print("\n對照:同一份凍結 200")
print("  Mamba 直接分類 1 通道 seed72        0.6421")
print("  Mamba 二階段回歸 1 通道 seed72       0.7314   目標函數效果 +0.089")
print("  Mamba 二階段回歸 2 通道 3 seed       0.7487±0.0120")
print("  TAP-CT 多任務 6ep(Codex,1 seed)   0.7253  分類頭 / 0.7171 比值頭")
print("\n  提醒:凍結 200 已被評分超過十二次,絕對名次帶樂觀偏差。")
print("  可信的是同 seed 的配對差,不是這張表的最大值。")
PY
note "DONE"
