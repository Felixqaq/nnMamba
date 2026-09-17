#!/usr/bin/env bash
# Does keeping the native in-plane detail help a frozen TAP-CT?
#
# TAP-CT resizes to 224 in-plane, which on this cohort is about 1.37 mm -- twice
# the Mamba pipeline's 2.74 mm, still twice the scanner's 0.6 mm. Airway walls run
# 0.5-1.5 mm, so they are the one COPD finding no channel trick can reconstruct
# from a downsampled volume; emphysema is already carried losslessly by the
# precomputed %LAA-950 density.
#
# The confound is stated up front: the encoder was pretrained at 224, where an
# 8x8 patch spans 11 mm; at 512 the same patch spans 4.8 mm, so every learned
# spatial relation shifts. A drop is therefore ambiguous -- it could be the
# resolution or the scale mismatch. A gain is not ambiguous, because it would
# have been won against that handicap.
#
# sw_batch_size is forced to 1: one 512 window peaks at 3.5 GiB on this 8 GiB
# card, and the script's default of 4 would not fit.
#
# Not `set -u`: conda's activate script references SYS_SYSROOT unbound and killed
# an earlier unattended runner here.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
STAMP=$(date +%Y%m%d_%H%M)
OUT=$REG/outputs/tapct_res512_$STAMP
EMB=$REG/embeddings/tapct_official200_r512
SRC=/home/felix/Research/nnMamba/classification/datasets/doctor_validation_official200_pft
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
BASE_EMB=$REG/embeddings/tapct_doctor_validation_official200/features.npz

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG/scripts" || { note "FATAL cannot cd to $REG/scripts"; exit 1; }

note "START  輸出 $OUT"
note "抽取 712 位 TAP-CT 特徵於 512 平面內解析度,預計約 1.4 小時"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
    while read -r l; do note "GPU 起始 $l"; done

note "EXTRACT_START"
python -u extract_tapct_embeddings.py \
    --source-root "$SRC" \
    --output-dir "$EMB" \
    --target-mode normal_v_abnormal \
    --resize-dim 512 \
    --sw-batch-size 1 \
    --dtype float16 >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    note "EXTRACT_FAILED exit=$rc  (137 = 被 OOM killer 砍掉)"
    tail -30 "$LOG" | tr '\r' '\n' | grep -viE "it/s|s/it" | tail -15 | \
        while read -r l; do note "    $l"; done
    note "DONE (抽取失敗,沒有探針結果)"
    exit $rc
fi
note "EXTRACT_OK"
python - "$EMB/extraction_config.json" <<'PY' 2>&1 | tee -a "$STATUS"
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
print("  抽取設定 resize_dim={} ({}), 病人 {}, 特徵維度 {}".format(
    c.get("resize_dim"), c.get("resize_dim_source"), c.get("num_cases"),
    c.get("feature_dim")))
PY

# Same probe, same split, same protocol as the 224 features, so the only thing
# that differs between the two result files is the input resolution.
for tag in r512 r224; do
    if [ "$tag" = "r512" ]; then FEAT=$EMB/features.npz; else FEAT=$BASE_EMB; fi
    note "PROBE_START $tag  features=$FEAT"
    python -u tapct_ratio_regression.py \
        --features "$FEAT" \
        --split-json "$SPLIT" \
        --pft-csv "$PFT" \
        --out "$OUT/probe_$tag.json" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then note "PROBE_OK $tag"; else
        note "PROBE_FAILED $tag exit=$rc"
        tail -20 "$LOG" | tr '\r' '\n' | tail -10 | while read -r l; do note "    $l"; done
    fi
done

note "COLLATE"
python - "$OUT" <<'PY' 2>&1 | tee -a "$STATUS"
import json, os, sys

root = sys.argv[1]
res = {}
for tag in ("r224", "r512"):
    path = os.path.join(root, "probe_%s.json" % tag)
    if os.path.exists(path):
        res[tag] = json.load(open(path, encoding="utf-8"))

print("")
print("=" * 82)
print("TAP-CT 平面內解析度:224 (約1.37mm) 對 512 (約0.6mm),凍結特徵、同一探針")
print("=" * 82)
if len(res) < 2:
    print("\n只有 %d 份結果,無法比較: %s" % (len(res), list(res)))
else:
    print("\n{:26s}{:>12}{:>12}{:>12}{:>12}".format(
        "", "224 OOF", "224 凍結", "512 OOF", "512 凍結"))
    print("-" * 74)
    a, b = res["r224"], res["r512"]
    print("{:26s}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}".format(
        "分類", a["classification"]["training_oof_auc"],
        a["classification"]["holdout"]["auc"],
        b["classification"]["training_oof_auc"],
        b["classification"]["holdout"]["auc"]))
    print("{:26s}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}".format(
        "二階段回歸,切 70", a["regression"]["training_oof_auc"],
        a["regression"]["holdout_at_label_rule_70"]["auc"],
        b["regression"]["training_oof_auc"],
        b["regression"]["holdout_at_label_rule_70"]["auc"]))
    print("\n比值預測誤差 (MAE, 點)")
    print("  224  訓練OOF {:.3f}  凍結 {:.3f}".format(
        a["regression"]["training_oof_mae"], a["regression"]["holdout_mae"]))
    print("  512  訓練OOF {:.3f}  凍結 {:.3f}".format(
        b["regression"]["training_oof_mae"], b["regression"]["holdout_mae"]))
    print("\n分難度看 AUC")
    for band in a.get("by_difficulty", {}):
        if band in b.get("by_difficulty", {}):
            print("  {:12s} 224 分類 {:.4f} 迴歸 {:.4f} | 512 分類 {:.4f} 迴歸 {:.4f}".format(
                band,
                a["by_difficulty"][band]["classification_auc"],
                a["by_difficulty"][band]["regression_auc"],
                b["by_difficulty"][band]["classification_auc"],
                b["by_difficulty"][band]["regression_auc"]))
    d = (b["regression"]["holdout_at_label_rule_70"]["auc"]
         - a["regression"]["holdout_at_label_rule_70"]["auc"])
    print("\n  解析度效果(512 - 224,二階段凍結 AUC): {:+.4f}".format(d))
    print("  512 是在不利條件下比較的:編碼器在 224 預訓練,patch physical")
    print("  尺寸從 11mm 變成 4.8mm。變好是乾淨的證據,變差則有兩種解釋。")

print("\n對照:同一份凍結 200")
print("  只用年齡性別身高                0.6302")
print("  Mamba 直接分類 1 通道           0.6421")
print("  Mamba 二階段回歸 2 通道 3 seed   0.7487±0.0120")
print("  TAP-CT LoRA 多任務 3 seed       0.7164±0.0099 (比值頭)")
print("\n  提醒:凍結 200 已被評分超過十八次,絕對名次帶樂觀偏差。")
PY
note "DONE"
