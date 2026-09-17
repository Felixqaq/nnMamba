#!/usr/bin/env bash
# Where does generalisation peak, and does stronger regularisation move it?
#
# Training loss on this task falls all the way to epoch 80 with no plateau
# (0.180 at 20, 0.097 at 40, 0.048 at 60, 0.043 at 80), so the epoch budget
# cannot be read off it -- the model is simply memorising further. Meanwhile
# in-sample ratio MAE is about 1.0 against 8.4 on held-out patients. That gap is
# the largest unexploited lever left, larger than resolution, architecture or
# ensembling, all of which have now been measured and found small.
#
# Each arm carves 100 of the 512 training patients out, trains on the remaining
# 412, and scores those 100 every 10 epochs. One run therefore yields a whole
# epoch-versus-generalisation curve rather than one number at a guessed budget.
#
# The frozen holdout is not touched by either arm. It has been scored more than
# eighteen times and every further look inflates whatever comes out of it; the
# epoch budget and the regularisation strength are chosen here, on training data,
# and only the winning configuration is later spent on it once.
#
# Not `set -u`: conda's activate script references SYS_SYSROOT unbound and killed
# an earlier unattended runner here. Not `set -e` around training: one crashed
# arm must not cancel the other.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
OUT=$REG/outputs/overfit_curves_$(date +%Y%m%d_%H%M)
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
MANIFEST=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/cohort_manifest.json
SRCDIR=/home/felix/Research/nnMamba/classification/datasets/doctor_validation_official200_pft
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
EPOCHS=80
SEED=72
VAL=100
EVERY=10

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG" || { note "FATAL cannot cd to $REG"; exit 1; }

note "START  輸出 $OUT"
note "兩條曲線,每條 80 epoch,內部驗證 $VAL 人,每 $EVERY epoch 評估一次"
note "凍結 200 全程不碰"

run_arm() {
    local tag=$1 cfg=$2
    note "ARM_START $tag  config=$cfg"
    python -u scripts/train_fixed_holdout_ratio_regression.py \
        --config "$cfg" \
        --split-json "$SPLIT" \
        --source-dir "$SRCDIR" \
        --manifest "$MANIFEST" \
        --pft-csv "$PFT" \
        --out "$OUT/$tag" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --internal-val-size "$VAL" \
        --eval-every "$EVERY" \
        --skip-holdout >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        note "ARM_OK $tag"
    else
        note "ARM_FAILED $tag exit=$rc  (137 = 被 OOM killer 砍掉)"
        tail -25 "$LOG" | tr '\r' '\n' | grep -viE "it/s|s/it" | tail -12 | \
            while read -r l; do note "    $l"; done
    fi
}

run_arm "A_baseline"  config.rq1.ratio_regression.2ch.yaml
run_arm "B_strongreg" config.rq1.ratio_regression.2ch.strongreg.yaml

note "COLLATE"
python - "$OUT" <<'PY' 2>&1 | tee -a "$STATUS"
import glob, json, os, sys

root = sys.argv[1]
curves = {}
for d in sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)):
    hits = glob.glob(os.path.join(d, "results_seed*.json"))
    if not hits:
        print("  %s 沒有結果檔" % os.path.basename(d))
        continue
    payload = json.load(open(hits[0], encoding="utf-8"))
    curves[os.path.basename(d)] = payload.get("internal_validation_curve", [])

print("")
print("=" * 78)
print("過擬合曲線:內部驗證 100 人,凍結 200 全程未碰")
print("=" * 78)
if not curves:
    print("\n沒有任何曲線")
    raise SystemExit(0)

names = sorted(curves)
print("\n內部驗證 AUC(切點 70)")
print("{:>7}".format("epoch") + "".join("{:>18}".format(n) for n in names))
print("-" * (7 + 18 * len(names)))
epochs = sorted({r["epoch"] for c in curves.values() for r in c})
for e in epochs:
    line = "{:>7}".format(e)
    for n in names:
        row = next((r for r in curves[n] if r["epoch"] == e), None)
        line += "{:>18}".format("{:.4f}".format(row["internal_auc_fixed70"])
                                if row else "-")
    print(line)

print("\n內部驗證比值 MAE(點)")
print("{:>7}".format("epoch") + "".join("{:>18}".format(n) for n in names))
print("-" * (7 + 18 * len(names)))
for e in epochs:
    line = "{:>7}".format(e)
    for n in names:
        row = next((r for r in curves[n] if r["epoch"] == e), None)
        line += "{:>18}".format("{:.3f}".format(row["internal_mae"]) if row else "-")
    print(line)

print("\n每條曲線的最佳點")
for n in names:
    c = curves[n]
    if not c:
        continue
    b_auc = max(c, key=lambda r: r["internal_auc_fixed70"])
    b_mae = min(c, key=lambda r: r["internal_mae"])
    last = c[-1]
    print("  {:14s} AUC 最佳 epoch {:>3} ({:.4f})   最終 epoch {:>3} ({:.4f})   差 {:+.4f}"
          .format(n, b_auc["epoch"], b_auc["internal_auc_fixed70"],
                  last["epoch"], last["internal_auc_fixed70"],
                  b_auc["internal_auc_fixed70"] - last["internal_auc_fixed70"]))
    print("  {:14s} MAE 最佳 epoch {:>3} ({:.3f})   最終 {:.3f}"
          .format("", b_mae["epoch"], b_mae["internal_mae"], last["internal_mae"]))

print("\n怎麼讀這張表")
print("  最佳 epoch 明顯早於 80  ->  減 epoch 有用,幅度就是那個「差」")
print("  B 的曲線整體高於 A      ->  加強正則有用")
print("  B 的最佳 epoch 晚於 A   ->  正則延後了過擬合,可以跑更久")
print("  兩者都沒動              ->  過擬合不是瓶頸,該換方向")
print("\n  參考:凍結 200 上 2 通道二階段 3 seed 是 0.7487±0.0120,")
print("  但那是 512 人訓練、80 epoch;這裡是 412 人,數字不能直接對照,")
print("  能對照的是兩條曲線之間的差。")
PY
note "DONE"
