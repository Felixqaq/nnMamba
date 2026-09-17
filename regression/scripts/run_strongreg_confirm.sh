#!/usr/bin/env bash
# Confirm the stronger-regularisation gain, then spend the frozen holdout once.
#
# One seed on 412 training patients put stronger regularisation +0.058 above the
# baseline on an internal 100, ahead at all eight epoch checkpoints. Seed spread
# on this cohort is about 0.010, so the gain is probably real -- but those eight
# points come from one training run, not eight independent ones, so two more
# seeds decide it.
#
# Phase 1 stays inside the training set and never touches the frozen 200.
# Phase 2 then trains on all 512 with the configuration chosen in phase 1 and
# scores the holdout. That ordering is the point: for the first time on this
# project the settings are fixed before the holdout is looked at, so the number
# phase 2 produces means what it says.
#
# Not `set -u`: conda's activate script references SYS_SYSROOT unbound and killed
# an earlier unattended runner here. Not `set -e` around training: one crashed
# seed must not cancel the rest.

set -o pipefail

REG=/home/felix/Research/nnMamba/regression
OUT=$REG/outputs/strongreg_$(date +%Y%m%d_%H%M)
SPLIT=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/split.json
MANIFEST=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831/cohort_manifest.json
SRCDIR=/home/felix/Research/nnMamba/classification/datasets/doctor_validation_official200_pft
PFT=/mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv
CFG=config.rq1.ratio_regression.2ch.strongreg.yaml
EPOCHS=80

mkdir -p "$OUT"
STATUS=$OUT/status.txt
LOG=$OUT/run.log

note() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" | tee -a "$STATUS"; }

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
cd "$REG" || { note "FATAL cannot cd to $REG"; exit 1; }

note "START  輸出 $OUT"
note "階段 1:412 人訓練,內部驗證,seed 73/74,不碰凍結 200"
note "階段 2:512 人訓練,seed 72/73/74,評分凍結 200"
note "設定 $CFG  epochs=$EPOCHS"
df -h /home | tail -1 | while read -r l; do note "磁碟 $l"; done

train() {
    local tag=$1 seed=$2
    shift 2
    note "RUN_START $tag seed=$seed"
    python -u scripts/train_fixed_holdout_ratio_regression.py \
        --config "$CFG" \
        --split-json "$SPLIT" \
        --source-dir "$SRCDIR" \
        --manifest "$MANIFEST" \
        --pft-csv "$PFT" \
        --out "$OUT/$tag" \
        --epochs "$EPOCHS" \
        --seed "$seed" \
        "$@" >> "$LOG" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        note "RUN_OK $tag seed=$seed"
    else
        note "RUN_FAILED $tag seed=$seed exit=$rc  (137 = 被 OOM killer 砍掉)"
        tail -25 "$LOG" | tr '\r' '\n' | grep -viE "it/s|s/it" | tail -12 | \
            while read -r l; do note "    $l"; done
    fi
}

note "PHASE1_START"
for seed in 73 74; do
    train "internal_seed$seed" "$seed" \
        --internal-val-size 100 --eval-every 10 --skip-holdout
done
note "PHASE1_DONE"

note "PHASE2_START  從這裡開始才會碰凍結 200"
for seed in 72 73 74; do
    train "holdout_seed$seed" "$seed"
done
note "PHASE2_DONE"

note "COLLATE"
python - "$OUT" "$REG/outputs/overfit_curves_20260904_1655" <<'PY' 2>&1 | tee -a "$STATUS"
import glob, json, os, sys
import numpy as np

root, prior = sys.argv[1], sys.argv[2]

def load(pattern):
    out = {}
    for d in sorted(p for p in glob.glob(pattern) if os.path.isdir(p)):
        hits = glob.glob(os.path.join(d, "results_seed*.json"))
        if hits:
            out[os.path.basename(d)] = json.load(open(hits[0], encoding="utf-8"))
    return out

runs = load(os.path.join(root, "*"))
old = load(os.path.join(prior, "*"))

print("")
print("=" * 80)
print("加強正則化:確認與最終評分")
print("=" * 80)

# ---- phase 1: internal curves, strong reg across three seeds ----
internal = {k: v for k, v in runs.items() if k.startswith("internal_")}
base = old.get("A_baseline", {}).get("internal_validation_curve", [])
seed72 = old.get("B_strongreg", {}).get("internal_validation_curve", [])
print("\n階段 1  內部驗證 100 人,AUC(切點 70)")
cols = ["A 基線 s72", "B 強正則 s72"] + ["B 強正則 s%s" % k.split("seed")[-1]
                                          for k in sorted(internal)]
print("{:>7}".format("epoch") + "".join("{:>15}".format(c) for c in cols))
print("-" * (7 + 15 * len(cols)))
curves = [base, seed72] + [internal[k]["internal_validation_curve"] for k in sorted(internal)]
epochs = sorted({r["epoch"] for c in curves for r in c})
for e in epochs:
    line = "{:>7}".format(e)
    for c in curves:
        row = next((r for r in c if r["epoch"] == e), None)
        line += "{:>15}".format("{:.4f}".format(row["internal_auc_fixed70"]) if row else "-")
    print(line)

b_final = [c[-1]["internal_auc_fixed70"] for c in curves[1:] if c]
if base and b_final:
    print("\n  基線 epoch80 {:.4f}".format(base[-1]["internal_auc_fixed70"]))
    print("  強正則 epoch80 {:.4f} ± {:.4f}  (n={})".format(
        np.mean(b_final), np.std(b_final), len(b_final)))
    print("  差 {:+.4f}".format(np.mean(b_final) - base[-1]["internal_auc_fixed70"]))

# ---- phase 2: the frozen holdout, scored once with settings already fixed ----
hold = {k: v for k, v in runs.items() if k.startswith("holdout_")}
print("\n階段 2  凍結 200(設定在階段 1 之前就固定,這是本專案第一次有紀律的評分)")
if not hold:
    print("  沒有結果")
else:
    print("{:16s}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "seed", "fx70AUC", "fx70bal", "fx70sen", "gliAUC", "glibal", "MAE"))
    print("-" * 76)
    acc = {k: [] for k in ("fa", "fb", "fs", "ga", "gb", "mae")}
    for k in sorted(hold):
        h = hold[k]["holdout"]
        f, g = h["fixed70_at_label_rule"], h["gli_at_label_rule"]
        r = hold[k]["holdout_ratio"]
        print("{:16s}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.3f}".format(
            k, f["auc"], f["balanced_accuracy"], f["sensitivity"],
            g["auc"], g["balanced_accuracy"], r["mae"]))
        acc["fa"].append(f["auc"]); acc["fb"].append(f["balanced_accuracy"])
        acc["fs"].append(f["sensitivity"]); acc["ga"].append(g["auc"])
        acc["gb"].append(g["balanced_accuracy"]); acc["mae"].append(r["mae"])
    if len(acc["fa"]) > 1:
        print("{:16s}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}".format("", "", "", "", "", "", ""))
        print("{:16s}".format("平均±SD") + "".join(
            "{:>10}".format("{:.4f}".format(np.mean(acc[k]))) for k in
            ("fa", "fb", "fs", "ga", "gb")) + "{:>10.3f}".format(np.mean(acc["mae"])))
        print("{:16s}".format("") + "".join(
            "{:>10}".format("±{:.4f}".format(np.std(acc[k]))) for k in
            ("fa", "fb", "fs", "ga", "gb")) + "{:>10.3f}".format(np.std(acc["mae"])))

print("\n對照:同一份凍結 200,基線正則化")
print("  2 通道二階段 3 seed   fixed70 0.7487±0.0120   GLI 0.7322±0.0098")
print("  1 通道二階段 3 seed   fixed70 0.7215±0.0102   GLI 0.6634±0.0078")
print("  單模型直接分類        fixed70 0.6421")
print("  只用年齡性別身高      fixed70 0.6302          GLI 0.5601")
PY
note "DONE"
