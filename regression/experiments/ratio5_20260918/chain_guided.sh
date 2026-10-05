#!/usr/bin/env bash
# After the plain-attention 1301 run (tmux attn1301) finishes, bring the guided
# code into the WSL working copy and run the guided arm. Additive only: two new
# files (the guided network and the guided trainer), the runner, and models.py,
# whose only changes are appended registrations. They are synced after attn1301
# ends because each of its seeds imports the model registry when it starts.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
D=/mnt/d/Felix/Hospital/nnMamba/regression
W=/home/felix/Research/nnMamba/regression

echo "=== waiting for attn1301: $(date) ==="
while tmux has-session -t attn1301 2>/dev/null; do sleep 120; done
echo "=== attn1301 ended: $(date) ==="
if ! grep -q "RUNATTNPOOL1301_EXIT=0" "$D/attn1301.log"; then
    echo "attn1301 did not finish cleanly; not starting the guided arm" >&2
    exit 1
fi

for f in networks/hybrid_mamba_guided_attnpool_regressor.py models.py \
         scripts/train_ratio_regression_guided.py \
         experiments/ratio5_20260918/run_guided_1301.py; do
    cp "$D/$f" "$W/$f"
    echo "synced $f"
done
# Existing code must be unchanged in content between the two copies.
for f in networks/hybrid_mamba_attention_regressor.py networks/hybrid_mamba_attnpool_regressor.py \
         core/config.py scripts/train_fixed_holdout_ratio_regression.py; do
    diff <(tr -d '\r' < "$D/$f") <(tr -d '\r' < "$W/$f") > /dev/null \
        || { echo "content of $f differs between D: and WSL; stopping" >&2; exit 1; }
    echo "same content: $f"
done

cd "$W"
python -u "$W/experiments/ratio5_20260918/run_guided_1301.py"
echo "CHAINGUIDED_EXIT=0"
echo "finished: $(date)"
