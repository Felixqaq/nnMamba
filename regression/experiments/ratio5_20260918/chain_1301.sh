#!/usr/bin/env bash
# Retrain the deployment recipe (mask-free 2-channel, five seeds) on the
# 1301-patient cohort, once the conversion and the attention-pooling experiment
# have both finished. The attention run keeps its 991-patient data so that its
# paired comparison stays clean; only then does the GPU move to the new cohort.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
W=/home/felix/Research/nnMamba/regression
LOG=/mnt/d/Felix/Hospital/nnMamba/regression/convert_20260929b.log

echo "=== waiting for conversion and chainattn: $(date) ==="
while tmux has-session -t convert 2>/dev/null || tmux has-session -t chainattn 2>/dev/null; do
    sleep 120
done
echo "=== both finished: $(date) ==="
tail -c 400 "$LOG" | tr '\r' '\n' | grep -E "converted=|on disk"
if ! grep -q "failed=0" "$LOG"; then
    echo "conversion did not report failed=0; not training" >&2
    exit 1
fi
cd "$W"
python -u "$W/experiments/ratio5_20260918/run_1301_maskfree.py"
echo "CHAIN1301_EXIT=0"
echo "finished: $(date)"
