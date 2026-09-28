#!/usr/bin/env bash
# After chain0922 finishes the seed baseline and the CV run, train the
# segmentation-free arm. The airway stage of chain0922 is paused on purpose
# (2026-09-22, user decision: the airway channel needs a segmentation network at
# inference, which the hospital deployment is meant to avoid): run_airway.py was
# renamed to run_airway.py.paused in the WSL copy, so chain0922's stage 3 exits
# at once with "can't open file" instead of segmenting 1149 patients.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
REG=/home/felix/Research/nnMamba/regression
EXP=$REG/experiments/ratio5_20260918
LOG=/mnt/d/Felix/Hospital/nnMamba/regression/chain_20260922.log

echo "=== waiting for chain0922 (baseline + CV): $(date) ==="
while tmux has-session -t chain0922 2>/dev/null; do sleep 120; done
if ! grep -q "RUNCV_EXIT=0" "$LOG"; then
    echo "chain0922 ended without finishing the CV run; not starting the mask-free arm" >&2
    exit 1
fi
cd "$REG"
echo "=== mask-free arm: $(date) ==="
python -u "$EXP/run_maskfree.py"
echo "CHAINMASKFREE_EXIT=0"
echo "finished: $(date)"
