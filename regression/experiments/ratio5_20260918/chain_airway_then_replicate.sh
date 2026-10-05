#!/usr/bin/env bash
# 1. airway-wall analysis (TotalSegmentator on 200 training patients, ~2 h GPU)
# 2. replication of guided vs average pooling with seeds 77-81 (~16 h GPU)
# Sequential because each needs most of the 8 GB card. The analysis failing does
# not block the replication; it only loses the analysis.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
W=/home/felix/Research/nnMamba/regression
EXP=$W/experiments/ratio5_20260918
cd "$W"

echo "=== [1/2] airway wall analysis: $(date) ==="
python -u "$EXP/airway_wall_value.py" && echo "AIRWAY_EXIT=0" || echo "AIRWAY_FAILED"

echo "=== [2/2] replication seeds 77-81: $(date) ==="
python -u "$EXP/run_replicate_s77.py" && echo "REPLICATE_EXIT=0" || echo "REPLICATE_FAILED"
echo "finished: $(date)"
