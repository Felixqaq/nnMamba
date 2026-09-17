#!/usr/bin/env bash
set -eo pipefail
source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
out=/mnt/d/Felix/Hospital/nnMamba/weights/ratio5_expanded_20260915
mkdir -p "$out"
echo "$$" > "$out/runner.pid"
exec python -u /mnt/d/Felix/Hospital/nnMamba/regression/experiments/ratio5_20260915/run.py >> "$out/runner.log" 2>&1