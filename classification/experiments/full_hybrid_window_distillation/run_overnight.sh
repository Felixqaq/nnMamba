#!/usr/bin/env bash
set -eo pipefail
artifact=/mnt/d/Felix/Hospital/nnMamba/weights/window_distillation
exec >> "$artifact/hybrid_full777_fp32.log" 2>&1
echo "Runner started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "$$" > "$artifact/hybrid_full777_fp32.pid"
source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /mnt/d/Felix/Hospital/nnMamba/classification
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec python -u -m experiments.full_hybrid_window_distillation.run \
  --manifest ../weights/window_distillation/cohort777/patients.csv \
  --source-summary /home/felix/Research/nnMamba/classification/datasets/normal_v_abnormal_fev1fvc70/build_summary.json \
  --output ../weights/window_distillation/hybrid_full777_fp32 \
  --hours "${1:-23}"
