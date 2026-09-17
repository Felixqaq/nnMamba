#!/usr/bin/env bash
# One COPDxNet member, 80 epochs, batch 2 -- a direction check before committing
# the ~20 hours the full calibrated pipeline would take. AUC is threshold-free,
# so a single member already says whether this architecture is competitive here.
set -o pipefail
source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /home/felix/Research/nnMamba/regression || exit 1

WIN=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_official200_20260831
OUT=/home/felix/Research/nnMamba/regression/outputs/copdxnet_probe_20260903
mkdir -p "$OUT"

echo "start: $(date)"
python -u scripts/train_fixed_holdout_ensemble.py \
    --config config.rq1.normal_v_abnormal.image.fev1fvc70.copdxnet.yaml \
    --split-json "$WIN/split.json" \
    --source-dir /home/felix/Research/nnMamba/classification/datasets/doctor_validation_official200_pft \
    --manifest "$WIN/cohort_manifest.json" \
    --out "$OUT" \
    --members 1 --epochs 80 --base-seed 72
echo "COPDXNET_PROBE_EXIT=$?"
echo "end: $(date)"
exec bash
