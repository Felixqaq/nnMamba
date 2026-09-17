#!/usr/bin/env bash
set -o pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

repo=/home/felix/Research/nnMamba
log="$repo/regression/outputs/doctor_validation_augmented208_easy200_20260829_build.log"
snapshot=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented208_easy200_20260829/fev1_fvc_frozen.csv

cd "$repo" || exit 1
python -u regression/scripts/build_fev1fvc70_dataset.py \
    --csv "$snapshot" \
    --cohorts copd117 \
    --workers 3 \
    2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "AUGMENT_BUILD_EXIT=$status" | tee -a "$log"
exit "$status"
