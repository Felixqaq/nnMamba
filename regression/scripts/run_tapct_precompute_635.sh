#!/usr/bin/env bash
set -eo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba
cd /home/felix/Research/nnMamba

output=regression/embeddings/tapct_doctor_validation_augmented635
log=regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/tapct_embedding.log
mkdir -p "$output/cases" "$(dirname "$log")"
cp -an regression/embeddings/tapct_doctor_validation_430/cases/. "$output/cases/"

python -u regression/scripts/extract_tapct_embeddings.py \
    --source-root classification/datasets/doctor_validation_augmented635_pft \
    --output-dir "$output" \
    --target-mode normal_v_abnormal \
    --device cuda \
    --dtype float16 \
    --depth-window 12 \
    --depth-stride 6 \
    --sw-batch-size 4 \
    --pooling mean_std_max \
    2>&1 | tee "$log"
