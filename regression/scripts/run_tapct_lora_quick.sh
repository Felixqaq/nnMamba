#!/usr/bin/env bash
# Sub-hour TAP-CT LoRA pilot on the exact official 512/200 split.
set -eEo pipefail

source /home/felix/miniconda3/etc/profile.d/conda.sh
conda activate nnMamba

REPO=/home/felix/Research/nnMamba
WINDOWS_REPO=/mnt/d/Felix/Hospital/nnMamba
OUTPUT="$WINDOWS_REPO/regression/outputs/tapct_lora_quick_official200_20260903"
CACHE="$REPO/regression/outputs/tapct_lora_quick_official200_20260903/cache"
LOG="$OUTPUT/tapct_lora_quick.log"

mkdir -p "$OUTPUT" "$CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u "$WINDOWS_REPO/regression/scripts/tapct_lora_quick.py" \
  --metadata-csv "$REPO/regression/embeddings/tapct_doctor_validation_official200/metadata.csv" \
  --split-json "$WINDOWS_REPO/regression/outputs/doctor_validation_official200_20260831/split.json" \
  --pft-csv /mnt/d/Felix/Hospital/copd_dataset/PFT_JPG/fev1_fvc.csv \
  --cache-dir "$CACHE" \
  --output-dir "$OUTPUT" \
  --epochs 6 \
  --cached-windows 4 \
  --sampled-windows 2 \
  --last-blocks 4 \
  --rank 8 \
  --lora-alpha 16 \
  --gradient-accumulation 8 \
  2>&1 | tee -a "$LOG"
