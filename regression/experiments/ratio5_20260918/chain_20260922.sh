#!/usr/bin/env bash
# 2026-09-22 chain: conversion -> 1149-patient seed baseline -> CV + OOF cutoff
# -> airway channel. Every stage is resumable; rerunning this script after a
# crash skips whatever already finished.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
REG=/home/felix/Research/nnMamba/regression
EXP=$REG/experiments/ratio5_20260918
LOG=/mnt/d/Felix/Hospital/nnMamba/regression

echo "=== waiting for conversion: $(date) ==="
while tmux has-session -t convert 2>/dev/null; do sleep 60; done
tail -n 3 "$LOG/convert_20260922.log" | tr '\r' '\n' | tail -3
if ! grep -q "failed=0" "$LOG/convert_20260922.log"; then
    echo "conversion did not report failed=0; stopping before training" >&2
    exit 1
fi

cd "$REG"
echo "=== [1/3] 1149-patient seed baseline: $(date) ==="
python -u "$EXP/run_1149.py"
echo "=== [2/3] CV ensemble + OOF cutoff: $(date) ==="
python -u "$EXP/run_cv.py"
echo "=== [3/3] airway channel: $(date) ==="
python -u "$EXP/run_airway.py"
echo "CHAIN20260922_EXIT=0"
echo "finished: $(date)"
