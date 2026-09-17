#!/usr/bin/env bash
set -eo pipefail

log=/home/felix/Research/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830/mamba_calibrated.log
output=/mnt/d/Felix/Hospital/nnMamba/regression/outputs/doctor_validation_augmented_ctqa_easy200_20260830
status="$output/training_status.txt"
excel="$output/doctor_validation_augmented206_easy200_calibrated.xlsx"

while true; do
    latest="waiting for the first epoch summary"
    if [[ -f "$log" ]]; then
        match=$(tr '\r' '\n' < "$log" | grep -E 'calibration fold|full-training member|seed=[0-9]+ epoch=[0-9]+/80 loss=|wrote ' | tail -n 3 || true)
        if [[ -n "$match" ]]; then
            latest="$match"
        fi
    fi
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "unavailable")
    state=RUNNING
    if [[ -f "$excel" ]]; then
        state=COMPLETE
    elif ! tmux has-session -t aug635cal 2>/dev/null; then
        state=FAILED_OR_STOPPED
    fi
    {
        echo "state=$state"
        echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "estimated_finish=2026-08-30 20:45-21:30 CST"
        echo "gpu_util_percent,memory_mib,temperature_c=$gpu"
        echo "latest_progress:"
        echo "$latest"
    } > "$status.tmp"
    mv "$status.tmp" "$status"
    if [[ "$state" != RUNNING ]]; then
        exit 0
    fi
    sleep 60
done
