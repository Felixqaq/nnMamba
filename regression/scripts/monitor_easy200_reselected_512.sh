#!/usr/bin/env bash
set -eo pipefail

name=doctor_validation_easy200_reselected_20260902
windows_output="/mnt/d/Felix/Hospital/nnMamba/regression/outputs/$name"
run_output="/home/felix/Research/nnMamba/regression/outputs/$name"
log="$run_output/easy200_reselected_experiment.log"
status="$windows_output/training_status.txt"
mamba="$run_output/mamba5_training_oof_calibrated_fixed80/mamba5_holdout_predictions.json"
qct="$run_output/qct/qct_features.csv"

while true; do
    if [[ -f "$qct" ]]; then
        state=READY_FOR_EXCEL
    elif tmux has-session -t =easy200r512 2>/dev/null; then
        if [[ -f "$mamba" ]]; then
            state=RUNNING_QCT
        else
            state=RUNNING_MAMBA
        fi
    else
        state=FAILED_OR_STOPPED
    fi

    latest="waiting for first progress line"
    if [[ -f "$log" ]]; then
        match=$(tr '\r' '\n' < "$log" | grep -E '\[[123]/3\]|calibration fold|full member|seed=[0-9]+ epoch=[0-9]+/80 loss=|segmenting|wrote ' | tail -n 5 || true)
        if [[ -n "$match" ]]; then
            latest="$match"
        fi
    fi
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "unavailable")
    {
        echo "state=$state"
        echo "updated=$(date '+%Y-%m-%d %H:%M:%S %Z')"
        echo "estimated_ready_for_excel=2026-09-02 23:00 to 2026-09-03 00:30 CST"
        echo "split=512 training / 200 deliberately easy validation"
        echo "gpu_util_percent,memory_mib,temperature_c=$gpu"
        echo "latest_progress:"
        echo "$latest"
    } > "$status.tmp"
    mv "$status.tmp" "$status"

    if [[ "$state" != RUNNING_MAMBA && "$state" != RUNNING_QCT ]]; then
        exit 0
    fi
    sleep 60
done
