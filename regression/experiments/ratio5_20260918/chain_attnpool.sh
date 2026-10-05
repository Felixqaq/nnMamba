#!/usr/bin/env bash
# Wait for the masked 1257 run, then bring the attention-pooling code into the
# WSL working copy and run the paired experiment.
#
# Additive only: a new network file (hybrid_mamba_attnpool_regressor.py) plus its
# registration in models.py (17 added lines, none changed). The parent model and
# core/config.py are not copied. models.py is synced only after the masked run
# ends, because each of its seeds imports the model registry when it starts.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate nnMamba
D=/mnt/d/Felix/Hospital/nnMamba/regression
W=/home/felix/Research/nnMamba/regression

echo "=== waiting for masked1257: $(date) ==="
while tmux has-session -t masked1257 2>/dev/null; do sleep 120; done
echo "=== masked1257 ended: $(date) ==="

for f in networks/hybrid_mamba_attnpool_regressor.py models.py \
         experiments/ratio5_20260918/run_attnpool.py \
         experiments/ratio5_20260918/attention_localization.py; do
    cp "$D/$f" "$W/$f"
    echo "synced $f"
done

# The parent model and config must have the same content in both copies, and the
# parent must still load a deployed-recipe checkpoint strictly. Carriage returns
# are ignored: git on Windows rewrites LF as CRLF on checkout, which changes the
# bytes but not the code (2026-09-29, core/config.py: 546 CRLF lines, same text).
for f in networks/hybrid_mamba_attention_regressor.py core/config.py; do
    diff <(tr -d '\r' < "$D/$f") <(tr -d '\r' < "$W/$f") > /dev/null \
        || { echo "content of $f differs between D: and WSL; stopping" >&2; exit 1; }
    echo "same content: $f"
done
cd "$W"
python - <<PY
import sys, torch
sys.path.insert(0, "$W")
from core.config import Config
from models import build_model
c = Config.from_yaml("$W/outputs/ratio5_expanded_20260929_maskfree/config.yaml")
m = build_model(c.model, output_dim=1)
b = torch.load("$W/outputs/ratio5_expanded_20260929_maskfree/seed72/regressor_seed72.pth",
               map_location="cpu", weights_only=False)
m.load_state_dict(b["state_dict"], strict=True)
print("parent model unchanged and loads strictly")
PY

python -u "$W/experiments/ratio5_20260918/run_attnpool.py"
echo "CHAINATTN_EXIT=0"
echo "finished: $(date)"
