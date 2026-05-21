# Reproducibility Checklist

## Environment

- AGENTS.md 要求工作前使用 conda env: `nnMamba`。
- 互動 shell 若 `conda activate nnMamba` 失敗，可用 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate nnMamba`。
- regression TAP-CT 相關環境也有 `regression/environment.tapct.yml` 可查。

## Core Commands

```bash
cd /home/felix/Research/nnMamba/classification
python train.py --config config.yaml
python evaluate.py --uuid <run_id> --config config.yaml
```

```bash
cd /home/felix/Research/nnMamba/regression
python train.py --config config.yaml
python train.py --config config.angle_3class.balanced_sampling.augmentation100.yaml
python evaluate.py --uuid <run_id> --config <matching_config.yaml>
python scripts/summarize_results.py figures/<task>/<run_id>/results.json
```

## Before A Table Goes Into The Paper

- Confirm the row points to a run page under `runs/` and that its `results.json` is embedded.
- Confirm the task/target definition matches the paper text.
- For classification, report mean ± std across folds for macro-F1, accuracy, balanced accuracy, and include confusion matrix when useful.
- For regression, report mean ± std across folds for MAE/RMSE/R2/Pearson.
- If using augmented or TAP-CT fusion results, cite the exact config and embedding feature bundle from the run page.
- `has_results=false` artifact folders in `07_artifact_inventory.md` should not be used as final quantitative evidence unless manually reconstructed from logs.
