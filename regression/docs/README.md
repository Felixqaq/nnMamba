# nnMamba Regression

This folder is for the CT regression pipeline that predicts the PFT collapse angle from CT.

## Environment

Use the existing conda environment:

```bash
conda activate nnMamba
```

If you prefer non-interactive execution:

```bash
conda run -n nnMamba python --version
```

## Source Data

The regression dataset is already organized by angle group under:

```text
by_angle_all/
├── abnormal_low_angle/
└── normal_high_angle/
```

These folder names are legacy names. For regression, the actual target semantics come from `patient_angle_classification_by_group.json`, especially the per-patient `low_angle` / `high_angle` labels and exact angle values.

Each CT filename is matched to its patient angle using:

```text
patient_angle_classification_by_group.json
```

## Helper Scripts

Run these from the repository root:

```bash
conda run -n nnMamba python regression/scripts/build_manifest.py
conda run -n nnMamba python regression/scripts/check_dataset.py
conda run -n nnMamba python regression/scripts/plot_dataset_overview.py
conda run -n nnMamba python regression/scripts/summarize_results.py regression/figures/<run_uuid>/results.json
```

The defaults are:

```text
source-root: ./by_angle_all
angle-json: ./patient_angle_classification_by_group.json
output-dir: ./regression/datasets/generated
```

## What These Scripts Do

`build_manifest.py` creates a JSON manifest with patient id, CT path, angle group, and regression target.

`check_dataset.py` verifies label coverage and reports the CT shape and angle range.

`plot_dataset_overview.py` creates paper-style overview figures for the regression dataset, including a histogram, boxplot, and summary table.

`summarize_results.py` reads a finished `results.json` and creates a summary CSV plus paper-style metric plots.

## Training

Run training from inside `regression/`:

```bash
cd regression
conda run -n nnMamba python train.py --config config.yaml
```

The loader will scan `../by_angle_all`, match patient IDs against `../patient_angle_classification_by_group.json`, and auto-save a manifest to `./datasets/generated/regression_manifest.json`.

## Evaluation

```bash
cd regression
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --fold 1 --config config.yaml
```

## Outputs

Training writes:

- `regression/weights/PFT_angle_regression/<run_uuid>/`
- `regression/train_log/PFT_angle_regression/<run_uuid>/`
- `regression/figures/PFT_angle_regression/<run_uuid>/`

Per-fold figures include loss/MAE/RMSE/R2/Pearson curves plus scatter, residual, error histogram, and Bland-Altman plots. Global figures aggregate all folds into `total_*.png`.
