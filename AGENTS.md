# Repository Guidelines

## Project Structure & Module Organization
`classification/` is the main maintained training pipeline for image classification. Keep reusable training and evaluation code in `classification/core/`, dataset and transform logic in `classification/data/`, architecture implementations in `classification/networks/`, and model selection in `classification/models.py`. Root-level `nnMamba.py` and `nnMamba4cls.py` hold core model definitions, while `nnunet/` contains the segmentation and landmark-detection stack. Generated artifacts belong in gitignored directories such as `weights/`, `train_log/`, `figures/`, and `graphs/`. Large medical-image inputs stay outside version control in folders such as `Normal/`, `Abnormal/`, and `classification/datasets/`.

## Build, Test, and Development Commands
Work from `classification/` when developing the classifier:

```bash
cd classification
python train.py --config config.yaml
python evaluate.py --uuid <run_id> --config config.yaml
python scripts/setup_copd_dataset.py
python scripts/check_image_sizes.py
```

`train.py` runs k-fold training, `evaluate.py` scores a saved checkpoint, `setup_copd_dataset.py` builds the COPD train/test split, and `check_image_sizes.py` verifies NIfTI dimensions. There is no separate build step.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, module docstrings, and type hints for new code. Follow the existing naming style: `snake_case` for functions, files, and YAML keys; `PascalCase` for classes; and task labels such as `NC_v_AD` and `Normal_v_Abnormal` exactly as defined in config and enums. Prefer extending the modular packages under `classification/core/` or `classification/data/` instead of adding one-off scripts at the repo root.

## Testing Guidelines
There is no formal `pytest` suite checked in today. Before opening a PR, run a smoke test on the affected path: at minimum, a short training or evaluation run that proves the code executes and writes outputs to the expected artifact directories. For data changes, validate directory layout and sample shapes with the scripts in `classification/scripts/`. Name new executable validation helpers `test_<subject>.py`.

## Commit & Pull Request Guidelines
Recent history uses short imperative commit subjects with prefixes like `feat:`, `fix:`, `refactor:`, and `chore:`. Keep each commit focused on one logical change. PRs should note the dataset or task affected, any `config.yaml` changes, and the expected impact on metrics or generated artifacts. Include representative plots or confusion matrices when changing training, evaluation, or visualization behavior, and never commit datasets, checkpoints, or generated figures.
