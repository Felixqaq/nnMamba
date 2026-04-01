# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nnMamba is a deep learning framework for 3D biomedical image analysis (segmentation, classification, landmark detection) that combines CNNs with State Space Models (Mamba/SSM). The key innovation is the **MICCSS block** (Mamba-In-Convolution with Channel-Spatial Siamese Learning) which models long-range voxel dependencies more efficiently than Transformers.

## Commands

### Classification

```bash
cd classification

# Train (uses config.yaml by default)
python train.py
python train.py --config my_config.yaml

# Evaluate a trained model by UUID
python evaluate.py --uuid nnMamba_2026-01-21_14:30:00
```

### Segmentation (nnU-Net framework)

Requires environment variables set before running:
```bash
export nnUNet_raw_data_base="/path/to/raw_data"
export nnUNet_preprocessed="/path/to/preprocessed"
export RESULTS_FOLDER="/path/to/results"

# Single GPU training
python -m nnunet.run.run_training nnMambaSeg nnUNetTrainerV2 TASK_NAME FOLD

# Multi-GPU (DDP)
python -m nnunet.run.run_training_DDP nnMambaSeg nnUNetTrainerV2 TASK_NAME FOLD

# Validation only
python -m nnunet.run.run_training nnMambaSeg nnUNetTrainerV2 TASK_NAME FOLD -val

# With pretrained weights
python -m nnunet.run.run_training nnMambaSeg nnUNetTrainerV2 TASK_NAME FOLD -pretrained_weights PATH/TO/checkpoint.model
```

### Dataset Validation Scripts
```bash
cd classification
python scripts/test_dataset.py      # Validate image loading, shapes, labels
python scripts/check_image_sizes.py # Verify preprocessing output
```

## Architecture

### Two Parallel Pipelines

**1. `classification/`** — Standalone classification pipeline with its own config, models, data loaders, and training logic. Entry points: `train.py`, `evaluate.py`.

**2. `nnunet/`** — Segmentation pipeline built on the nnU-Net framework. Entry points under `nnunet/run/`.

Standalone model files `nnMamba.py` and `nnMamba4cls.py` in the root are reference implementations not used by either pipeline directly.

### Core Mamba Block (`MambaLayer`)
The building block used in all models:
1. 1×1 Conv projection → BatchNorm + ReLU
2. Flatten spatial dims → `(B, L, C)` sequence
3. Run Mamba SSM in **4 directions** (original + 3 flips over different axes)
4. Average all 4 directional outputs (the "siamese" part)
5. Residual connection + projection back

### Classification Model (`nnMambaEncoder`)
- Stem: DoubleConv (stride=2) → MambaLayer
- Encoder: 3 residual stages, each doubling channels (stride=2 downsampling), interleaved with MambaLayers
- Head: Global avg pool of each encoder stage → concatenate → MLP classifier
- Default: 32 base channels, d_state=8, d_conv=4, expand=2

### Segmentation Model (`nnMambaSeg`)
- U-Net encoder-decoder with MambaLayers at each encoder level
- Skip connections scaled by learned channel-attention weights
- 4 encoder levels + 4 decoder levels with skip connections
- d_state=16, d_conv=4, expand=2

### Model Registry
`classification/models.py` uses a `MODEL_REGISTRY` dict for dynamic instantiation. To add a new model: implement it under `classification/networks/` and register it in `MODEL_REGISTRY`.

## Configuration

`classification/config.yaml` controls all classification experiments:

```yaml
model:
  name: nnmamba    # nnmamba | densenet | vit | crate

task: Normal_v_Abnormal  # NC_v_AD | sMCI_v_pMCI | Normal_v_COPD | Normal_v_Abnormal

training:
  k_folds: 5
  epochs: 50
  batch_size: 12
  learning_rate: 0.0001

data:
  image_size: [112, 136, 112]  # All NIfTI inputs are resized to this

paths:
  weights: ../weights      # Checkpoints saved here
  logs: ../train_log
  figures: ../figures      # Per-run visualizations
  graphs: ../graphs        # Training curves
```

## Training Details

**Classification** uses 5-fold stratified cross-validation. Labels are inferred from parent directory names (e.g., `Normal/` → 0, `Abnormal/` → 1). Models are saved by UUID (`nnMamba_YYYY-MM-DD_HH:MM:SS`). Best checkpoint is selected by AUC.

**Segmentation** follows the standard nnU-Net workflow: experiment planning → preprocessing → 5-fold training → post-processing.

## Key Dependencies

- `mamba-ssm`: The core SSM library — must be installed for any model to run
- `nibabel`: NIfTI file loading
- `torchmetrics`: Standardized metrics (Accuracy, AUC, AUROC)
- `batchgenerators`: Data augmentation for the nnU-Net pipeline
- `scikit-image`: Image resizing in the classification data pipeline
