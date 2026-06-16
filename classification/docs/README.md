# nnMamba Classification

Medical image classification using nnMamba, DenseNet, ViT, and CRATE architectures.

## Quick Start

```bash
# Edit config.yaml to set your task and hyperparameters
python train.py

# Evaluate a trained model and generate Grad-CAM figures by default
python evaluate.py --uuid nnMamba_2026-01-21_14:30:00
```

Training automatically writes Grad-CAM overlays for each fold's best model to:

```
../figures/{task}/{uuid}/gradcam/fold{N}/
```

Use `python train.py --no-gradcam` or set `gradcam.enabled: false` for a
metrics-only training run.

Evaluation also writes Grad-CAM overlays to:

```
../figures/{task}/{uuid}/gradcam/
```

Use `--no-gradcam` for a metrics-only evaluation. Use `--gradcam-layer` to
override the target layer with any name from `model.named_modules()`.

## Configuration

All settings are in `config.yaml`:

```yaml
model:
  name: nnmamba    # nnmamba | densenet | vit | crate

training:
  epochs: 50
  batch_size: 12
  k_folds: 5

gradcam:
  enabled: true
  max_samples: 8

task: Normal_v_Abnormal  # NC_v_AD | sMCI_v_pMCI | Normal_v_COPD | Normal_v_Abnormal
```

## Project Structure

```
classification/
├── config.yaml          # Configuration file
├── train.py             # Training entry point
├── evaluate.py          # Evaluation entry point
├── models.py            # Model registry
│
├── core/                # Training modules
│   ├── config.py        # Config loader
│   ├── trainer.py       # Training loop
│   ├── evaluator.py     # Metrics
│   ├── checkpoints.py   # Save/load weights
│   └── visualizer.py    # Training plots
│
├── data/                # Data handling
│   ├── dataset.py       # MRIDataset class
│   ├── loader.py        # DataLoader helper
│   └── transforms.py    # Data transforms
│
├── networks/            # Model architectures
│   ├── ssm_nnMamba.py
│   ├── conv_Densenet121.py
│   ├── tr_ViT.py
│   └── tr_crate.py
│
└── scripts/             # Utility scripts
    ├── setup_copd_dataset.py
    ├── check_image_sizes.py
    └── test_dataset.py
```

## Dataset Setup

### Normal vs Abnormal (COPD)

Place your data in the parent directory:
```
nnMamba/
├── Normal/       # Normal CT scans (.nii or .nii.gz)
├── Abnormal/     # COPD CT scans
└── classification/
```

### ADNI Dataset

```
classification/datasets/
├── adni1/
│   ├── NC/
│   └── AD/
└── adni2/
    ├── NC/
    └── AD/
```

## Output

Training produces:
- **Weights**: `../weights/{task}/{uuid}/best_weight.pth`
- **Logs**: `../train_log/{uuid}.txt`
- **Figures**: `../figures/{uuid}_fold{N}_*.png`
- **Grad-CAM**: `../figures/{task}/{uuid}/gradcam/fold{N}/*.png`

## Troubleshooting

**GPU not detected**
```bash
nvidia-smi  # Check GPU
conda activate nnMamba
```

**Out of memory**
Edit `config.yaml`:
```yaml
training:
  batch_size: 4  # Reduce batch size
```
