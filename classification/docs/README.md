# nnMamba Classification

Medical image classification using nnMamba, DenseNet, ViT, and CRATE architectures.

## Quick Start

```bash
# Edit config.yaml to set your task and hyperparameters
python train.py

# Evaluate a trained model
python evaluate.py --uuid nnMamba_2026-01-21_14:30:00
```

## Configuration

All settings are in `config.yaml`:

```yaml
model:
  name: nnmamba    # nnmamba | densenet | vit | crate

training:
  epochs: 50
  batch_size: 12
  k_folds: 5

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
