# Config: regression/config.gold.yaml

- Source: [regression/config.gold.yaml](/home/felix/Research/nnMamba/regression/config.gold.yaml)
- Size: 2.3 KB

## Parsed Summary
| field | value |
| --- | --- |
| task | GOLD_stage_classification |
| model.name | hybrid_mamba_attention |
| data.target_mode | gold |
| training.epochs | 160 |
| training.batch_size | 12 |
| training.k_folds | 3 |
| training.learning_rate | 0.0001 |
| training.loss | auto |
| data.balanced_sampling | true |
| data.augmentation.enabled | true |
| experiment.name | GOLD balanced sampling + aug200/class |

## Full YAML
```yaml
# nnMamba GOLD Stage Classification Configuration
# ===============================================

experiment:
  name: GOLD balanced sampling + aug200/class
  description: Virtual train-fold augmentation brings all four GOLD classes to 200 samples before per-epoch balanced sampling.

model:
  name: hybrid_mamba_attention  # options: mamba | hybrid_mamba_attention | swinunetr
  in_channels: 1
  num_classes: 4
  hidden_dim: 128
  dropout: 0.3
  base_channels: 32
  blocks: 3
  attn_heads: 8
  attn_layers: 1
  attn_mlp_ratio: 2.0
  attn_dropout: 0.1

  # SwinUNETR-only settings. Keep these here so switching models only needs
  # changing `name`.
  feature_size: 24
  depths: [2, 2, 2, 2]
  num_heads: [3, 6, 12, 24]
  window_size: 4
  patch_size: 2
  use_checkpoint: false
  use_v2: true

training:
  epochs: 160
  batch_size: 12
  eval_batch_size: 12
  swin_batch_size: 5
  swin_eval_batch_size: 6
  learning_rate: 0.0001
  weight_decay: 0.001
  k_folds: 3
  eval_interval: 5
  save_interval: 10
  seed: 42
  loss: auto
  clip_grad_norm: 1.0
  amp: true
  track_train_metrics: false
  class_weight_mode: none

early_stopping:
  enabled: true
  patience: 6
  min_delta: 0.005

data:
  target_mode: gold
  source_dir: ../by_angle_all
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json
  angle_split_manifest: ../by_angle_all/reclassification_manifest.json
  manifest: ./datasets/generated/gold_manifest.json
  image_size: [112, 136, 112]
  intensity_window: [-1000.0, 400.0]
  input_normalization: zscore
  target_normalization: none
  cache_data: true
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
  angle_bin_count: 5
  balanced_sampling: true
  augmentation:
    enabled: true
    balance_to_majority: false
    target_per_class: 100
    probability: 1.0
    # GOLD stages are one-based; include all stages so every train fold reaches 200/class.
    gold_stages: [1, 2, 3, 4]
    rotation_degrees: 7.0
    translation_fraction: 0.05
    scale_range: [0.95, 1.05]
    intensity_scale_range: [0.95, 1.05]
    intensity_shift_range: [-0.1, 0.1]
    noise_std: 0.03

paths:
  weights: ./weights
  logs: ./train_log
  figures: ./figures
  graphs: ./graphs

task: GOLD_stage_classification

resume:
  enabled: false
  uuid: null
  start_fold: 0

gpu:
  device_id: "0"

```
