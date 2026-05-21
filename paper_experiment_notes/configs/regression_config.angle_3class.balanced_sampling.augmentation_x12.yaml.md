# Config: regression/config.angle_3class.balanced_sampling.augmentation_x12.yaml

- Source: [regression/config.angle_3class.balanced_sampling.augmentation_x12.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.augmentation_x12.yaml)
- Size: 2.5 KB

## Parsed Summary
| field | value |
| --- | --- |
| task | Angle_3class_classification |
| model.name | hybrid_mamba_attention |
| data.target_mode | angle_3class |
| training.epochs | 160 |
| training.batch_size | 12 |
| training.k_folds | 5 |
| training.learning_rate | 0.0001 |
| training.loss | auto |
| data.balanced_sampling | true |
| data.augmentation.enabled | true |
| experiment.name | Balanced sampling + augx12/epoch |

## Full YAML
```yaml
# nnMamba Angle Three-Class Classification with Balance-Then-Augment x12
# =====================================================================

experiment:
  name: Balanced sampling + augx12/epoch
  description: Each epoch first undersamples all classes to the minority count, then uses 12 total views per selected sample.

model:
  name: hybrid_mamba_attention  # options: mamba | hybrid_mamba_attention | swinunetr
  in_channels: 1
  num_classes: 3
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
  swin_batch_size: 8
  swin_eval_batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.001
  k_folds: 5
  eval_interval: 5
  save_interval: 10
  seed: 42
  loss: auto
  clip_grad_norm: 1.0
  amp: false
  track_train_metrics: false
  class_weight_mode: none

early_stopping:
  enabled: true
  patience: 6
  min_delta: 0.005

data:
  target_mode: angle_3class
  source_dir: ../by_angle_all
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json
  angle_split_manifest: ../by_angle_all/reclassification_manifest.json
  manifest: ./datasets/generated/angle_3class_manifest.json
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
    # First sample a balanced epoch base set, then expand each selected sample
    # to 12 total views: 1 original view + 11 random augmented views.
    balance_then_augment: true
    views_per_sample: 12
    probability: 1.0
    # Zero-based angle_3class labels: 0=Abnormal, 1=Intermediate, 2=Normal.
    class_indices: [0, 1, 2]
    rotation_degrees: 5.0
    translation_fraction: 0.03
    scale_range: [0.97, 1.03]
    intensity_scale_range: [0.98, 1.02]
    intensity_shift_range: [-0.05, 0.05]
    noise_std: 0.02

paths:
  weights: ./weights
  logs: ./train_log
  figures: ./figures
  graphs: ./graphs

task: Angle_3class_classification

resume:
  enabled: false
  uuid: null
  start_fold: 0

gpu:
  device_id: "0"

```
