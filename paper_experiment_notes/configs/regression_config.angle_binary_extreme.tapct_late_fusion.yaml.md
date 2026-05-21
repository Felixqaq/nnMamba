# Config: regression/config.angle_binary_extreme.tapct_late_fusion.yaml

- Source: [regression/config.angle_binary_extreme.tapct_late_fusion.yaml](/home/felix/Research/nnMamba/regression/config.angle_binary_extreme.tapct_late_fusion.yaml)
- Size: 2.4 KB

## Parsed Summary
| field | value |
| --- | --- |
| task | Angle_extreme_binary_classification |
| model.name | hybrid_mamba_tapct_fusion |
| data.target_mode | angle_binary_extreme |
| training.epochs | 160 |
| training.batch_size | 12 |
| training.k_folds | 5 |
| training.learning_rate | 0.0001 |
| training.loss | auto |
| data.balanced_sampling | true |
| data.augmentation.enabled | true |
| experiment.name | TAP-CT late fusion extreme binary + aug100/class |

## Full YAML
```yaml
# nnMamba Angle Extreme Binary Classification with TAP-CT Late Fusion
# ===================================================================

experiment:
  name: TAP-CT late fusion extreme binary + aug100/class
  description: Gray-zone excluded binary classification using Hybrid Mamba-Attention CT features fused with frozen TAP-CT-S patient-level embeddings.

model:
  name: hybrid_mamba_tapct_fusion
  in_channels: 1
  num_classes: 2
  hidden_dim: 256
  dropout: 0.3
  base_channels: 32
  blocks: 3
  attn_heads: 8
  attn_layers: 1
  attn_mlp_ratio: 2.0
  attn_dropout: 0.1
  tapct_embedding_dim: 1152
  fusion_projection_dim: 128
  fusion_dropout: 0.1

  # SwinUNETR-only settings. Kept for consistency with other configs.
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
  target_mode: angle_binary_extreme
  source_dir: ../by_angle_all
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json
  angle_split_manifest: ../by_angle_all/reclassification_manifest.json
  manifest: ./datasets/generated/angle_binary_extreme_manifest.tapct_late_fusion.json
  tapct_features: ./embeddings/tapct_s_3d/features.npz
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
    # Zero-based angle_binary_extreme labels: 0=Abnormal/emphysema-like, 1=Normal-like.
    class_indices: [0, 1]
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

task: Angle_extreme_binary_classification

resume:
  enabled: false
  uuid: null
  start_fold: 0

gpu:
  device_id: "0"

```
