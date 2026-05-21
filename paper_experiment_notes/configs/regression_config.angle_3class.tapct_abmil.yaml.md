# Config: regression/config.angle_3class.tapct_abmil.yaml

- Source: [regression/config.angle_3class.tapct_abmil.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.tapct_abmil.yaml)
- Size: 2.4 KB

## Parsed Summary
| field | value |
| --- | --- |
| task | Angle_3class_classification |
| model.name | tapct_abmil |
| data.target_mode | angle_3class |
| training.epochs | 120 |
| training.batch_size | 12 |
| training.k_folds | 5 |
| training.learning_rate | 0.0003 |
| training.loss | auto |
| data.balanced_sampling | true |
| data.augmentation.enabled | false |
| experiment.name | TAP-CT ABMIL pooled S-3D |

## Full YAML
```yaml
# TAP-CT frozen embedding + ABMIL scan-level three-class classification
# =====================================================================
#
# This is the paper-style scan-level classifier head:
# frozen TAP-CT features -> attention-based MIL pooling -> classifier.
# The checked-in TAP-CT bundles currently contain one pooled/CLS-like vector
# per scan, so this config runs single-instance ABMIL out of the box. To run
# true per-window MIL, re-extract TAP-CT with --save-window-embeddings and set
# data.tapct_feature_key to window_embeddings.

experiment:
  name: TAP-CT ABMIL pooled S-3D
  description: Frozen TAP-CT-S scan embeddings are classified with a gated attention-based MIL head; CT volumes are not loaded.

model:
  name: tapct_abmil
  num_classes: 3
  hidden_dim: 128
  dropout: 0.25
  tapct_embedding_dim: 1152
  tapct_attention_dim: 128
  tapct_gated_attention: true

training:
  epochs: 120
  batch_size: 12
  eval_batch_size: 12
  swin_batch_size: 12
  swin_eval_batch_size: 12
  learning_rate: 0.0003
  weight_decay: 0.001
  k_folds: 5
  eval_interval: 5
  save_interval: 10
  seed: 42
  loss: auto
  clip_grad_norm: 1.0
  amp: false
  track_train_metrics: false
  class_weight_mode: balanced

early_stopping:
  enabled: true
  patience: 8
  min_delta: 0.005

data:
  target_mode: angle_3class
  source_dir: ../by_angle_all
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json
  angle_split_manifest: ../by_angle_all/reclassification_manifest.json
  manifest: ./datasets/generated/angle_3class_manifest.tapct_abmil.json
  tapct_features: ./embeddings/tapct_s_3d/features.npz
  tapct_feature_key: features
  tapct_allow_single_instance_fallback: false
  load_ct: false
  image_size: [112, 136, 112]
  intensity_window: [-1000.0, 400.0]
  input_normalization: zscore
  target_normalization: none
  cache_data: true
  num_workers: 0
  pin_memory: true
  prefetch_factor: 2
  angle_bin_count: 5
  balanced_sampling: true
  augmentation:
    enabled: false
    balance_to_majority: false
    target_per_class: null
    probability: 0.0
    class_indices: [0, 1, 2]
    rotation_degrees: 0.0
    translation_fraction: 0.0
    scale_range: [1.0, 1.0]
    intensity_scale_range: [1.0, 1.0]
    intensity_shift_range: [0.0, 0.0]
    noise_std: 0.0

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
