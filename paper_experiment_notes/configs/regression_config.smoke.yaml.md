# Config: regression/config.smoke.yaml

- Source: [regression/config.smoke.yaml](/home/felix/Research/nnMamba/regression/config.smoke.yaml)
- Size: 1.6 KB

## Parsed Summary
| field | value |
| --- | --- |
| task | PFT_angle_regression |
| model.name | hybrid_mamba_attention |
| data.target_mode | angle |
| training.epochs | 15 |
| training.batch_size | 12 |
| training.k_folds | 2 |
| training.learning_rate | 0.0001 |
| training.loss | auto |
| data.balanced_sampling |  |
| data.augmentation.enabled |  |
| experiment.name |  |

## Full YAML
```yaml
# Quick smoke config for validating regression training stability.

model:
  name: hybrid_mamba_attention  # options: mamba | hybrid_mamba_attention | swinunetr
  in_channels: 1
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
  epochs: 15
  batch_size: 12
  eval_batch_size: 12
  swin_batch_size: 4
  swin_eval_batch_size: 5
  learning_rate: 0.0001
  weight_decay: 0.001
  k_folds: 2
  eval_interval: 5
  save_interval: 5
  seed: 42
  loss: auto
  clip_grad_norm: 1.0
  amp: true
  track_train_metrics: false

data:
  target_mode: angle  # angle | gold | angle_3class | angle_binary_extreme
  source_dir: ../by_angle_all
  labels_json: ../patient_angle_classification_by_group.json
  pft_json: ../pft.json
  angle_split_manifest: ../by_angle_all/reclassification_manifest.json
  manifest: ./datasets/generated/regression_manifest.smoke.json
  image_size: [112, 136, 112]
  intensity_window: [-1000.0, 400.0]
  input_normalization: zscore
  target_normalization: zscore
  cache_data: true
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
  angle_bin_count: 5

paths:
  weights: ./weights
  logs: ./train_log
  figures: ./figures
  graphs: ./graphs

task: PFT_angle_regression  # change to GOLD_stage_classification for GOLD runs

resume:
  enabled: false
  uuid: null
  start_fold: 0

gpu:
  device_id: "0"

```
