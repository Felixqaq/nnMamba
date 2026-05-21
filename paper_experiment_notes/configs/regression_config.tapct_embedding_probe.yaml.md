# Config: regression/config.tapct_embedding_probe.yaml

- Source: [regression/config.tapct_embedding_probe.yaml](/home/felix/Research/nnMamba/regression/config.tapct_embedding_probe.yaml)
- Size: 1.1 KB

## Parsed Summary
| field | value |
| --- | --- |
| task |  |
| model.name |  |
| data.target_mode |  |
| training.epochs |  |
| training.batch_size |  |
| training.k_folds |  |
| training.learning_rate |  |
| training.loss |  |
| data.balanced_sampling |  |
| data.augmentation.enabled |  |
| experiment.name | tapct_b_3d_embedding_probe |

## Full YAML
```yaml
# TAP-CT frozen embedding + sklearn probe 一鍵流程
#
# 用法：
#   conda activate tapct
#   python regression/scripts/run_tapct_embedding_probe.py \
#     --config regression/config.tapct_embedding_probe.yaml

experiment:
  name: tapct_b_3d_embedding_probe

run:
  extract_embeddings: true
  train_probe: true

tapct:
  model_id: fomofo/tap-ct-b-3d
  device: cuda
  # TAP-CT-B 建議使用 float32；fp16 在 smoke run 中曾出現 NaN。
  dtype: float32

data:
  source_root: by_angle_all
  labels_json: patient_angle_classification_by_group.json
  pft_json: pft.json

embedding:
  output_dir: regression/embeddings/tapct_b_3d
  depth_window: 12
  depth_stride: 6
  sw_batch_size: 1
  pooling: mean_std_max
  max_cases: null
  patient_ids: []
  force: false
  save_window_embeddings: false

probe:
  # features 預設會使用 embedding.output_dir/features.npz；通常不用改。
  features: null
  metadata: null
  output_dir: regression/figures/TAPCT_embedding_probes/tapct_b_3d_yaml
  target: all
  model: all
  n_splits: 5
  seed: 42
  ridge_alpha: 1.0
  plots: true
  plot_dpi: 180

```
