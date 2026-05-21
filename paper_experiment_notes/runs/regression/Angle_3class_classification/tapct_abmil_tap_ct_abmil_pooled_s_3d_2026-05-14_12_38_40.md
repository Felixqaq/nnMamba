# Run: tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40

## 一句話用途
- 方法名稱: `TAP-CT ABMIL pooled S-3D`
- 實驗描述: Frozen TAP-CT-S scan embeddings are classified with a gated attention-based MIL head; CT volumes are not loaded.
- 任務: `Angle_3class_classification`
- 模型: `tapct_abmil`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T12:39:30.620319`
- 訓練時間: `2026-05-14T12:38:40.662734` -> `2026-05-14T12:39:30.016636`
- 訓練耗時: 0.0137 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.71319 |
| std_accuracy | 0.08458 |
| mean_macro_f1 | 0.58834 |
| std_macro_f1 | 0.13805 |
| mean_macro_precision | 0.6177 |
| std_macro_precision | 0.1229 |
| mean_macro_recall | 0.64 |
| std_macro_recall | 0.20359 |
| mean_balanced_accuracy | 0.64 |
| std_balanced_accuracy | 0.20359 |

## Total Confusion Matrix
```json
[
  [
    11,
    1,
    2
  ],
  [
    0,
    2,
    3
  ],
  [
    9,
    4,
    34
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 60 | 0.64286 | 0.43333 | 0.6 | 0.37778 | 0.37778 |
| 2 | 45 | 0.61538 | 0.72028 | 0.79167 | 0.81481 | 0.81481 |
| 3 | 70 | 0.76923 | 0.50292 | 0.48889 | 0.51852 | 0.51852 |
| 4 | 95 | 0.69231 | 0.5 | 0.48571 | 0.55556 | 0.55556 |
| 5 | 90 | 0.84615 | 0.78519 | 0.72222 | 0.93333 | 0.93333 |

## Training Config Embedded In Result
```json
{
  "epochs": 120,
  "batch_size": 12,
  "learning_rate": 0.0003,
  "weight_decay": 0.001,
  "k_folds": 5,
  "seed": 42,
  "loss": "auto",
  "num_classes": 3,
  "target_mode": "angle_3class",
  "tapct_features": "embeddings/tapct_s_3d/features.npz",
  "tapct_embedding_dim": 1152,
  "fusion_projection_dim": 128
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_confusion_matrix.png) — 211.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_loss.png) — 157.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_summary.png) — 339.9 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_confusion_matrix.png) — 210.4 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_loss.png) — 150.7 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_summary.png) — 390.2 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_confusion_matrix.png) — 214.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_loss.png) — 155.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_summary.png) — 348.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_confusion_matrix.png) — 210.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_loss.png) — 157.0 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_summary.png) — 327.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_confusion_matrix.png) — 212.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_loss.png) — 165.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_summary.png) — 365.0 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/metric_boxplot.png) — 128.3 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/total_confusion_matrix.png) — 223.9 KB

### prediction_files
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1.log) — 3.9 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2.log) — 3.3 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3.log) — 4.2 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4.log) — 4.5 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5.log) — 4.5 KB

### weights
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch10_weights-2026-05-14_12:38:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch10_weights-2026-05-14_12:38:41.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch20_weights-2026-05-14_12:38:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch20_weights-2026-05-14_12:38:42.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch30_weights-2026-05-14_12:38:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch30_weights-2026-05-14_12:38:43.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch40_weights-2026-05-14_12:38:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch40_weights-2026-05-14_12:38:44.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch50_weights-2026-05-14_12:38:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch50_weights-2026-05-14_12:38:45.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch60_weights-2026-05-14_12:38:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch60_weights-2026-05-14_12:38:46.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch70_weights-2026-05-14_12:38:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch70_weights-2026-05-14_12:38:47.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch80_weights-2026-05-14_12:38:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch80_weights-2026-05-14_12:38:47.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch90_weights-2026-05-14_12:38:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold1_epoch90_weights-2026-05-14_12:38:48.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch10_weights-2026-05-14_12:38:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch10_weights-2026-05-14_12:38:49.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch20_weights-2026-05-14_12:38:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch20_weights-2026-05-14_12:38:51.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch30_weights-2026-05-14_12:38:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch30_weights-2026-05-14_12:38:52.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch40_weights-2026-05-14_12:38:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch40_weights-2026-05-14_12:38:53.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch50_weights-2026-05-14_12:38:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch50_weights-2026-05-14_12:38:54.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch60_weights-2026-05-14_12:38:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch60_weights-2026-05-14_12:38:55.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch70_weights-2026-05-14_12:38:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch70_weights-2026-05-14_12:38:55.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch80_weights-2026-05-14_12:38:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold2_epoch80_weights-2026-05-14_12:38:56.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch100_weights-2026-05-14_12:39:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch100_weights-2026-05-14_12:39:06.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch10_weights-2026-05-14_12:38:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch10_weights-2026-05-14_12:38:57.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch20_weights-2026-05-14_12:38:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch20_weights-2026-05-14_12:38:58.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch30_weights-2026-05-14_12:38:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch30_weights-2026-05-14_12:38:59.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch40_weights-2026-05-14_12:39:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch40_weights-2026-05-14_12:39:00.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch50_weights-2026-05-14_12:39:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch50_weights-2026-05-14_12:39:01.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch60_weights-2026-05-14_12:39:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch60_weights-2026-05-14_12:39:02.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch70_weights-2026-05-14_12:39:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch70_weights-2026-05-14_12:39:04.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch80_weights-2026-05-14_12:39:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch80_weights-2026-05-14_12:39:04.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch90_weights-2026-05-14_12:39:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold3_epoch90_weights-2026-05-14_12:39:05.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch100_weights-2026-05-14_12:39:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch100_weights-2026-05-14_12:39:15.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch10_weights-2026-05-14_12:39:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch10_weights-2026-05-14_12:39:07.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch110_weights-2026-05-14_12:39:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch110_weights-2026-05-14_12:39:16.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch120_weights-2026-05-14_12:39:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch120_weights-2026-05-14_12:39:17.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch20_weights-2026-05-14_12:39:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch20_weights-2026-05-14_12:39:08.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch30_weights-2026-05-14_12:39:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch30_weights-2026-05-14_12:39:09.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch40_weights-2026-05-14_12:39:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch40_weights-2026-05-14_12:39:10.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch50_weights-2026-05-14_12:39:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch50_weights-2026-05-14_12:39:10.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch60_weights-2026-05-14_12:39:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch60_weights-2026-05-14_12:39:11.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch70_weights-2026-05-14_12:39:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch70_weights-2026-05-14_12:39:12.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch80_weights-2026-05-14_12:39:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch80_weights-2026-05-14_12:39:14.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch90_weights-2026-05-14_12:39:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold4_epoch90_weights-2026-05-14_12:39:15.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch100_weights-2026-05-14_12:39:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch100_weights-2026-05-14_12:39:27.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch10_weights-2026-05-14_12:39:18.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch10_weights-2026-05-14_12:39:18.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch110_weights-2026-05-14_12:39:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch110_weights-2026-05-14_12:39:28.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch120_weights-2026-05-14_12:39:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch120_weights-2026-05-14_12:39:29.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch20_weights-2026-05-14_12:39:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch20_weights-2026-05-14_12:39:19.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch30_weights-2026-05-14_12:39:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch30_weights-2026-05-14_12:39:20.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch40_weights-2026-05-14_12:39:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch40_weights-2026-05-14_12:39:21.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch50_weights-2026-05-14_12:39:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch50_weights-2026-05-14_12:39:22.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch60_weights-2026-05-14_12:39:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch60_weights-2026-05-14_12:39:23.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch70_weights-2026-05-14_12:39:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch70_weights-2026-05-14_12:39:24.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch80_weights-2026-05-14_12:39:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch80_weights-2026-05-14_12:39:25.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch90_weights-2026-05-14_12:39:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40/fold5_epoch90_weights-2026-05-14_12:39:26.pth) — 789.1 KB

## Full results.json
```json
{
  "meta": {
    "uuid": "tapct_abmil_tap_ct_abmil_pooled_s_3d_2026-05-14_12:38:40",
    "model": "tapct_abmil",
    "experiment": {
      "name": "TAP-CT ABMIL pooled S-3D",
      "description": "Frozen TAP-CT-S scan embeddings are classified with a gated attention-based MIL head; CT volumes are not loaded."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T12:39:30.620319",
    "training_started_at": "2026-05-14T12:38:40.662734",
    "training_finished_at": "2026-05-14T12:39:30.016636",
    "training_duration_seconds": 49.354,
    "training_duration_hours": 0.0137,
    "class_names": [
      "Emphysema/Abnormal (<=131 deg)",
      "Intermediate (132-151 deg)",
      "Normal (>=152 deg)"
    ],
    "config": {
      "epochs": 120,
      "batch_size": 12,
      "learning_rate": 0.0003,
      "weight_decay": 0.001,
      "k_folds": 5,
      "seed": 42,
      "loss": "auto",
      "num_classes": 3,
      "target_mode": "angle_3class",
      "tapct_features": "embeddings/tapct_s_3d/features.npz",
      "tapct_embedding_dim": 1152,
      "fusion_projection_dim": 128
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 60,
      "accuracy": 0.64286,
      "macro_f1": 0.43333,
      "macro_precision": 0.6,
      "macro_recall": 0.37778,
      "balanced_accuracy": 0.37778,
      "confusion_matrix": [
        [
          1,
          1,
          1
        ],
        [
          0,
          0,
          1
        ],
        [
          0,
          2,
          8
        ]
      ]
    },
    {
      "fold": 2,
      "best_epoch": 45,
      "accuracy": 0.61538,
      "macro_f1": 0.72028,
      "macro_precision": 0.79167,
      "macro_recall": 0.81481,
      "balanced_accuracy": 0.81481,
      "confusion_matrix": [
        [
          3,
          0,
          0
        ],
        [
          0,
          1,
          0
        ],
        [
          5,
          0,
          4
        ]
      ]
    },
    {
      "fold": 3,
      "best_epoch": 70,
      "accuracy": 0.76923,
      "macro_f1": 0.50292,
      "macro_precision": 0.48889,
      "macro_recall": 0.51852,
      "balanced_accuracy": 0.51852,
      "confusion_matrix": [
        [
          2,
          0,
          1
        ],
        [
          0,
          0,
          1
        ],
        [
          1,
          0,
          8
        ]
      ]
    },
    {
      "fold": 4,
      "best_epoch": 95,
      "accuracy": 0.69231,
      "macro_f1": 0.5,
      "macro_precision": 0.48571,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556,
      "confusion_matrix": [
        [
          3,
          0,
          0
        ],
        [
          0,
          0,
          1
        ],
        [
          2,
          1,
          6
        ]
      ]
    },
    {
      "fold": 5,
      "best_epoch": 90,
      "accuracy": 0.84615,
      "macro_f1": 0.78519,
      "macro_precision": 0.72222,
      "macro_recall": 0.93333,
      "balanced_accuracy": 0.93333,
      "confusion_matrix": [
        [
          2,
          0,
          0
        ],
        [
          0,
          1,
          0
        ],
        [
          1,
          1,
          8
        ]
      ]
    }
  ],
  "summary": {
    "mean_accuracy": 0.71319,
    "std_accuracy": 0.08458,
    "mean_macro_f1": 0.58834,
    "std_macro_f1": 0.13805,
    "mean_macro_precision": 0.6177,
    "std_macro_precision": 0.1229,
    "mean_macro_recall": 0.64,
    "std_macro_recall": 0.20359,
    "mean_balanced_accuracy": 0.64,
    "std_balanced_accuracy": 0.20359
  },
  "total_confusion_matrix": [
    [
      11,
      1,
      2
    ],
    [
      0,
      2,
      3
    ],
    [
      9,
      4,
      34
    ]
  ]
}
```
