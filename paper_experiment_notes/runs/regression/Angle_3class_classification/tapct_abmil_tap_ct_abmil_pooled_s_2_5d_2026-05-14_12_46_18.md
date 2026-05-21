# Run: tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18

## 一句話用途
- 方法名稱: `TAP-CT ABMIL pooled S-2.5D`
- 實驗描述: Frozen TAP-CT-S-2.5D scan embeddings are classified with a gated attention-based MIL head; CT volumes are not loaded during classifier training.
- 任務: `Angle_3class_classification`
- 模型: `tapct_abmil`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T12:47:03.013473`
- 訓練時間: `2026-05-14T12:46:18.855422` -> `2026-05-14T12:47:02.548530`
- 訓練耗時: 0.0121 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.74176 |
| std_accuracy | 0.08 |
| mean_macro_f1 | 0.58513 |
| std_macro_f1 | 0.04929 |
| mean_macro_precision | 0.58682 |
| std_macro_precision | 0.0555 |
| mean_macro_recall | 0.68518 |
| std_macro_recall | 0.13263 |
| mean_balanced_accuracy | 0.68518 |
| std_balanced_accuracy | 0.13263 |

## Total Confusion Matrix
```json
[
  [
    13,
    0,
    1
  ],
  [
    1,
    2,
    2
  ],
  [
    8,
    5,
    34
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 55 | 0.78571 | 0.55238 | 0.60606 | 0.52222 | 0.52222 |
| 2 | 40 | 0.61538 | 0.62735 | 0.64286 | 0.81481 | 0.81481 |
| 3 | 70 | 0.76923 | 0.51389 | 0.5 | 0.59259 | 0.59259 |
| 4 | 60 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 5 | 70 | 0.69231 | 0.65 | 0.63889 | 0.86667 | 0.86667 |

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
  "tapct_features": "embeddings/tapct_s_2_5d/features.npz",
  "tapct_embedding_dim": 1152,
  "fusion_projection_dim": 128
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_confusion_matrix.png) — 215.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_loss.png) — 154.4 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_summary.png) — 359.4 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_confusion_matrix.png) — 221.4 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_loss.png) — 154.2 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_summary.png) — 329.4 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_confusion_matrix.png) — 213.2 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_loss.png) — 160.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_summary.png) — 342.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_confusion_matrix.png) — 214.3 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_loss.png) — 155.1 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_summary.png) — 358.9 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_confusion_matrix.png) — 211.2 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_loss.png) — 167.8 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_summary.png) — 362.0 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/metric_boxplot.png) — 139.6 KB
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/total_confusion_matrix.png) — 224.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1.log) — 3.7 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3.log) — 4.2 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4.log) — 3.9 KB
- [regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5.log) — 4.2 KB

### weights
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch10_weights-2026-05-14_12:46:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch10_weights-2026-05-14_12:46:19.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch20_weights-2026-05-14_12:46:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch20_weights-2026-05-14_12:46:20.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch30_weights-2026-05-14_12:46:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch30_weights-2026-05-14_12:46:21.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch40_weights-2026-05-14_12:46:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch40_weights-2026-05-14_12:46:22.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch50_weights-2026-05-14_12:46:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch50_weights-2026-05-14_12:46:23.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch60_weights-2026-05-14_12:46:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch60_weights-2026-05-14_12:46:24.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch70_weights-2026-05-14_12:46:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch70_weights-2026-05-14_12:46:25.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch80_weights-2026-05-14_12:46:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch80_weights-2026-05-14_12:46:26.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch90_weights-2026-05-14_12:46:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold1_epoch90_weights-2026-05-14_12:46:27.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch10_weights-2026-05-14_12:46:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch10_weights-2026-05-14_12:46:28.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch20_weights-2026-05-14_12:46:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch20_weights-2026-05-14_12:46:29.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch30_weights-2026-05-14_12:46:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch30_weights-2026-05-14_12:46:30.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch40_weights-2026-05-14_12:46:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch40_weights-2026-05-14_12:46:31.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch50_weights-2026-05-14_12:46:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch50_weights-2026-05-14_12:46:32.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch60_weights-2026-05-14_12:46:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch60_weights-2026-05-14_12:46:33.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch70_weights-2026-05-14_12:46:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold2_epoch70_weights-2026-05-14_12:46:33.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch100_weights-2026-05-14_12:46:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch100_weights-2026-05-14_12:46:43.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch10_weights-2026-05-14_12:46:35.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch10_weights-2026-05-14_12:46:35.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch20_weights-2026-05-14_12:46:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch20_weights-2026-05-14_12:46:36.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch30_weights-2026-05-14_12:46:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch30_weights-2026-05-14_12:46:37.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch40_weights-2026-05-14_12:46:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch40_weights-2026-05-14_12:46:38.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch50_weights-2026-05-14_12:46:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch50_weights-2026-05-14_12:46:39.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch60_weights-2026-05-14_12:46:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch60_weights-2026-05-14_12:46:40.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch70_weights-2026-05-14_12:46:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch70_weights-2026-05-14_12:46:41.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch80_weights-2026-05-14_12:46:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch80_weights-2026-05-14_12:46:41.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch90_weights-2026-05-14_12:46:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold3_epoch90_weights-2026-05-14_12:46:42.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch10_weights-2026-05-14_12:46:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch10_weights-2026-05-14_12:46:44.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch20_weights-2026-05-14_12:46:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch20_weights-2026-05-14_12:46:45.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch30_weights-2026-05-14_12:46:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch30_weights-2026-05-14_12:46:46.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch40_weights-2026-05-14_12:46:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch40_weights-2026-05-14_12:46:47.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch50_weights-2026-05-14_12:46:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch50_weights-2026-05-14_12:46:49.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch60_weights-2026-05-14_12:46:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch60_weights-2026-05-14_12:46:50.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch70_weights-2026-05-14_12:46:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch70_weights-2026-05-14_12:46:50.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch80_weights-2026-05-14_12:46:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch80_weights-2026-05-14_12:46:51.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch90_weights-2026-05-14_12:46:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold4_epoch90_weights-2026-05-14_12:46:52.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_best_weight.pth) — 788.7 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch100_weights-2026-05-14_12:47:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch100_weights-2026-05-14_12:47:01.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch10_weights-2026-05-14_12:46:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch10_weights-2026-05-14_12:46:53.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch20_weights-2026-05-14_12:46:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch20_weights-2026-05-14_12:46:54.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch30_weights-2026-05-14_12:46:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch30_weights-2026-05-14_12:46:55.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch40_weights-2026-05-14_12:46:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch40_weights-2026-05-14_12:46:56.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch50_weights-2026-05-14_12:46:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch50_weights-2026-05-14_12:46:57.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch60_weights-2026-05-14_12:46:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch60_weights-2026-05-14_12:46:58.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch70_weights-2026-05-14_12:46:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch70_weights-2026-05-14_12:46:59.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch80_weights-2026-05-14_12:47:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch80_weights-2026-05-14_12:47:00.pth) — 789.1 KB
- [regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch90_weights-2026-05-14_12:47:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18/fold5_epoch90_weights-2026-05-14_12:47:01.pth) — 789.1 KB

## Full results.json
```json
{
  "meta": {
    "uuid": "tapct_abmil_tap_ct_abmil_pooled_s_2_5d_2026-05-14_12:46:18",
    "model": "tapct_abmil",
    "experiment": {
      "name": "TAP-CT ABMIL pooled S-2.5D",
      "description": "Frozen TAP-CT-S-2.5D scan embeddings are classified with a gated attention-based MIL head; CT volumes are not loaded during classifier training."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T12:47:03.013473",
    "training_started_at": "2026-05-14T12:46:18.855422",
    "training_finished_at": "2026-05-14T12:47:02.548530",
    "training_duration_seconds": 43.693,
    "training_duration_hours": 0.0121,
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
      "tapct_features": "embeddings/tapct_s_2_5d/features.npz",
      "tapct_embedding_dim": 1152,
      "fusion_projection_dim": 128
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 55,
      "accuracy": 0.78571,
      "macro_f1": 0.55238,
      "macro_precision": 0.60606,
      "macro_recall": 0.52222,
      "balanced_accuracy": 0.52222,
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
          0,
          1,
          9
        ]
      ]
    },
    {
      "fold": 2,
      "best_epoch": 40,
      "accuracy": 0.61538,
      "macro_f1": 0.62735,
      "macro_precision": 0.64286,
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
          4,
          1,
          4
        ]
      ]
    },
    {
      "fold": 3,
      "best_epoch": 70,
      "accuracy": 0.76923,
      "macro_f1": 0.51389,
      "macro_precision": 0.5,
      "macro_recall": 0.59259,
      "balanced_accuracy": 0.59259,
      "confusion_matrix": [
        [
          3,
          0,
          0
        ],
        [
          1,
          0,
          0
        ],
        [
          2,
          0,
          7
        ]
      ]
    },
    {
      "fold": 4,
      "best_epoch": 60,
      "accuracy": 0.84615,
      "macro_f1": 0.58201,
      "macro_precision": 0.5463,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963,
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
          1,
          0,
          8
        ]
      ]
    },
    {
      "fold": 5,
      "best_epoch": 70,
      "accuracy": 0.69231,
      "macro_f1": 0.65,
      "macro_precision": 0.63889,
      "macro_recall": 0.86667,
      "balanced_accuracy": 0.86667,
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
          3,
          6
        ]
      ]
    }
  ],
  "summary": {
    "mean_accuracy": 0.74176,
    "std_accuracy": 0.08,
    "mean_macro_f1": 0.58513,
    "std_macro_f1": 0.04929,
    "mean_macro_precision": 0.58682,
    "std_macro_precision": 0.0555,
    "mean_macro_recall": 0.68518,
    "std_macro_recall": 0.13263,
    "mean_balanced_accuracy": 0.68518,
    "std_balanced_accuracy": 0.13263
  },
  "total_confusion_matrix": [
    [
      13,
      0,
      1
    ],
    [
      1,
      2,
      2
    ],
    [
      8,
      5,
      34
    ]
  ]
}
```
