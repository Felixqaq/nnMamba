# Run: hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25

## 一句話用途
- 方法名稱: `TAP-CT late fusion extreme binary + aug100/class`
- 實驗描述: Gray-zone excluded binary classification using Hybrid Mamba-Attention CT features fused with frozen TAP-CT-S patient-level embeddings.
- 任務: `Angle_extreme_binary_classification`
- 模型: `hybrid_mamba_tapct_fusion`
- target_mode/task_type: `angle_binary_extreme`
- 結果時間: `2026-05-13T17:48:37.061414`
- 訓練時間: `2026-05-13T17:32:25.436860` -> `2026-05-13T17:48:36.577361`
- 訓練耗時: 0.2698 hours
- 原始 results.json: [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/results.json)

## 類別定義
- 0: Abnormal/emphysema-like (AC <=131 deg)
- 1: Normal-like (AC >=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.91923 |
| std_accuracy | 0.0718 |
| mean_macro_f1 | 0.8763 |
| std_macro_f1 | 0.1186 |
| mean_macro_precision | 0.91389 |
| std_macro_precision | 0.08352 |
| mean_macro_recall | 0.87778 |
| std_macro_recall | 0.13333 |
| mean_balanced_accuracy | 0.87778 |
| std_balanced_accuracy | 0.13333 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 0.84615 | 0.70455 | 0.91667 | 0.66667 | 0.66667 |
| 2 | 5 | 0.83333 | 0.77778 | 0.77778 | 0.77778 | 0.77778 |
| 3 | 5 | 1 | 1 | 1 | 1 | 1 |
| 4 | 25 | 0.91667 | 0.89916 | 0.875 | 0.94444 | 0.94444 |
| 5 | 20 | 1 | 1 | 1 | 1 | 1 |

## Training Config Embedded In Result
```json
{
  "epochs": 160,
  "batch_size": 12,
  "learning_rate": 0.0001,
  "weight_decay": 0.001,
  "k_folds": 5,
  "seed": 42,
  "loss": "auto",
  "num_classes": 2,
  "target_mode": "angle_binary_extreme",
  "tapct_features": "embeddings/tapct_s_3d/features.npz",
  "tapct_embedding_dim": 1152,
  "fusion_projection_dim": 128,
  "abnormal_rule": "AC <= 131 deg",
  "excluded_gray_zone": "132 <= AC < 152 deg",
  "normal_rule": "AC >= 152 deg"
}
```

## Artifact Index
### figures
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_confusion_matrix.png) — 202.0 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_loss.png) — 147.0 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_summary.png) — 312.1 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_confusion_matrix.png) — 202.7 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_loss.png) — 163.9 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_summary.png) — 321.1 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_confusion_matrix.png) — 207.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_loss.png) — 163.3 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_summary.png) — 326.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_confusion_matrix.png) — 204.3 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_loss.png) — 159.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_summary.png) — 385.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_confusion_matrix.png) — 203.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_loss.png) — 166.1 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_summary.png) — 344.9 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/metric_boxplot.png) — 143.6 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/total_confusion_matrix.png) — 217.7 KB

### prediction_files
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1.log) — 1.8 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4.log) — 2.2 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5.log) — 2.0 KB

### weights
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch10_weights-2026-05-13_17:33:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch10_weights-2026-05-13_17:33:13.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch20_weights-2026-05-13_17:33:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch20_weights-2026-05-13_17:33:56.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch30_weights-2026-05-13_17:34:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch30_weights-2026-05-13_17:34:38.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch40_weights-2026-05-13_17:35:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold1_epoch40_weights-2026-05-13_17:35:21.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch10_weights-2026-05-13_17:36:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch10_weights-2026-05-13_17:36:24.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch20_weights-2026-05-13_17:37:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch20_weights-2026-05-13_17:37:07.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch30_weights-2026-05-13_17:37:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold2_epoch30_weights-2026-05-13_17:37:50.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch10_weights-2026-05-13_17:38:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch10_weights-2026-05-13_17:38:56.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch20_weights-2026-05-13_17:39:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch20_weights-2026-05-13_17:39:39.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch30_weights-2026-05-13_17:40:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold3_epoch30_weights-2026-05-13_17:40:23.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch10_weights-2026-05-13_17:41:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch10_weights-2026-05-13_17:41:30.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch20_weights-2026-05-13_17:42:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch20_weights-2026-05-13_17:42:15.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch30_weights-2026-05-13_17:42:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch30_weights-2026-05-13_17:42:59.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch40_weights-2026-05-13_17:43:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch40_weights-2026-05-13_17:43:44.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch50_weights-2026-05-13_17:44:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold4_epoch50_weights-2026-05-13_17:44:29.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch10_weights-2026-05-13_17:45:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch10_weights-2026-05-13_17:45:37.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch20_weights-2026-05-13_17:46:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch20_weights-2026-05-13_17:46:22.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch30_weights-2026-05-13_17:47:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch30_weights-2026-05-13_17:47:06.pth) — 5.2 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch40_weights-2026-05-13_17:47:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25/fold5_epoch40_weights-2026-05-13_17:47:51.pth) — 5.2 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25",
    "model": "hybrid_mamba_tapct_fusion",
    "experiment": {
      "name": "TAP-CT late fusion extreme binary + aug100/class",
      "description": "Gray-zone excluded binary classification using Hybrid Mamba-Attention CT features fused with frozen TAP-CT-S patient-level embeddings."
    },
    "task": "Angle_extreme_binary_classification",
    "task_type": "angle_binary_extreme",
    "timestamp": "2026-05-13T17:48:37.061414",
    "training_started_at": "2026-05-13T17:32:25.436860",
    "training_finished_at": "2026-05-13T17:48:36.577361",
    "training_duration_seconds": 971.141,
    "training_duration_hours": 0.2698,
    "class_names": [
      "Abnormal/emphysema-like (AC <=131 deg)",
      "Normal-like (AC >=152 deg)"
    ],
    "config": {
      "epochs": 160,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.001,
      "k_folds": 5,
      "seed": 42,
      "loss": "auto",
      "num_classes": 2,
      "target_mode": "angle_binary_extreme",
      "tapct_features": "embeddings/tapct_s_3d/features.npz",
      "tapct_embedding_dim": 1152,
      "fusion_projection_dim": 128,
      "abnormal_rule": "AC <= 131 deg",
      "excluded_gray_zone": "132 <= AC < 152 deg",
      "normal_rule": "AC >= 152 deg"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 15,
      "accuracy": 0.84615,
      "macro_f1": 0.70455,
      "macro_precision": 0.91667,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.83333,
      "macro_f1": 0.77778,
      "macro_precision": 0.77778,
      "macro_recall": 0.77778,
      "balanced_accuracy": 0.77778
    },
    {
      "fold": 3,
      "best_epoch": 5,
      "accuracy": 1.0,
      "macro_f1": 1.0,
      "macro_precision": 1.0,
      "macro_recall": 1.0,
      "balanced_accuracy": 1.0
    },
    {
      "fold": 4,
      "best_epoch": 25,
      "accuracy": 0.91667,
      "macro_f1": 0.89916,
      "macro_precision": 0.875,
      "macro_recall": 0.94444,
      "balanced_accuracy": 0.94444
    },
    {
      "fold": 5,
      "best_epoch": 20,
      "accuracy": 1.0,
      "macro_f1": 1.0,
      "macro_precision": 1.0,
      "macro_recall": 1.0,
      "balanced_accuracy": 1.0
    }
  ],
  "summary": {
    "mean_accuracy": 0.91923,
    "std_accuracy": 0.0718,
    "mean_macro_f1": 0.8763,
    "std_macro_f1": 0.1186,
    "mean_macro_precision": 0.91389,
    "std_macro_precision": 0.08352,
    "mean_macro_recall": 0.87778,
    "std_macro_recall": 0.13333,
    "mean_balanced_accuracy": 0.87778,
    "std_balanced_accuracy": 0.13333
  }
}
```
