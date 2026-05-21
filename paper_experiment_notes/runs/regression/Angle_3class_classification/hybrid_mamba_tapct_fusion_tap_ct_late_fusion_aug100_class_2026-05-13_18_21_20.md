# Run: hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20

## 一句話用途
- 方法名稱: `TAP-CT late fusion + aug100/class`
- 實驗描述: Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-B patient-level embeddings before the classification head; train-fold augmentation balances each class to 100 samples.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_tapct_fusion`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-13T18:43:58.934118`
- 訓練時間: `2026-05-13T18:21:20.118427` -> `2026-05-13T18:43:58.441983`
- 訓練耗時: 0.3773 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.82088 |
| std_accuracy | 0.09386 |
| mean_macro_f1 | 0.61406 |
| std_macro_f1 | 0.14033 |
| mean_macro_precision | 0.63896 |
| std_macro_precision | 0.10146 |
| mean_macro_recall | 0.63259 |
| std_macro_recall | 0.18026 |
| mean_balanced_accuracy | 0.63259 |
| std_balanced_accuracy | 0.18026 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.64286 | 0.42063 | 0.57576 | 0.37778 | 0.37778 |
| 2 | 15 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 3 | 15 | 0.92308 | 0.64912 | 0.63333 | 0.66667 | 0.66667 |
| 4 | 15 | 0.84615 | 0.56667 | 0.60606 | 0.55556 | 0.55556 |
| 5 | 10 | 0.84615 | 0.85185 | 0.83333 | 0.93333 | 0.93333 |

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
  "num_classes": 3,
  "target_mode": "angle_3class",
  "tapct_features": "embeddings/tapct_b_3d/features.npz",
  "tapct_embedding_dim": 2304,
  "fusion_projection_dim": 128
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_confusion_matrix.png) — 218.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_loss.png) — 119.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_summary.png) — 286.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_confusion_matrix.png) — 218.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_loss.png) — 119.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_summary.png) — 298.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_confusion_matrix.png) — 220.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_loss.png) — 123.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_summary.png) — 325.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_confusion_matrix.png) — 219.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_loss.png) — 117.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_summary.png) — 302.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_confusion_matrix.png) — 219.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_loss.png) — 127.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_summary.png) — 290.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/metric_boxplot.png) — 139.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/total_confusion_matrix.png) — 231.7 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5.log) — 1.6 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_best_weight.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch10_weights-2026-05-13_18:22:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch10_weights-2026-05-13_18:22:26.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch20_weights-2026-05-13_18:23:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch20_weights-2026-05-13_18:23:27.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch30_weights-2026-05-13_18:24:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold1_epoch30_weights-2026-05-13_18:24:28.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_best_weight.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch10_weights-2026-05-13_18:26:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch10_weights-2026-05-13_18:26:02.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch20_weights-2026-05-13_18:27:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch20_weights-2026-05-13_18:27:05.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch30_weights-2026-05-13_18:28:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch30_weights-2026-05-13_18:28:09.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch40_weights-2026-05-13_18:29:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold2_epoch40_weights-2026-05-13_18:29:14.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_best_weight.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch10_weights-2026-05-13_18:30:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch10_weights-2026-05-13_18:30:51.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch20_weights-2026-05-13_18:31:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch20_weights-2026-05-13_18:31:56.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch30_weights-2026-05-13_18:33:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch30_weights-2026-05-13_18:33:02.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch40_weights-2026-05-13_18:34:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold3_epoch40_weights-2026-05-13_18:34:07.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_best_weight.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch10_weights-2026-05-13_18:35:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch10_weights-2026-05-13_18:35:45.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch20_weights-2026-05-13_18:36:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch20_weights-2026-05-13_18:36:51.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch30_weights-2026-05-13_18:37:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch30_weights-2026-05-13_18:37:56.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch40_weights-2026-05-13_18:39:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold4_epoch40_weights-2026-05-13_18:39:02.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_best_weight.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch10_weights-2026-05-13_18:40:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch10_weights-2026-05-13_18:40:41.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch20_weights-2026-05-13_18:41:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch20_weights-2026-05-13_18:41:47.pth) — 5.7 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch30_weights-2026-05-13_18:42:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/fold5_epoch30_weights-2026-05-13_18:42:52.pth) — 5.7 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20",
    "model": "hybrid_mamba_tapct_fusion",
    "experiment": {
      "name": "TAP-CT late fusion + aug100/class",
      "description": "Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-B patient-level embeddings before the classification head; train-fold augmentation balances each class to 100 samples."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-13T18:43:58.934118",
    "training_started_at": "2026-05-13T18:21:20.118427",
    "training_finished_at": "2026-05-13T18:43:58.441983",
    "training_duration_seconds": 1358.324,
    "training_duration_hours": 0.3773,
    "class_names": [
      "Emphysema/Abnormal (<=131 deg)",
      "Intermediate (132-151 deg)",
      "Normal (>=152 deg)"
    ],
    "config": {
      "epochs": 160,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.001,
      "k_folds": 5,
      "seed": 42,
      "loss": "auto",
      "num_classes": 3,
      "target_mode": "angle_3class",
      "tapct_features": "embeddings/tapct_b_3d/features.npz",
      "tapct_embedding_dim": 2304,
      "fusion_projection_dim": 128
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 5,
      "accuracy": 0.64286,
      "macro_f1": 0.42063,
      "macro_precision": 0.57576,
      "macro_recall": 0.37778,
      "balanced_accuracy": 0.37778
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "accuracy": 0.84615,
      "macro_f1": 0.58201,
      "macro_precision": 0.5463,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 3,
      "best_epoch": 15,
      "accuracy": 0.92308,
      "macro_f1": 0.64912,
      "macro_precision": 0.63333,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    },
    {
      "fold": 4,
      "best_epoch": 15,
      "accuracy": 0.84615,
      "macro_f1": 0.56667,
      "macro_precision": 0.60606,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 5,
      "best_epoch": 10,
      "accuracy": 0.84615,
      "macro_f1": 0.85185,
      "macro_precision": 0.83333,
      "macro_recall": 0.93333,
      "balanced_accuracy": 0.93333
    }
  ],
  "summary": {
    "mean_accuracy": 0.82088,
    "std_accuracy": 0.09386,
    "mean_macro_f1": 0.61406,
    "std_macro_f1": 0.14033,
    "mean_macro_precision": 0.63896,
    "std_macro_precision": 0.10146,
    "mean_macro_recall": 0.63259,
    "std_macro_recall": 0.18026,
    "mean_balanced_accuracy": 0.63259,
    "std_balanced_accuracy": 0.18026
  }
}
```
