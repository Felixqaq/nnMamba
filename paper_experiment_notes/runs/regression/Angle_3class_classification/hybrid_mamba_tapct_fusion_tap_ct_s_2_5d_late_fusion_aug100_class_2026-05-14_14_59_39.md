# Run: hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39

## 一句話用途
- 方法名稱: `TAP-CT-S-2.5D late fusion + aug100/class`
- 實驗描述: Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-S-2.5D patient-level embeddings before the original MLP classification head; train-fold augmentation balances each class to 100 samples.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_tapct_fusion`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T15:20:59.532883`
- 訓練時間: `2026-05-14T14:59:39.572799` -> `2026-05-14T15:20:59.056881`
- 訓練耗時: 0.3554 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.83517 |
| std_accuracy | 0.08311 |
| mean_macro_f1 | 0.58843 |
| std_macro_f1 | 0.0756 |
| mean_macro_precision | 0.60941 |
| std_macro_precision | 0.03552 |
| mean_macro_recall | 0.59333 |
| std_macro_recall | 0.09516 |
| mean_balanced_accuracy | 0.59333 |
| std_balanced_accuracy | 0.09516 |

## Total Confusion Matrix
```json
[
  [
    12,
    1,
    1
  ],
  [
    0,
    0,
    5
  ],
  [
    1,
    3,
    43
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.71429 | 0.45238 | 0.60606 | 0.41111 | 0.41111 |
| 2 | 25 | 0.76923 | 0.56022 | 0.54167 | 0.59259 | 0.59259 |
| 3 | 20 | 0.92308 | 0.64912 | 0.63333 | 0.66667 | 0.66667 |
| 4 | 10 | 0.84615 | 0.62963 | 0.62963 | 0.62963 | 0.62963 |
| 5 | 25 | 0.92308 | 0.65079 | 0.63636 | 0.66667 | 0.66667 |

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
  "tapct_features": "embeddings/tapct_s_2_5d/features.npz",
  "tapct_embedding_dim": 1152,
  "fusion_projection_dim": 128
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_confusion_matrix.png) — 221.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_loss.png) — 144.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_summary.png) — 301.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_confusion_matrix.png) — 219.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_loss.png) — 133.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_summary.png) — 321.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_confusion_matrix.png) — 225.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_loss.png) — 152.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_summary.png) — 344.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_confusion_matrix.png) — 223.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_loss.png) — 152.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_summary.png) — 357.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_confusion_matrix.png) — 220.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_loss.png) — 133.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_summary.png) — 360.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/metric_boxplot.png) — 137.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/total_confusion_matrix.png) — 235.0 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2.log) — 2.2 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3.log) — 2.0 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5.log) — 2.2 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch10_weights-2026-05-14_15:00:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch10_weights-2026-05-14_15:00:33.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch20_weights-2026-05-14_15:01:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch20_weights-2026-05-14_15:01:27.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch30_weights-2026-05-14_15:02:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold1_epoch30_weights-2026-05-14_15:02:21.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch10_weights-2026-05-14_15:04:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch10_weights-2026-05-14_15:04:09.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch20_weights-2026-05-14_15:05:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch20_weights-2026-05-14_15:05:02.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch30_weights-2026-05-14_15:05:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch30_weights-2026-05-14_15:05:55.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch40_weights-2026-05-14_15:06:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch40_weights-2026-05-14_15:06:48.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch50_weights-2026-05-14_15:07:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold2_epoch50_weights-2026-05-14_15:07:41.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch10_weights-2026-05-14_15:09:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch10_weights-2026-05-14_15:09:02.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch20_weights-2026-05-14_15:09:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch20_weights-2026-05-14_15:09:56.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch30_weights-2026-05-14_15:10:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch30_weights-2026-05-14_15:10:48.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch40_weights-2026-05-14_15:11:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold3_epoch40_weights-2026-05-14_15:11:42.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch10_weights-2026-05-14_15:13:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch10_weights-2026-05-14_15:13:29.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch20_weights-2026-05-14_15:14:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch20_weights-2026-05-14_15:14:22.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch30_weights-2026-05-14_15:15:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold4_epoch30_weights-2026-05-14_15:15:15.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch10_weights-2026-05-14_15:17:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch10_weights-2026-05-14_15:17:01.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch20_weights-2026-05-14_15:17:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch20_weights-2026-05-14_15:17:54.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch30_weights-2026-05-14_15:18:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch30_weights-2026-05-14_15:18:47.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch40_weights-2026-05-14_15:19:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch40_weights-2026-05-14_15:19:39.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch50_weights-2026-05-14_15:20:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39/fold5_epoch50_weights-2026-05-14_15:20:32.pth) — 5.2 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:59:39",
    "model": "hybrid_mamba_tapct_fusion",
    "experiment": {
      "name": "TAP-CT-S-2.5D late fusion + aug100/class",
      "description": "Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-S-2.5D patient-level embeddings before the original MLP classification head; train-fold augmentation balances each class to 100 samples."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T15:20:59.532883",
    "training_started_at": "2026-05-14T14:59:39.572799",
    "training_finished_at": "2026-05-14T15:20:59.056881",
    "training_duration_seconds": 1279.484,
    "training_duration_hours": 0.3554,
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
      "tapct_features": "embeddings/tapct_s_2_5d/features.npz",
      "tapct_embedding_dim": 1152,
      "fusion_projection_dim": 128
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 10,
      "accuracy": 0.71429,
      "macro_f1": 0.45238,
      "macro_precision": 0.60606,
      "macro_recall": 0.41111,
      "balanced_accuracy": 0.41111,
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
          1,
          9
        ]
      ]
    },
    {
      "fold": 2,
      "best_epoch": 25,
      "accuracy": 0.76923,
      "macro_f1": 0.56022,
      "macro_precision": 0.54167,
      "macro_recall": 0.59259,
      "balanced_accuracy": 0.59259,
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
          1,
          7
        ]
      ]
    },
    {
      "fold": 3,
      "best_epoch": 20,
      "accuracy": 0.92308,
      "macro_f1": 0.64912,
      "macro_precision": 0.63333,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667,
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
          0,
          0,
          9
        ]
      ]
    },
    {
      "fold": 4,
      "best_epoch": 10,
      "accuracy": 0.84615,
      "macro_f1": 0.62963,
      "macro_precision": 0.62963,
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
          0,
          1,
          8
        ]
      ]
    },
    {
      "fold": 5,
      "best_epoch": 25,
      "accuracy": 0.92308,
      "macro_f1": 0.65079,
      "macro_precision": 0.63636,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667,
      "confusion_matrix": [
        [
          2,
          0,
          0
        ],
        [
          0,
          0,
          1
        ],
        [
          0,
          0,
          10
        ]
      ]
    }
  ],
  "summary": {
    "mean_accuracy": 0.83517,
    "std_accuracy": 0.08311,
    "mean_macro_f1": 0.58843,
    "std_macro_f1": 0.0756,
    "mean_macro_precision": 0.60941,
    "std_macro_precision": 0.03552,
    "mean_macro_recall": 0.59333,
    "std_macro_recall": 0.09516,
    "mean_balanced_accuracy": 0.59333,
    "std_balanced_accuracy": 0.09516
  },
  "total_confusion_matrix": [
    [
      12,
      1,
      1
    ],
    [
      0,
      0,
      5
    ],
    [
      1,
      3,
      43
    ]
  ]
}
```
