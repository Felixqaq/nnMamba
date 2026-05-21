# Run: hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45

## 一句話用途
- 方法名稱: `TAP-CT-S-2.5D late fusion + aug100/class`
- 實驗描述: Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-S-2.5D patient-level embeddings before the original MLP classification head; train-fold augmentation balances each class to 100 samples.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_tapct_fusion`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T14:58:04.138331`
- 訓練時間: `2026-05-14T14:30:45.287598` -> `2026-05-14T14:58:03.645374`
- 訓練耗時: 0.4551 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.8044 |
| std_accuracy | 0.08747 |
| mean_macro_f1 | 0.54481 |
| std_macro_f1 | 0.07336 |
| mean_macro_precision | 0.56943 |
| std_macro_precision | 0.06542 |
| mean_macro_recall | 0.56371 |
| std_macro_recall | 0.0876 |
| mean_balanced_accuracy | 0.56371 |
| std_balanced_accuracy | 0.0876 |

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
    0,
    5
  ],
  [
    4,
    1,
    42
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 0.71429 | 0.45238 | 0.60606 | 0.41111 | 0.41111 |
| 2 | 15 | 0.69231 | 0.47222 | 0.45238 | 0.55556 | 0.55556 |
| 3 | 20 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 4 | 15 | 0.84615 | 0.56667 | 0.60606 | 0.55556 | 0.55556 |
| 5 | 20 | 0.92308 | 0.65079 | 0.63636 | 0.66667 | 0.66667 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_confusion_matrix.png) — 221.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_loss.png) — 147.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_summary.png) — 300.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_confusion_matrix.png) — 221.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_loss.png) — 125.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_summary.png) — 279.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_confusion_matrix.png) — 223.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_loss.png) — 146.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_summary.png) — 305.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_confusion_matrix.png) — 224.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_loss.png) — 146.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_summary.png) — 306.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_confusion_matrix.png) — 220.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_loss.png) — 151.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_summary.png) — 332.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/metric_boxplot.png) — 136.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/total_confusion_matrix.png) — 234.2 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3.log) — 2.0 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5.log) — 2.0 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch10_weights-2026-05-14_14:31:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch10_weights-2026-05-14_14:31:57.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch20_weights-2026-05-14_14:33:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch20_weights-2026-05-14_14:33:03.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch30_weights-2026-05-14_14:34:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch30_weights-2026-05-14_14:34:08.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch40_weights-2026-05-14_14:35:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold1_epoch40_weights-2026-05-14_14:35:15.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch10_weights-2026-05-14_14:36:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch10_weights-2026-05-14_14:36:57.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch20_weights-2026-05-14_14:38:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch20_weights-2026-05-14_14:38:05.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch30_weights-2026-05-14_14:39:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch30_weights-2026-05-14_14:39:15.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch40_weights-2026-05-14_14:40:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold2_epoch40_weights-2026-05-14_14:40:25.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch10_weights-2026-05-14_14:42:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch10_weights-2026-05-14_14:42:10.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch20_weights-2026-05-14_14:43:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch20_weights-2026-05-14_14:43:20.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch30_weights-2026-05-14_14:44:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch30_weights-2026-05-14_14:44:30.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch40_weights-2026-05-14_14:45:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold3_epoch40_weights-2026-05-14_14:45:41.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch10_weights-2026-05-14_14:48:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch10_weights-2026-05-14_14:48:01.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch20_weights-2026-05-14_14:49:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch20_weights-2026-05-14_14:49:12.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch30_weights-2026-05-14_14:50:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch30_weights-2026-05-14_14:50:22.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch40_weights-2026-05-14_14:51:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold4_epoch40_weights-2026-05-14_14:51:32.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_best_weight.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch10_weights-2026-05-14_14:53:18.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch10_weights-2026-05-14_14:53:18.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch20_weights-2026-05-14_14:54:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch20_weights-2026-05-14_14:54:29.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch30_weights-2026-05-14_14:55:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch30_weights-2026-05-14_14:55:40.pth) — 5.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch40_weights-2026-05-14_14:56:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45/fold5_epoch40_weights-2026-05-14_14:56:51.pth) — 5.2 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_fusion_tap_ct_s_2_5d_late_fusion_aug100_class_2026-05-14_14:30:45",
    "model": "hybrid_mamba_tapct_fusion",
    "experiment": {
      "name": "TAP-CT-S-2.5D late fusion + aug100/class",
      "description": "Hybrid Mamba-Attention CT features are concatenated with frozen TAP-CT-S-2.5D patient-level embeddings before the original MLP classification head; train-fold augmentation balances each class to 100 samples."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T14:58:04.138331",
    "training_started_at": "2026-05-14T14:30:45.287598",
    "training_finished_at": "2026-05-14T14:58:03.645374",
    "training_duration_seconds": 1638.358,
    "training_duration_hours": 0.4551,
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
      "best_epoch": 15,
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
      "best_epoch": 15,
      "accuracy": 0.69231,
      "macro_f1": 0.47222,
      "macro_precision": 0.45238,
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
          3,
          0,
          6
        ]
      ]
    },
    {
      "fold": 3,
      "best_epoch": 20,
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
      "fold": 4,
      "best_epoch": 15,
      "accuracy": 0.84615,
      "macro_f1": 0.56667,
      "macro_precision": 0.60606,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556,
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
          0,
          9
        ]
      ]
    },
    {
      "fold": 5,
      "best_epoch": 20,
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
    "mean_accuracy": 0.8044,
    "std_accuracy": 0.08747,
    "mean_macro_f1": 0.54481,
    "std_macro_f1": 0.07336,
    "mean_macro_precision": 0.56943,
    "std_macro_precision": 0.06542,
    "mean_macro_recall": 0.56371,
    "std_macro_recall": 0.0876,
    "mean_balanced_accuracy": 0.56371,
    "std_balanced_accuracy": 0.0876
  },
  "total_confusion_matrix": [
    [
      11,
      1,
      2
    ],
    [
      0,
      0,
      5
    ],
    [
      4,
      1,
      42
    ]
  ]
}
```
