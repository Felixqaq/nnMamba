# Run: hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19

## 一句話用途
- 方法名稱: `TAP-CT attention fusion + aug100/class`
- 實驗描述: Hybrid Mamba-Attention CT features and frozen TAP-CT-B patient-level embeddings are reweighted by modality attention, concatenated, then classified by the original late-fusion MLP head.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_tapct_attention_fusion`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T14:20:38.905267`
- 訓練時間: `2026-05-14T14:03:19.848962` -> `2026-05-14T14:20:38.439163`
- 訓練耗時: 0.2885 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.74286 |
| std_accuracy | 0.11369 |
| mean_macro_f1 | 0.51326 |
| std_macro_f1 | 0.12278 |
| mean_macro_precision | 0.50336 |
| std_macro_precision | 0.14135 |
| mean_macro_recall | 0.59704 |
| std_macro_recall | 0.15025 |
| mean_balanced_accuracy | 0.59704 |
| std_balanced_accuracy | 0.15025 |

## Total Confusion Matrix
```json
[
  [
    11,
    0,
    3
  ],
  [
    0,
    1,
    4
  ],
  [
    4,
    6,
    37
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.71429 | 0.27778 | 0.2381 | 0.33333 | 0.33333 |
| 2 | 15 | 0.76923 | 0.52451 | 0.49167 | 0.59259 | 0.59259 |
| 3 | 10 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 4 | 5 | 0.84615 | 0.62963 | 0.62963 | 0.62963 | 0.62963 |
| 5 | 5 | 0.53846 | 0.55238 | 0.61111 | 0.8 | 0.8 |

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
  "fusion_projection_dim": 128,
  "tapct_attention_dim": 128,
  "tapct_gated_attention": true
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_confusion_matrix.png) — 218.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_loss.png) — 117.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_summary.png) — 266.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_confusion_matrix.png) — 219.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_loss.png) — 116.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_summary.png) — 291.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_confusion_matrix.png) — 221.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_loss.png) — 120.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_summary.png) — 316.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_confusion_matrix.png) — 220.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_loss.png) — 120.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_summary.png) — 287.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_confusion_matrix.png) — 215.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_loss.png) — 120.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_summary.png) — 302.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/metric_boxplot.png) — 136.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/total_confusion_matrix.png) — 232.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5.log) — 1.4 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_best_weight.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch10_weights-2026-05-14_14:04:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch10_weights-2026-05-14_14:04:17.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch20_weights-2026-05-14_14:05:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch20_weights-2026-05-14_14:05:13.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch30_weights-2026-05-14_14:06:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold1_epoch30_weights-2026-05-14_14:06:08.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_best_weight.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch10_weights-2026-05-14_14:07:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch10_weights-2026-05-14_14:07:31.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch20_weights-2026-05-14_14:08:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch20_weights-2026-05-14_14:08:26.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch30_weights-2026-05-14_14:09:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch30_weights-2026-05-14_14:09:20.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch40_weights-2026-05-14_14:10:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold2_epoch40_weights-2026-05-14_14:10:15.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_best_weight.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch10_weights-2026-05-14_14:11:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch10_weights-2026-05-14_14:11:37.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch20_weights-2026-05-14_14:12:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch20_weights-2026-05-14_14:12:31.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch30_weights-2026-05-14_14:13:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold3_epoch30_weights-2026-05-14_14:13:25.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_best_weight.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch10_weights-2026-05-14_14:15:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch10_weights-2026-05-14_14:15:14.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch20_weights-2026-05-14_14:16:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch20_weights-2026-05-14_14:16:08.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch30_weights-2026-05-14_14:17:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold4_epoch30_weights-2026-05-14_14:17:02.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_best_weight.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch10_weights-2026-05-14_14:18:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch10_weights-2026-05-14_14:18:23.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch20_weights-2026-05-14_14:19:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch20_weights-2026-05-14_14:19:17.pth) — 6.2 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch30_weights-2026-05-14_14:20:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19/fold5_epoch30_weights-2026-05-14_14:20:11.pth) — 6.2 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_attention_fusion_tap_ct_attention_fusion_aug100_class_2026-05-14_14:03:19",
    "model": "hybrid_mamba_tapct_attention_fusion",
    "experiment": {
      "name": "TAP-CT attention fusion + aug100/class",
      "description": "Hybrid Mamba-Attention CT features and frozen TAP-CT-B patient-level embeddings are reweighted by modality attention, concatenated, then classified by the original late-fusion MLP head."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T14:20:38.905267",
    "training_started_at": "2026-05-14T14:03:19.848962",
    "training_finished_at": "2026-05-14T14:20:38.439163",
    "training_duration_seconds": 1038.59,
    "training_duration_hours": 0.2885,
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
      "fusion_projection_dim": 128,
      "tapct_attention_dim": 128,
      "tapct_gated_attention": true
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 5,
      "accuracy": 0.71429,
      "macro_f1": 0.27778,
      "macro_precision": 0.2381,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333,
      "confusion_matrix": [
        [
          0,
          0,
          3
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
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "accuracy": 0.76923,
      "macro_f1": 0.52451,
      "macro_precision": 0.49167,
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
          2,
          0,
          7
        ]
      ]
    },
    {
      "fold": 3,
      "best_epoch": 10,
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
      "best_epoch": 5,
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
      "best_epoch": 5,
      "accuracy": 0.53846,
      "macro_f1": 0.55238,
      "macro_precision": 0.61111,
      "macro_recall": 0.8,
      "balanced_accuracy": 0.8,
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
          5,
          4
        ]
      ]
    }
  ],
  "summary": {
    "mean_accuracy": 0.74286,
    "std_accuracy": 0.11369,
    "mean_macro_f1": 0.51326,
    "std_macro_f1": 0.12278,
    "mean_macro_precision": 0.50336,
    "std_macro_precision": 0.14135,
    "mean_macro_recall": 0.59704,
    "std_macro_recall": 0.15025,
    "mean_balanced_accuracy": 0.59704,
    "std_balanced_accuracy": 0.15025
  },
  "total_confusion_matrix": [
    [
      11,
      0,
      3
    ],
    [
      0,
      1,
      4
    ],
    [
      4,
      6,
      37
    ]
  ]
}
```
