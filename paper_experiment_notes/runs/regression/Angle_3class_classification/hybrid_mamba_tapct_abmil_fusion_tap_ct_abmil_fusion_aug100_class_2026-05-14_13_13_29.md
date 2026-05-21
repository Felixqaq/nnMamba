# Run: hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29

## 一句話用途
- 方法名稱: `TAP-CT ABMIL fusion + aug100/class`
- 實驗描述: Hybrid Mamba-Attention CT features and frozen TAP-CT-B patient-level embeddings are projected as modality instances, pooled by gated ABMIL attention, then classified.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_tapct_abmil_fusion`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-14T13:29:19.366943`
- 訓練時間: `2026-05-14T13:13:29.708120` -> `2026-05-14T13:29:18.588483`
- 訓練耗時: 0.2636 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.81868 |
| std_accuracy | 0.03405 |
| mean_macro_f1 | 0.53621 |
| std_macro_f1 | 0.04306 |
| mean_macro_precision | 0.56315 |
| std_macro_precision | 0.04727 |
| mean_macro_recall | 0.5563 |
| std_macro_recall | 0.06287 |
| mean_balanced_accuracy | 0.5563 |
| std_balanced_accuracy | 0.06287 |

## Total Confusion Matrix
```json
[
  [
    10,
    0,
    4
  ],
  [
    0,
    0,
    5
  ],
  [
    3,
    0,
    44
  ]
]
```

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.78571 | 0.45652 | 0.58974 | 0.44444 | 0.44444 |
| 2 | 5 | 0.76923 | 0.52451 | 0.49167 | 0.59259 | 0.59259 |
| 3 | 5 | 0.84615 | 0.56667 | 0.60606 | 0.55556 | 0.55556 |
| 4 | 5 | 0.84615 | 0.56667 | 0.60606 | 0.55556 | 0.55556 |
| 5 | 5 | 0.84615 | 0.56667 | 0.52222 | 0.63333 | 0.63333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_confusion_matrix.png) — 215.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_loss.png) — 112.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_summary.png) — 294.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_confusion_matrix.png) — 217.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_loss.png) — 111.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_summary.png) — 294.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_confusion_matrix.png) — 220.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_loss.png) — 112.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_summary.png) — 298.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_confusion_matrix.png) — 219.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_loss.png) — 112.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_summary.png) — 273.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_confusion_matrix.png) — 219.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_loss.png) — 112.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_summary.png) — 270.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/metric_boxplot.png) — 142.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/total_confusion_matrix.png) — 232.7 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5.log) — 1.4 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_best_weight.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch10_weights-2026-05-14_13:14:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch10_weights-2026-05-14_13:14:25.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch20_weights-2026-05-14_13:15:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch20_weights-2026-05-14_13:15:19.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch30_weights-2026-05-14_13:16:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold1_epoch30_weights-2026-05-14_13:16:13.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_best_weight.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch10_weights-2026-05-14_13:17:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch10_weights-2026-05-14_13:17:59.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch20_weights-2026-05-14_13:18:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch20_weights-2026-05-14_13:18:51.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch30_weights-2026-05-14_13:19:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold2_epoch30_weights-2026-05-14_13:19:43.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_best_weight.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch10_weights-2026-05-14_13:21:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch10_weights-2026-05-14_13:21:01.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch20_weights-2026-05-14_13:21:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch20_weights-2026-05-14_13:21:52.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch30_weights-2026-05-14_13:22:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold3_epoch30_weights-2026-05-14_13:22:44.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_best_weight.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch10_weights-2026-05-14_13:24:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch10_weights-2026-05-14_13:24:01.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch20_weights-2026-05-14_13:24:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch20_weights-2026-05-14_13:24:53.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch30_weights-2026-05-14_13:25:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold4_epoch30_weights-2026-05-14_13:25:45.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_best_weight.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch10_weights-2026-05-14_13:27:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch10_weights-2026-05-14_13:27:04.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch20_weights-2026-05-14_13:27:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch20_weights-2026-05-14_13:27:57.pth) — 5.6 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch30_weights-2026-05-14_13:28:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29/fold5_epoch30_weights-2026-05-14_13:28:50.pth) — 5.6 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_tapct_abmil_fusion_tap_ct_abmil_fusion_aug100_class_2026-05-14_13:13:29",
    "model": "hybrid_mamba_tapct_abmil_fusion",
    "experiment": {
      "name": "TAP-CT ABMIL fusion + aug100/class",
      "description": "Hybrid Mamba-Attention CT features and frozen TAP-CT-B patient-level embeddings are projected as modality instances, pooled by gated ABMIL attention, then classified."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-14T13:29:19.366943",
    "training_started_at": "2026-05-14T13:13:29.708120",
    "training_finished_at": "2026-05-14T13:29:18.588483",
    "training_duration_seconds": 948.88,
    "training_duration_hours": 0.2636,
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
      "best_epoch": 10,
      "accuracy": 0.78571,
      "macro_f1": 0.45652,
      "macro_precision": 0.58974,
      "macro_recall": 0.44444,
      "balanced_accuracy": 0.44444,
      "confusion_matrix": [
        [
          1,
          0,
          2
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
      "best_epoch": 5,
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
      "best_epoch": 5,
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
      "fold": 4,
      "best_epoch": 5,
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
      "best_epoch": 5,
      "accuracy": 0.84615,
      "macro_f1": 0.56667,
      "macro_precision": 0.52222,
      "macro_recall": 0.63333,
      "balanced_accuracy": 0.63333,
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
          1,
          0,
          9
        ]
      ]
    }
  ],
  "summary": {
    "mean_accuracy": 0.81868,
    "std_accuracy": 0.03405,
    "mean_macro_f1": 0.53621,
    "std_macro_f1": 0.04306,
    "mean_macro_precision": 0.56315,
    "std_macro_precision": 0.04727,
    "mean_macro_recall": 0.5563,
    "std_macro_recall": 0.06287,
    "mean_balanced_accuracy": 0.5563,
    "std_balanced_accuracy": 0.06287
  },
  "total_confusion_matrix": [
    [
      10,
      0,
      4
    ],
    [
      0,
      0,
      5
    ],
    [
      3,
      0,
      44
    ]
  ]
}
```
