# Run: hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00

## 一句話用途
- 方法名稱: `Balanced sampling + aug100/class`
- 實驗描述: Virtual train-fold augmentation brings all three classes to 100 samples before per-epoch balanced sampling.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-05T11:01:41.745312`
- 訓練時間: `2026-05-05T10:31:00.944001` -> `2026-05-05T11:01:41.259734`
- 訓練耗時: 0.5112 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.72637 |
| std_accuracy | 0.21064 |
| mean_macro_f1 | 0.51062 |
| std_macro_f1 | 0.04234 |
| mean_macro_precision | 0.54916 |
| std_macro_precision | 0.07349 |
| mean_macro_recall | 0.54445 |
| std_macro_recall | 0.07371 |
| mean_balanced_accuracy | 0.54445 |
| std_balanced_accuracy | 0.07371 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | 0.78571 | 0.45652 | 0.58974 | 0.44444 | 0.44444 |
| 2 | 5 | 0.30769 | 0.46667 | 0.41667 | 0.66667 | 0.66667 |
| 3 | 30 | 0.84615 | 0.53801 | 0.52222 | 0.55556 | 0.55556 |
| 4 | 30 | 0.84615 | 0.56667 | 0.60606 | 0.55556 | 0.55556 |
| 5 | 50 | 0.84615 | 0.52525 | 0.61111 | 0.5 | 0.5 |

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
  "num_classes": 3
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_confusion_matrix.png) — 216.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_loss.png) — 163.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_summary.png) — 360.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_confusion_matrix.png) — 220.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_loss.png) — 148.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_summary.png) — 351.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_confusion_matrix.png) — 220.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_loss.png) — 183.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_summary.png) — 363.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_confusion_matrix.png) — 219.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_loss.png) — 165.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_summary.png) — 371.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_confusion_matrix.png) — 215.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_loss.png) — 188.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_summary.png) — 373.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/metric_boxplot.png) — 132.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/total_confusion_matrix.png) — 231.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1.log) — 2.2 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5.log) — 3.1 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch10_weights-2026-05-05_10:32:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch10_weights-2026-05-05_10:32:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch20_weights-2026-05-05_10:33:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch20_weights-2026-05-05_10:33:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch30_weights-2026-05-05_10:34:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch30_weights-2026-05-05_10:34:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch40_weights-2026-05-05_10:35:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch40_weights-2026-05-05_10:35:14.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch50_weights-2026-05-05_10:36:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold1_epoch50_weights-2026-05-05_10:36:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch10_weights-2026-05-05_10:37:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch10_weights-2026-05-05_10:37:49.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch20_weights-2026-05-05_10:38:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch20_weights-2026-05-05_10:38:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch30_weights-2026-05-05_10:39:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold2_epoch30_weights-2026-05-05_10:39:54.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch10_weights-2026-05-05_10:41:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch10_weights-2026-05-05_10:41:29.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch20_weights-2026-05-05_10:42:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch20_weights-2026-05-05_10:42:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch30_weights-2026-05-05_10:43:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch30_weights-2026-05-05_10:43:37.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch40_weights-2026-05-05_10:44:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch40_weights-2026-05-05_10:44:40.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch50_weights-2026-05-05_10:45:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold3_epoch50_weights-2026-05-05_10:45:44.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch10_weights-2026-05-05_10:47:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch10_weights-2026-05-05_10:47:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch20_weights-2026-05-05_10:48:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch20_weights-2026-05-05_10:48:54.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch30_weights-2026-05-05_10:49:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch30_weights-2026-05-05_10:49:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch40_weights-2026-05-05_10:51:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch40_weights-2026-05-05_10:51:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch50_weights-2026-05-05_10:52:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold4_epoch50_weights-2026-05-05_10:52:06.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch10_weights-2026-05-05_10:54:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch10_weights-2026-05-05_10:54:14.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch20_weights-2026-05-05_10:55:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch20_weights-2026-05-05_10:55:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch30_weights-2026-05-05_10:56:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch30_weights-2026-05-05_10:56:22.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch40_weights-2026-05-05_10:57:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch40_weights-2026-05-05_10:57:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch50_weights-2026-05-05_10:58:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch50_weights-2026-05-05_10:58:30.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch60_weights-2026-05-05_10:59:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch60_weights-2026-05-05_10:59:33.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch70_weights-2026-05-05_11:00:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00/fold5_epoch70_weights-2026-05-05_11:00:37.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_aug100_class_2026-05-05_10:31:00",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + aug100/class",
      "description": "Virtual train-fold augmentation brings all three classes to 100 samples before per-epoch balanced sampling."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-05T11:01:41.745312",
    "training_started_at": "2026-05-05T10:31:00.944001",
    "training_finished_at": "2026-05-05T11:01:41.259734",
    "training_duration_seconds": 1840.316,
    "training_duration_hours": 0.5112,
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
      "num_classes": 3
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 25,
      "accuracy": 0.78571,
      "macro_f1": 0.45652,
      "macro_precision": 0.58974,
      "macro_recall": 0.44444,
      "balanced_accuracy": 0.44444
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.30769,
      "macro_f1": 0.46667,
      "macro_precision": 0.41667,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "accuracy": 0.84615,
      "macro_f1": 0.53801,
      "macro_precision": 0.52222,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 4,
      "best_epoch": 30,
      "accuracy": 0.84615,
      "macro_f1": 0.56667,
      "macro_precision": 0.60606,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 5,
      "best_epoch": 50,
      "accuracy": 0.84615,
      "macro_f1": 0.52525,
      "macro_precision": 0.61111,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    }
  ],
  "summary": {
    "mean_accuracy": 0.72637,
    "std_accuracy": 0.21064,
    "mean_macro_f1": 0.51062,
    "std_macro_f1": 0.04234,
    "mean_macro_precision": 0.54916,
    "std_macro_precision": 0.07349,
    "mean_macro_recall": 0.54445,
    "std_macro_recall": 0.07371,
    "mean_balanced_accuracy": 0.54445,
    "std_balanced_accuracy": 0.07371
  }
}
```
