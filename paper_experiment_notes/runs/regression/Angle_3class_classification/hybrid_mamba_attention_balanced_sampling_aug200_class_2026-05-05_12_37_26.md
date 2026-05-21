# Run: hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26

## 一句話用途
- 方法名稱: `Balanced sampling + aug200/class`
- 實驗描述: Virtual train-fold augmentation brings all three classes to 200 samples before per-epoch balanced sampling.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-05T13:28:38.900809`
- 訓練時間: `2026-05-05T12:37:26.072654` -> `2026-05-05T13:28:38.429927`
- 訓練耗時: 0.8534 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.75714 |
| std_accuracy | 0.05881 |
| mean_macro_f1 | 0.50992 |
| std_macro_f1 | 0.04043 |
| mean_macro_precision | 0.56351 |
| std_macro_precision | 0.04294 |
| mean_macro_recall | 0.50074 |
| std_macro_recall | 0.04999 |
| mean_balanced_accuracy | 0.50074 |
| std_balanced_accuracy | 0.04999 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 0.78571 | 0.50794 | 0.49495 | 0.52222 | 0.52222 |
| 2 | 15 | 0.69231 | 0.43333 | 0.57576 | 0.40741 | 0.40741 |
| 3 | 10 | 0.69231 | 0.53571 | 0.53571 | 0.55556 | 0.55556 |
| 4 | 10 | 0.76923 | 0.54737 | 0.6 | 0.51852 | 0.51852 |
| 5 | 80 | 0.84615 | 0.52525 | 0.61111 | 0.5 | 0.5 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_confusion_matrix.png) — 220.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_loss.png) — 178.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_summary.png) — 367.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_confusion_matrix.png) — 219.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_loss.png) — 159.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_summary.png) — 332.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_confusion_matrix.png) — 216.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_loss.png) — 161.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_summary.png) — 344.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_confusion_matrix.png) — 219.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_loss.png) — 160.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_summary.png) — 351.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_confusion_matrix.png) — 216.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_loss.png) — 179.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_summary.png) — 400.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/metric_boxplot.png) — 133.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/total_confusion_matrix.png) — 232.5 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1.log) — 2.0 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5.log) — 4.2 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch10_weights-2026-05-05_12:39:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch10_weights-2026-05-05_12:39:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch20_weights-2026-05-05_12:41:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch20_weights-2026-05-05_12:41:01.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch30_weights-2026-05-05_12:42:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch30_weights-2026-05-05_12:42:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch40_weights-2026-05-05_12:44:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold1_epoch40_weights-2026-05-05_12:44:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch10_weights-2026-05-05_12:48:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch10_weights-2026-05-05_12:48:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch20_weights-2026-05-05_12:49:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch20_weights-2026-05-05_12:49:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch30_weights-2026-05-05_12:51:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch30_weights-2026-05-05_12:51:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch40_weights-2026-05-05_12:53:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold2_epoch40_weights-2026-05-05_12:53:30.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch10_weights-2026-05-05_12:56:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch10_weights-2026-05-05_12:56:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch20_weights-2026-05-05_12:58:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch20_weights-2026-05-05_12:58:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch30_weights-2026-05-05_12:59:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold3_epoch30_weights-2026-05-05_12:59:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch10_weights-2026-05-05_13:03:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch10_weights-2026-05-05_13:03:22.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch20_weights-2026-05-05_13:05:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch20_weights-2026-05-05_13:05:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch30_weights-2026-05-05_13:06:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold4_epoch30_weights-2026-05-05_13:06:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch100_weights-2026-05-05_13:26:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch100_weights-2026-05-05_13:26:50.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch10_weights-2026-05-05_13:10:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch10_weights-2026-05-05_13:10:33.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch20_weights-2026-05-05_13:12:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch20_weights-2026-05-05_13:12:21.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch30_weights-2026-05-05_13:14:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch30_weights-2026-05-05_13:14:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch40_weights-2026-05-05_13:15:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch40_weights-2026-05-05_13:15:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch50_weights-2026-05-05_13:17:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch50_weights-2026-05-05_13:17:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch60_weights-2026-05-05_13:19:35.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch60_weights-2026-05-05_13:19:35.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch70_weights-2026-05-05_13:21:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch70_weights-2026-05-05_13:21:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch80_weights-2026-05-05_13:23:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch80_weights-2026-05-05_13:23:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch90_weights-2026-05-05_13:25:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26/fold5_epoch90_weights-2026-05-05_13:25:01.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_aug200_class_2026-05-05_12:37:26",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + aug200/class",
      "description": "Virtual train-fold augmentation brings all three classes to 200 samples before per-epoch balanced sampling."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-05T13:28:38.900809",
    "training_started_at": "2026-05-05T12:37:26.072654",
    "training_finished_at": "2026-05-05T13:28:38.429927",
    "training_duration_seconds": 3072.357,
    "training_duration_hours": 0.8534,
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
      "best_epoch": 20,
      "accuracy": 0.78571,
      "macro_f1": 0.50794,
      "macro_precision": 0.49495,
      "macro_recall": 0.52222,
      "balanced_accuracy": 0.52222
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "accuracy": 0.69231,
      "macro_f1": 0.43333,
      "macro_precision": 0.57576,
      "macro_recall": 0.40741,
      "balanced_accuracy": 0.40741
    },
    {
      "fold": 3,
      "best_epoch": 10,
      "accuracy": 0.69231,
      "macro_f1": 0.53571,
      "macro_precision": 0.53571,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 4,
      "best_epoch": 10,
      "accuracy": 0.76923,
      "macro_f1": 0.54737,
      "macro_precision": 0.6,
      "macro_recall": 0.51852,
      "balanced_accuracy": 0.51852
    },
    {
      "fold": 5,
      "best_epoch": 80,
      "accuracy": 0.84615,
      "macro_f1": 0.52525,
      "macro_precision": 0.61111,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    }
  ],
  "summary": {
    "mean_accuracy": 0.75714,
    "std_accuracy": 0.05881,
    "mean_macro_f1": 0.50992,
    "std_macro_f1": 0.04043,
    "mean_macro_precision": 0.56351,
    "std_macro_precision": 0.04294,
    "mean_macro_recall": 0.50074,
    "std_macro_recall": 0.04999,
    "mean_balanced_accuracy": 0.50074,
    "std_balanced_accuracy": 0.04999
  }
}
```
