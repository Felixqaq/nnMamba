# Run: hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46

## 一句話用途
- 方法名稱: `Balanced sampling + aug75/class`
- 實驗描述: Virtual train-fold augmentation brings all three classes to 75 samples before per-epoch balanced sampling.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-30T15:22:11.677856`
- 訓練時間: `2026-04-30T14:52:46.021136` -> `2026-04-30T15:22:11.202151`
- 訓練耗時: 0.4903 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.6967 |
| std_accuracy | 0.10914 |
| mean_macro_f1 | 0.45017 |
| std_macro_f1 | 0.06951 |
| mean_macro_precision | 0.50611 |
| std_macro_precision | 0.09671 |
| mean_macro_recall | 0.45704 |
| std_macro_recall | 0.06327 |
| mean_balanced_accuracy | 0.45704 |
| std_balanced_accuracy | 0.06327 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 75 | 0.71429 | 0.43939 | 0.58333 | 0.41111 | 0.41111 |
| 2 | 50 | 0.53846 | 0.37229 | 0.39167 | 0.48148 | 0.48148 |
| 3 | 55 | 0.84615 | 0.53801 | 0.52222 | 0.55556 | 0.55556 |
| 4 | 30 | 0.61538 | 0.37895 | 0.4 | 0.37037 | 0.37037 |
| 5 | 45 | 0.76923 | 0.52222 | 0.63333 | 0.46667 | 0.46667 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_confusion_matrix.png) — 218.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_loss.png) — 178.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_summary.png) — 393.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_confusion_matrix.png) — 213.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_loss.png) — 176.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_summary.png) — 360.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_confusion_matrix.png) — 219.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_loss.png) — 192.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_summary.png) — 396.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_confusion_matrix.png) — 213.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_loss.png) — 155.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_summary.png) — 355.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_confusion_matrix.png) — 217.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_loss.png) — 194.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_summary.png) — 403.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/metric_boxplot.png) — 128.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/total_confusion_matrix.png) — 230.5 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1.log) — 4.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3.log) — 3.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5.log) — 2.9 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch100_weights-2026-04-30_15:00:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch100_weights-2026-04-30_15:00:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch10_weights-2026-04-30_14:53:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch10_weights-2026-04-30_14:53:31.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch20_weights-2026-04-30_14:54:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch20_weights-2026-04-30_14:54:16.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch30_weights-2026-04-30_14:55:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch30_weights-2026-04-30_14:55:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch40_weights-2026-04-30_14:55:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch40_weights-2026-04-30_14:55:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch50_weights-2026-04-30_14:56:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch50_weights-2026-04-30_14:56:30.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch60_weights-2026-04-30_14:57:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch60_weights-2026-04-30_14:57:14.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch70_weights-2026-04-30_14:57:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch70_weights-2026-04-30_14:57:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch80_weights-2026-04-30_14:58:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch80_weights-2026-04-30_14:58:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch90_weights-2026-04-30_14:59:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold1_epoch90_weights-2026-04-30_14:59:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch10_weights-2026-04-30_15:01:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch10_weights-2026-04-30_15:01:19.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch20_weights-2026-04-30_15:02:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch20_weights-2026-04-30_15:02:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch30_weights-2026-04-30_15:02:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch30_weights-2026-04-30_15:02:47.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch40_weights-2026-04-30_15:03:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch40_weights-2026-04-30_15:03:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch50_weights-2026-04-30_15:04:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch50_weights-2026-04-30_15:04:16.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch60_weights-2026-04-30_15:04:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch60_weights-2026-04-30_15:04:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch70_weights-2026-04-30_15:05:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold2_epoch70_weights-2026-04-30_15:05:37.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch10_weights-2026-04-30_15:06:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch10_weights-2026-04-30_15:06:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch20_weights-2026-04-30_15:07:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch20_weights-2026-04-30_15:07:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch30_weights-2026-04-30_15:08:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch30_weights-2026-04-30_15:08:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch40_weights-2026-04-30_15:09:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch40_weights-2026-04-30_15:09:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch50_weights-2026-04-30_15:09:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch50_weights-2026-04-30_15:09:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch60_weights-2026-04-30_15:10:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch60_weights-2026-04-30_15:10:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch70_weights-2026-04-30_15:11:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch70_weights-2026-04-30_15:11:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch80_weights-2026-04-30_15:12:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold3_epoch80_weights-2026-04-30_15:12:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch10_weights-2026-04-30_15:13:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch10_weights-2026-04-30_15:13:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch20_weights-2026-04-30_15:14:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch20_weights-2026-04-30_15:14:01.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch30_weights-2026-04-30_15:14:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch30_weights-2026-04-30_15:14:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch40_weights-2026-04-30_15:15:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch40_weights-2026-04-30_15:15:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch50_weights-2026-04-30_15:16:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold4_epoch50_weights-2026-04-30_15:16:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch10_weights-2026-04-30_15:17:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch10_weights-2026-04-30_15:17:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch20_weights-2026-04-30_15:18:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch20_weights-2026-04-30_15:18:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch30_weights-2026-04-30_15:18:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch30_weights-2026-04-30_15:18:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch40_weights-2026-04-30_15:19:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch40_weights-2026-04-30_15:19:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch50_weights-2026-04-30_15:20:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch50_weights-2026-04-30_15:20:25.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch60_weights-2026-04-30_15:21:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch60_weights-2026-04-30_15:21:07.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch70_weights-2026-04-30_15:21:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46/fold5_epoch70_weights-2026-04-30_15:21:49.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_aug75_class_2026-04-30_14:52:46",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + aug75/class",
      "description": "Virtual train-fold augmentation brings all three classes to 75 samples before per-epoch balanced sampling."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-30T15:22:11.677856",
    "training_started_at": "2026-04-30T14:52:46.021136",
    "training_finished_at": "2026-04-30T15:22:11.202151",
    "training_duration_seconds": 1765.181,
    "training_duration_hours": 0.4903,
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
      "best_epoch": 75,
      "accuracy": 0.71429,
      "macro_f1": 0.43939,
      "macro_precision": 0.58333,
      "macro_recall": 0.41111,
      "balanced_accuracy": 0.41111
    },
    {
      "fold": 2,
      "best_epoch": 50,
      "accuracy": 0.53846,
      "macro_f1": 0.37229,
      "macro_precision": 0.39167,
      "macro_recall": 0.48148,
      "balanced_accuracy": 0.48148
    },
    {
      "fold": 3,
      "best_epoch": 55,
      "accuracy": 0.84615,
      "macro_f1": 0.53801,
      "macro_precision": 0.52222,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 4,
      "best_epoch": 30,
      "accuracy": 0.61538,
      "macro_f1": 0.37895,
      "macro_precision": 0.4,
      "macro_recall": 0.37037,
      "balanced_accuracy": 0.37037
    },
    {
      "fold": 5,
      "best_epoch": 45,
      "accuracy": 0.76923,
      "macro_f1": 0.52222,
      "macro_precision": 0.63333,
      "macro_recall": 0.46667,
      "balanced_accuracy": 0.46667
    }
  ],
  "summary": {
    "mean_accuracy": 0.6967,
    "std_accuracy": 0.10914,
    "mean_macro_f1": 0.45017,
    "std_macro_f1": 0.06951,
    "mean_macro_precision": 0.50611,
    "std_macro_precision": 0.09671,
    "mean_macro_recall": 0.45704,
    "std_macro_recall": 0.06327,
    "mean_balanced_accuracy": 0.45704,
    "std_balanced_accuracy": 0.06327
  }
}
```
