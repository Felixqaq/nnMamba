# Run: hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05

## 一句話用途
- 方法名稱: `Balanced sampling + aug300/class`
- 實驗描述: Virtual train-fold augmentation brings all three classes to 300 samples before per-epoch balanced sampling.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-07T10:54:52.668291`
- 訓練時間: `2026-05-07T09:38:05.312058` -> `2026-05-07T10:54:52.182644`
- 訓練耗時: 1.2797 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.75934 |
| std_accuracy | 0.11725 |
| mean_macro_f1 | 0.57129 |
| std_macro_f1 | 0.07481 |
| mean_macro_precision | 0.55862 |
| std_macro_precision | 0.07898 |
| mean_macro_recall | 0.63185 |
| std_macro_recall | 0.11603 |
| mean_balanced_accuracy | 0.63185 |
| std_balanced_accuracy | 0.11603 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 0.64286 | 0.43609 | 0.42593 | 0.45556 | 0.45556 |
| 2 | 15 | 0.61538 | 0.62735 | 0.64286 | 0.81481 | 0.81481 |
| 3 | 70 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 4 | 15 | 0.76923 | 0.56022 | 0.54167 | 0.59259 | 0.59259 |
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
  "target_mode": "angle_3class"
}
```

## Artifact Index
### figures
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_confusion_matrix.png) — 216.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_loss.png) — 149.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_summary.png) — 343.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_confusion_matrix.png) — 226.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_loss.png) — 143.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_summary.png) — 360.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_confusion_matrix.png) — 220.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_loss.png) — 146.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_summary.png) — 377.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_confusion_matrix.png) — 215.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_loss.png) — 139.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_summary.png) — 362.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_confusion_matrix.png) — 217.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_loss.png) — 145.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_summary.png) — 362.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/metric_boxplot.png) — 135.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/total_confusion_matrix.png) — 232.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3.log) — 3.9 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5.log) — 2.0 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch10_weights-2026-05-07_09:40:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch10_weights-2026-05-07_09:40:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch20_weights-2026-05-07_09:43:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch20_weights-2026-05-07_09:43:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch30_weights-2026-05-07_09:46:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch30_weights-2026-05-07_09:46:16.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch40_weights-2026-05-07_09:48:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold1_epoch40_weights-2026-05-07_09:48:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch10_weights-2026-05-07_09:53:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch10_weights-2026-05-07_09:53:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch20_weights-2026-05-07_09:55:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch20_weights-2026-05-07_09:55:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch30_weights-2026-05-07_09:58:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch30_weights-2026-05-07_09:58:29.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch40_weights-2026-05-07_10:01:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold2_epoch40_weights-2026-05-07_10:01:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch10_weights-2026-05-07_10:05:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch10_weights-2026-05-07_10:05:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch20_weights-2026-05-07_10:07:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch20_weights-2026-05-07_10:07:55.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch30_weights-2026-05-07_10:10:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch30_weights-2026-05-07_10:10:38.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch40_weights-2026-05-07_10:13:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch40_weights-2026-05-07_10:13:19.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch50_weights-2026-05-07_10:16:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch50_weights-2026-05-07_10:16:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch60_weights-2026-05-07_10:18:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch60_weights-2026-05-07_10:18:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch70_weights-2026-05-07_10:21:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch70_weights-2026-05-07_10:21:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch80_weights-2026-05-07_10:24:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch80_weights-2026-05-07_10:24:05.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch90_weights-2026-05-07_10:26:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold3_epoch90_weights-2026-05-07_10:26:47.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch10_weights-2026-05-07_10:32:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch10_weights-2026-05-07_10:32:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch20_weights-2026-05-07_10:34:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch20_weights-2026-05-07_10:34:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch30_weights-2026-05-07_10:37:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch30_weights-2026-05-07_10:37:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch40_weights-2026-05-07_10:40:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold4_epoch40_weights-2026-05-07_10:40:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch10_weights-2026-05-07_10:44:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch10_weights-2026-05-07_10:44:14.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch20_weights-2026-05-07_10:46:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch20_weights-2026-05-07_10:46:55.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch30_weights-2026-05-07_10:49:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch30_weights-2026-05-07_10:49:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch40_weights-2026-05-07_10:52:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05/fold5_epoch40_weights-2026-05-07_10:52:13.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-07_09:38:05",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + aug300/class",
      "description": "Virtual train-fold augmentation brings all three classes to 300 samples before per-epoch balanced sampling."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-07T10:54:52.668291",
    "training_started_at": "2026-05-07T09:38:05.312058",
    "training_finished_at": "2026-05-07T10:54:52.182644",
    "training_duration_seconds": 4606.871,
    "training_duration_hours": 1.2797,
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
      "target_mode": "angle_3class"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 15,
      "accuracy": 0.64286,
      "macro_f1": 0.43609,
      "macro_precision": 0.42593,
      "macro_recall": 0.45556,
      "balanced_accuracy": 0.45556
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "accuracy": 0.61538,
      "macro_f1": 0.62735,
      "macro_precision": 0.64286,
      "macro_recall": 0.81481,
      "balanced_accuracy": 0.81481
    },
    {
      "fold": 3,
      "best_epoch": 70,
      "accuracy": 0.84615,
      "macro_f1": 0.58201,
      "macro_precision": 0.5463,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 4,
      "best_epoch": 15,
      "accuracy": 0.76923,
      "macro_f1": 0.56022,
      "macro_precision": 0.54167,
      "macro_recall": 0.59259,
      "balanced_accuracy": 0.59259
    },
    {
      "fold": 5,
      "best_epoch": 20,
      "accuracy": 0.92308,
      "macro_f1": 0.65079,
      "macro_precision": 0.63636,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    }
  ],
  "summary": {
    "mean_accuracy": 0.75934,
    "std_accuracy": 0.11725,
    "mean_macro_f1": 0.57129,
    "std_macro_f1": 0.07481,
    "mean_macro_precision": 0.55862,
    "std_macro_precision": 0.07898,
    "mean_macro_recall": 0.63185,
    "std_macro_recall": 0.11603,
    "mean_balanced_accuracy": 0.63185,
    "std_balanced_accuracy": 0.11603
  }
}
```
