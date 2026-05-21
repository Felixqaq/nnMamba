# Run: hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34

## 一句話用途
- 方法名稱: `Balanced sampling + aug50/class`
- 實驗描述: Virtual train-fold augmentation brings all three classes to 50 samples before per-epoch balanced sampling.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-30T14:51:09.177533`
- 訓練時間: `2026-04-30T14:33:34.722081` -> `2026-04-30T14:51:08.687181`
- 訓練耗時: 0.2928 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.72747 |
| std_accuracy | 0.18537 |
| mean_macro_f1 | 0.56051 |
| std_macro_f1 | 0.1674 |
| mean_macro_precision | 0.6046 |
| std_macro_precision | 0.19204 |
| mean_macro_recall | 0.61926 |
| std_macro_recall | 0.13422 |
| mean_balanced_accuracy | 0.61926 |
| std_balanced_accuracy | 0.13422 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 70 | 0.71429 | 0.45238 | 0.60606 | 0.41111 | 0.41111 |
| 2 | 5 | 0.38462 | 0.38889 | 0.42222 | 0.62963 | 0.62963 |
| 3 | 30 | 0.84615 | 0.56373 | 0.53333 | 0.62963 | 0.62963 |
| 4 | 35 | 0.76923 | 0.52451 | 0.49167 | 0.59259 | 0.59259 |
| 5 | 60 | 0.92308 | 0.87302 | 0.9697 | 0.83333 | 0.83333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_confusion_matrix.png) — 216.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_loss.png) — 188.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_summary.png) — 399.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_confusion_matrix.png) — 226.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_loss.png) — 165.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_summary.png) — 358.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_confusion_matrix.png) — 218.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_loss.png) — 167.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_summary.png) — 378.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_confusion_matrix.png) — 215.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_loss.png) — 184.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_summary.png) — 370.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_confusion_matrix.png) — 215.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_loss.png) — 224.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_summary.png) — 402.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/metric_boxplot.png) — 137.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/total_confusion_matrix.png) — 228.9 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1.log) — 3.9 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4.log) — 2.5 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5.log) — 3.5 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch10_weights-2026-04-30_14:34:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch10_weights-2026-04-30_14:34:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch20_weights-2026-04-30_14:34:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch20_weights-2026-04-30_14:34:41.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch30_weights-2026-04-30_14:35:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch30_weights-2026-04-30_14:35:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch40_weights-2026-04-30_14:35:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch40_weights-2026-04-30_14:35:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch50_weights-2026-04-30_14:36:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch50_weights-2026-04-30_14:36:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch60_weights-2026-04-30_14:36:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch60_weights-2026-04-30_14:36:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch70_weights-2026-04-30_14:37:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch70_weights-2026-04-30_14:37:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch80_weights-2026-04-30_14:37:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch80_weights-2026-04-30_14:37:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch90_weights-2026-04-30_14:38:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold1_epoch90_weights-2026-04-30_14:38:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch10_weights-2026-04-30_14:39:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch10_weights-2026-04-30_14:39:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch20_weights-2026-04-30_14:39:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch20_weights-2026-04-30_14:39:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch30_weights-2026-04-30_14:40:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold2_epoch30_weights-2026-04-30_14:40:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch10_weights-2026-04-30_14:40:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch10_weights-2026-04-30_14:40:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch20_weights-2026-04-30_14:41:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch20_weights-2026-04-30_14:41:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch30_weights-2026-04-30_14:41:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch30_weights-2026-04-30_14:41:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch40_weights-2026-04-30_14:42:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch40_weights-2026-04-30_14:42:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch50_weights-2026-04-30_14:42:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold3_epoch50_weights-2026-04-30_14:42:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch10_weights-2026-04-30_14:43:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch10_weights-2026-04-30_14:43:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch20_weights-2026-04-30_14:44:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch20_weights-2026-04-30_14:44:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch30_weights-2026-04-30_14:44:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch30_weights-2026-04-30_14:44:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch40_weights-2026-04-30_14:45:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch40_weights-2026-04-30_14:45:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch50_weights-2026-04-30_14:45:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch50_weights-2026-04-30_14:45:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch60_weights-2026-04-30_14:46:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold4_epoch60_weights-2026-04-30_14:46:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch10_weights-2026-04-30_14:47:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch10_weights-2026-04-30_14:47:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch20_weights-2026-04-30_14:47:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch20_weights-2026-04-30_14:47:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch30_weights-2026-04-30_14:48:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch30_weights-2026-04-30_14:48:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch40_weights-2026-04-30_14:48:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch40_weights-2026-04-30_14:48:41.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch50_weights-2026-04-30_14:49:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch50_weights-2026-04-30_14:49:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch60_weights-2026-04-30_14:49:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch60_weights-2026-04-30_14:49:40.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch70_weights-2026-04-30_14:50:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch70_weights-2026-04-30_14:50:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch80_weights-2026-04-30_14:50:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34/fold5_epoch80_weights-2026-04-30_14:50:39.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_aug50_class_2026-04-30_14:33:34",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + aug50/class",
      "description": "Virtual train-fold augmentation brings all three classes to 50 samples before per-epoch balanced sampling."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-30T14:51:09.177533",
    "training_started_at": "2026-04-30T14:33:34.722081",
    "training_finished_at": "2026-04-30T14:51:08.687181",
    "training_duration_seconds": 1053.965,
    "training_duration_hours": 0.2928,
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
      "best_epoch": 70,
      "accuracy": 0.71429,
      "macro_f1": 0.45238,
      "macro_precision": 0.60606,
      "macro_recall": 0.41111,
      "balanced_accuracy": 0.41111
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.38462,
      "macro_f1": 0.38889,
      "macro_precision": 0.42222,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "accuracy": 0.84615,
      "macro_f1": 0.56373,
      "macro_precision": 0.53333,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 4,
      "best_epoch": 35,
      "accuracy": 0.76923,
      "macro_f1": 0.52451,
      "macro_precision": 0.49167,
      "macro_recall": 0.59259,
      "balanced_accuracy": 0.59259
    },
    {
      "fold": 5,
      "best_epoch": 60,
      "accuracy": 0.92308,
      "macro_f1": 0.87302,
      "macro_precision": 0.9697,
      "macro_recall": 0.83333,
      "balanced_accuracy": 0.83333
    }
  ],
  "summary": {
    "mean_accuracy": 0.72747,
    "std_accuracy": 0.18537,
    "mean_macro_f1": 0.56051,
    "std_macro_f1": 0.1674,
    "mean_macro_precision": 0.6046,
    "std_macro_precision": 0.19204,
    "mean_macro_recall": 0.61926,
    "std_macro_recall": 0.13422,
    "mean_balanced_accuracy": 0.61926,
    "std_balanced_accuracy": 0.13422
  }
}
```
