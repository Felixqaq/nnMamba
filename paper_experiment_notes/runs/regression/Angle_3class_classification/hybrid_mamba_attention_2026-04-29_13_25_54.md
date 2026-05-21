# Run: hybrid_mamba_attention_2026-04-29_13:25:54

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-29T13:36:58.979917`
- 訓練時間: `2026-04-29T13:25:54.007012` -> `2026-04-29T13:36:58.531364`
- 訓練耗時: 0.1846 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.63516 |
| std_accuracy | 0.21572 |
| mean_macro_f1 | 0.50424 |
| std_macro_f1 | 0.15411 |
| mean_macro_precision | 0.57699 |
| std_macro_precision | 0.12881 |
| mean_macro_recall | 0.56296 |
| std_macro_recall | 0.11158 |
| mean_balanced_accuracy | 0.56296 |
| std_balanced_accuracy | 0.11158 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 0.71429 | 0.54497 | 0.54167 | 0.56667 | 0.56667 |
| 2 | 5 | 0.23077 | 0.28889 | 0.69697 | 0.48148 | 0.48148 |
| 3 | 40 | 0.84615 | 0.53801 | 0.52222 | 0.55556 | 0.55556 |
| 4 | 65 | 0.61538 | 0.40196 | 0.38333 | 0.44444 | 0.44444 |
| 5 | 40 | 0.76923 | 0.74737 | 0.74074 | 0.76667 | 0.76667 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_confusion_matrix.png) — 192.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_loss.png) — 149.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_summary.png) — 365.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_confusion_matrix.png) — 196.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_loss.png) — 117.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_summary.png) — 317.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_confusion_matrix.png) — 197.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_loss.png) — 116.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_summary.png) — 311.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_confusion_matrix.png) — 193.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_loss.png) — 111.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_summary.png) — 298.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_confusion_matrix.png) — 195.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_loss.png) — 176.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_summary.png) — 385.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/metric_boxplot.png) — 111.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/total_confusion_matrix.png) — 204.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1.log) — 2.7 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3.log) — 2.7 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4.log) — 3.7 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5.log) — 2.7 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch10_weights-2026-04-29_13:26:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch10_weights-2026-04-29_13:26:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch20_weights-2026-04-29_13:26:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch20_weights-2026-04-29_13:26:38.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch30_weights-2026-04-29_13:26:58.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch30_weights-2026-04-29_13:26:58.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch40_weights-2026-04-29_13:27:18.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch40_weights-2026-04-29_13:27:18.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch50_weights-2026-04-29_13:27:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch50_weights-2026-04-29_13:27:38.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch60_weights-2026-04-29_13:27:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold1_epoch60_weights-2026-04-29_13:27:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch10_weights-2026-04-29_13:28:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch10_weights-2026-04-29_13:28:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch20_weights-2026-04-29_13:28:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch20_weights-2026-04-29_13:28:53.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch30_weights-2026-04-29_13:29:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold2_epoch30_weights-2026-04-29_13:29:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch10_weights-2026-04-29_13:29:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch10_weights-2026-04-29_13:29:38.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch20_weights-2026-04-29_13:29:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch20_weights-2026-04-29_13:29:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch30_weights-2026-04-29_13:30:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch30_weights-2026-04-29_13:30:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch40_weights-2026-04-29_13:30:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch40_weights-2026-04-29_13:30:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch50_weights-2026-04-29_13:30:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch50_weights-2026-04-29_13:30:52.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch60_weights-2026-04-29_13:31:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold3_epoch60_weights-2026-04-29_13:31:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch10_weights-2026-04-29_13:31:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch10_weights-2026-04-29_13:31:49.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch20_weights-2026-04-29_13:32:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch20_weights-2026-04-29_13:32:08.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch30_weights-2026-04-29_13:32:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch30_weights-2026-04-29_13:32:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch40_weights-2026-04-29_13:32:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch40_weights-2026-04-29_13:32:47.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch50_weights-2026-04-29_13:33:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch50_weights-2026-04-29_13:33:06.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch60_weights-2026-04-29_13:33:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch60_weights-2026-04-29_13:33:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch70_weights-2026-04-29_13:33:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch70_weights-2026-04-29_13:33:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch80_weights-2026-04-29_13:34:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch80_weights-2026-04-29_13:34:06.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch90_weights-2026-04-29_13:34:25.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold4_epoch90_weights-2026-04-29_13:34:25.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch10_weights-2026-04-29_13:34:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch10_weights-2026-04-29_13:34:56.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch20_weights-2026-04-29_13:35:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch20_weights-2026-04-29_13:35:16.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch30_weights-2026-04-29_13:35:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch30_weights-2026-04-29_13:35:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch40_weights-2026-04-29_13:35:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch40_weights-2026-04-29_13:35:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch50_weights-2026-04-29_13:36:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch50_weights-2026-04-29_13:36:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch60_weights-2026-04-29_13:36:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:25:54/fold5_epoch60_weights-2026-04-29_13:36:38.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-29_13:25:54",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-29T13:36:58.979917",
    "training_started_at": "2026-04-29T13:25:54.007012",
    "training_finished_at": "2026-04-29T13:36:58.531364",
    "training_duration_seconds": 664.524,
    "training_duration_hours": 0.1846,
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
      "best_epoch": 40,
      "accuracy": 0.71429,
      "macro_f1": 0.54497,
      "macro_precision": 0.54167,
      "macro_recall": 0.56667,
      "balanced_accuracy": 0.56667
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.23077,
      "macro_f1": 0.28889,
      "macro_precision": 0.69697,
      "macro_recall": 0.48148,
      "balanced_accuracy": 0.48148
    },
    {
      "fold": 3,
      "best_epoch": 40,
      "accuracy": 0.84615,
      "macro_f1": 0.53801,
      "macro_precision": 0.52222,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 4,
      "best_epoch": 65,
      "accuracy": 0.61538,
      "macro_f1": 0.40196,
      "macro_precision": 0.38333,
      "macro_recall": 0.44444,
      "balanced_accuracy": 0.44444
    },
    {
      "fold": 5,
      "best_epoch": 40,
      "accuracy": 0.76923,
      "macro_f1": 0.74737,
      "macro_precision": 0.74074,
      "macro_recall": 0.76667,
      "balanced_accuracy": 0.76667
    }
  ],
  "summary": {
    "mean_accuracy": 0.63516,
    "std_accuracy": 0.21572,
    "mean_macro_f1": 0.50424,
    "std_macro_f1": 0.15411,
    "mean_macro_precision": 0.57699,
    "std_macro_precision": 0.12881,
    "mean_macro_recall": 0.56296,
    "std_macro_recall": 0.11158,
    "mean_balanced_accuracy": 0.56296,
    "std_balanced_accuracy": 0.11158
  }
}
```
