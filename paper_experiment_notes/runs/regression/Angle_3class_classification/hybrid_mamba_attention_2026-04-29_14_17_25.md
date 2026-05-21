# Run: hybrid_mamba_attention_2026-04-29_14:17:25

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-29T14:44:58.680508`
- 訓練時間: `2026-04-29T14:17:25.803139` -> `2026-04-29T14:44:58.096082`
- 訓練耗時: 0.459 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.78791 |
| std_accuracy | 0.10177 |
| mean_macro_f1 | 0.58609 |
| std_macro_f1 | 0.15456 |
| mean_macro_precision | 0.58918 |
| std_macro_precision | 0.19637 |
| mean_macro_recall | 0.62 |
| std_macro_recall | 0.11541 |
| mean_balanced_accuracy | 0.62 |
| std_balanced_accuracy | 0.11541 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | 0.78571 | 0.56642 | 0.5463 | 0.6 | 0.6 |
| 2 | 20 | 0.61538 | 0.42222 | 0.42063 | 0.51852 | 0.51852 |
| 3 | 25 | 0.76923 | 0.48677 | 0.46296 | 0.51852 | 0.51852 |
| 4 | 50 | 0.84615 | 0.58201 | 0.5463 | 0.62963 | 0.62963 |
| 5 | 55 | 0.92308 | 0.87302 | 0.9697 | 0.83333 | 0.83333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_confusion_matrix.png) — 194.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_loss.png) — 147.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_summary.png) — 336.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_confusion_matrix.png) — 192.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_loss.png) — 126.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_summary.png) — 317.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_confusion_matrix.png) — 196.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_loss.png) — 161.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_summary.png) — 381.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_confusion_matrix.png) — 195.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_loss.png) — 142.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_summary.png) — 344.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_confusion_matrix.png) — 193.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_loss.png) — 156.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_summary.png) — 337.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/metric_boxplot.png) — 112.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/total_confusion_matrix.png) — 206.6 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1.log) — 2.2 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2.log) — 2.0 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3.log) — 2.2 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5.log) — 3.3 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch10_weights-2026-04-29_14:18:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch10_weights-2026-04-29_14:18:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch20_weights-2026-04-29_14:19:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch20_weights-2026-04-29_14:19:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch30_weights-2026-04-29_14:20:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch30_weights-2026-04-29_14:20:01.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch40_weights-2026-04-29_14:20:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch40_weights-2026-04-29_14:20:53.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch50_weights-2026-04-29_14:21:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold1_epoch50_weights-2026-04-29_14:21:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch10_weights-2026-04-29_14:23:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch10_weights-2026-04-29_14:23:02.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch20_weights-2026-04-29_14:23:53.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch20_weights-2026-04-29_14:23:53.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch30_weights-2026-04-29_14:24:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch30_weights-2026-04-29_14:24:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch40_weights-2026-04-29_14:25:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold2_epoch40_weights-2026-04-29_14:25:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch10_weights-2026-04-29_14:27:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch10_weights-2026-04-29_14:27:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch20_weights-2026-04-29_14:28:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch20_weights-2026-04-29_14:28:06.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch30_weights-2026-04-29_14:28:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch30_weights-2026-04-29_14:28:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch40_weights-2026-04-29_14:29:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch40_weights-2026-04-29_14:29:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch50_weights-2026-04-29_14:30:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold3_epoch50_weights-2026-04-29_14:30:39.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch10_weights-2026-04-29_14:31:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch10_weights-2026-04-29_14:31:55.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch20_weights-2026-04-29_14:32:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch20_weights-2026-04-29_14:32:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch30_weights-2026-04-29_14:33:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch30_weights-2026-04-29_14:33:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch40_weights-2026-04-29_14:34:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch40_weights-2026-04-29_14:34:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch50_weights-2026-04-29_14:35:18.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch50_weights-2026-04-29_14:35:18.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch60_weights-2026-04-29_14:36:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch60_weights-2026-04-29_14:36:08.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch70_weights-2026-04-29_14:36:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold4_epoch70_weights-2026-04-29_14:36:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch10_weights-2026-04-29_14:38:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch10_weights-2026-04-29_14:38:40.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch20_weights-2026-04-29_14:39:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch20_weights-2026-04-29_14:39:31.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch30_weights-2026-04-29_14:40:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch30_weights-2026-04-29_14:40:21.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch40_weights-2026-04-29_14:41:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch40_weights-2026-04-29_14:41:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch50_weights-2026-04-29_14:42:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch50_weights-2026-04-29_14:42:01.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch60_weights-2026-04-29_14:42:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch60_weights-2026-04-29_14:42:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch70_weights-2026-04-29_14:43:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch70_weights-2026-04-29_14:43:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch80_weights-2026-04-29_14:44:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:17:25/fold5_epoch80_weights-2026-04-29_14:44:32.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-29_14:17:25",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-29T14:44:58.680508",
    "training_started_at": "2026-04-29T14:17:25.803139",
    "training_finished_at": "2026-04-29T14:44:58.096082",
    "training_duration_seconds": 1652.293,
    "training_duration_hours": 0.459,
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
      "macro_f1": 0.56642,
      "macro_precision": 0.5463,
      "macro_recall": 0.6,
      "balanced_accuracy": 0.6
    },
    {
      "fold": 2,
      "best_epoch": 20,
      "accuracy": 0.61538,
      "macro_f1": 0.42222,
      "macro_precision": 0.42063,
      "macro_recall": 0.51852,
      "balanced_accuracy": 0.51852
    },
    {
      "fold": 3,
      "best_epoch": 25,
      "accuracy": 0.76923,
      "macro_f1": 0.48677,
      "macro_precision": 0.46296,
      "macro_recall": 0.51852,
      "balanced_accuracy": 0.51852
    },
    {
      "fold": 4,
      "best_epoch": 50,
      "accuracy": 0.84615,
      "macro_f1": 0.58201,
      "macro_precision": 0.5463,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 5,
      "best_epoch": 55,
      "accuracy": 0.92308,
      "macro_f1": 0.87302,
      "macro_precision": 0.9697,
      "macro_recall": 0.83333,
      "balanced_accuracy": 0.83333
    }
  ],
  "summary": {
    "mean_accuracy": 0.78791,
    "std_accuracy": 0.10177,
    "mean_macro_f1": 0.58609,
    "std_macro_f1": 0.15456,
    "mean_macro_precision": 0.58918,
    "std_macro_precision": 0.19637,
    "mean_macro_recall": 0.62,
    "std_macro_recall": 0.11541,
    "mean_balanced_accuracy": 0.62,
    "std_balanced_accuracy": 0.11541
  }
}
```
