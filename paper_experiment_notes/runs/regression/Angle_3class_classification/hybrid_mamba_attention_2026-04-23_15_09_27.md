# Run: hybrid_mamba_attention_2026-04-23_15:09:27

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-23T15:20:53.086604`
- 訓練時間: `2026-04-23T15:09:27.490212` -> `2026-04-23T15:20:52.629460`
- 訓練耗時: 0.1903 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.68461 |
| std_accuracy | 0.12016 |
| mean_macro_f1 | 0.56503 |
| std_macro_f1 | 0.12827 |
| mean_macro_precision | 0.59122 |
| std_macro_precision | 0.08652 |
| mean_macro_recall | 0.60222 |
| std_macro_recall | 0.19586 |
| mean_balanced_accuracy | 0.60222 |
| std_balanced_accuracy | 0.19586 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 | 0.5 | 0.37719 | 0.55556 | 0.31111 | 0.31111 |
| 2 | 50 | 0.76923 | 0.73889 | 0.7 | 0.88889 | 0.88889 |
| 3 | 50 | 0.84615 | 0.56373 | 0.53333 | 0.62963 | 0.62963 |
| 4 | 45 | 0.69231 | 0.48148 | 0.48148 | 0.48148 | 0.48148 |
| 5 | 65 | 0.61538 | 0.66387 | 0.68571 | 0.7 | 0.7 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_confusion_matrix.png) — 192.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_loss.png) — 129.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_summary.png) — 320.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_confusion_matrix.png) — 193.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_loss.png) — 151.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_summary.png) — 353.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_confusion_matrix.png) — 196.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_loss.png) — 158.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_summary.png) — 334.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_confusion_matrix.png) — 192.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_loss.png) — 135.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_summary.png) — 354.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_confusion_matrix.png) — 192.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_loss.png) — 149.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_summary.png) — 352.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/metric_boxplot.png) — 111.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/total_confusion_matrix.png) — 206.4 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1.log) — 2.5 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4.log) — 2.9 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5.log) — 3.7 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch10_weights-2026-04-23_15:09:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch10_weights-2026-04-23_15:09:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch20_weights-2026-04-23_15:10:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch20_weights-2026-04-23_15:10:04.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch30_weights-2026-04-23_15:10:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch30_weights-2026-04-23_15:10:22.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch40_weights-2026-04-23_15:10:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch40_weights-2026-04-23_15:10:40.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch50_weights-2026-04-23_15:10:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch50_weights-2026-04-23_15:10:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch60_weights-2026-04-23_15:11:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold1_epoch60_weights-2026-04-23_15:11:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch10_weights-2026-04-23_15:11:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch10_weights-2026-04-23_15:11:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch20_weights-2026-04-23_15:11:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch20_weights-2026-04-23_15:11:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch30_weights-2026-04-23_15:12:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch30_weights-2026-04-23_15:12:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch40_weights-2026-04-23_15:12:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch40_weights-2026-04-23_15:12:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch50_weights-2026-04-23_15:12:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch50_weights-2026-04-23_15:12:52.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch60_weights-2026-04-23_15:13:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch60_weights-2026-04-23_15:13:08.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch70_weights-2026-04-23_15:13:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold2_epoch70_weights-2026-04-23_15:13:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch10_weights-2026-04-23_15:14:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch10_weights-2026-04-23_15:14:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch20_weights-2026-04-23_15:14:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch20_weights-2026-04-23_15:14:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch30_weights-2026-04-23_15:14:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch30_weights-2026-04-23_15:14:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch40_weights-2026-04-23_15:14:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch40_weights-2026-04-23_15:14:52.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch50_weights-2026-04-23_15:15:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch50_weights-2026-04-23_15:15:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch60_weights-2026-04-23_15:15:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch60_weights-2026-04-23_15:15:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch70_weights-2026-04-23_15:15:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold3_epoch70_weights-2026-04-23_15:15:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch10_weights-2026-04-23_15:16:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch10_weights-2026-04-23_15:16:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch20_weights-2026-04-23_15:16:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch20_weights-2026-04-23_15:16:34.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch30_weights-2026-04-23_15:16:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch30_weights-2026-04-23_15:16:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch40_weights-2026-04-23_15:17:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch40_weights-2026-04-23_15:17:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch50_weights-2026-04-23_15:17:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch50_weights-2026-04-23_15:17:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch60_weights-2026-04-23_15:17:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch60_weights-2026-04-23_15:17:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch70_weights-2026-04-23_15:18:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold4_epoch70_weights-2026-04-23_15:18:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch10_weights-2026-04-23_15:18:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch10_weights-2026-04-23_15:18:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch20_weights-2026-04-23_15:18:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch20_weights-2026-04-23_15:18:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch30_weights-2026-04-23_15:19:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch30_weights-2026-04-23_15:19:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch40_weights-2026-04-23_15:19:17.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch40_weights-2026-04-23_15:19:17.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch50_weights-2026-04-23_15:19:35.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch50_weights-2026-04-23_15:19:35.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch60_weights-2026-04-23_15:19:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch60_weights-2026-04-23_15:19:52.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch70_weights-2026-04-23_15:20:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch70_weights-2026-04-23_15:20:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch80_weights-2026-04-23_15:20:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch80_weights-2026-04-23_15:20:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch90_weights-2026-04-23_15:20:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_15:09:27/fold5_epoch90_weights-2026-04-23_15:20:43.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-23_15:09:27",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-23T15:20:53.086604",
    "training_started_at": "2026-04-23T15:09:27.490212",
    "training_finished_at": "2026-04-23T15:20:52.629460",
    "training_duration_seconds": 685.139,
    "training_duration_hours": 0.1903,
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
      "best_epoch": 35,
      "accuracy": 0.5,
      "macro_f1": 0.37719,
      "macro_precision": 0.55556,
      "macro_recall": 0.31111,
      "balanced_accuracy": 0.31111
    },
    {
      "fold": 2,
      "best_epoch": 50,
      "accuracy": 0.76923,
      "macro_f1": 0.73889,
      "macro_precision": 0.7,
      "macro_recall": 0.88889,
      "balanced_accuracy": 0.88889
    },
    {
      "fold": 3,
      "best_epoch": 50,
      "accuracy": 0.84615,
      "macro_f1": 0.56373,
      "macro_precision": 0.53333,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 4,
      "best_epoch": 45,
      "accuracy": 0.69231,
      "macro_f1": 0.48148,
      "macro_precision": 0.48148,
      "macro_recall": 0.48148,
      "balanced_accuracy": 0.48148
    },
    {
      "fold": 5,
      "best_epoch": 65,
      "accuracy": 0.61538,
      "macro_f1": 0.66387,
      "macro_precision": 0.68571,
      "macro_recall": 0.7,
      "balanced_accuracy": 0.7
    }
  ],
  "summary": {
    "mean_accuracy": 0.68461,
    "std_accuracy": 0.12016,
    "mean_macro_f1": 0.56503,
    "std_macro_f1": 0.12827,
    "mean_macro_precision": 0.59122,
    "std_macro_precision": 0.08652,
    "mean_macro_recall": 0.60222,
    "std_macro_recall": 0.19586,
    "mean_balanced_accuracy": 0.60222,
    "std_balanced_accuracy": 0.19586
  }
}
```
