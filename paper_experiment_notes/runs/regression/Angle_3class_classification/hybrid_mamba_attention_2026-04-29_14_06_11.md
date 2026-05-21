# Run: hybrid_mamba_attention_2026-04-29_14:06:11

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-29T14:11:15.951496`
- 訓練時間: `2026-04-29T14:06:11.182856` -> `2026-04-29T14:11:15.482602`
- 訓練耗時: 0.0845 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.41319 |
| std_accuracy | 0.26241 |
| mean_macro_f1 | 0.35915 |
| std_macro_f1 | 0.15953 |
| mean_macro_precision | 0.40243 |
| std_macro_precision | 0.11493 |
| mean_macro_recall | 0.5037 |
| std_macro_recall | 0.15289 |
| mean_balanced_accuracy | 0.5037 |
| std_balanced_accuracy | 0.15289 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.14286 | 0.21429 | 0.35897 | 0.44444 | 0.44444 |
| 2 | 10 | 0.76923 | 0.61905 | 0.58333 | 0.66667 | 0.66667 |
| 3 | 55 | 0.69231 | 0.44974 | 0.42593 | 0.48148 | 0.48148 |
| 4 | 15 | 0.23077 | 0.18788 | 0.41667 | 0.25926 | 0.25926 |
| 5 | 10 | 0.23077 | 0.32479 | 0.22727 | 0.66667 | 0.66667 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_confusion_matrix.png) — 193.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_loss.png) — 144.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_summary.png) — 325.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_confusion_matrix.png) — 197.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_loss.png) — 108.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_summary.png) — 306.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_confusion_matrix.png) — 194.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_loss.png) — 162.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_summary.png) — 359.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_confusion_matrix.png) — 191.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_loss.png) — 183.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_summary.png) — 304.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_confusion_matrix.png) — 196.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_loss.png) — 139.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_summary.png) — 319.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/metric_boxplot.png) — 113.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/total_confusion_matrix.png) — 203.0 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3.log) — 3.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5.log) — 1.6 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch10_weights-2026-04-29_14:06:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch10_weights-2026-04-29_14:06:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch20_weights-2026-04-29_14:06:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch20_weights-2026-04-29_14:06:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch30_weights-2026-04-29_14:06:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold1_epoch30_weights-2026-04-29_14:06:49.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch10_weights-2026-04-29_14:07:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch10_weights-2026-04-29_14:07:07.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch20_weights-2026-04-29_14:07:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch20_weights-2026-04-29_14:07:19.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch30_weights-2026-04-29_14:07:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold2_epoch30_weights-2026-04-29_14:07:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch10_weights-2026-04-29_14:07:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch10_weights-2026-04-29_14:07:56.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch20_weights-2026-04-29_14:08:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch20_weights-2026-04-29_14:08:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch30_weights-2026-04-29_14:08:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch30_weights-2026-04-29_14:08:23.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch40_weights-2026-04-29_14:08:35.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch40_weights-2026-04-29_14:08:35.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch50_weights-2026-04-29_14:08:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch50_weights-2026-04-29_14:08:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch60_weights-2026-04-29_14:09:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch60_weights-2026-04-29_14:09:00.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch70_weights-2026-04-29_14:09:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch70_weights-2026-04-29_14:09:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch80_weights-2026-04-29_14:09:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold3_epoch80_weights-2026-04-29_14:09:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch10_weights-2026-04-29_14:09:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch10_weights-2026-04-29_14:09:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch20_weights-2026-04-29_14:09:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch20_weights-2026-04-29_14:09:56.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch30_weights-2026-04-29_14:10:08.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch30_weights-2026-04-29_14:10:08.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch40_weights-2026-04-29_14:10:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold4_epoch40_weights-2026-04-29_14:10:20.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch10_weights-2026-04-29_14:10:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch10_weights-2026-04-29_14:10:39.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch20_weights-2026-04-29_14:10:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch20_weights-2026-04-29_14:10:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch30_weights-2026-04-29_14:11:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:06:11/fold5_epoch30_weights-2026-04-29_14:11:03.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-29_14:06:11",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-29T14:11:15.951496",
    "training_started_at": "2026-04-29T14:06:11.182856",
    "training_finished_at": "2026-04-29T14:11:15.482602",
    "training_duration_seconds": 304.3,
    "training_duration_hours": 0.0845,
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
      "best_epoch": 5,
      "accuracy": 0.14286,
      "macro_f1": 0.21429,
      "macro_precision": 0.35897,
      "macro_recall": 0.44444,
      "balanced_accuracy": 0.44444
    },
    {
      "fold": 2,
      "best_epoch": 10,
      "accuracy": 0.76923,
      "macro_f1": 0.61905,
      "macro_precision": 0.58333,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    },
    {
      "fold": 3,
      "best_epoch": 55,
      "accuracy": 0.69231,
      "macro_f1": 0.44974,
      "macro_precision": 0.42593,
      "macro_recall": 0.48148,
      "balanced_accuracy": 0.48148
    },
    {
      "fold": 4,
      "best_epoch": 15,
      "accuracy": 0.23077,
      "macro_f1": 0.18788,
      "macro_precision": 0.41667,
      "macro_recall": 0.25926,
      "balanced_accuracy": 0.25926
    },
    {
      "fold": 5,
      "best_epoch": 10,
      "accuracy": 0.23077,
      "macro_f1": 0.32479,
      "macro_precision": 0.22727,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    }
  ],
  "summary": {
    "mean_accuracy": 0.41319,
    "std_accuracy": 0.26241,
    "mean_macro_f1": 0.35915,
    "std_macro_f1": 0.15953,
    "mean_macro_precision": 0.40243,
    "std_macro_precision": 0.11493,
    "mean_macro_recall": 0.5037,
    "std_macro_recall": 0.15289,
    "mean_balanced_accuracy": 0.5037,
    "std_balanced_accuracy": 0.15289
  }
}
```
