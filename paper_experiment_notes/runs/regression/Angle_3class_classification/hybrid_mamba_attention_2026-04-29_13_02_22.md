# Run: hybrid_mamba_attention_2026-04-29_13:02:22

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-29T13:03:54.739061`
- 訓練時間: `2026-04-29T13:02:22.819208` -> `2026-04-29T13:03:54.268406`
- 訓練耗時: 0.0254 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.63956 |
| std_accuracy | 0.11617 |
| mean_macro_f1 | 0.29105 |
| std_macro_f1 | 0.07151 |
| mean_macro_precision | 0.27526 |
| std_macro_precision | 0.09327 |
| mean_macro_recall | 0.3437 |
| std_macro_recall | 0.10152 |
| mean_balanced_accuracy | 0.3437 |
| std_balanced_accuracy | 0.10152 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 20 | 0.42857 | 0.2 | 0.2 | 0.2 | 0.2 |
| 2 | 15 | 0.69231 | 0.27273 | 0.23077 | 0.33333 | 0.33333 |
| 3 | 30 | 0.61538 | 0.41991 | 0.45833 | 0.51852 | 0.51852 |
| 4 | 10 | 0.69231 | 0.27273 | 0.23077 | 0.33333 | 0.33333 |
| 5 | 45 | 0.76923 | 0.28986 | 0.25641 | 0.33333 | 0.33333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_confusion_matrix.png) — 192.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_loss.png) — 154.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_summary.png) — 317.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_confusion_matrix.png) — 197.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_loss.png) — 125.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_summary.png) — 275.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_confusion_matrix.png) — 192.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_loss.png) — 127.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_summary.png) — 313.3 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_confusion_matrix.png) — 196.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_loss.png) — 120.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_summary.png) — 267.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_confusion_matrix.png) — 194.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_loss.png) — 135.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_summary.png) — 315.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/metric_boxplot.png) — 108.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/total_confusion_matrix.png) — 206.7 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1.log) — 2.0 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2.log) — 1.8 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5.log) — 2.9 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch10_weights-2026-04-29_13:02:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch10_weights-2026-04-29_13:02:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch20_weights-2026-04-29_13:02:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch20_weights-2026-04-29_13:02:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch30_weights-2026-04-29_13:02:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch30_weights-2026-04-29_13:02:39.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch40_weights-2026-04-29_13:02:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold1_epoch40_weights-2026-04-29_13:02:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch10_weights-2026-04-29_13:02:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch10_weights-2026-04-29_13:02:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch20_weights-2026-04-29_13:02:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch20_weights-2026-04-29_13:02:51.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch30_weights-2026-04-29_13:02:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch30_weights-2026-04-29_13:02:54.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch40_weights-2026-04-29_13:02:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold2_epoch40_weights-2026-04-29_13:02:57.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch10_weights-2026-04-29_13:03:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch10_weights-2026-04-29_13:03:02.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch20_weights-2026-04-29_13:03:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch20_weights-2026-04-29_13:03:05.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch30_weights-2026-04-29_13:03:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch30_weights-2026-04-29_13:03:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch40_weights-2026-04-29_13:03:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch40_weights-2026-04-29_13:03:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch50_weights-2026-04-29_13:03:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold3_epoch50_weights-2026-04-29_13:03:15.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch10_weights-2026-04-29_13:03:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch10_weights-2026-04-29_13:03:21.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch20_weights-2026-04-29_13:03:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch20_weights-2026-04-29_13:03:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch30_weights-2026-04-29_13:03:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold4_epoch30_weights-2026-04-29_13:03:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch10_weights-2026-04-29_13:03:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch10_weights-2026-04-29_13:03:33.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch20_weights-2026-04-29_13:03:36.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch20_weights-2026-04-29_13:03:36.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch30_weights-2026-04-29_13:03:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch30_weights-2026-04-29_13:03:39.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch40_weights-2026-04-29_13:03:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch40_weights-2026-04-29_13:03:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch50_weights-2026-04-29_13:03:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch50_weights-2026-04-29_13:03:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch60_weights-2026-04-29_13:03:49.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch60_weights-2026-04-29_13:03:49.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch70_weights-2026-04-29_13:03:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_13:02:22/fold5_epoch70_weights-2026-04-29_13:03:52.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-29_13:02:22",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-29T13:03:54.739061",
    "training_started_at": "2026-04-29T13:02:22.819208",
    "training_finished_at": "2026-04-29T13:03:54.268406",
    "training_duration_seconds": 91.449,
    "training_duration_hours": 0.0254,
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
      "accuracy": 0.42857,
      "macro_f1": 0.2,
      "macro_precision": 0.2,
      "macro_recall": 0.2,
      "balanced_accuracy": 0.2
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "accuracy": 0.69231,
      "macro_f1": 0.27273,
      "macro_precision": 0.23077,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "accuracy": 0.61538,
      "macro_f1": 0.41991,
      "macro_precision": 0.45833,
      "macro_recall": 0.51852,
      "balanced_accuracy": 0.51852
    },
    {
      "fold": 4,
      "best_epoch": 10,
      "accuracy": 0.69231,
      "macro_f1": 0.27273,
      "macro_precision": 0.23077,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 5,
      "best_epoch": 45,
      "accuracy": 0.76923,
      "macro_f1": 0.28986,
      "macro_precision": 0.25641,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    }
  ],
  "summary": {
    "mean_accuracy": 0.63956,
    "std_accuracy": 0.11617,
    "mean_macro_f1": 0.29105,
    "std_macro_f1": 0.07151,
    "mean_macro_precision": 0.27526,
    "std_macro_precision": 0.09327,
    "mean_macro_recall": 0.3437,
    "std_macro_recall": 0.10152,
    "mean_balanced_accuracy": 0.3437,
    "std_balanced_accuracy": 0.10152
  }
}
```
