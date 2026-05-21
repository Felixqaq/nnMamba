# Run: hybrid_mamba_attention_2026-04-23_14:40:43

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-23T14:44:27.888035`
- 訓練時間: `2026-04-23T14:40:43.293523` -> `2026-04-23T14:44:27.430244`
- 訓練耗時: 0.0623 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.71209 |
| std_accuracy | 0.02981 |
| mean_macro_f1 | 0.27958 |
| std_macro_f1 | 0.00839 |
| mean_macro_precision | 0.24103 |
| std_macro_precision | 0.01256 |
| mean_macro_recall | 0.33333 |
| std_macro_recall | 0 |
| mean_balanced_accuracy | 0.33333 |
| std_balanced_accuracy | 0 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | 0.71429 | 0.28986 | 0.25641 | 0.33333 | 0.33333 |
| 2 | 5 | 0.69231 | 0.27273 | 0.23077 | 0.33333 | 0.33333 |
| 3 | 5 | 0.69231 | 0.27273 | 0.23077 | 0.33333 | 0.33333 |
| 4 | 5 | 0.69231 | 0.27273 | 0.23077 | 0.33333 | 0.33333 |
| 5 | 5 | 0.76923 | 0.28986 | 0.25641 | 0.33333 | 0.33333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_confusion_matrix.png) — 193.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_loss.png) — 133.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_summary.png) — 268.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_confusion_matrix.png) — 197.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_loss.png) — 165.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_summary.png) — 273.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_confusion_matrix.png) — 197.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_loss.png) — 167.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_summary.png) — 276.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_confusion_matrix.png) — 196.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_loss.png) — 175.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_summary.png) — 279.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_confusion_matrix.png) — 194.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_loss.png) — 136.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_summary.png) — 267.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/metric_boxplot.png) — 97.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/total_confusion_matrix.png) — 202.4 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1.log) — 2.2 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5.log) — 1.4 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch10_weights-2026-04-23_14:41:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch10_weights-2026-04-23_14:41:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch20_weights-2026-04-23_14:41:14.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch20_weights-2026-04-23_14:41:14.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch30_weights-2026-04-23_14:41:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch30_weights-2026-04-23_14:41:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch40_weights-2026-04-23_14:41:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch40_weights-2026-04-23_14:41:37.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch50_weights-2026-04-23_14:41:48.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold1_epoch50_weights-2026-04-23_14:41:48.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch10_weights-2026-04-23_14:42:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch10_weights-2026-04-23_14:42:05.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch20_weights-2026-04-23_14:42:16.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch20_weights-2026-04-23_14:42:16.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch30_weights-2026-04-23_14:42:27.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold2_epoch30_weights-2026-04-23_14:42:27.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch10_weights-2026-04-23_14:42:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch10_weights-2026-04-23_14:42:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch20_weights-2026-04-23_14:42:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch20_weights-2026-04-23_14:42:54.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch30_weights-2026-04-23_14:43:05.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold3_epoch30_weights-2026-04-23_14:43:05.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch10_weights-2026-04-23_14:43:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch10_weights-2026-04-23_14:43:21.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch20_weights-2026-04-23_14:43:32.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch20_weights-2026-04-23_14:43:32.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch30_weights-2026-04-23_14:43:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold4_epoch30_weights-2026-04-23_14:43:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch10_weights-2026-04-23_14:43:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch10_weights-2026-04-23_14:43:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch20_weights-2026-04-23_14:44:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch20_weights-2026-04-23_14:44:10.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch30_weights-2026-04-23_14:44:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-23_14:40:43/fold5_epoch30_weights-2026-04-23_14:44:21.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-23_14:40:43",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-23T14:44:27.888035",
    "training_started_at": "2026-04-23T14:40:43.293523",
    "training_finished_at": "2026-04-23T14:44:27.430244",
    "training_duration_seconds": 224.137,
    "training_duration_hours": 0.0623,
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
      "accuracy": 0.71429,
      "macro_f1": 0.28986,
      "macro_precision": 0.25641,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.69231,
      "macro_f1": 0.27273,
      "macro_precision": 0.23077,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 3,
      "best_epoch": 5,
      "accuracy": 0.69231,
      "macro_f1": 0.27273,
      "macro_precision": 0.23077,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 4,
      "best_epoch": 5,
      "accuracy": 0.69231,
      "macro_f1": 0.27273,
      "macro_precision": 0.23077,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    },
    {
      "fold": 5,
      "best_epoch": 5,
      "accuracy": 0.76923,
      "macro_f1": 0.28986,
      "macro_precision": 0.25641,
      "macro_recall": 0.33333,
      "balanced_accuracy": 0.33333
    }
  ],
  "summary": {
    "mean_accuracy": 0.71209,
    "std_accuracy": 0.02981,
    "mean_macro_f1": 0.27958,
    "std_macro_f1": 0.00839,
    "mean_macro_precision": 0.24103,
    "std_macro_precision": 0.01256,
    "mean_macro_recall": 0.33333,
    "std_macro_recall": 0.0,
    "mean_balanced_accuracy": 0.33333,
    "std_balanced_accuracy": 0.0
  }
}
```
