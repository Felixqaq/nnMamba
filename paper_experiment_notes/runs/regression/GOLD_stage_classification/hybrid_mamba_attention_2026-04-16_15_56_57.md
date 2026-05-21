# Run: hybrid_mamba_attention_2026-04-16_15:56:57

## 一句話用途
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-04-16T16:00:27.963293`
- 訓練時間: `2026-04-16T15:56:57.321073` -> `2026-04-16T16:00:27.444286`
- 訓練耗時: 0.0584 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.5 |
| std_accuracy | 0.09072 |
| mean_macro_f1 | 0.37849 |
| std_macro_f1 | 0.09163 |
| mean_macro_precision | 0.43783 |
| std_macro_precision | 0.14006 |
| mean_macro_recall | 0.42292 |
| std_macro_recall | 0.05504 |
| mean_balanced_accuracy | 0.42292 |
| std_balanced_accuracy | 0.05504 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65 | 0.38889 | 0.30154 | 0.3125 | 0.39375 | 0.39375 |
| 2 | 30 | 0.61111 | 0.50725 | 0.63333 | 0.5 | 0.5 |
| 3 | 35 | 0.5 | 0.32667 | 0.36765 | 0.375 | 0.375 |

## Training Config Embedded In Result
```json
{
  "epochs": 80,
  "batch_size": 12,
  "learning_rate": 0.0001,
  "weight_decay": 0.001,
  "k_folds": 3,
  "seed": 42,
  "loss": "auto",
  "num_classes": 4
}
```

## Artifact Index
### figures
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_confusion_matrix.png) — 197.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_loss.png) — 160.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_summary.png) — 343.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_confusion_matrix.png) — 187.3 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_loss.png) — 151.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_summary.png) — 341.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_confusion_matrix.png) — 187.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_loss.png) — 122.3 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_summary.png) — 279.3 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/metric_boxplot.png) — 108.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/total_confusion_matrix.png) — 204.9 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=18
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=18
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=18

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1.log) — 2.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2.log) — 2.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3.log) — 2.9 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch10_weights-2026-04-16_15:57:07.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch10_weights-2026-04-16_15:57:07.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch20_weights-2026-04-16_15:57:16.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch20_weights-2026-04-16_15:57:16.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch30_weights-2026-04-16_15:57:25.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch30_weights-2026-04-16_15:57:25.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch40_weights-2026-04-16_15:57:34.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch40_weights-2026-04-16_15:57:34.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch50_weights-2026-04-16_15:57:42.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch50_weights-2026-04-16_15:57:42.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch60_weights-2026-04-16_15:57:51.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch60_weights-2026-04-16_15:57:51.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch70_weights-2026-04-16_15:58:00.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch70_weights-2026-04-16_15:58:00.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch80_weights-2026-04-16_15:58:08.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold1_epoch80_weights-2026-04-16_15:58:08.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch10_weights-2026-04-16_15:58:17.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch10_weights-2026-04-16_15:58:17.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch20_weights-2026-04-16_15:58:26.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch20_weights-2026-04-16_15:58:26.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch30_weights-2026-04-16_15:58:35.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch30_weights-2026-04-16_15:58:35.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch40_weights-2026-04-16_15:58:43.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch40_weights-2026-04-16_15:58:43.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch50_weights-2026-04-16_15:58:51.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch50_weights-2026-04-16_15:58:51.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch60_weights-2026-04-16_15:58:59.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch60_weights-2026-04-16_15:58:59.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch70_weights-2026-04-16_15:59:07.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch70_weights-2026-04-16_15:59:07.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch80_weights-2026-04-16_15:59:16.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold2_epoch80_weights-2026-04-16_15:59:16.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch10_weights-2026-04-16_15:59:25.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch10_weights-2026-04-16_15:59:25.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch20_weights-2026-04-16_15:59:34.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch20_weights-2026-04-16_15:59:34.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch30_weights-2026-04-16_15:59:43.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch30_weights-2026-04-16_15:59:43.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch40_weights-2026-04-16_15:59:52.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch40_weights-2026-04-16_15:59:52.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch50_weights-2026-04-16_16:00:00.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch50_weights-2026-04-16_16:00:00.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch60_weights-2026-04-16_16:00:09.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch60_weights-2026-04-16_16:00:09.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch70_weights-2026-04-16_16:00:17.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch70_weights-2026-04-16_16:00:17.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch80_weights-2026-04-16_16:00:26.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-16_15:56:57/fold3_epoch80_weights-2026-04-16_16:00:26.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-16_15:56:57",
    "model": "hybrid_mamba_attention",
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-04-16T16:00:27.963293",
    "training_started_at": "2026-04-16T15:56:57.321073",
    "training_finished_at": "2026-04-16T16:00:27.444286",
    "training_duration_seconds": 210.123,
    "training_duration_hours": 0.0584,
    "class_names": [
      "GOLD 1 (Mild)",
      "GOLD 2 (Moderate)",
      "GOLD 3 (Severe)",
      "GOLD 4 (Very Severe)"
    ],
    "config": {
      "epochs": 80,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.001,
      "k_folds": 3,
      "seed": 42,
      "loss": "auto",
      "num_classes": 4
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 65,
      "accuracy": 0.38889,
      "macro_f1": 0.30154,
      "macro_precision": 0.3125,
      "macro_recall": 0.39375,
      "balanced_accuracy": 0.39375
    },
    {
      "fold": 2,
      "best_epoch": 30,
      "accuracy": 0.61111,
      "macro_f1": 0.50725,
      "macro_precision": 0.63333,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 3,
      "best_epoch": 35,
      "accuracy": 0.5,
      "macro_f1": 0.32667,
      "macro_precision": 0.36765,
      "macro_recall": 0.375,
      "balanced_accuracy": 0.375
    }
  ],
  "summary": {
    "mean_accuracy": 0.5,
    "std_accuracy": 0.09072,
    "mean_macro_f1": 0.37849,
    "std_macro_f1": 0.09163,
    "mean_macro_precision": 0.43783,
    "std_macro_precision": 0.14006,
    "mean_macro_recall": 0.42292,
    "std_macro_recall": 0.05504,
    "mean_balanced_accuracy": 0.42292,
    "std_balanced_accuracy": 0.05504
  }
}
```
