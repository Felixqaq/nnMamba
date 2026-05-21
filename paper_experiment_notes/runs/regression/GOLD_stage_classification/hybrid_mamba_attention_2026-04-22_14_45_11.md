# Run: hybrid_mamba_attention_2026-04-22_14:45:11

## 一句話用途
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-04-22T14:52:45.446813`
- 訓練時間: `2026-04-22T14:45:11.136726` -> `2026-04-22T14:52:44.875487`
- 訓練耗時: 0.126 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.57576 |
| std_accuracy | 0.08571 |
| mean_macro_f1 | 0.43553 |
| std_macro_f1 | 0.05321 |
| mean_macro_precision | 0.43559 |
| std_macro_precision | 0.07008 |
| mean_macro_recall | 0.5 |
| std_macro_recall | 0.07795 |
| mean_balanced_accuracy | 0.5 |
| std_balanced_accuracy | 0.07795 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 70 | 0.63636 | 0.47821 | 0.51786 | 0.52083 | 0.52083 |
| 2 | 70 | 0.45455 | 0.36051 | 0.34659 | 0.39583 | 0.39583 |
| 3 | 15 | 0.63636 | 0.46786 | 0.44231 | 0.58333 | 0.58333 |

## Training Config Embedded In Result
```json
{
  "epochs": 160,
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
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_confusion_matrix.png) — 186.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_loss.png) — 132.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_summary.png) — 344.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_confusion_matrix.png) — 186.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_loss.png) — 159.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_summary.png) — 374.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_confusion_matrix.png) — 188.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_loss.png) — 129.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_summary.png) — 340.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/metric_boxplot.png) — 107.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/total_confusion_matrix.png) — 196.9 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1.log) — 3.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2.log) — 3.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3.log) — 1.8 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch10_weights-2026-04-22_14:45:31.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch10_weights-2026-04-22_14:45:31.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch20_weights-2026-04-22_14:45:50.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch20_weights-2026-04-22_14:45:50.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch30_weights-2026-04-22_14:46:09.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch30_weights-2026-04-22_14:46:09.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch40_weights-2026-04-22_14:46:27.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch40_weights-2026-04-22_14:46:27.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch50_weights-2026-04-22_14:46:45.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch50_weights-2026-04-22_14:46:45.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch60_weights-2026-04-22_14:47:03.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch60_weights-2026-04-22_14:47:03.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch70_weights-2026-04-22_14:47:22.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch70_weights-2026-04-22_14:47:22.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch80_weights-2026-04-22_14:47:39.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch80_weights-2026-04-22_14:47:39.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch90_weights-2026-04-22_14:47:57.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold1_epoch90_weights-2026-04-22_14:47:57.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch10_weights-2026-04-22_14:48:34.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch10_weights-2026-04-22_14:48:34.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch20_weights-2026-04-22_14:48:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch20_weights-2026-04-22_14:48:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch30_weights-2026-04-22_14:49:12.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch30_weights-2026-04-22_14:49:12.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch40_weights-2026-04-22_14:49:30.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch40_weights-2026-04-22_14:49:30.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch50_weights-2026-04-22_14:49:49.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch50_weights-2026-04-22_14:49:49.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch60_weights-2026-04-22_14:50:08.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch60_weights-2026-04-22_14:50:08.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch70_weights-2026-04-22_14:50:27.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch70_weights-2026-04-22_14:50:27.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch80_weights-2026-04-22_14:50:46.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch80_weights-2026-04-22_14:50:46.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch90_weights-2026-04-22_14:51:04.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold2_epoch90_weights-2026-04-22_14:51:04.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch10_weights-2026-04-22_14:51:41.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch10_weights-2026-04-22_14:51:41.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch20_weights-2026-04-22_14:52:00.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch20_weights-2026-04-22_14:52:00.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch30_weights-2026-04-22_14:52:17.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch30_weights-2026-04-22_14:52:17.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch40_weights-2026-04-22_14:52:35.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:45:11/fold3_epoch40_weights-2026-04-22_14:52:35.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-22_14:45:11",
    "model": "hybrid_mamba_attention",
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-04-22T14:52:45.446813",
    "training_started_at": "2026-04-22T14:45:11.136726",
    "training_finished_at": "2026-04-22T14:52:44.875487",
    "training_duration_seconds": 453.739,
    "training_duration_hours": 0.126,
    "class_names": [
      "GOLD 1 (Mild)",
      "GOLD 2 (Moderate)",
      "GOLD 3 (Severe)",
      "GOLD 4 (Very Severe)"
    ],
    "config": {
      "epochs": 160,
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
      "best_epoch": 70,
      "accuracy": 0.63636,
      "macro_f1": 0.47821,
      "macro_precision": 0.51786,
      "macro_recall": 0.52083,
      "balanced_accuracy": 0.52083
    },
    {
      "fold": 2,
      "best_epoch": 70,
      "accuracy": 0.45455,
      "macro_f1": 0.36051,
      "macro_precision": 0.34659,
      "macro_recall": 0.39583,
      "balanced_accuracy": 0.39583
    },
    {
      "fold": 3,
      "best_epoch": 15,
      "accuracy": 0.63636,
      "macro_f1": 0.46786,
      "macro_precision": 0.44231,
      "macro_recall": 0.58333,
      "balanced_accuracy": 0.58333
    }
  ],
  "summary": {
    "mean_accuracy": 0.57576,
    "std_accuracy": 0.08571,
    "mean_macro_f1": 0.43553,
    "std_macro_f1": 0.05321,
    "mean_macro_precision": 0.43559,
    "std_macro_precision": 0.07008,
    "mean_macro_recall": 0.5,
    "std_macro_recall": 0.07795,
    "mean_balanced_accuracy": 0.5,
    "std_balanced_accuracy": 0.07795
  }
}
```
