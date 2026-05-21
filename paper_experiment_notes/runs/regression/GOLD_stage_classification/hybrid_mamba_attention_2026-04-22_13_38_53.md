# Run: hybrid_mamba_attention_2026-04-22_13:38:53

## 一句話用途
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-04-22T13:45:36.874826`
- 訓練時間: `2026-04-22T13:38:53.780030` -> `2026-04-22T13:45:36.329683`
- 訓練耗時: 0.1118 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.51515 |
| std_accuracy | 0.1868 |
| mean_macro_f1 | 0.46388 |
| std_macro_f1 | 0.16446 |
| mean_macro_precision | 0.53311 |
| std_macro_precision | 0.16408 |
| mean_macro_recall | 0.51944 |
| std_macro_recall | 0.08335 |
| mean_balanced_accuracy | 0.51944 |
| std_balanced_accuracy | 0.08335 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 75 | 0.72727 | 0.63111 | 0.66987 | 0.62083 | 0.62083 |
| 2 | 35 | 0.54545 | 0.52024 | 0.62708 | 0.52083 | 0.52083 |
| 3 | 20 | 0.27273 | 0.24028 | 0.30238 | 0.41667 | 0.41667 |

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
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_confusion_matrix.png) — 185.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_loss.png) — 182.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_summary.png) — 378.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_confusion_matrix.png) — 187.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_loss.png) — 157.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_summary.png) — 355.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_confusion_matrix.png) — 189.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_loss.png) — 161.6 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_summary.png) — 330.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/metric_boxplot.png) — 98.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/total_confusion_matrix.png) — 207.3 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1.log) — 4.1 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2.log) — 2.5 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3.log) — 2.0 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch100_weights-2026-04-22_13:41:57.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch100_weights-2026-04-22_13:41:57.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch10_weights-2026-04-22_13:39:13.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch10_weights-2026-04-22_13:39:13.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch20_weights-2026-04-22_13:39:31.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch20_weights-2026-04-22_13:39:31.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch30_weights-2026-04-22_13:39:50.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch30_weights-2026-04-22_13:39:50.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch40_weights-2026-04-22_13:40:08.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch40_weights-2026-04-22_13:40:08.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch50_weights-2026-04-22_13:40:27.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch50_weights-2026-04-22_13:40:27.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch60_weights-2026-04-22_13:40:45.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch60_weights-2026-04-22_13:40:45.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch70_weights-2026-04-22_13:41:04.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch70_weights-2026-04-22_13:41:04.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch80_weights-2026-04-22_13:41:22.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch80_weights-2026-04-22_13:41:22.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch90_weights-2026-04-22_13:41:40.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold1_epoch90_weights-2026-04-22_13:41:40.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch10_weights-2026-04-22_13:42:24.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch10_weights-2026-04-22_13:42:24.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch20_weights-2026-04-22_13:42:43.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch20_weights-2026-04-22_13:42:43.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch30_weights-2026-04-22_13:43:01.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch30_weights-2026-04-22_13:43:01.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch40_weights-2026-04-22_13:43:19.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch40_weights-2026-04-22_13:43:19.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch50_weights-2026-04-22_13:43:38.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch50_weights-2026-04-22_13:43:38.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch60_weights-2026-04-22_13:43:56.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold2_epoch60_weights-2026-04-22_13:43:56.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch10_weights-2026-04-22_13:44:23.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch10_weights-2026-04-22_13:44:23.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch20_weights-2026-04-22_13:44:42.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch20_weights-2026-04-22_13:44:42.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch30_weights-2026-04-22_13:45:00.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch30_weights-2026-04-22_13:45:00.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch40_weights-2026-04-22_13:45:18.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_13:38:53/fold3_epoch40_weights-2026-04-22_13:45:18.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-22_13:38:53",
    "model": "hybrid_mamba_attention",
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-04-22T13:45:36.874826",
    "training_started_at": "2026-04-22T13:38:53.780030",
    "training_finished_at": "2026-04-22T13:45:36.329683",
    "training_duration_seconds": 402.55,
    "training_duration_hours": 0.1118,
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
      "best_epoch": 75,
      "accuracy": 0.72727,
      "macro_f1": 0.63111,
      "macro_precision": 0.66987,
      "macro_recall": 0.62083,
      "balanced_accuracy": 0.62083
    },
    {
      "fold": 2,
      "best_epoch": 35,
      "accuracy": 0.54545,
      "macro_f1": 0.52024,
      "macro_precision": 0.62708,
      "macro_recall": 0.52083,
      "balanced_accuracy": 0.52083
    },
    {
      "fold": 3,
      "best_epoch": 20,
      "accuracy": 0.27273,
      "macro_f1": 0.24028,
      "macro_precision": 0.30238,
      "macro_recall": 0.41667,
      "balanced_accuracy": 0.41667
    }
  ],
  "summary": {
    "mean_accuracy": 0.51515,
    "std_accuracy": 0.1868,
    "mean_macro_f1": 0.46388,
    "std_macro_f1": 0.16446,
    "mean_macro_precision": 0.53311,
    "std_macro_precision": 0.16408,
    "mean_macro_recall": 0.51944,
    "std_macro_recall": 0.08335,
    "mean_balanced_accuracy": 0.51944,
    "std_balanced_accuracy": 0.08335
  }
}
```
