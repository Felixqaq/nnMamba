# Run: hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13

## 一句話用途
- 方法名稱: `GOLD balanced sampling + aug200/class`
- 實驗描述: Virtual train-fold augmentation brings all four GOLD classes to 200 samples before per-epoch balanced sampling.
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-05-06T15:31:02.871691`
- 訓練時間: `2026-05-06T15:04:13.891873` -> `2026-05-06T15:31:02.354405`
- 訓練耗時: 0.4468 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.56061 |
| std_accuracy | 0.07725 |
| mean_macro_f1 | 0.4436 |
| std_macro_f1 | 0.06737 |
| mean_macro_precision | 0.52479 |
| std_macro_precision | 0.14612 |
| mean_macro_recall | 0.46111 |
| std_macro_recall | 0.03068 |
| mean_balanced_accuracy | 0.46111 |
| std_balanced_accuracy | 0.03068 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.63636 | 0.4375 | 0.65 | 0.425 | 0.425 |
| 2 | 80 | 0.59091 | 0.52899 | 0.60455 | 0.5 | 0.5 |
| 3 | 20 | 0.45455 | 0.3643 | 0.31981 | 0.45833 | 0.45833 |

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
  "num_classes": 4,
  "target_mode": "gold"
}
```

## Artifact Index
### figures
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_confusion_matrix.png) — 214.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_loss.png) — 147.5 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_summary.png) — 330.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_confusion_matrix.png) — 216.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_loss.png) — 188.6 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_summary.png) — 413.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_confusion_matrix.png) — 214.5 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_loss.png) — 155.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_summary.png) — 346.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/metric_boxplot.png) — 136.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/total_confusion_matrix.png) — 224.9 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1.log) — 1.6 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2.log) — 4.2 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3.log) — 2.0 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch10_weights-2026-05-06_15:05:28.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch10_weights-2026-05-06_15:05:28.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch20_weights-2026-05-06_15:06:41.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch20_weights-2026-05-06_15:06:41.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch30_weights-2026-05-06_15:07:54.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold1_epoch30_weights-2026-05-06_15:07:54.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch100_weights-2026-05-06_15:23:14.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch100_weights-2026-05-06_15:23:14.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch10_weights-2026-05-06_15:10:17.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch10_weights-2026-05-06_15:10:17.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch20_weights-2026-05-06_15:11:29.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch20_weights-2026-05-06_15:11:29.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch30_weights-2026-05-06_15:12:41.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch30_weights-2026-05-06_15:12:41.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch40_weights-2026-05-06_15:13:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch40_weights-2026-05-06_15:13:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch50_weights-2026-05-06_15:15:05.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch50_weights-2026-05-06_15:15:05.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch60_weights-2026-05-06_15:17:56.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch60_weights-2026-05-06_15:17:56.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch70_weights-2026-05-06_15:19:12.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch70_weights-2026-05-06_15:19:12.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch80_weights-2026-05-06_15:20:33.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch80_weights-2026-05-06_15:20:33.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch90_weights-2026-05-06_15:21:54.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold2_epoch90_weights-2026-05-06_15:21:54.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch10_weights-2026-05-06_15:25:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch10_weights-2026-05-06_15:25:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch20_weights-2026-05-06_15:27:13.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch20_weights-2026-05-06_15:27:13.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch30_weights-2026-05-06_15:28:32.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch30_weights-2026-05-06_15:28:32.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch40_weights-2026-05-06_15:29:49.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13/fold3_epoch40_weights-2026-05-06_15:29:49.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "GOLD balanced sampling + aug200/class",
      "description": "Virtual train-fold augmentation brings all four GOLD classes to 200 samples before per-epoch balanced sampling."
    },
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-05-06T15:31:02.871691",
    "training_started_at": "2026-05-06T15:04:13.891873",
    "training_finished_at": "2026-05-06T15:31:02.354405",
    "training_duration_seconds": 1608.463,
    "training_duration_hours": 0.4468,
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
      "num_classes": 4,
      "target_mode": "gold"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 10,
      "accuracy": 0.63636,
      "macro_f1": 0.4375,
      "macro_precision": 0.65,
      "macro_recall": 0.425,
      "balanced_accuracy": 0.425
    },
    {
      "fold": 2,
      "best_epoch": 80,
      "accuracy": 0.59091,
      "macro_f1": 0.52899,
      "macro_precision": 0.60455,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 3,
      "best_epoch": 20,
      "accuracy": 0.45455,
      "macro_f1": 0.3643,
      "macro_precision": 0.31981,
      "macro_recall": 0.45833,
      "balanced_accuracy": 0.45833
    }
  ],
  "summary": {
    "mean_accuracy": 0.56061,
    "std_accuracy": 0.07725,
    "mean_macro_f1": 0.4436,
    "std_macro_f1": 0.06737,
    "mean_macro_precision": 0.52479,
    "std_macro_precision": 0.14612,
    "mean_macro_recall": 0.46111,
    "std_macro_recall": 0.03068,
    "mean_balanced_accuracy": 0.46111,
    "std_balanced_accuracy": 0.03068
  }
}
```
