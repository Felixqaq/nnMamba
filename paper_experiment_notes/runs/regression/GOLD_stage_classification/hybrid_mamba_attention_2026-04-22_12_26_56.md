# Run: hybrid_mamba_attention_2026-04-22_12:26:56

## 一句話用途
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-04-22T12:34:46.419752`
- 訓練時間: `2026-04-22T12:26:56.282410` -> `2026-04-22T12:34:45.728167`
- 訓練耗時: 0.1304 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.4697 |
| std_accuracy | 0.14051 |
| mean_macro_f1 | 0.35846 |
| std_macro_f1 | 0.01901 |
| mean_macro_precision | 0.3888 |
| std_macro_precision | 0.00613 |
| mean_macro_recall | 0.39445 |
| std_macro_recall | 0.03408 |
| mean_balanced_accuracy | 0.39445 |
| std_balanced_accuracy | 0.03408 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65 | 0.59091 | 0.38406 | 0.38203 | 0.39167 | 0.39167 |
| 2 | 80 | 0.54545 | 0.33854 | 0.3875 | 0.35417 | 0.35417 |
| 3 | 25 | 0.27273 | 0.35278 | 0.39687 | 0.4375 | 0.4375 |

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
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_confusion_matrix.png) — 185.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_loss.png) — 156.5 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_summary.png) — 364.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_confusion_matrix.png) — 185.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_loss.png) — 156.5 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_summary.png) — 324.3 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_confusion_matrix.png) — 187.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_loss.png) — 174.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_summary.png) — 368.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/metric_boxplot.png) — 107.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/total_confusion_matrix.png) — 208.8 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1.log) — 2.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2.log) — 2.9 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3.log) — 2.9 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch10_weights-2026-04-22_12:27:22.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch10_weights-2026-04-22_12:27:22.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch20_weights-2026-04-22_12:27:41.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch20_weights-2026-04-22_12:27:41.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch30_weights-2026-04-22_12:28:00.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch30_weights-2026-04-22_12:28:00.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch40_weights-2026-04-22_12:28:18.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch40_weights-2026-04-22_12:28:18.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch50_weights-2026-04-22_12:28:37.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch50_weights-2026-04-22_12:28:37.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch60_weights-2026-04-22_12:28:56.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch60_weights-2026-04-22_12:28:56.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch70_weights-2026-04-22_12:29:14.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch70_weights-2026-04-22_12:29:14.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch80_weights-2026-04-22_12:29:33.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold1_epoch80_weights-2026-04-22_12:29:33.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch10_weights-2026-04-22_12:29:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch10_weights-2026-04-22_12:29:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch20_weights-2026-04-22_12:30:12.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch20_weights-2026-04-22_12:30:12.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch30_weights-2026-04-22_12:30:31.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch30_weights-2026-04-22_12:30:31.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch40_weights-2026-04-22_12:30:50.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch40_weights-2026-04-22_12:30:50.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch50_weights-2026-04-22_12:31:10.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch50_weights-2026-04-22_12:31:10.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch60_weights-2026-04-22_12:31:29.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch60_weights-2026-04-22_12:31:29.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch70_weights-2026-04-22_12:31:49.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch70_weights-2026-04-22_12:31:49.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch80_weights-2026-04-22_12:32:09.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold2_epoch80_weights-2026-04-22_12:32:09.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch10_weights-2026-04-22_12:32:29.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch10_weights-2026-04-22_12:32:29.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch20_weights-2026-04-22_12:32:49.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch20_weights-2026-04-22_12:32:49.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch30_weights-2026-04-22_12:33:08.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch30_weights-2026-04-22_12:33:08.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch40_weights-2026-04-22_12:33:28.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch40_weights-2026-04-22_12:33:28.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch50_weights-2026-04-22_12:33:47.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch50_weights-2026-04-22_12:33:47.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch60_weights-2026-04-22_12:34:06.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch60_weights-2026-04-22_12:34:06.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch70_weights-2026-04-22_12:34:25.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch70_weights-2026-04-22_12:34:25.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch80_weights-2026-04-22_12:34:44.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:26:56/fold3_epoch80_weights-2026-04-22_12:34:44.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-22_12:26:56",
    "model": "hybrid_mamba_attention",
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-04-22T12:34:46.419752",
    "training_started_at": "2026-04-22T12:26:56.282410",
    "training_finished_at": "2026-04-22T12:34:45.728167",
    "training_duration_seconds": 469.446,
    "training_duration_hours": 0.1304,
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
      "accuracy": 0.59091,
      "macro_f1": 0.38406,
      "macro_precision": 0.38203,
      "macro_recall": 0.39167,
      "balanced_accuracy": 0.39167
    },
    {
      "fold": 2,
      "best_epoch": 80,
      "accuracy": 0.54545,
      "macro_f1": 0.33854,
      "macro_precision": 0.3875,
      "macro_recall": 0.35417,
      "balanced_accuracy": 0.35417
    },
    {
      "fold": 3,
      "best_epoch": 25,
      "accuracy": 0.27273,
      "macro_f1": 0.35278,
      "macro_precision": 0.39687,
      "macro_recall": 0.4375,
      "balanced_accuracy": 0.4375
    }
  ],
  "summary": {
    "mean_accuracy": 0.4697,
    "std_accuracy": 0.14051,
    "mean_macro_f1": 0.35846,
    "std_macro_f1": 0.01901,
    "mean_macro_precision": 0.3888,
    "std_macro_precision": 0.00613,
    "mean_macro_recall": 0.39445,
    "std_macro_recall": 0.03408,
    "mean_balanced_accuracy": 0.39445,
    "std_balanced_accuracy": 0.03408
  }
}
```
