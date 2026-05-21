# Run: hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40

## 一句話用途
- 方法名稱: `GOLD balanced sampling + aug36/class`
- 實驗描述: Virtual train-fold augmentation brings all four GOLD classes to 36 samples before per-epoch balanced sampling.
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-05-07T11:32:03.340537`
- 訓練時間: `2026-05-07T11:17:40.808063` -> `2026-05-07T11:32:02.714356`
- 訓練耗時: 0.2394 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.56061 |
| std_accuracy | 0.11338 |
| mean_macro_f1 | 0.53231 |
| std_macro_f1 | 0.04788 |
| mean_macro_precision | 0.67006 |
| std_macro_precision | 0.03969 |
| mean_macro_recall | 0.52222 |
| std_macro_recall | 0.04632 |
| mean_balanced_accuracy | 0.52222 |
| std_balanced_accuracy | 0.04632 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 65 | 0.68182 | 0.6 | 0.69048 | 0.56667 | 0.56667 |
| 2 | 25 | 0.59091 | 0.5 | 0.61458 | 0.45833 | 0.45833 |
| 3 | 30 | 0.40909 | 0.49692 | 0.70513 | 0.54167 | 0.54167 |

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
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_confusion_matrix.png) — 213.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_loss.png) — 179.7 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_summary.png) — 376.3 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_confusion_matrix.png) — 214.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_loss.png) — 179.5 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_summary.png) — 366.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_confusion_matrix.png) — 214.8 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_loss.png) — 146.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_summary.png) — 356.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/metric_boxplot.png) — 134.9 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/total_confusion_matrix.png) — 237.5 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1.log) — 3.7 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2.log) — 2.2 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3.log) — 2.3 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch10_weights-2026-05-07_11:18:22.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch10_weights-2026-05-07_11:18:22.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch20_weights-2026-05-07_11:19:01.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch20_weights-2026-05-07_11:19:01.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch30_weights-2026-05-07_11:19:38.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch30_weights-2026-05-07_11:19:38.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch40_weights-2026-05-07_11:20:15.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch40_weights-2026-05-07_11:20:15.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch50_weights-2026-05-07_11:20:52.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch50_weights-2026-05-07_11:20:52.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch60_weights-2026-05-07_11:21:30.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch60_weights-2026-05-07_11:21:30.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch70_weights-2026-05-07_11:22:10.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch70_weights-2026-05-07_11:22:10.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch80_weights-2026-05-07_11:22:50.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch80_weights-2026-05-07_11:22:50.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch90_weights-2026-05-07_11:23:30.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold1_epoch90_weights-2026-05-07_11:23:30.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch10_weights-2026-05-07_11:24:33.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch10_weights-2026-05-07_11:24:33.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch20_weights-2026-05-07_11:25:15.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch20_weights-2026-05-07_11:25:15.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch30_weights-2026-05-07_11:25:57.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch30_weights-2026-05-07_11:25:57.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch40_weights-2026-05-07_11:26:39.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch40_weights-2026-05-07_11:26:39.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch50_weights-2026-05-07_11:27:22.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold2_epoch50_weights-2026-05-07_11:27:22.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch10_weights-2026-05-07_11:28:27.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch10_weights-2026-05-07_11:28:27.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch20_weights-2026-05-07_11:29:10.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch20_weights-2026-05-07_11:29:10.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch30_weights-2026-05-07_11:29:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch30_weights-2026-05-07_11:29:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch40_weights-2026-05-07_11:30:36.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch40_weights-2026-05-07_11:30:36.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch50_weights-2026-05-07_11:31:19.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40/fold3_epoch50_weights-2026-05-07_11:31:19.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "GOLD balanced sampling + aug36/class",
      "description": "Virtual train-fold augmentation brings all four GOLD classes to 36 samples before per-epoch balanced sampling."
    },
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-05-07T11:32:03.340537",
    "training_started_at": "2026-05-07T11:17:40.808063",
    "training_finished_at": "2026-05-07T11:32:02.714356",
    "training_duration_seconds": 861.906,
    "training_duration_hours": 0.2394,
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
      "best_epoch": 65,
      "accuracy": 0.68182,
      "macro_f1": 0.6,
      "macro_precision": 0.69048,
      "macro_recall": 0.56667,
      "balanced_accuracy": 0.56667
    },
    {
      "fold": 2,
      "best_epoch": 25,
      "accuracy": 0.59091,
      "macro_f1": 0.5,
      "macro_precision": 0.61458,
      "macro_recall": 0.45833,
      "balanced_accuracy": 0.45833
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "accuracy": 0.40909,
      "macro_f1": 0.49692,
      "macro_precision": 0.70513,
      "macro_recall": 0.54167,
      "balanced_accuracy": 0.54167
    }
  ],
  "summary": {
    "mean_accuracy": 0.56061,
    "std_accuracy": 0.11338,
    "mean_macro_f1": 0.53231,
    "std_macro_f1": 0.04788,
    "mean_macro_precision": 0.67006,
    "std_macro_precision": 0.03969,
    "mean_macro_recall": 0.52222,
    "std_macro_recall": 0.04632,
    "mean_balanced_accuracy": 0.52222,
    "std_balanced_accuracy": 0.04632
  }
}
```
