# Run: hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43

## 一句話用途
- 方法名稱: `Angle extreme binary baseline`
- 實驗描述: Gray-zone excluded binary classification using AC <=131 degrees as abnormal/emphysema-like and AC >=152 degrees as normal-like.
- 任務: `Angle_extreme_binary_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_binary_extreme`
- 結果時間: `2026-05-05T15:53:10.225722`
- 訓練時間: `2026-05-05T15:49:43.654198` -> `2026-05-05T15:53:09.797134`
- 訓練耗時: 0.0573 hours
- 原始 results.json: [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/results.json)

## 類別定義
- 0: Abnormal/emphysema-like (AC <=131 deg)
- 1: Normal-like (AC >=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.77051 |
| std_accuracy | 0.03228 |
| mean_macro_f1 | 0.43501 |
| std_macro_f1 | 0.01006 |
| mean_macro_precision | 0.38526 |
| std_macro_precision | 0.01614 |
| mean_macro_recall | 0.5 |
| std_macro_recall | 0 |
| mean_balanced_accuracy | 0.5 |
| std_balanced_accuracy | 0 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.76923 | 0.43478 | 0.38462 | 0.5 | 0.5 |
| 2 | 5 | 0.75 | 0.42857 | 0.375 | 0.5 | 0.5 |
| 3 | 5 | 0.75 | 0.42857 | 0.375 | 0.5 | 0.5 |
| 4 | 5 | 0.75 | 0.42857 | 0.375 | 0.5 | 0.5 |
| 5 | 5 | 0.83333 | 0.45455 | 0.41667 | 0.5 | 0.5 |

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
  "num_classes": 2,
  "target_mode": "angle_binary_extreme",
  "abnormal_rule": "AC <= 131 deg",
  "excluded_gray_zone": "132 <= AC < 152 deg",
  "normal_rule": "AC >= 152 deg"
}
```

## Artifact Index
### figures
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_confusion_matrix.png) — 190.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_loss.png) — 168.7 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_summary.png) — 302.0 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_confusion_matrix.png) — 194.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_loss.png) — 179.1 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_summary.png) — 307.3 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_confusion_matrix.png) — 194.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_loss.png) — 180.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_summary.png) — 309.6 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_confusion_matrix.png) — 193.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_loss.png) — 156.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_summary.png) — 289.6 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_confusion_matrix.png) — 191.1 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_loss.png) — 160.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_summary.png) — 291.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/metric_boxplot.png) — 120.7 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/total_confusion_matrix.png) — 197.9 KB

### prediction_files
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5.log) — 1.4 KB

### weights
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch10_weights-2026-05-05_15:49:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch10_weights-2026-05-05_15:49:57.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch20_weights-2026-05-05_15:50:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch20_weights-2026-05-05_15:50:09.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch30_weights-2026-05-05_15:50:21.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold1_epoch30_weights-2026-05-05_15:50:21.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch10_weights-2026-05-05_15:50:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch10_weights-2026-05-05_15:50:39.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch20_weights-2026-05-05_15:50:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch20_weights-2026-05-05_15:50:50.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch30_weights-2026-05-05_15:51:02.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold2_epoch30_weights-2026-05-05_15:51:02.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch10_weights-2026-05-05_15:51:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch10_weights-2026-05-05_15:51:19.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch20_weights-2026-05-05_15:51:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch20_weights-2026-05-05_15:51:31.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch30_weights-2026-05-05_15:51:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold3_epoch30_weights-2026-05-05_15:51:42.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch10_weights-2026-05-05_15:52:00.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch10_weights-2026-05-05_15:52:00.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch20_weights-2026-05-05_15:52:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch20_weights-2026-05-05_15:52:11.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch30_weights-2026-05-05_15:52:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold4_epoch30_weights-2026-05-05_15:52:22.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch10_weights-2026-05-05_15:52:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch10_weights-2026-05-05_15:52:40.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch20_weights-2026-05-05_15:52:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch20_weights-2026-05-05_15:52:51.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch30_weights-2026-05-05_15:53:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43/fold5_epoch30_weights-2026-05-05_15:53:03.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Angle extreme binary baseline",
      "description": "Gray-zone excluded binary classification using AC <=131 degrees as abnormal/emphysema-like and AC >=152 degrees as normal-like."
    },
    "task": "Angle_extreme_binary_classification",
    "task_type": "angle_binary_extreme",
    "timestamp": "2026-05-05T15:53:10.225722",
    "training_started_at": "2026-05-05T15:49:43.654198",
    "training_finished_at": "2026-05-05T15:53:09.797134",
    "training_duration_seconds": 206.143,
    "training_duration_hours": 0.0573,
    "class_names": [
      "Abnormal/emphysema-like (AC <=131 deg)",
      "Normal-like (AC >=152 deg)"
    ],
    "config": {
      "epochs": 160,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.001,
      "k_folds": 5,
      "seed": 42,
      "loss": "auto",
      "num_classes": 2,
      "target_mode": "angle_binary_extreme",
      "abnormal_rule": "AC <= 131 deg",
      "excluded_gray_zone": "132 <= AC < 152 deg",
      "normal_rule": "AC >= 152 deg"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 5,
      "accuracy": 0.76923,
      "macro_f1": 0.43478,
      "macro_precision": 0.38462,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 2,
      "best_epoch": 5,
      "accuracy": 0.75,
      "macro_f1": 0.42857,
      "macro_precision": 0.375,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 3,
      "best_epoch": 5,
      "accuracy": 0.75,
      "macro_f1": 0.42857,
      "macro_precision": 0.375,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 4,
      "best_epoch": 5,
      "accuracy": 0.75,
      "macro_f1": 0.42857,
      "macro_precision": 0.375,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 5,
      "best_epoch": 5,
      "accuracy": 0.83333,
      "macro_f1": 0.45455,
      "macro_precision": 0.41667,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    }
  ],
  "summary": {
    "mean_accuracy": 0.77051,
    "std_accuracy": 0.03228,
    "mean_macro_f1": 0.43501,
    "std_macro_f1": 0.01006,
    "mean_macro_precision": 0.38526,
    "std_macro_precision": 0.01614,
    "mean_macro_recall": 0.5,
    "std_macro_recall": 0.0,
    "mean_balanced_accuracy": 0.5,
    "std_balanced_accuracy": 0.0
  }
}
```
