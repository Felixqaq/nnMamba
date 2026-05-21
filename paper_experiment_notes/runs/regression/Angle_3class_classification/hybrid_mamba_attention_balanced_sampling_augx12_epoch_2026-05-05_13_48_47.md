# Run: hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47

## 一句話用途
- 方法名稱: `Balanced sampling + augx12/epoch`
- 實驗描述: Each epoch first undersamples all classes to the minority count, then uses 12 total views per selected sample.
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-05-05T14:03:53.108221`
- 訓練時間: `2026-05-05T13:48:47.385365` -> `2026-05-05T14:03:52.615582`
- 訓練耗時: 0.2515 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.42308 |
| std_accuracy | 0.0973 |
| mean_macro_f1 | 0.34325 |
| std_macro_f1 | 0.10997 |
| mean_macro_precision | 0.391 |
| std_macro_precision | 0.09205 |
| mean_macro_recall | 0.47704 |
| std_macro_recall | 0.16197 |
| mean_balanced_accuracy | 0.47704 |
| std_balanced_accuracy | 0.16197 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 0.5 | 0.22222 | 0.21212 | 0.23333 | 0.23333 |
| 2 | 30 | 0.30769 | 0.46667 | 0.41667 | 0.66667 | 0.66667 |
| 3 | 30 | 0.53846 | 0.40513 | 0.47619 | 0.48148 | 0.48148 |
| 4 | 5 | 0.30769 | 0.2 | 0.41667 | 0.37037 | 0.37037 |
| 5 | 5 | 0.46154 | 0.42222 | 0.43333 | 0.63333 | 0.63333 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_confusion_matrix.png) — 216.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_loss.png) — 187.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_summary.png) — 357.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_confusion_matrix.png) — 219.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_loss.png) — 196.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_summary.png) — 354.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_confusion_matrix.png) — 227.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_loss.png) — 206.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_summary.png) — 381.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_confusion_matrix.png) — 217.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_loss.png) — 159.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_summary.png) — 307.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_confusion_matrix.png) — 224.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_loss.png) — 180.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_summary.png) — 390.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/metric_boxplot.png) — 130.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/total_confusion_matrix.png) — 225.1 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3.log) — 2.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4.log) — 1.4 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5.log) — 1.4 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch10_weights-2026-05-05_13:49:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch10_weights-2026-05-05_13:49:23.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch20_weights-2026-05-05_13:49:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch20_weights-2026-05-05_13:49:56.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch30_weights-2026-05-05_13:50:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold1_epoch30_weights-2026-05-05_13:50:30.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch10_weights-2026-05-05_13:51:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch10_weights-2026-05-05_13:51:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch20_weights-2026-05-05_13:52:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch20_weights-2026-05-05_13:52:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch30_weights-2026-05-05_13:53:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch30_weights-2026-05-05_13:53:04.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch40_weights-2026-05-05_13:53:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch40_weights-2026-05-05_13:53:43.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch50_weights-2026-05-05_13:54:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold2_epoch50_weights-2026-05-05_13:54:22.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch10_weights-2026-05-05_13:55:42.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch10_weights-2026-05-05_13:55:42.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch20_weights-2026-05-05_13:56:23.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch20_weights-2026-05-05_13:56:23.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch30_weights-2026-05-05_13:57:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch30_weights-2026-05-05_13:57:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch40_weights-2026-05-05_13:57:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch40_weights-2026-05-05_13:57:44.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch50_weights-2026-05-05_13:58:24.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold3_epoch50_weights-2026-05-05_13:58:24.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch10_weights-2026-05-05_13:59:46.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch10_weights-2026-05-05_13:59:46.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch20_weights-2026-05-05_14:00:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch20_weights-2026-05-05_14:00:26.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch30_weights-2026-05-05_14:01:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold4_epoch30_weights-2026-05-05_14:01:07.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch10_weights-2026-05-05_14:02:09.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch10_weights-2026-05-05_14:02:09.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch20_weights-2026-05-05_14:02:50.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch20_weights-2026-05-05_14:02:50.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch30_weights-2026-05-05_14:03:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47/fold5_epoch30_weights-2026-05-05_14:03:31.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_balanced_sampling_augx12_epoch_2026-05-05_13:48:47",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Balanced sampling + augx12/epoch",
      "description": "Each epoch first undersamples all classes to the minority count, then uses 12 total views per selected sample."
    },
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-05-05T14:03:53.108221",
    "training_started_at": "2026-05-05T13:48:47.385365",
    "training_finished_at": "2026-05-05T14:03:52.615582",
    "training_duration_seconds": 905.23,
    "training_duration_hours": 0.2515,
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
      "best_epoch": 10,
      "accuracy": 0.5,
      "macro_f1": 0.22222,
      "macro_precision": 0.21212,
      "macro_recall": 0.23333,
      "balanced_accuracy": 0.23333
    },
    {
      "fold": 2,
      "best_epoch": 30,
      "accuracy": 0.30769,
      "macro_f1": 0.46667,
      "macro_precision": 0.41667,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "accuracy": 0.53846,
      "macro_f1": 0.40513,
      "macro_precision": 0.47619,
      "macro_recall": 0.48148,
      "balanced_accuracy": 0.48148
    },
    {
      "fold": 4,
      "best_epoch": 5,
      "accuracy": 0.30769,
      "macro_f1": 0.2,
      "macro_precision": 0.41667,
      "macro_recall": 0.37037,
      "balanced_accuracy": 0.37037
    },
    {
      "fold": 5,
      "best_epoch": 5,
      "accuracy": 0.46154,
      "macro_f1": 0.42222,
      "macro_precision": 0.43333,
      "macro_recall": 0.63333,
      "balanced_accuracy": 0.63333
    }
  ],
  "summary": {
    "mean_accuracy": 0.42308,
    "std_accuracy": 0.0973,
    "mean_macro_f1": 0.34325,
    "std_macro_f1": 0.10997,
    "mean_macro_precision": 0.391,
    "std_macro_precision": 0.09205,
    "mean_macro_recall": 0.47704,
    "std_macro_recall": 0.16197,
    "mean_balanced_accuracy": 0.47704,
    "std_balanced_accuracy": 0.16197
  }
}
```
