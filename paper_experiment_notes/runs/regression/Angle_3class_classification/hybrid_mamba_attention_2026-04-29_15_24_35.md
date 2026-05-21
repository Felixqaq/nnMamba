# Run: hybrid_mamba_attention_2026-04-29_15:24:35

## 一句話用途
- 任務: `Angle_3class_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_3class`
- 結果時間: `2026-04-29T15:52:14.528697`
- 訓練時間: `2026-04-29T15:24:35.805982` -> `2026-04-29T15:52:14.068763`
- 訓練耗時: 0.4606 hours
- 原始 results.json: [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/results.json)

## 類別定義
- 0: Emphysema/Abnormal (<=131 deg)
- 1: Intermediate (132-151 deg)
- 2: Normal (>=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.75934 |
| std_accuracy | 0.10668 |
| mean_macro_f1 | 0.46771 |
| std_macro_f1 | 0.13136 |
| mean_macro_precision | 0.44151 |
| std_macro_precision | 0.1349 |
| mean_macro_recall | 0.51185 |
| std_macro_recall | 0.13823 |
| mean_balanced_accuracy | 0.51185 |
| std_balanced_accuracy | 0.13823 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 50 | 0.64286 | 0.26087 | 0.23077 | 0.3 | 0.3 |
| 2 | 10 | 0.69231 | 0.48889 | 0.40909 | 0.62963 | 0.62963 |
| 3 | 35 | 0.84615 | 0.53801 | 0.52222 | 0.55556 | 0.55556 |
| 4 | 55 | 0.69231 | 0.4 | 0.40909 | 0.40741 | 0.40741 |
| 5 | 20 | 0.92308 | 0.65079 | 0.63636 | 0.66667 | 0.66667 |

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
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_confusion_matrix.png) — 196.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_loss.png) — 159.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_summary.png) — 331.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_confusion_matrix.png) — 196.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_loss.png) — 139.8 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_summary.png) — 328.4 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_confusion_matrix.png) — 197.1 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_loss.png) — 151.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_summary.png) — 355.0 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_confusion_matrix.png) — 195.5 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_loss.png) — 150.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_summary.png) — 330.7 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_confusion_matrix.png) — 193.9 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_loss.png) — 135.2 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_summary.png) — 347.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/metric_boxplot.png) — 113.6 KB
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/total_confusion_matrix.png) — 207.5 KB

### prediction_files
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=14
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1.log) — 3.1 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2.log) — 1.6 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3.log) — 2.5 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4.log) — 3.3 KB
- [regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5.log) — 2.0 KB

### weights
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch10_weights-2026-04-29_15:25:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch10_weights-2026-04-29_15:25:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch20_weights-2026-04-29_15:26:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch20_weights-2026-04-29_15:26:20.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch30_weights-2026-04-29_15:27:13.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch30_weights-2026-04-29_15:27:13.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch40_weights-2026-04-29_15:28:06.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch40_weights-2026-04-29_15:28:06.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch50_weights-2026-04-29_15:28:59.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch50_weights-2026-04-29_15:28:59.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch60_weights-2026-04-29_15:29:52.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch60_weights-2026-04-29_15:29:52.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch70_weights-2026-04-29_15:30:44.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold1_epoch70_weights-2026-04-29_15:30:44.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch10_weights-2026-04-29_15:32:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch10_weights-2026-04-29_15:32:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch20_weights-2026-04-29_15:33:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch20_weights-2026-04-29_15:33:20.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch30_weights-2026-04-29_15:34:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold2_epoch30_weights-2026-04-29_15:34:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch10_weights-2026-04-29_15:35:54.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch10_weights-2026-04-29_15:35:54.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch20_weights-2026-04-29_15:36:45.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch20_weights-2026-04-29_15:36:45.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch30_weights-2026-04-29_15:37:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch30_weights-2026-04-29_15:37:37.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch40_weights-2026-04-29_15:38:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch40_weights-2026-04-29_15:38:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch50_weights-2026-04-29_15:39:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch50_weights-2026-04-29_15:39:20.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch60_weights-2026-04-29_15:40:12.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold3_epoch60_weights-2026-04-29_15:40:12.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch10_weights-2026-04-29_15:41:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch10_weights-2026-04-29_15:41:28.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch20_weights-2026-04-29_15:42:20.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch20_weights-2026-04-29_15:42:20.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch30_weights-2026-04-29_15:43:11.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch30_weights-2026-04-29_15:43:11.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch40_weights-2026-04-29_15:44:03.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch40_weights-2026-04-29_15:44:03.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch50_weights-2026-04-29_15:44:55.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch50_weights-2026-04-29_15:44:55.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch60_weights-2026-04-29_15:45:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch60_weights-2026-04-29_15:45:47.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch70_weights-2026-04-29_15:46:38.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch70_weights-2026-04-29_15:46:38.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch80_weights-2026-04-29_15:47:29.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold4_epoch80_weights-2026-04-29_15:47:29.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch10_weights-2026-04-29_15:48:47.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch10_weights-2026-04-29_15:48:47.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch20_weights-2026-04-29_15:49:39.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch20_weights-2026-04-29_15:49:39.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch30_weights-2026-04-29_15:50:30.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch30_weights-2026-04-29_15:50:30.pth) — 4.3 MB
- [regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch40_weights-2026-04-29_15:51:22.pth](/home/felix/Research/nnMamba/regression/weights/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/fold5_epoch40_weights-2026-04-29_15:51:22.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-29_15:24:35",
    "model": "hybrid_mamba_attention",
    "task": "Angle_3class_classification",
    "task_type": "angle_3class",
    "timestamp": "2026-04-29T15:52:14.528697",
    "training_started_at": "2026-04-29T15:24:35.805982",
    "training_finished_at": "2026-04-29T15:52:14.068763",
    "training_duration_seconds": 1658.263,
    "training_duration_hours": 0.4606,
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
      "best_epoch": 50,
      "accuracy": 0.64286,
      "macro_f1": 0.26087,
      "macro_precision": 0.23077,
      "macro_recall": 0.3,
      "balanced_accuracy": 0.3
    },
    {
      "fold": 2,
      "best_epoch": 10,
      "accuracy": 0.69231,
      "macro_f1": 0.48889,
      "macro_precision": 0.40909,
      "macro_recall": 0.62963,
      "balanced_accuracy": 0.62963
    },
    {
      "fold": 3,
      "best_epoch": 35,
      "accuracy": 0.84615,
      "macro_f1": 0.53801,
      "macro_precision": 0.52222,
      "macro_recall": 0.55556,
      "balanced_accuracy": 0.55556
    },
    {
      "fold": 4,
      "best_epoch": 55,
      "accuracy": 0.69231,
      "macro_f1": 0.4,
      "macro_precision": 0.40909,
      "macro_recall": 0.40741,
      "balanced_accuracy": 0.40741
    },
    {
      "fold": 5,
      "best_epoch": 20,
      "accuracy": 0.92308,
      "macro_f1": 0.65079,
      "macro_precision": 0.63636,
      "macro_recall": 0.66667,
      "balanced_accuracy": 0.66667
    }
  ],
  "summary": {
    "mean_accuracy": 0.75934,
    "std_accuracy": 0.10668,
    "mean_macro_f1": 0.46771,
    "std_macro_f1": 0.13136,
    "mean_macro_precision": 0.44151,
    "std_macro_precision": 0.1349,
    "mean_macro_recall": 0.51185,
    "std_macro_recall": 0.13823,
    "mean_balanced_accuracy": 0.51185,
    "std_balanced_accuracy": 0.13823
  }
}
```
