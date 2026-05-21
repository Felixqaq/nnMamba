# Run: hybrid_mamba_attention_2026-04-22_12:59:09

## 一句話用途
- 任務: `GOLD_stage_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `gold`
- 結果時間: `2026-04-22T13:13:16.277682`
- 訓練時間: `2026-04-22T12:59:09.242340` -> `2026-04-22T13:13:15.746415`
- 訓練耗時: 0.2351 hours
- 原始 results.json: [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/results.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/results.json)

## 類別定義
- 0: GOLD 1 (Mild)
- 1: GOLD 2 (Moderate)
- 2: GOLD 3 (Severe)
- 3: GOLD 4 (Very Severe)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.5 |
| std_accuracy | 0.06428 |
| mean_macro_f1 | 0.33575 |
| std_macro_f1 | 0.09381 |
| mean_macro_precision | 0.37514 |
| std_macro_precision | 0.15364 |
| mean_macro_recall | 0.39028 |
| std_macro_recall | 0.10432 |
| mean_balanced_accuracy | 0.39028 |
| std_balanced_accuracy | 0.10432 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 140 | 0.59091 | 0.43669 | 0.55833 | 0.42083 | 0.42083 |
| 2 | 70 | 0.45455 | 0.35982 | 0.38474 | 0.5 | 0.5 |
| 3 | 25 | 0.45455 | 0.21073 | 0.18235 | 0.25 | 0.25 |

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
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_confusion_matrix.png) — 189.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_loss.png) — 172.1 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_summary.png) — 390.4 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_confusion_matrix.png) — 187.2 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_loss.png) — 158.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_summary.png) — 337.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_confusion_matrix.png) — 192.6 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_loss.png) — 179.6 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_summary.png) — 319.0 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/metric_boxplot.png) — 114.9 KB
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/total_confusion_matrix.png) — 195.3 KB

### prediction_files
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22
- [regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=22

### summary_files
_No files found._

### logs
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1.log) — 6.1 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2.log) — 6.1 KB
- [regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3.log](/home/felix/Research/nnMamba/regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3.log) — 6.1 KB

### weights
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch100_weights-2026-04-22_13:02:06.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch100_weights-2026-04-22_13:02:06.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch10_weights-2026-04-22_12:59:28.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch10_weights-2026-04-22_12:59:28.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch110_weights-2026-04-22_13:02:23.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch110_weights-2026-04-22_13:02:23.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch120_weights-2026-04-22_13:02:42.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch120_weights-2026-04-22_13:02:42.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch130_weights-2026-04-22_13:02:59.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch130_weights-2026-04-22_13:02:59.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch140_weights-2026-04-22_13:03:17.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch140_weights-2026-04-22_13:03:17.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch150_weights-2026-04-22_13:03:34.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch150_weights-2026-04-22_13:03:34.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch160_weights-2026-04-22_13:03:51.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch160_weights-2026-04-22_13:03:51.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch20_weights-2026-04-22_12:59:46.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch20_weights-2026-04-22_12:59:46.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch30_weights-2026-04-22_13:00:04.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch30_weights-2026-04-22_13:00:04.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch40_weights-2026-04-22_13:00:21.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch40_weights-2026-04-22_13:00:21.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch50_weights-2026-04-22_13:00:39.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch50_weights-2026-04-22_13:00:39.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch60_weights-2026-04-22_13:00:57.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch60_weights-2026-04-22_13:00:57.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch70_weights-2026-04-22_13:01:14.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch70_weights-2026-04-22_13:01:14.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch80_weights-2026-04-22_13:01:32.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch80_weights-2026-04-22_13:01:32.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch90_weights-2026-04-22_13:01:49.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold1_epoch90_weights-2026-04-22_13:01:49.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch100_weights-2026-04-22_13:06:48.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch100_weights-2026-04-22_13:06:48.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch10_weights-2026-04-22_13:04:10.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch10_weights-2026-04-22_13:04:10.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch110_weights-2026-04-22_13:07:06.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch110_weights-2026-04-22_13:07:06.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch120_weights-2026-04-22_13:07:23.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch120_weights-2026-04-22_13:07:23.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch130_weights-2026-04-22_13:07:40.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch130_weights-2026-04-22_13:07:40.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch140_weights-2026-04-22_13:07:58.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch140_weights-2026-04-22_13:07:58.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch150_weights-2026-04-22_13:08:16.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch150_weights-2026-04-22_13:08:16.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch160_weights-2026-04-22_13:08:33.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch160_weights-2026-04-22_13:08:33.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch20_weights-2026-04-22_13:04:28.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch20_weights-2026-04-22_13:04:28.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch30_weights-2026-04-22_13:04:45.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch30_weights-2026-04-22_13:04:45.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch40_weights-2026-04-22_13:05:03.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch40_weights-2026-04-22_13:05:03.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch50_weights-2026-04-22_13:05:20.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch50_weights-2026-04-22_13:05:20.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch60_weights-2026-04-22_13:05:38.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch60_weights-2026-04-22_13:05:38.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch70_weights-2026-04-22_13:05:56.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch70_weights-2026-04-22_13:05:56.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch80_weights-2026-04-22_13:06:14.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch80_weights-2026-04-22_13:06:14.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch90_weights-2026-04-22_13:06:31.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold2_epoch90_weights-2026-04-22_13:06:31.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch100_weights-2026-04-22_13:11:28.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch100_weights-2026-04-22_13:11:28.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch10_weights-2026-04-22_13:08:52.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch10_weights-2026-04-22_13:08:52.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch110_weights-2026-04-22_13:11:46.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch110_weights-2026-04-22_13:11:46.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch120_weights-2026-04-22_13:12:04.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch120_weights-2026-04-22_13:12:04.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch130_weights-2026-04-22_13:12:21.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch130_weights-2026-04-22_13:12:21.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch140_weights-2026-04-22_13:12:39.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch140_weights-2026-04-22_13:12:39.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch150_weights-2026-04-22_13:12:56.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch150_weights-2026-04-22_13:12:56.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch160_weights-2026-04-22_13:13:14.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch160_weights-2026-04-22_13:13:14.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch20_weights-2026-04-22_13:09:09.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch20_weights-2026-04-22_13:09:09.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch30_weights-2026-04-22_13:09:27.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch30_weights-2026-04-22_13:09:27.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch40_weights-2026-04-22_13:09:45.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch40_weights-2026-04-22_13:09:45.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch50_weights-2026-04-22_13:10:02.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch50_weights-2026-04-22_13:10:02.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch60_weights-2026-04-22_13:10:19.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch60_weights-2026-04-22_13:10:19.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch70_weights-2026-04-22_13:10:36.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch70_weights-2026-04-22_13:10:36.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch80_weights-2026-04-22_13:10:53.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch80_weights-2026-04-22_13:10:53.pth) — 4.3 MB
- [regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch90_weights-2026-04-22_13:11:11.pth](/home/felix/Research/nnMamba/regression/weights/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:59:09/fold3_epoch90_weights-2026-04-22_13:11:11.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-22_12:59:09",
    "model": "hybrid_mamba_attention",
    "task": "GOLD_stage_classification",
    "task_type": "gold",
    "timestamp": "2026-04-22T13:13:16.277682",
    "training_started_at": "2026-04-22T12:59:09.242340",
    "training_finished_at": "2026-04-22T13:13:15.746415",
    "training_duration_seconds": 846.504,
    "training_duration_hours": 0.2351,
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
      "best_epoch": 140,
      "accuracy": 0.59091,
      "macro_f1": 0.43669,
      "macro_precision": 0.55833,
      "macro_recall": 0.42083,
      "balanced_accuracy": 0.42083
    },
    {
      "fold": 2,
      "best_epoch": 70,
      "accuracy": 0.45455,
      "macro_f1": 0.35982,
      "macro_precision": 0.38474,
      "macro_recall": 0.5,
      "balanced_accuracy": 0.5
    },
    {
      "fold": 3,
      "best_epoch": 25,
      "accuracy": 0.45455,
      "macro_f1": 0.21073,
      "macro_precision": 0.18235,
      "macro_recall": 0.25,
      "balanced_accuracy": 0.25
    }
  ],
  "summary": {
    "mean_accuracy": 0.5,
    "std_accuracy": 0.06428,
    "mean_macro_f1": 0.33575,
    "std_macro_f1": 0.09381,
    "mean_macro_precision": 0.37514,
    "std_macro_precision": 0.15364,
    "mean_macro_recall": 0.39028,
    "std_macro_recall": 0.10432,
    "mean_balanced_accuracy": 0.39028,
    "std_balanced_accuracy": 0.10432
  }
}
```
