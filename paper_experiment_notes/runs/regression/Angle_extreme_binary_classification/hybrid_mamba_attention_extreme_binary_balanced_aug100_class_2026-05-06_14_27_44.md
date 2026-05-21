# Run: hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44

## 一句話用途
- 方法名稱: `Extreme binary + balanced aug100/class`
- 實驗描述: Gray-zone excluded binary classification with virtual train-fold augmentation to 100 samples per class before per-epoch balanced sampling.
- 任務: `Angle_extreme_binary_classification`
- 模型: `hybrid_mamba_attention`
- target_mode/task_type: `angle_binary_extreme`
- 結果時間: `2026-05-06T14:41:10.649192`
- 訓練時間: `2026-05-06T14:27:44.471399` -> `2026-05-06T14:41:10.177263`
- 訓練耗時: 0.2238 hours
- 原始 results.json: [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/results.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/results.json)

## 類別定義
- 0: Abnormal/emphysema-like (AC <=131 deg)
- 1: Normal-like (AC >=152 deg)

## Summary
| metric | value |
| --- | --- |
| mean_accuracy | 0.86923 |
| std_accuracy | 0.06557 |
| mean_macro_f1 | 0.81773 |
| std_macro_f1 | 0.07612 |
| mean_macro_precision | 0.83508 |
| std_macro_precision | 0.09156 |
| mean_macro_recall | 0.82889 |
| std_macro_recall | 0.09631 |
| mean_balanced_accuracy | 0.82889 |
| std_balanced_accuracy | 0.09631 |

## Fold Metrics
| fold | best_epoch | accuracy | macro_f1 | macro_precision | macro_recall | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 0.84615 | 0.78333 | 0.78333 | 0.78333 | 0.78333 |
| 2 | 10 | 0.75 | 0.69748 | 0.6875 | 0.72222 | 0.72222 |
| 3 | 5 | 0.91667 | 0.89916 | 0.875 | 0.94444 | 0.94444 |
| 4 | 30 | 0.91667 | 0.89916 | 0.875 | 0.94444 | 0.94444 |
| 5 | 20 | 0.91667 | 0.80952 | 0.95455 | 0.75 | 0.75 |

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
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_confusion_matrix.png) — 195.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_loss.png) — 171.3 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_summary.png) — 369.9 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_confusion_matrix.png) — 193.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_loss.png) — 188.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_summary.png) — 392.3 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_confusion_matrix.png) — 197.2 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_loss.png) — 181.5 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_summary.png) — 332.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_confusion_matrix.png) — 196.5 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_loss.png) — 188.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_summary.png) — 369.5 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_confusion_matrix.png) — 194.0 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_loss.png) — 180.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_summary.png) — 381.4 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/metric_boxplot.png) — 134.8 KB
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/total_confusion_matrix.png](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/total_confusion_matrix.png) — 207.1 KB

### prediction_files
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=13
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12
- [regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_predictions.json) — json keys=['fold', 'accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'class_names', 'confusion_matrix', 'predictions']; rows=12

### summary_files
_No files found._

### logs
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2.log) — 1.6 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3.log) — 1.4 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4.log) — 2.3 KB
- [regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5.log](/home/felix/Research/nnMamba/regression/train_log/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5.log) — 2.0 KB

### weights
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch10_weights-2026-05-06_14:28:26.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch10_weights-2026-05-06_14:28:26.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch20_weights-2026-05-06_14:29:01.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch20_weights-2026-05-06_14:29:01.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch30_weights-2026-05-06_14:29:37.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold1_epoch30_weights-2026-05-06_14:29:37.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch10_weights-2026-05-06_14:30:31.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch10_weights-2026-05-06_14:30:31.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch20_weights-2026-05-06_14:31:07.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch20_weights-2026-05-06_14:31:07.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch30_weights-2026-05-06_14:31:43.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold2_epoch30_weights-2026-05-06_14:31:43.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch10_weights-2026-05-06_14:32:57.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch10_weights-2026-05-06_14:32:57.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch20_weights-2026-05-06_14:33:34.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch20_weights-2026-05-06_14:33:34.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch30_weights-2026-05-06_14:34:10.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold3_epoch30_weights-2026-05-06_14:34:10.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch10_weights-2026-05-06_14:35:04.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch10_weights-2026-05-06_14:35:04.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch20_weights-2026-05-06_14:35:40.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch20_weights-2026-05-06_14:35:40.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch30_weights-2026-05-06_14:36:15.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch30_weights-2026-05-06_14:36:15.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch40_weights-2026-05-06_14:36:51.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch40_weights-2026-05-06_14:36:51.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch50_weights-2026-05-06_14:37:28.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold4_epoch50_weights-2026-05-06_14:37:28.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch10_weights-2026-05-06_14:38:41.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch10_weights-2026-05-06_14:38:41.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch20_weights-2026-05-06_14:39:19.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch20_weights-2026-05-06_14:39:19.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch30_weights-2026-05-06_14:39:56.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch30_weights-2026-05-06_14:39:56.pth) — 4.3 MB
- [regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch40_weights-2026-05-06_14:40:33.pth](/home/felix/Research/nnMamba/regression/weights/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44/fold5_epoch40_weights-2026-05-06_14:40:33.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44",
    "model": "hybrid_mamba_attention",
    "experiment": {
      "name": "Extreme binary + balanced aug100/class",
      "description": "Gray-zone excluded binary classification with virtual train-fold augmentation to 100 samples per class before per-epoch balanced sampling."
    },
    "task": "Angle_extreme_binary_classification",
    "task_type": "angle_binary_extreme",
    "timestamp": "2026-05-06T14:41:10.649192",
    "training_started_at": "2026-05-06T14:27:44.471399",
    "training_finished_at": "2026-05-06T14:41:10.177263",
    "training_duration_seconds": 805.706,
    "training_duration_hours": 0.2238,
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
      "accuracy": 0.84615,
      "macro_f1": 0.78333,
      "macro_precision": 0.78333,
      "macro_recall": 0.78333,
      "balanced_accuracy": 0.78333
    },
    {
      "fold": 2,
      "best_epoch": 10,
      "accuracy": 0.75,
      "macro_f1": 0.69748,
      "macro_precision": 0.6875,
      "macro_recall": 0.72222,
      "balanced_accuracy": 0.72222
    },
    {
      "fold": 3,
      "best_epoch": 5,
      "accuracy": 0.91667,
      "macro_f1": 0.89916,
      "macro_precision": 0.875,
      "macro_recall": 0.94444,
      "balanced_accuracy": 0.94444
    },
    {
      "fold": 4,
      "best_epoch": 30,
      "accuracy": 0.91667,
      "macro_f1": 0.89916,
      "macro_precision": 0.875,
      "macro_recall": 0.94444,
      "balanced_accuracy": 0.94444
    },
    {
      "fold": 5,
      "best_epoch": 20,
      "accuracy": 0.91667,
      "macro_f1": 0.80952,
      "macro_precision": 0.95455,
      "macro_recall": 0.75,
      "balanced_accuracy": 0.75
    }
  ],
  "summary": {
    "mean_accuracy": 0.86923,
    "std_accuracy": 0.06557,
    "mean_macro_f1": 0.81773,
    "std_macro_f1": 0.07612,
    "mean_macro_precision": 0.83508,
    "std_macro_precision": 0.09156,
    "mean_macro_recall": 0.82889,
    "std_macro_recall": 0.09631,
    "mean_balanced_accuracy": 0.82889,
    "std_balanced_accuracy": 0.09631
  }
}
```
