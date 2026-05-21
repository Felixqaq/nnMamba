# Run: hybrid_mamba_attention_2026-04-09_14:05:42

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T14:11:43.484980`
- 訓練時間: `2026-04-09T14:05:42.237190` -> `2026-04-09T14:11:42.419740`
- 訓練耗時: 0.1001 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 16.44861 |
| std_mae | 2.94086 |
| mean_rmse | 22.1694 |
| std_rmse | 2.71803 |
| mean_r2 | 0.1728 |
| std_r2 | 0.1073 |
| mean_pearson | 0.57414 |
| std_pearson | 0.08829 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 15.66992 | 21.26646 | 0.18877 | 0.49844 | 3.60888 |
| 2 | 65 | 17.6591 | 23.35691 | 0.11471 | 0.46669 | 5.04563 |
| 3 | 15 | 19.14919 | 24.41901 | 0.10034 | 0.57497 | 11.11562 |
| 4 | 40 | 11.07634 | 17.2663 | 0.37521 | 0.61489 | -0.63138 |
| 5 | 25 | 18.6885 | 24.5383 | 0.08497 | 0.71569 | 10.81668 |

## Training Config Embedded In Result
```json
{
  "epochs": 80,
  "batch_size": 12,
  "learning_rate": 0.0001,
  "weight_decay": 0.001,
  "k_folds": 5,
  "seed": 42,
  "loss": "smooth_l1"
}
```

## Artifact Index
### figures
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_bland_altman.png) — 123.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_error_hist.png) — 102.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_loss.png) — 138.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_residuals.png) — 93.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_scatter.png) — 194.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_summary.png) — 424.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_bland_altman.png) — 124.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_error_hist.png) — 96.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_loss.png) — 162.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_residuals.png) — 95.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_scatter.png) — 217.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_summary.png) — 465.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_bland_altman.png) — 118.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_error_hist.png) — 90.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_loss.png) — 137.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_residuals.png) — 92.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_scatter.png) — 206.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_summary.png) — 452.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_bland_altman.png) — 129.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_error_hist.png) — 85.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_loss.png) — 155.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_residuals.png) — 94.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_scatter.png) — 210.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_summary.png) — 460.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_bland_altman.png) — 121.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_error_hist.png) — 99.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_loss.png) — 150.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_residuals.png) — 89.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_scatter.png) — 207.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_summary.png) — 438.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/metric_boxplot.png) — 95.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_bland_altman.png) — 160.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_error_hist.png) — 90.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_residuals.png) — 121.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/total_scatter.png) — 242.0 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch10_weights-2026-04-09_14:05:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch10_weights-2026-04-09_14:05:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch20_weights-2026-04-09_14:06:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch20_weights-2026-04-09_14:06:02.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch30_weights-2026-04-09_14:06:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch30_weights-2026-04-09_14:06:10.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch40_weights-2026-04-09_14:06:18.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch40_weights-2026-04-09_14:06:18.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch50_weights-2026-04-09_14:06:26.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch50_weights-2026-04-09_14:06:26.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch60_weights-2026-04-09_14:06:34.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch60_weights-2026-04-09_14:06:34.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch70_weights-2026-04-09_14:06:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch70_weights-2026-04-09_14:06:42.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch80_weights-2026-04-09_14:06:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold1_epoch80_weights-2026-04-09_14:06:51.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch10_weights-2026-04-09_14:07:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch10_weights-2026-04-09_14:07:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch20_weights-2026-04-09_14:07:12.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch20_weights-2026-04-09_14:07:12.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch30_weights-2026-04-09_14:07:21.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch30_weights-2026-04-09_14:07:21.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch40_weights-2026-04-09_14:07:30.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch40_weights-2026-04-09_14:07:30.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch50_weights-2026-04-09_14:07:39.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch50_weights-2026-04-09_14:07:39.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch60_weights-2026-04-09_14:07:48.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch60_weights-2026-04-09_14:07:48.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch70_weights-2026-04-09_14:07:58.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch70_weights-2026-04-09_14:07:58.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch80_weights-2026-04-09_14:08:06.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold2_epoch80_weights-2026-04-09_14:08:06.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch10_weights-2026-04-09_14:08:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch10_weights-2026-04-09_14:08:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch20_weights-2026-04-09_14:08:26.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch20_weights-2026-04-09_14:08:26.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch30_weights-2026-04-09_14:08:34.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch30_weights-2026-04-09_14:08:34.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch40_weights-2026-04-09_14:08:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch40_weights-2026-04-09_14:08:42.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch50_weights-2026-04-09_14:08:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch50_weights-2026-04-09_14:08:51.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch60_weights-2026-04-09_14:08:59.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch60_weights-2026-04-09_14:08:59.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch70_weights-2026-04-09_14:09:07.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch70_weights-2026-04-09_14:09:07.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch80_weights-2026-04-09_14:09:15.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold3_epoch80_weights-2026-04-09_14:09:15.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch10_weights-2026-04-09_14:09:26.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch10_weights-2026-04-09_14:09:26.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch20_weights-2026-04-09_14:09:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch20_weights-2026-04-09_14:09:35.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch30_weights-2026-04-09_14:09:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch30_weights-2026-04-09_14:09:45.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch40_weights-2026-04-09_14:09:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch40_weights-2026-04-09_14:09:54.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch50_weights-2026-04-09_14:10:03.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch50_weights-2026-04-09_14:10:03.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch60_weights-2026-04-09_14:10:11.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch60_weights-2026-04-09_14:10:11.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch70_weights-2026-04-09_14:10:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch70_weights-2026-04-09_14:10:20.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch80_weights-2026-04-09_14:10:29.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold4_epoch80_weights-2026-04-09_14:10:29.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch10_weights-2026-04-09_14:10:40.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch10_weights-2026-04-09_14:10:40.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch20_weights-2026-04-09_14:10:50.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch20_weights-2026-04-09_14:10:50.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch30_weights-2026-04-09_14:10:59.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch30_weights-2026-04-09_14:10:59.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch40_weights-2026-04-09_14:11:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch40_weights-2026-04-09_14:11:08.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch50_weights-2026-04-09_14:11:16.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch50_weights-2026-04-09_14:11:16.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch60_weights-2026-04-09_14:11:24.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch60_weights-2026-04-09_14:11:24.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch70_weights-2026-04-09_14:11:32.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch70_weights-2026-04-09_14:11:32.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch80_weights-2026-04-09_14:11:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:05:42/fold5_epoch80_weights-2026-04-09_14:11:41.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:05:42",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T14:11:43.484980",
    "training_started_at": "2026-04-09T14:05:42.237190",
    "training_finished_at": "2026-04-09T14:11:42.419740",
    "training_duration_seconds": 360.183,
    "training_duration_hours": 0.1001,
    "config": {
      "epochs": 80,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.001,
      "k_folds": 5,
      "seed": 42,
      "loss": "smooth_l1"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 15,
      "mae": 15.66992,
      "rmse": 21.26646,
      "r2": 0.18877,
      "pearson": 0.49844,
      "mean_error": 3.60888
    },
    {
      "fold": 2,
      "best_epoch": 65,
      "mae": 17.6591,
      "rmse": 23.35691,
      "r2": 0.11471,
      "pearson": 0.46669,
      "mean_error": 5.04563
    },
    {
      "fold": 3,
      "best_epoch": 15,
      "mae": 19.14919,
      "rmse": 24.41901,
      "r2": 0.10034,
      "pearson": 0.57497,
      "mean_error": 11.11562
    },
    {
      "fold": 4,
      "best_epoch": 40,
      "mae": 11.07634,
      "rmse": 17.2663,
      "r2": 0.37521,
      "pearson": 0.61489,
      "mean_error": -0.63138
    },
    {
      "fold": 5,
      "best_epoch": 25,
      "mae": 18.6885,
      "rmse": 24.5383,
      "r2": 0.08497,
      "pearson": 0.71569,
      "mean_error": 10.81668
    }
  ],
  "summary": {
    "mean_mae": 16.44861,
    "std_mae": 2.94086,
    "mean_rmse": 22.1694,
    "std_rmse": 2.71803,
    "mean_r2": 0.1728,
    "std_r2": 0.1073,
    "mean_pearson": 0.57414,
    "std_pearson": 0.08829
  }
}
```
