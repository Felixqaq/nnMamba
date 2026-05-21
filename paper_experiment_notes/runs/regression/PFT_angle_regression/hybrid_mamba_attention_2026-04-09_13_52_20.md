# Run: hybrid_mamba_attention_2026-04-09_13:52:20

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T13:58:07.330523`
- 訓練時間: `2026-04-09T13:52:20.234456` -> `2026-04-09T13:58:06.193415`
- 訓練耗時: 0.0961 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 17.89599 |
| std_mae | 3.25584 |
| mean_rmse | 23.0391 |
| std_rmse | 3.93281 |
| mean_r2 | 0.08743 |
| std_r2 | 0.24851 |
| mean_pearson | 0.54386 |
| std_pearson | 0.13355 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 15.30494 | 21.97836 | 0.13355 | 0.50402 | 6.20855 |
| 2 | 50 | 13.16412 | 17.49525 | 0.5033 | 0.77236 | -6.22517 |
| 3 | 5 | 21.55385 | 25.79782 | -0.00412 | 0.3558 | 3.41918 |
| 4 | 10 | 18.34978 | 21.05888 | 0.07059 | 0.54624 | 0.8779 |
| 5 | 45 | 21.10728 | 28.86521 | -0.26619 | 0.54086 | 16.67899 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_bland_altman.png) — 120.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_error_hist.png) — 87.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_loss.png) — 154.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_residuals.png) — 91.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_scatter.png) — 193.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_summary.png) — 423.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_bland_altman.png) — 129.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_error_hist.png) — 89.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_loss.png) — 169.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_residuals.png) — 92.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_scatter.png) — 232.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_summary.png) — 437.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_bland_altman.png) — 125.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_error_hist.png) — 85.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_loss.png) — 141.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_residuals.png) — 95.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_scatter.png) — 194.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_summary.png) — 403.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_bland_altman.png) — 122.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_error_hist.png) — 84.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_loss.png) — 142.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_residuals.png) — 89.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_scatter.png) — 187.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_summary.png) — 404.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_bland_altman.png) — 123.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_error_hist.png) — 100.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_loss.png) — 156.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_residuals.png) — 91.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_scatter.png) — 202.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_summary.png) — 460.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/metric_boxplot.png) — 102.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_bland_altman.png) — 162.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_error_hist.png) — 96.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_residuals.png) — 123.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/total_scatter.png) — 235.5 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch10_weights-2026-04-09_13:52:30.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch10_weights-2026-04-09_13:52:30.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch20_weights-2026-04-09_13:52:39.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch20_weights-2026-04-09_13:52:39.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch30_weights-2026-04-09_13:52:48.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch30_weights-2026-04-09_13:52:48.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch40_weights-2026-04-09_13:52:56.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch40_weights-2026-04-09_13:52:56.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch50_weights-2026-04-09_13:53:04.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch50_weights-2026-04-09_13:53:04.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch60_weights-2026-04-09_13:53:12.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch60_weights-2026-04-09_13:53:12.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch70_weights-2026-04-09_13:53:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch70_weights-2026-04-09_13:53:20.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch80_weights-2026-04-09_13:53:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold1_epoch80_weights-2026-04-09_13:53:28.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch10_weights-2026-04-09_13:53:39.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch10_weights-2026-04-09_13:53:39.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch20_weights-2026-04-09_13:53:49.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch20_weights-2026-04-09_13:53:49.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch30_weights-2026-04-09_13:53:57.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch30_weights-2026-04-09_13:53:57.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch40_weights-2026-04-09_13:54:05.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch40_weights-2026-04-09_13:54:05.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch50_weights-2026-04-09_13:54:14.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch50_weights-2026-04-09_13:54:14.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch60_weights-2026-04-09_13:54:22.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch60_weights-2026-04-09_13:54:22.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch70_weights-2026-04-09_13:54:30.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch70_weights-2026-04-09_13:54:30.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch80_weights-2026-04-09_13:54:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold2_epoch80_weights-2026-04-09_13:54:38.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch10_weights-2026-04-09_13:54:47.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch10_weights-2026-04-09_13:54:47.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch20_weights-2026-04-09_13:54:56.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch20_weights-2026-04-09_13:54:56.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch30_weights-2026-04-09_13:55:04.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch30_weights-2026-04-09_13:55:04.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch40_weights-2026-04-09_13:55:12.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch40_weights-2026-04-09_13:55:12.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch50_weights-2026-04-09_13:55:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch50_weights-2026-04-09_13:55:20.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch60_weights-2026-04-09_13:55:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch60_weights-2026-04-09_13:55:28.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch70_weights-2026-04-09_13:55:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch70_weights-2026-04-09_13:55:36.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch80_weights-2026-04-09_13:55:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold3_epoch80_weights-2026-04-09_13:55:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch10_weights-2026-04-09_13:55:55.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch10_weights-2026-04-09_13:55:55.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch20_weights-2026-04-09_13:56:03.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch20_weights-2026-04-09_13:56:03.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch30_weights-2026-04-09_13:56:11.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch30_weights-2026-04-09_13:56:11.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch40_weights-2026-04-09_13:56:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch40_weights-2026-04-09_13:56:19.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch50_weights-2026-04-09_13:56:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch50_weights-2026-04-09_13:56:27.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch60_weights-2026-04-09_13:56:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch60_weights-2026-04-09_13:56:35.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch70_weights-2026-04-09_13:56:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch70_weights-2026-04-09_13:56:42.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch80_weights-2026-04-09_13:56:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold4_epoch80_weights-2026-04-09_13:56:51.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch10_weights-2026-04-09_13:57:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch10_weights-2026-04-09_13:57:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch20_weights-2026-04-09_13:57:11.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch20_weights-2026-04-09_13:57:11.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch30_weights-2026-04-09_13:57:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch30_weights-2026-04-09_13:57:19.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch40_weights-2026-04-09_13:57:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch40_weights-2026-04-09_13:57:28.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch50_weights-2026-04-09_13:57:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch50_weights-2026-04-09_13:57:38.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch60_weights-2026-04-09_13:57:47.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch60_weights-2026-04-09_13:57:47.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch70_weights-2026-04-09_13:57:56.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch70_weights-2026-04-09_13:57:56.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch80_weights-2026-04-09_13:58:05.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:52:20/fold5_epoch80_weights-2026-04-09_13:58:05.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_13:52:20",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T13:58:07.330523",
    "training_started_at": "2026-04-09T13:52:20.234456",
    "training_finished_at": "2026-04-09T13:58:06.193415",
    "training_duration_seconds": 345.959,
    "training_duration_hours": 0.0961,
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
      "mae": 15.30494,
      "rmse": 21.97836,
      "r2": 0.13355,
      "pearson": 0.50402,
      "mean_error": 6.20855
    },
    {
      "fold": 2,
      "best_epoch": 50,
      "mae": 13.16412,
      "rmse": 17.49525,
      "r2": 0.5033,
      "pearson": 0.77236,
      "mean_error": -6.22517
    },
    {
      "fold": 3,
      "best_epoch": 5,
      "mae": 21.55385,
      "rmse": 25.79782,
      "r2": -0.00412,
      "pearson": 0.3558,
      "mean_error": 3.41918
    },
    {
      "fold": 4,
      "best_epoch": 10,
      "mae": 18.34978,
      "rmse": 21.05888,
      "r2": 0.07059,
      "pearson": 0.54624,
      "mean_error": 0.8779
    },
    {
      "fold": 5,
      "best_epoch": 45,
      "mae": 21.10728,
      "rmse": 28.86521,
      "r2": -0.26619,
      "pearson": 0.54086,
      "mean_error": 16.67899
    }
  ],
  "summary": {
    "mean_mae": 17.89599,
    "std_mae": 3.25584,
    "mean_rmse": 23.0391,
    "std_rmse": 3.93281,
    "mean_r2": 0.08743,
    "std_r2": 0.24851,
    "mean_pearson": 0.54386,
    "std_pearson": 0.13355
  }
}
```
