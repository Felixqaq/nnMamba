# Run: hybrid_mamba_attention_2026-04-09_14:28:08

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T14:33:51.234006`
- 訓練時間: `2026-04-09T14:28:08.830185` -> `2026-04-09T14:33:50.133975`
- 訓練耗時: 0.0948 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 18.45693 |
| std_mae | 1.8067 |
| mean_rmse | 26.17976 |
| std_rmse | 2.22181 |
| mean_r2 | -0.16713 |
| std_r2 | 0.18032 |
| mean_pearson | 0.24266 |
| std_pearson | 0.23205 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 30 | 16.15305 | 22.71596 | 0.07442 | 0.31115 | 3.52333 |
| 2 | 15 | 19.10121 | 25.14733 | -0.02622 | 0.24736 | 7.18642 |
| 3 | 20 | 20.372 | 29.2462 | -0.29051 | -0.16302 | 13.32641 |
| 4 | 50 | 16.47209 | 26.12226 | -0.43008 | 0.25948 | 1.09119 |
| 5 | 15 | 20.18631 | 27.66703 | -0.16325 | 0.55834 | 16.17455 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_bland_altman.png) — 124.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_error_hist.png) — 85.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_loss.png) — 158.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_residuals.png) — 94.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_scatter.png) — 184.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_summary.png) — 425.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_bland_altman.png) — 121.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_error_hist.png) — 85.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_loss.png) — 138.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_residuals.png) — 95.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_scatter.png) — 199.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_summary.png) — 419.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_bland_altman.png) — 128.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_error_hist.png) — 89.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_loss.png) — 146.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_residuals.png) — 95.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_scatter.png) — 197.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_summary.png) — 431.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_bland_altman.png) — 128.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_error_hist.png) — 88.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_loss.png) — 151.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_residuals.png) — 93.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_scatter.png) — 193.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_summary.png) — 419.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_bland_altman.png) — 120.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_error_hist.png) — 99.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_loss.png) — 150.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_residuals.png) — 94.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_scatter.png) — 206.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_summary.png) — 429.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/metric_boxplot.png) — 98.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_bland_altman.png) — 163.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_error_hist.png) — 103.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_residuals.png) — 128.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/total_scatter.png) — 232.6 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_best_weight.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch10_weights-2026-04-09_14:28:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch10_weights-2026-04-09_14:28:19.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch20_weights-2026-04-09_14:28:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch20_weights-2026-04-09_14:28:28.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch30_weights-2026-04-09_14:28:37.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch30_weights-2026-04-09_14:28:37.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch40_weights-2026-04-09_14:28:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch40_weights-2026-04-09_14:28:45.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch50_weights-2026-04-09_14:28:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch50_weights-2026-04-09_14:28:53.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch60_weights-2026-04-09_14:29:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch60_weights-2026-04-09_14:29:01.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch70_weights-2026-04-09_14:29:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch70_weights-2026-04-09_14:29:09.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch80_weights-2026-04-09_14:29:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold1_epoch80_weights-2026-04-09_14:29:17.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_best_weight.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch10_weights-2026-04-09_14:29:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch10_weights-2026-04-09_14:29:27.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch20_weights-2026-04-09_14:29:37.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch20_weights-2026-04-09_14:29:37.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch30_weights-2026-04-09_14:29:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch30_weights-2026-04-09_14:29:45.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch40_weights-2026-04-09_14:29:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch40_weights-2026-04-09_14:29:53.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch50_weights-2026-04-09_14:30:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch50_weights-2026-04-09_14:30:01.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch60_weights-2026-04-09_14:30:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch60_weights-2026-04-09_14:30:09.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch70_weights-2026-04-09_14:30:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch70_weights-2026-04-09_14:30:17.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch80_weights-2026-04-09_14:30:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold2_epoch80_weights-2026-04-09_14:30:25.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_best_weight.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch10_weights-2026-04-09_14:30:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch10_weights-2026-04-09_14:30:35.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch20_weights-2026-04-09_14:30:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch20_weights-2026-04-09_14:30:45.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch30_weights-2026-04-09_14:30:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch30_weights-2026-04-09_14:30:53.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch40_weights-2026-04-09_14:31:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch40_weights-2026-04-09_14:31:01.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch50_weights-2026-04-09_14:31:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch50_weights-2026-04-09_14:31:09.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch60_weights-2026-04-09_14:31:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch60_weights-2026-04-09_14:31:17.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch70_weights-2026-04-09_14:31:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch70_weights-2026-04-09_14:31:25.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch80_weights-2026-04-09_14:31:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold3_epoch80_weights-2026-04-09_14:31:33.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_best_weight.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch10_weights-2026-04-09_14:31:43.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch10_weights-2026-04-09_14:31:43.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch20_weights-2026-04-09_14:31:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch20_weights-2026-04-09_14:31:52.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch30_weights-2026-04-09_14:32:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch30_weights-2026-04-09_14:32:00.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch40_weights-2026-04-09_14:32:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch40_weights-2026-04-09_14:32:08.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch50_weights-2026-04-09_14:32:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch50_weights-2026-04-09_14:32:17.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch60_weights-2026-04-09_14:32:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch60_weights-2026-04-09_14:32:25.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch70_weights-2026-04-09_14:32:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch70_weights-2026-04-09_14:32:33.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch80_weights-2026-04-09_14:32:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold4_epoch80_weights-2026-04-09_14:32:41.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_best_weight.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch10_weights-2026-04-09_14:32:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch10_weights-2026-04-09_14:32:51.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch20_weights-2026-04-09_14:33:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch20_weights-2026-04-09_14:33:00.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch30_weights-2026-04-09_14:33:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch30_weights-2026-04-09_14:33:08.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch40_weights-2026-04-09_14:33:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch40_weights-2026-04-09_14:33:17.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch50_weights-2026-04-09_14:33:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch50_weights-2026-04-09_14:33:25.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch60_weights-2026-04-09_14:33:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch60_weights-2026-04-09_14:33:33.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch70_weights-2026-04-09_14:33:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch70_weights-2026-04-09_14:33:41.pth) — 4.5 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch80_weights-2026-04-09_14:33:49.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:28:08/fold5_epoch80_weights-2026-04-09_14:33:49.pth) — 4.5 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:28:08",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T14:33:51.234006",
    "training_started_at": "2026-04-09T14:28:08.830185",
    "training_finished_at": "2026-04-09T14:33:50.133975",
    "training_duration_seconds": 341.304,
    "training_duration_hours": 0.0948,
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
      "best_epoch": 30,
      "mae": 16.15305,
      "rmse": 22.71596,
      "r2": 0.07442,
      "pearson": 0.31115,
      "mean_error": 3.52333
    },
    {
      "fold": 2,
      "best_epoch": 15,
      "mae": 19.10121,
      "rmse": 25.14733,
      "r2": -0.02622,
      "pearson": 0.24736,
      "mean_error": 7.18642
    },
    {
      "fold": 3,
      "best_epoch": 20,
      "mae": 20.372,
      "rmse": 29.2462,
      "r2": -0.29051,
      "pearson": -0.16302,
      "mean_error": 13.32641
    },
    {
      "fold": 4,
      "best_epoch": 50,
      "mae": 16.47209,
      "rmse": 26.12226,
      "r2": -0.43008,
      "pearson": 0.25948,
      "mean_error": 1.09119
    },
    {
      "fold": 5,
      "best_epoch": 15,
      "mae": 20.18631,
      "rmse": 27.66703,
      "r2": -0.16325,
      "pearson": 0.55834,
      "mean_error": 16.17455
    }
  ],
  "summary": {
    "mean_mae": 18.45693,
    "std_mae": 1.8067,
    "mean_rmse": 26.17976,
    "std_rmse": 2.22181,
    "mean_r2": -0.16713,
    "std_r2": 0.18032,
    "mean_pearson": 0.24266,
    "std_pearson": 0.23205
  }
}
```
