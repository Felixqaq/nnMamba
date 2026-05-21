# Run: hybrid_mamba_attention_2026-04-09_14:19:08

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T14:25:05.884821`
- 訓練時間: `2026-04-09T14:19:08.923538` -> `2026-04-09T14:25:04.812102`
- 訓練耗時: 0.0989 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 16.13509 |
| std_mae | 1.59995 |
| mean_rmse | 21.64588 |
| std_rmse | 3.5763 |
| mean_r2 | 0.14738 |
| std_r2 | 0.39114 |
| mean_pearson | 0.46973 |
| std_pearson | 0.24879 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 | 18.08765 | 24.39116 | -0.06713 | 0.11859 | 1.59674 |
| 2 | 30 | 13.59639 | 16.8768 | 0.53779 | 0.73718 | 1.27599 |
| 3 | 70 | 15.58716 | 19.49517 | 0.42658 | 0.65599 | 1.27523 |
| 4 | 40 | 17.59613 | 26.92139 | -0.51892 | 0.22518 | -0.85816 |
| 5 | 55 | 15.80811 | 20.54488 | 0.35856 | 0.6117 | -3.19851 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_bland_altman.png) — 119.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_error_hist.png) — 100.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_loss.png) — 152.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_residuals.png) — 96.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_scatter.png) — 187.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_summary.png) — 431.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_bland_altman.png) — 129.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_error_hist.png) — 92.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_loss.png) — 147.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_residuals.png) — 93.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_scatter.png) — 226.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_summary.png) — 444.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_bland_altman.png) — 133.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_error_hist.png) — 99.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_loss.png) — 160.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_residuals.png) — 93.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_scatter.png) — 224.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_summary.png) — 431.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_bland_altman.png) — 126.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_error_hist.png) — 88.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_loss.png) — 161.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_residuals.png) — 90.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_scatter.png) — 198.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_summary.png) — 476.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_bland_altman.png) — 129.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_error_hist.png) — 100.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_loss.png) — 169.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_residuals.png) — 94.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_scatter.png) — 221.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_summary.png) — 450.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/metric_boxplot.png) — 95.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_bland_altman.png) — 164.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_error_hist.png) — 97.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_residuals.png) — 127.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/total_scatter.png) — 252.5 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch10_weights-2026-04-09_14:19:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch10_weights-2026-04-09_14:19:19.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch20_weights-2026-04-09_14:19:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch20_weights-2026-04-09_14:19:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch30_weights-2026-04-09_14:19:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch30_weights-2026-04-09_14:19:36.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch40_weights-2026-04-09_14:19:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch40_weights-2026-04-09_14:19:45.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch50_weights-2026-04-09_14:19:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch50_weights-2026-04-09_14:19:53.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch60_weights-2026-04-09_14:20:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch60_weights-2026-04-09_14:20:01.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch70_weights-2026-04-09_14:20:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch70_weights-2026-04-09_14:20:09.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch80_weights-2026-04-09_14:20:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold1_epoch80_weights-2026-04-09_14:20:17.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch10_weights-2026-04-09_14:20:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch10_weights-2026-04-09_14:20:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch20_weights-2026-04-09_14:20:37.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch20_weights-2026-04-09_14:20:37.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch30_weights-2026-04-09_14:20:46.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch30_weights-2026-04-09_14:20:46.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch40_weights-2026-04-09_14:20:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch40_weights-2026-04-09_14:20:54.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch50_weights-2026-04-09_14:21:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch50_weights-2026-04-09_14:21:02.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch60_weights-2026-04-09_14:21:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch60_weights-2026-04-09_14:21:10.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch70_weights-2026-04-09_14:21:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch70_weights-2026-04-09_14:21:19.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch80_weights-2026-04-09_14:21:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold2_epoch80_weights-2026-04-09_14:21:27.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch10_weights-2026-04-09_14:21:37.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch10_weights-2026-04-09_14:21:37.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch20_weights-2026-04-09_14:21:48.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch20_weights-2026-04-09_14:21:48.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch30_weights-2026-04-09_14:21:58.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch30_weights-2026-04-09_14:21:58.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch40_weights-2026-04-09_14:22:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch40_weights-2026-04-09_14:22:08.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch50_weights-2026-04-09_14:22:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch50_weights-2026-04-09_14:22:17.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch60_weights-2026-04-09_14:22:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch60_weights-2026-04-09_14:22:25.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch70_weights-2026-04-09_14:22:34.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch70_weights-2026-04-09_14:22:34.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch80_weights-2026-04-09_14:22:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold3_epoch80_weights-2026-04-09_14:22:42.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch10_weights-2026-04-09_14:22:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch10_weights-2026-04-09_14:22:53.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch20_weights-2026-04-09_14:23:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch20_weights-2026-04-09_14:23:01.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch30_weights-2026-04-09_14:23:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch30_weights-2026-04-09_14:23:10.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch40_weights-2026-04-09_14:23:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch40_weights-2026-04-09_14:23:19.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch50_weights-2026-04-09_14:23:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch50_weights-2026-04-09_14:23:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch60_weights-2026-04-09_14:23:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch60_weights-2026-04-09_14:23:36.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch70_weights-2026-04-09_14:23:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch70_weights-2026-04-09_14:23:44.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch80_weights-2026-04-09_14:23:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold4_epoch80_weights-2026-04-09_14:23:52.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch10_weights-2026-04-09_14:24:03.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch10_weights-2026-04-09_14:24:03.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch20_weights-2026-04-09_14:24:12.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch20_weights-2026-04-09_14:24:12.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch30_weights-2026-04-09_14:24:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch30_weights-2026-04-09_14:24:20.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch40_weights-2026-04-09_14:24:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch40_weights-2026-04-09_14:24:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch50_weights-2026-04-09_14:24:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch50_weights-2026-04-09_14:24:38.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch60_weights-2026-04-09_14:24:47.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch60_weights-2026-04-09_14:24:47.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch70_weights-2026-04-09_14:24:55.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch70_weights-2026-04-09_14:24:55.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch80_weights-2026-04-09_14:25:03.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:19:08/fold5_epoch80_weights-2026-04-09_14:25:03.pth) — 4.9 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:19:08",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T14:25:05.884821",
    "training_started_at": "2026-04-09T14:19:08.923538",
    "training_finished_at": "2026-04-09T14:25:04.812102",
    "training_duration_seconds": 355.889,
    "training_duration_hours": 0.0989,
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
      "mae": 18.08765,
      "rmse": 24.39116,
      "r2": -0.06713,
      "pearson": 0.11859,
      "mean_error": 1.59674
    },
    {
      "fold": 2,
      "best_epoch": 30,
      "mae": 13.59639,
      "rmse": 16.8768,
      "r2": 0.53779,
      "pearson": 0.73718,
      "mean_error": 1.27599
    },
    {
      "fold": 3,
      "best_epoch": 70,
      "mae": 15.58716,
      "rmse": 19.49517,
      "r2": 0.42658,
      "pearson": 0.65599,
      "mean_error": 1.27523
    },
    {
      "fold": 4,
      "best_epoch": 40,
      "mae": 17.59613,
      "rmse": 26.92139,
      "r2": -0.51892,
      "pearson": 0.22518,
      "mean_error": -0.85816
    },
    {
      "fold": 5,
      "best_epoch": 55,
      "mae": 15.80811,
      "rmse": 20.54488,
      "r2": 0.35856,
      "pearson": 0.6117,
      "mean_error": -3.19851
    }
  ],
  "summary": {
    "mean_mae": 16.13509,
    "std_mae": 1.59995,
    "mean_rmse": 21.64588,
    "std_rmse": 3.5763,
    "mean_r2": 0.14738,
    "std_r2": 0.39114,
    "mean_pearson": 0.46973,
    "std_pearson": 0.24879
  }
}
```
