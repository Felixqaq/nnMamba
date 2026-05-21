# Run: hybrid_mamba_attention_2026-04-09_13:23:43

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T13:29:19.906922`
- 訓練時間: `2026-04-09T13:23:43.915503` -> `2026-04-09T13:29:18.928013`
- 訓練耗時: 0.0931 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 16.27248 |
| std_mae | 2.98635 |
| mean_rmse | 21.9338 |
| std_rmse | 3.54859 |
| mean_r2 | 0.17102 |
| std_r2 | 0.21656 |
| mean_pearson | 0.35182 |
| std_pearson | 0.44336 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 17.93149 | 23.86351 | -0.02146 | -0.50722 | 3.2994 |
| 2 | 55 | 12.42704 | 16.23583 | 0.57224 | 0.75822 | -0.36492 |
| 3 | 15 | 18.54242 | 24.52549 | 0.09248 | 0.4701 | 8.80094 |
| 4 | 75 | 12.92198 | 19.41061 | 0.21038 | 0.58909 | 0.09373 |
| 5 | 40 | 19.53946 | 25.63357 | 0.00146 | 0.44891 | 11.33104 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_bland_altman.png) — 124.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_error_hist.png) — 85.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_loss.png) — 156.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_residuals.png) — 93.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_scatter.png) — 179.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_summary.png) — 410.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_bland_altman.png) — 127.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_error_hist.png) — 101.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_loss.png) — 177.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_residuals.png) — 93.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_scatter.png) — 228.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_summary.png) — 475.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_bland_altman.png) — 124.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_error_hist.png) — 96.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_loss.png) — 138.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_residuals.png) — 93.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_scatter.png) — 205.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_summary.png) — 464.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_bland_altman.png) — 125.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_error_hist.png) — 85.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_loss.png) — 139.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_residuals.png) — 96.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_scatter.png) — 223.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_summary.png) — 447.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_bland_altman.png) — 123.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_error_hist.png) — 99.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_loss.png) — 158.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_residuals.png) — 91.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_scatter.png) — 203.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_summary.png) — 475.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/metric_boxplot.png) — 97.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_bland_altman.png) — 163.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_error_hist.png) — 95.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_residuals.png) — 126.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/total_scatter.png) — 239.8 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch10_weights-2026-04-09_13:23:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch10_weights-2026-04-09_13:23:54.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch20_weights-2026-04-09_13:24:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch20_weights-2026-04-09_13:24:02.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch30_weights-2026-04-09_13:24:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch30_weights-2026-04-09_13:24:10.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch40_weights-2026-04-09_13:24:18.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch40_weights-2026-04-09_13:24:18.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch50_weights-2026-04-09_13:24:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch50_weights-2026-04-09_13:24:27.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch60_weights-2026-04-09_13:24:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch60_weights-2026-04-09_13:24:35.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch70_weights-2026-04-09_13:24:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch70_weights-2026-04-09_13:24:42.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch80_weights-2026-04-09_13:24:50.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold1_epoch80_weights-2026-04-09_13:24:50.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch10_weights-2026-04-09_13:25:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch10_weights-2026-04-09_13:25:00.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch20_weights-2026-04-09_13:25:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch20_weights-2026-04-09_13:25:09.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch30_weights-2026-04-09_13:25:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch30_weights-2026-04-09_13:25:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch40_weights-2026-04-09_13:25:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch40_weights-2026-04-09_13:25:27.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch50_weights-2026-04-09_13:25:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch50_weights-2026-04-09_13:25:35.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch60_weights-2026-04-09_13:25:43.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch60_weights-2026-04-09_13:25:43.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch70_weights-2026-04-09_13:25:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch70_weights-2026-04-09_13:25:51.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch80_weights-2026-04-09_13:25:59.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold2_epoch80_weights-2026-04-09_13:25:59.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch10_weights-2026-04-09_13:26:07.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch10_weights-2026-04-09_13:26:07.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch20_weights-2026-04-09_13:26:16.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch20_weights-2026-04-09_13:26:16.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch30_weights-2026-04-09_13:26:23.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch30_weights-2026-04-09_13:26:23.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch40_weights-2026-04-09_13:26:31.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch40_weights-2026-04-09_13:26:31.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch50_weights-2026-04-09_13:26:39.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch50_weights-2026-04-09_13:26:39.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch60_weights-2026-04-09_13:26:47.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch60_weights-2026-04-09_13:26:47.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch70_weights-2026-04-09_13:26:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch70_weights-2026-04-09_13:26:54.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch80_weights-2026-04-09_13:27:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold3_epoch80_weights-2026-04-09_13:27:02.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch10_weights-2026-04-09_13:27:12.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch10_weights-2026-04-09_13:27:12.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch20_weights-2026-04-09_13:27:21.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch20_weights-2026-04-09_13:27:21.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch30_weights-2026-04-09_13:27:29.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch30_weights-2026-04-09_13:27:29.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch40_weights-2026-04-09_13:27:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch40_weights-2026-04-09_13:27:38.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch50_weights-2026-04-09_13:27:46.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch50_weights-2026-04-09_13:27:46.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch60_weights-2026-04-09_13:27:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch60_weights-2026-04-09_13:27:53.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch70_weights-2026-04-09_13:28:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch70_weights-2026-04-09_13:28:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch80_weights-2026-04-09_13:28:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold4_epoch80_weights-2026-04-09_13:28:10.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch10_weights-2026-04-09_13:28:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch10_weights-2026-04-09_13:28:20.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch20_weights-2026-04-09_13:28:29.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch20_weights-2026-04-09_13:28:29.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch30_weights-2026-04-09_13:28:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch30_weights-2026-04-09_13:28:38.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch40_weights-2026-04-09_13:28:46.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch40_weights-2026-04-09_13:28:46.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch50_weights-2026-04-09_13:28:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch50_weights-2026-04-09_13:28:54.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch60_weights-2026-04-09_13:29:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch60_weights-2026-04-09_13:29:02.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch70_weights-2026-04-09_13:29:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch70_weights-2026-04-09_13:29:10.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch80_weights-2026-04-09_13:29:18.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/fold5_epoch80_weights-2026-04-09_13:29:18.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_13:23:43",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T13:29:19.906922",
    "training_started_at": "2026-04-09T13:23:43.915503",
    "training_finished_at": "2026-04-09T13:29:18.928013",
    "training_duration_seconds": 335.013,
    "training_duration_hours": 0.0931,
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
      "best_epoch": 10,
      "mae": 17.93149,
      "rmse": 23.86351,
      "r2": -0.02146,
      "pearson": -0.50722,
      "mean_error": 3.2994
    },
    {
      "fold": 2,
      "best_epoch": 55,
      "mae": 12.42704,
      "rmse": 16.23583,
      "r2": 0.57224,
      "pearson": 0.75822,
      "mean_error": -0.36492
    },
    {
      "fold": 3,
      "best_epoch": 15,
      "mae": 18.54242,
      "rmse": 24.52549,
      "r2": 0.09248,
      "pearson": 0.4701,
      "mean_error": 8.80094
    },
    {
      "fold": 4,
      "best_epoch": 75,
      "mae": 12.92198,
      "rmse": 19.41061,
      "r2": 0.21038,
      "pearson": 0.58909,
      "mean_error": 0.09373
    },
    {
      "fold": 5,
      "best_epoch": 40,
      "mae": 19.53946,
      "rmse": 25.63357,
      "r2": 0.00146,
      "pearson": 0.44891,
      "mean_error": 11.33104
    }
  ],
  "summary": {
    "mean_mae": 16.27248,
    "std_mae": 2.98635,
    "mean_rmse": 21.9338,
    "std_rmse": 3.54859,
    "mean_r2": 0.17102,
    "std_r2": 0.21656,
    "mean_pearson": 0.35182,
    "std_pearson": 0.44336
  }
}
```
