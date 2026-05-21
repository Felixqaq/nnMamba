# Run: hybrid_mamba_attention_2026-04-09_14:57:03

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T15:02:55.074697`
- 訓練時間: `2026-04-09T14:57:03.936277` -> `2026-04-09T15:02:54.004316`
- 訓練耗時: 0.0972 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 16.57387 |
| std_mae | 0.91024 |
| mean_rmse | 23.04624 |
| std_rmse | 2.32081 |
| mean_r2 | 0.10395 |
| std_r2 | 0.09716 |
| mean_pearson | 0.47104 |
| std_pearson | 0.05309 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 | 16.88201 | 22.0682 | 0.12645 | 0.38878 | 1.03785 |
| 2 | 30 | 16.89795 | 24.66698 | 0.01261 | 0.53203 | 12.22662 |
| 3 | 25 | 17.92764 | 26.11425 | -0.02891 | 0.43034 | 6.73563 |
| 4 | 35 | 15.31828 | 19.31339 | 0.21827 | 0.49824 | -0.04869 |
| 5 | 45 | 15.84347 | 23.06837 | 0.19131 | 0.50579 | 6.36377 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_bland_altman.png) — 121.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_error_hist.png) — 84.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_loss.png) — 148.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_residuals.png) — 91.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_scatter.png) — 193.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_summary.png) — 404.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_bland_altman.png) — 118.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_error_hist.png) — 84.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_loss.png) — 190.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_residuals.png) — 92.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_scatter.png) — 211.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_summary.png) — 452.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_bland_altman.png) — 127.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_error_hist.png) — 100.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_loss.png) — 161.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_residuals.png) — 91.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_scatter.png) — 216.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_summary.png) — 426.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_bland_altman.png) — 132.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_error_hist.png) — 99.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_loss.png) — 150.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_residuals.png) — 92.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_scatter.png) — 206.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_summary.png) — 428.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_bland_altman.png) — 126.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_error_hist.png) — 86.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_loss.png) — 150.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_residuals.png) — 96.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_scatter.png) — 217.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_summary.png) — 434.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/metric_boxplot.png) — 93.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_bland_altman.png) — 162.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_error_hist.png) — 96.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_residuals.png) — 126.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/total_scatter.png) — 244.3 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch10_weights-2026-04-09_14:57:14.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch10_weights-2026-04-09_14:57:14.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch20_weights-2026-04-09_14:57:23.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch20_weights-2026-04-09_14:57:23.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch30_weights-2026-04-09_14:57:32.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch30_weights-2026-04-09_14:57:32.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch40_weights-2026-04-09_14:57:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch40_weights-2026-04-09_14:57:41.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch50_weights-2026-04-09_14:57:49.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch50_weights-2026-04-09_14:57:49.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch60_weights-2026-04-09_14:57:58.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch60_weights-2026-04-09_14:57:58.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch70_weights-2026-04-09_14:58:06.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch70_weights-2026-04-09_14:58:06.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch80_weights-2026-04-09_14:58:15.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold1_epoch80_weights-2026-04-09_14:58:15.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch10_weights-2026-04-09_14:58:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch10_weights-2026-04-09_14:58:25.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch20_weights-2026-04-09_14:58:34.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch20_weights-2026-04-09_14:58:34.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch30_weights-2026-04-09_14:58:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch30_weights-2026-04-09_14:58:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch40_weights-2026-04-09_14:58:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch40_weights-2026-04-09_14:58:53.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch50_weights-2026-04-09_14:59:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch50_weights-2026-04-09_14:59:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch60_weights-2026-04-09_14:59:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch60_weights-2026-04-09_14:59:09.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch70_weights-2026-04-09_14:59:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch70_weights-2026-04-09_14:59:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch80_weights-2026-04-09_14:59:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold2_epoch80_weights-2026-04-09_14:59:25.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch10_weights-2026-04-09_14:59:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch10_weights-2026-04-09_14:59:35.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch20_weights-2026-04-09_14:59:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch20_weights-2026-04-09_14:59:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch30_weights-2026-04-09_14:59:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch30_weights-2026-04-09_14:59:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch40_weights-2026-04-09_15:00:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch40_weights-2026-04-09_15:00:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch50_weights-2026-04-09_15:00:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch50_weights-2026-04-09_15:00:09.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch60_weights-2026-04-09_15:00:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch60_weights-2026-04-09_15:00:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch70_weights-2026-04-09_15:00:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch70_weights-2026-04-09_15:00:25.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch80_weights-2026-04-09_15:00:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold3_epoch80_weights-2026-04-09_15:00:33.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch10_weights-2026-04-09_15:00:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch10_weights-2026-04-09_15:00:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch20_weights-2026-04-09_15:00:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch20_weights-2026-04-09_15:00:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch30_weights-2026-04-09_15:01:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch30_weights-2026-04-09_15:01:00.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch40_weights-2026-04-09_15:01:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch40_weights-2026-04-09_15:01:09.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch50_weights-2026-04-09_15:01:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch50_weights-2026-04-09_15:01:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch60_weights-2026-04-09_15:01:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch60_weights-2026-04-09_15:01:25.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch70_weights-2026-04-09_15:01:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch70_weights-2026-04-09_15:01:33.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch80_weights-2026-04-09_15:01:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold4_epoch80_weights-2026-04-09_15:01:41.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch10_weights-2026-04-09_15:01:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch10_weights-2026-04-09_15:01:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch20_weights-2026-04-09_15:02:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch20_weights-2026-04-09_15:02:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch30_weights-2026-04-09_15:02:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch30_weights-2026-04-09_15:02:10.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch40_weights-2026-04-09_15:02:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch40_weights-2026-04-09_15:02:20.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch50_weights-2026-04-09_15:02:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch50_weights-2026-04-09_15:02:28.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch60_weights-2026-04-09_15:02:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch60_weights-2026-04-09_15:02:36.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch70_weights-2026-04-09_15:02:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch70_weights-2026-04-09_15:02:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch80_weights-2026-04-09_15:02:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:57:03/fold5_epoch80_weights-2026-04-09_15:02:53.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:57:03",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T15:02:55.074697",
    "training_started_at": "2026-04-09T14:57:03.936277",
    "training_finished_at": "2026-04-09T15:02:54.004316",
    "training_duration_seconds": 350.068,
    "training_duration_hours": 0.0972,
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
      "best_epoch": 35,
      "mae": 16.88201,
      "rmse": 22.0682,
      "r2": 0.12645,
      "pearson": 0.38878,
      "mean_error": 1.03785
    },
    {
      "fold": 2,
      "best_epoch": 30,
      "mae": 16.89795,
      "rmse": 24.66698,
      "r2": 0.01261,
      "pearson": 0.53203,
      "mean_error": 12.22662
    },
    {
      "fold": 3,
      "best_epoch": 25,
      "mae": 17.92764,
      "rmse": 26.11425,
      "r2": -0.02891,
      "pearson": 0.43034,
      "mean_error": 6.73563
    },
    {
      "fold": 4,
      "best_epoch": 35,
      "mae": 15.31828,
      "rmse": 19.31339,
      "r2": 0.21827,
      "pearson": 0.49824,
      "mean_error": -0.04869
    },
    {
      "fold": 5,
      "best_epoch": 45,
      "mae": 15.84347,
      "rmse": 23.06837,
      "r2": 0.19131,
      "pearson": 0.50579,
      "mean_error": 6.36377
    }
  ],
  "summary": {
    "mean_mae": 16.57387,
    "std_mae": 0.91024,
    "mean_rmse": 23.04624,
    "std_rmse": 2.32081,
    "mean_r2": 0.10395,
    "std_r2": 0.09716,
    "mean_pearson": 0.47104,
    "std_pearson": 0.05309
  }
}
```
