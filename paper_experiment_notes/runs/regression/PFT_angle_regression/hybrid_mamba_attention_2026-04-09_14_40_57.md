# Run: hybrid_mamba_attention_2026-04-09_14:40:57

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T14:46:26.097546`
- 訓練時間: `2026-04-09T14:40:57.069130` -> `2026-04-09T14:46:25.024525`
- 訓練耗時: 0.0911 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 17.85291 |
| std_mae | 1.62602 |
| mean_rmse | 24.56909 |
| std_rmse | 2.58396 |
| mean_r2 | -0.03367 |
| std_r2 | 0.21023 |
| mean_pearson | 0.38972 |
| std_pearson | 0.1932 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 16.31718 | 21.55095 | 0.16692 | 0.55888 | 3.41542 |
| 2 | 45 | 16.21369 | 21.60391 | 0.24261 | 0.54983 | 5.97568 |
| 3 | 30 | 20.53856 | 26.83305 | -0.08633 | 0.03158 | 7.5668 |
| 4 | 25 | 17.44881 | 25.11719 | -0.32215 | 0.36112 | -3.9935 |
| 5 | 45 | 18.74629 | 27.74036 | -0.16942 | 0.44721 | 15.05325 |

## Training Config Embedded In Result
```json
{
  "epochs": 80,
  "batch_size": 12,
  "learning_rate": 0.0001,
  "weight_decay": 0.0005,
  "k_folds": 5,
  "seed": 42,
  "loss": "smooth_l1"
}
```

## Artifact Index
### figures
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_bland_altman.png) — 121.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_error_hist.png) — 83.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_loss.png) — 141.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_residuals.png) — 91.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_scatter.png) — 192.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_summary.png) — 408.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_bland_altman.png) — 130.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_error_hist.png) — 99.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_loss.png) — 147.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_residuals.png) — 93.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_scatter.png) — 217.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_summary.png) — 445.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_bland_altman.png) — 128.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_error_hist.png) — 85.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_loss.png) — 161.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_residuals.png) — 95.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_scatter.png) — 192.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_summary.png) — 446.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_bland_altman.png) — 127.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_error_hist.png) — 86.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_loss.png) — 157.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_residuals.png) — 91.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_scatter.png) — 204.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_summary.png) — 436.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_bland_altman.png) — 125.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_error_hist.png) — 86.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_loss.png) — 157.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_residuals.png) — 94.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_scatter.png) — 211.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_summary.png) — 463.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/metric_boxplot.png) — 98.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_bland_altman.png) — 165.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_error_hist.png) — 98.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_residuals.png) — 128.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/total_scatter.png) — 239.8 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch10_weights-2026-04-09_14:41:07.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch10_weights-2026-04-09_14:41:07.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch20_weights-2026-04-09_14:41:15.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch20_weights-2026-04-09_14:41:15.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch30_weights-2026-04-09_14:41:23.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch30_weights-2026-04-09_14:41:23.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch40_weights-2026-04-09_14:41:30.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch40_weights-2026-04-09_14:41:30.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch50_weights-2026-04-09_14:41:38.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch50_weights-2026-04-09_14:41:38.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch60_weights-2026-04-09_14:41:46.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch60_weights-2026-04-09_14:41:46.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch70_weights-2026-04-09_14:41:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch70_weights-2026-04-09_14:41:53.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch80_weights-2026-04-09_14:42:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold1_epoch80_weights-2026-04-09_14:42:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch10_weights-2026-04-09_14:42:11.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch10_weights-2026-04-09_14:42:11.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch20_weights-2026-04-09_14:42:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch20_weights-2026-04-09_14:42:19.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch30_weights-2026-04-09_14:42:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch30_weights-2026-04-09_14:42:27.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch40_weights-2026-04-09_14:42:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch40_weights-2026-04-09_14:42:36.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch50_weights-2026-04-09_14:42:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch50_weights-2026-04-09_14:42:45.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch60_weights-2026-04-09_14:42:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch60_weights-2026-04-09_14:42:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch70_weights-2026-04-09_14:43:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch70_weights-2026-04-09_14:43:00.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch80_weights-2026-04-09_14:43:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold2_epoch80_weights-2026-04-09_14:43:08.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch10_weights-2026-04-09_14:43:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch10_weights-2026-04-09_14:43:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch20_weights-2026-04-09_14:43:24.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch20_weights-2026-04-09_14:43:24.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch30_weights-2026-04-09_14:43:34.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch30_weights-2026-04-09_14:43:34.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch40_weights-2026-04-09_14:43:42.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch40_weights-2026-04-09_14:43:42.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch50_weights-2026-04-09_14:43:50.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch50_weights-2026-04-09_14:43:50.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch60_weights-2026-04-09_14:43:57.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch60_weights-2026-04-09_14:43:57.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch70_weights-2026-04-09_14:44:05.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch70_weights-2026-04-09_14:44:05.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch80_weights-2026-04-09_14:44:13.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold3_epoch80_weights-2026-04-09_14:44:13.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch10_weights-2026-04-09_14:44:23.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch10_weights-2026-04-09_14:44:23.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch20_weights-2026-04-09_14:44:30.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch20_weights-2026-04-09_14:44:30.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch30_weights-2026-04-09_14:44:39.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch30_weights-2026-04-09_14:44:39.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch40_weights-2026-04-09_14:44:47.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch40_weights-2026-04-09_14:44:47.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch50_weights-2026-04-09_14:44:54.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch50_weights-2026-04-09_14:44:54.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch60_weights-2026-04-09_14:45:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch60_weights-2026-04-09_14:45:02.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch70_weights-2026-04-09_14:45:09.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch70_weights-2026-04-09_14:45:09.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch80_weights-2026-04-09_14:45:17.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold4_epoch80_weights-2026-04-09_14:45:17.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_best_weight.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch10_weights-2026-04-09_14:45:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch10_weights-2026-04-09_14:45:28.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch20_weights-2026-04-09_14:45:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch20_weights-2026-04-09_14:45:36.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch30_weights-2026-04-09_14:45:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch30_weights-2026-04-09_14:45:44.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch40_weights-2026-04-09_14:45:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch40_weights-2026-04-09_14:45:52.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch50_weights-2026-04-09_14:46:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch50_weights-2026-04-09_14:46:01.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch60_weights-2026-04-09_14:46:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch60_weights-2026-04-09_14:46:08.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch70_weights-2026-04-09_14:46:16.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch70_weights-2026-04-09_14:46:16.pth) — 4.3 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch80_weights-2026-04-09_14:46:24.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:40:57/fold5_epoch80_weights-2026-04-09_14:46:24.pth) — 4.3 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:40:57",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T14:46:26.097546",
    "training_started_at": "2026-04-09T14:40:57.069130",
    "training_finished_at": "2026-04-09T14:46:25.024525",
    "training_duration_seconds": 327.955,
    "training_duration_hours": 0.0911,
    "config": {
      "epochs": 80,
      "batch_size": 12,
      "learning_rate": 0.0001,
      "weight_decay": 0.0005,
      "k_folds": 5,
      "seed": 42,
      "loss": "smooth_l1"
    }
  },
  "folds": [
    {
      "fold": 1,
      "best_epoch": 10,
      "mae": 16.31718,
      "rmse": 21.55095,
      "r2": 0.16692,
      "pearson": 0.55888,
      "mean_error": 3.41542
    },
    {
      "fold": 2,
      "best_epoch": 45,
      "mae": 16.21369,
      "rmse": 21.60391,
      "r2": 0.24261,
      "pearson": 0.54983,
      "mean_error": 5.97568
    },
    {
      "fold": 3,
      "best_epoch": 30,
      "mae": 20.53856,
      "rmse": 26.83305,
      "r2": -0.08633,
      "pearson": 0.03158,
      "mean_error": 7.5668
    },
    {
      "fold": 4,
      "best_epoch": 25,
      "mae": 17.44881,
      "rmse": 25.11719,
      "r2": -0.32215,
      "pearson": 0.36112,
      "mean_error": -3.9935
    },
    {
      "fold": 5,
      "best_epoch": 45,
      "mae": 18.74629,
      "rmse": 27.74036,
      "r2": -0.16942,
      "pearson": 0.44721,
      "mean_error": 15.05325
    }
  ],
  "summary": {
    "mean_mae": 17.85291,
    "std_mae": 1.62602,
    "mean_rmse": 24.56909,
    "std_rmse": 2.58396,
    "mean_r2": -0.03367,
    "std_r2": 0.21023,
    "mean_pearson": 0.38972,
    "std_pearson": 0.1932
  }
}
```
