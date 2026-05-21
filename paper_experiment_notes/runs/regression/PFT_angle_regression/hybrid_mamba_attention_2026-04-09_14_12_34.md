# Run: hybrid_mamba_attention_2026-04-09_14:12:34

## 一句話用途
- 任務: `PFT_angle_regression`
- 模型: `hybrid_mamba_attention`
- 結果時間: `2026-04-09T14:18:18.879109`
- 訓練時間: `2026-04-09T14:12:34.013110` -> `2026-04-09T14:18:17.810868`
- 訓練耗時: 0.0955 hours
- 原始 results.json: [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/results.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/results.json)

## Summary
| metric | value |
| --- | --- |
| mean_mae | 17.70489 |
| std_mae | 4.29311 |
| mean_rmse | 22.63559 |
| std_rmse | 3.6867 |
| mean_r2 | 0.12399 |
| std_r2 | 0.21592 |
| mean_pearson | 0.30556 |
| std_pearson | 0.35868 |

## Fold Metrics
| fold | best_epoch | mae | rmse | r2 | pearson | mean_error |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 5 | 19.79508 | 23.69438 | -0.00703 | 0.00451 | -1.96732 |
| 2 | 40 | 13.17981 | 17.76836 | 0.48767 | 0.82831 | 8.24872 |
| 3 | 15 | 20.48294 | 26.72956 | -0.07797 | 0.31171 | 7.48461 |
| 4 | 70 | 12.07748 | 18.86335 | 0.25428 | 0.54757 | -0.20223 |
| 5 | 5 | 22.98915 | 26.12232 | -0.03698 | -0.16429 | 4.82002 |

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
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_bland_altman.png) — 123.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_error_hist.png) — 85.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_loss.png) — 160.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_residuals.png) — 94.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_scatter.png) — 178.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_summary.png) — 411.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_bland_altman.png) — 127.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_error_hist.png) — 97.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_loss.png) — 146.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_residuals.png) — 87.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_scatter.png) — 223.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_summary.png) — 452.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_bland_altman.png) — 126.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_error_hist.png) — 86.8 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_loss.png) — 146.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_residuals.png) — 95.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_scatter.png) — 194.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_summary.png) — 426.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_bland_altman.png) — 123.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_error_hist.png) — 88.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_loss.png) — 179.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_residuals.png) — 96.9 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_scatter.png) — 211.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_summary.png) — 471.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_bland_altman.png) — 124.3 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_error_hist.png) — 99.2 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_loss.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_loss.png) — 134.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_residuals.png) — 91.7 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_scatter.png) — 190.5 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_summary.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_summary.png) — 386.0 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/metric_boxplot.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/metric_boxplot.png) — 97.1 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_bland_altman.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_bland_altman.png) — 161.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_error_hist.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_error_hist.png) — 96.6 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_residuals.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_residuals.png) — 124.4 KB
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_scatter.png](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/total_scatter.png) — 230.2 KB

### prediction_files
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=11
- [regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_predictions.json](/home/felix/Research/nnMamba/regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_predictions.json) — json keys=['fold', 'mae', 'rmse', 'r2', 'pearson', 'mean_error', 'predictions']; rows=10

### summary_files
_No files found._

### logs
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4.log) — 2.8 KB
- [regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5.log](/home/felix/Research/nnMamba/regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5.log) — 2.8 KB

### weights
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch10_weights-2026-04-09_14:12:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch10_weights-2026-04-09_14:12:44.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch20_weights-2026-04-09_14:12:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch20_weights-2026-04-09_14:12:52.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch30_weights-2026-04-09_14:13:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch30_weights-2026-04-09_14:13:00.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch40_weights-2026-04-09_14:13:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch40_weights-2026-04-09_14:13:08.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch50_weights-2026-04-09_14:13:16.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch50_weights-2026-04-09_14:13:16.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch60_weights-2026-04-09_14:13:25.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch60_weights-2026-04-09_14:13:25.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch70_weights-2026-04-09_14:13:33.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch70_weights-2026-04-09_14:13:33.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch80_weights-2026-04-09_14:13:41.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold1_epoch80_weights-2026-04-09_14:13:41.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch10_weights-2026-04-09_14:13:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch10_weights-2026-04-09_14:13:51.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch20_weights-2026-04-09_14:14:01.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch20_weights-2026-04-09_14:14:01.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch30_weights-2026-04-09_14:14:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch30_weights-2026-04-09_14:14:10.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch40_weights-2026-04-09_14:14:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch40_weights-2026-04-09_14:14:20.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch50_weights-2026-04-09_14:14:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch50_weights-2026-04-09_14:14:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch60_weights-2026-04-09_14:14:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch60_weights-2026-04-09_14:14:36.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch70_weights-2026-04-09_14:14:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch70_weights-2026-04-09_14:14:44.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch80_weights-2026-04-09_14:14:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold2_epoch80_weights-2026-04-09_14:14:52.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch10_weights-2026-04-09_14:15:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch10_weights-2026-04-09_14:15:02.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch20_weights-2026-04-09_14:15:11.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch20_weights-2026-04-09_14:15:11.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch30_weights-2026-04-09_14:15:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch30_weights-2026-04-09_14:15:19.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch40_weights-2026-04-09_14:15:27.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch40_weights-2026-04-09_14:15:27.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch50_weights-2026-04-09_14:15:35.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch50_weights-2026-04-09_14:15:35.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch60_weights-2026-04-09_14:15:43.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch60_weights-2026-04-09_14:15:43.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch70_weights-2026-04-09_14:15:51.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch70_weights-2026-04-09_14:15:51.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch80_weights-2026-04-09_14:15:59.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold3_epoch80_weights-2026-04-09_14:15:59.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch10_weights-2026-04-09_14:16:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch10_weights-2026-04-09_14:16:10.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch20_weights-2026-04-09_14:16:20.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch20_weights-2026-04-09_14:16:20.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch30_weights-2026-04-09_14:16:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch30_weights-2026-04-09_14:16:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch40_weights-2026-04-09_14:16:37.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch40_weights-2026-04-09_14:16:37.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch50_weights-2026-04-09_14:16:45.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch50_weights-2026-04-09_14:16:45.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch60_weights-2026-04-09_14:16:53.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch60_weights-2026-04-09_14:16:53.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch70_weights-2026-04-09_14:17:02.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch70_weights-2026-04-09_14:17:02.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch80_weights-2026-04-09_14:17:10.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold4_epoch80_weights-2026-04-09_14:17:10.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_best_weight.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_best_weight.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch10_weights-2026-04-09_14:17:19.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch10_weights-2026-04-09_14:17:19.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch20_weights-2026-04-09_14:17:28.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch20_weights-2026-04-09_14:17:28.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch30_weights-2026-04-09_14:17:36.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch30_weights-2026-04-09_14:17:36.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch40_weights-2026-04-09_14:17:44.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch40_weights-2026-04-09_14:17:44.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch50_weights-2026-04-09_14:17:52.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch50_weights-2026-04-09_14:17:52.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch60_weights-2026-04-09_14:18:00.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch60_weights-2026-04-09_14:18:00.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch70_weights-2026-04-09_14:18:08.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch70_weights-2026-04-09_14:18:08.pth) — 4.9 MB
- [regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch80_weights-2026-04-09_14:18:16.pth](/home/felix/Research/nnMamba/regression/weights/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:12:34/fold5_epoch80_weights-2026-04-09_14:18:16.pth) — 4.9 MB

## Full results.json
```json
{
  "meta": {
    "uuid": "hybrid_mamba_attention_2026-04-09_14:12:34",
    "model": "hybrid_mamba_attention",
    "task": "PFT_angle_regression",
    "timestamp": "2026-04-09T14:18:18.879109",
    "training_started_at": "2026-04-09T14:12:34.013110",
    "training_finished_at": "2026-04-09T14:18:17.810868",
    "training_duration_seconds": 343.798,
    "training_duration_hours": 0.0955,
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
      "best_epoch": 5,
      "mae": 19.79508,
      "rmse": 23.69438,
      "r2": -0.00703,
      "pearson": 0.00451,
      "mean_error": -1.96732
    },
    {
      "fold": 2,
      "best_epoch": 40,
      "mae": 13.17981,
      "rmse": 17.76836,
      "r2": 0.48767,
      "pearson": 0.82831,
      "mean_error": 8.24872
    },
    {
      "fold": 3,
      "best_epoch": 15,
      "mae": 20.48294,
      "rmse": 26.72956,
      "r2": -0.07797,
      "pearson": 0.31171,
      "mean_error": 7.48461
    },
    {
      "fold": 4,
      "best_epoch": 70,
      "mae": 12.07748,
      "rmse": 18.86335,
      "r2": 0.25428,
      "pearson": 0.54757,
      "mean_error": -0.20223
    },
    {
      "fold": 5,
      "best_epoch": 5,
      "mae": 22.98915,
      "rmse": 26.12232,
      "r2": -0.03698,
      "pearson": -0.16429,
      "mean_error": 4.82002
    }
  ],
  "summary": {
    "mean_mae": 17.70489,
    "std_mae": 4.29311,
    "mean_rmse": 22.63559,
    "std_rmse": 3.6867,
    "mean_r2": 0.12399,
    "std_r2": 0.21592,
    "mean_pearson": 0.30556,
    "std_pearson": 0.35868
  }
}
```
