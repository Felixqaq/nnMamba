# RQ1/2/3 — 誠實版重算（最終 epoch）

本檔案由 `scripts/recompute_honest_final_epoch.py` 從 `train_log/run_all_rq/*.log`
重算而來，**不修改** `results.json` 或 `rq_overview.md`。

## 為什麼要重算

trainer 回報的是 `best_fold_result` — 一折 100 epochs、每 5 epoch 評一次，
共 20 個評估點取**最大值**。但資料只切 train/val 兩份（`loader.py` 的
`StratifiedKFold`），那個 val 折就是 held-out 測試折，所以 max 是在測試資料上挑的。
`rq_overview.md` 宣稱的協定是「最終 epoch 評估」，與程式實際行為不符。
下表改用每折第 100 epoch 的評估值，即文件宣稱的固定預算協定。

## 分類任務

| run | 誠實版 Acc (mean±std) | 報告值 Acc | 差距 |
|-----|----------------------|-----------|------|
| config.rq1.normal_v_abnormal.fusion | 0.820 ± 0.072 | 0.866 | +0.046 |
| config.rq1.normal_v_abnormal.image | 0.760 ± 0.140 | 0.879 | +0.119 |
| config.rq2a.angle_3class.fusion | 0.744 ± 0.051 | 0.774 | +0.030 |
| config.rq2a.angle_3class.image | 0.395 ± 0.273 | 0.518 | +0.123 |
| config.rq2b.angle_binary.fusion | 0.821 ± 0.108 | 0.871 | +0.050 |
| config.rq2b.angle_binary.image | 0.836 ± 0.091 | 0.885 | +0.049 |
| config.rq3.oi_emphysema.fusion | 0.834 ± 0.088 | 0.834 | +-0.000 |
| config.rq3.oi_emphysema.image | 0.711 ± 0.061 | 0.803 | +0.092 |

## 迴歸任務 (RQ2c)

| run | 誠實版 MAE (mean±std) | 誠實版 R² | 報告值 MAE | 差距 |
|-----|----------------------|----------|-----------|------|
| config.rq2c.angle_reg.fusion | 17.69 ± 4.83 | -0.088 | 14.61 | -3.08 |
| config.rq2c.angle_reg.image | 17.62 ± 1.05 | -0.115 | 15.48 | -2.14 |

## 每折明細

- `config.rq1.normal_v_abnormal.fusion`: 0.7143, 0.9231, 0.8461, 0.7692, 0.8461
- `config.rq1.normal_v_abnormal.image`: 0.5714, 0.9231, 0.8461, 0.6154, 0.8461
- `config.rq2a.angle_3class.fusion`: 0.6429, 0.7692, 0.7692, 0.7692, 0.7692
- `config.rq2a.angle_3class.image`: 0.3571, 0.3077, 0.9231, 0.2308, 0.1538
- `config.rq2b.angle_binary.fusion`: 0.7692, 0.8333, 0.6667, 0.8333, 1.0000
- `config.rq2b.angle_binary.image`: 0.8461, 0.7500, 0.7500, 1.0000, 0.8333
- `config.rq2c.angle_reg.fusion`: 14.9800, 23.5452, 14.3574, 12.1252, 23.4254
- `config.rq2c.angle_reg.image`: 17.1475, 17.4118, 19.2288, 18.1828, 16.1070
- `config.rq3.oi_emphysema.fusion`: 0.7857, 0.9231, 0.8461, 0.9231, 0.6923
- `config.rq3.oi_emphysema.image`: 0.7857, 0.6923, 0.6923, 0.6154, 0.7692
