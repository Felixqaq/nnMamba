# CT Regression 使用說明

這個資料夾是把原本的 CT classification 流程改成 regression 流程，用 3D Mamba 網路從 CT 預測 PFT 塌陷角度。

## 1. 環境

使用既有的 conda 環境：

```bash
conda activate nnMamba
```

如果你想用非互動方式執行：

```bash
conda run -n nnMamba python --version
```

## 2. 資料來源

目前 pipeline 會讀這兩個來源：

```text
../by_angle_all/
../patient_angle_classification_by_group.json
```

其中：

- `by_angle_all/` 放 CT 影像
- `patient_angle_classification_by_group.json` 放每位病人的角度標註
- 程式會從 CT 檔名抓病人 ID，再去 JSON 對應角度

注意：

- `by_angle_all/abnormal_low_angle/` 和 `by_angle_all/normal_high_angle/` 是歷史命名
- regression 真正使用的是 JSON 裡的實際角度值，不是直接把資料夾名當 regression target

## 3. 先檢查資料

在 repo root 執行：

```bash
conda run -n nnMamba python regression/scripts/build_manifest.py
conda run -n nnMamba python regression/scripts/check_dataset.py
conda run -n nnMamba python regression/scripts/plot_dataset_overview.py
```

這三個指令會做：

- 建立 manifest：`regression/datasets/generated/regression_manifest.json`
- 檢查資料是否都有 label、角度範圍是否正常
- 產生資料概覽圖

## 4. 開始訓練

進到 `regression/` 後執行：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.yaml
python train.py --config config.hybrid.yaml
```

預設設定在 [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml)。

目前預設是：

- 5-fold cross validation
- model: `nnmamba_regressor`
- loss: `smooth_l1`
- target: CT -> 單一角度值

資料前處理相關設定在 `data:` 區塊：

- `intensity_window`: 先對 CT 做 HU clipping，預設 `[-1000, 400]`
- `input_normalization`: CT 輸入正規化方式，預設 `zscore`
- `target_normalization`: 角度 label 的正規化方式，預設 `zscore`
  使用 `zscore` 時，現在是用整個資料集的 angle mean/std 計算，不再依 fold 分開算

如果你想停用 CT 輸入正規化，但保留 intensity window，可以改成：

```yaml
data:
  intensity_window: [-1000.0, 400.0]
  input_normalization: none
  target_normalization: zscore
```

## 5. 評估模型

訓練完後，假設 run id 是 `<run_uuid>`：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
```

如果只想看某一個 fold：

```bash
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --fold 1 --config config.yaml
```

## 6. 結果在哪裡

輸出會放在：

```text
regression/weights/PFT_angle_regression/<run_uuid>/
regression/train_log/PFT_angle_regression/<run_uuid>/
regression/figures/PFT_angle_regression/<run_uuid>/
```

圖表包含：

- loss curve
- MAE / RMSE / R2 / Pearson 曲線
- prediction scatter
- residual plot
- error histogram
- Bland-Altman plot
- 全 fold 的 total summary 圖

## 7. 匯總結果

如果某次訓練已經產生 `results.json`，可以再做 summary：

```bash
conda run -n nnMamba python regression/scripts/summarize_results.py regression/figures/PFT_angle_regression/<run_uuid>/results.json
```

會輸出：

- `summary.csv`
- `metric_boxplot.png`
- `metric_barplot.png`

## 8. 你最常會改的地方

- 改訓練參數： [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml)
- 改資料載入： [loader.py](/home/felix/Research/nnMamba/regression/data/loader.py)
- 改 model： [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py)
- 改訓練流程： [trainer.py](/home/felix/Research/nnMamba/regression/core/trainer.py)

## 9. 自動調參

如果你想讓 hybrid model 在大約一小時內自動跑一輪參數搜尋，可以用：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python scripts/tune_hybrid.py --base-config config.hybrid.yaml --budget-minutes 60
```

這個腳本會：

- 以 `config.hybrid.yaml` 為基底
- 自動嘗試一組適合一小時預算的 hybrid 候選設定
- 每次訓練後讀取 `results.json`
- 輸出 `leaderboard.csv`、`leaderboard.json`、`best_config.yaml`

預設輸出位置：

```text
regression/figures/PFT_angle_regression/tuning_runs/<timestamp>/
```

如果只想先看它打算跑哪些候選，不要真的開始訓練：

```bash
conda run -n nnMamba python scripts/tune_hybrid.py --base-config config.hybrid.yaml --dry-run
```

## 10. 最小使用流程

如果你只想照順序跑一次，用這組：

```bash
cd /home/felix/Research/nnMamba
conda run -n nnMamba python regression/scripts/build_manifest.py
conda run -n nnMamba python regression/scripts/check_dataset.py
cd regression
conda run -n nnMamba python train.py --config config.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
```
