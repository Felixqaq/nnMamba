# CT Regression / GOLD 分類 使用說明

這個資料夾現在支援兩種任務，而且共用同一套 CT 輸入 pipeline 與三個模型：

- `angle`：原本的 regression，從 CT 預測 PFT 塌陷角度
- `gold`：新的四分類，從 CT 預測 `GOLD 1 ~ 4`

你可以直接改 yaml 切換，不需要換 `train.py` / `evaluate.py` 入口。

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

目前 pipeline 會讀這些來源：

```text
../by_angle_all/
../patient_angle_classification_by_group.json
../pft.json
```

其中：

- `by_angle_all/` 放 CT 影像
- `patient_angle_classification_by_group.json` 放每位病人的角度標註
- `pft.json` 放每位病人的 GOLD 分級
- 程式會從 CT 檔名抓病人 ID，再去 JSON 對應 target

注意：

- `by_angle_all/abnormal_low_angle/` 和 `by_angle_all/normal_high_angle/` 是歷史命名
- regression 真正使用的是 angle JSON 裡的實際角度值
- GOLD 四分類真正使用的是 `pft.json` 的 `GOLD 1 ~ 4`

## 3. 先檢查資料

在 repo root 執行：

```bash
conda run -n nnMamba python regression/scripts/build_manifest.py
conda run -n nnMamba python regression/scripts/check_dataset.py
conda run -n nnMamba python regression/scripts/plot_dataset_overview.py
```

這三個指令主要還是做 regression 資料完整性檢查。

## 4. 如何用 yaml 切換任務

你只要改這幾個欄位：

- `data.target_mode: angle | gold`
- `task: PFT_angle_regression | GOLD_stage_classification`
- `model.name: mamba | hybrid_mamba_attention | swinunetr`

`training.loss` 已經支援 `auto`：

- `angle` 時會自動用 regression loss
- `gold` 時會自動用 `cross_entropy`

已經附一份可直接使用的 GOLD 範例設定：

- [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml)：原本 regression
- [config.hybrid.preset.yaml](/home/felix/Research/nnMamba/regression/config.hybrid.preset.yaml)：hybrid 的可選 preset
- [config.gold.yaml](/home/felix/Research/nnMamba/regression/config.gold.yaml)：GOLD 四分類範例

## 5. 開始訓練

進到 `regression/` 後執行：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.yaml
conda run -n nnMamba python train.py --config config.gold.yaml
```

目前三個模型都支援同一套切換方式：

- `mamba`
- `hybrid_mamba_attention`
- `swinunetr`

如果是 `gold` 模式：

- 輸入一樣是 CT
- 輸出會變成 4 類 logits
- validation 會看 `Accuracy / Macro-F1 / Balanced Accuracy`
- 每 fold 會輸出 confusion matrix

資料前處理相關設定在 `data:` 區塊：

- `intensity_window`: 先對 CT 做 HU clipping，預設 `[-1000, 400]`
- `input_normalization`: CT 輸入正規化方式
- `target_normalization`: 只對 regression target 生效

## 6. 評估模型

訓練完後，假設 run id 是 `<run_uuid>`：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.gold.yaml
```

如果只想看某一個 fold：

```bash
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --fold 1 --config config.gold.yaml
```

## 7. 結果在哪裡

輸出會放在：

```text
regression/weights/<task>/<run_uuid>/
regression/train_log/<task>/<run_uuid>/
regression/figures/<task>/<run_uuid>/
```

regression 模式圖表包含：

- loss curve
- MAE / RMSE / R2 / Pearson 曲線
- prediction scatter
- residual plot
- error histogram
- Bland-Altman plot
- 全 fold 的 total summary 圖

GOLD 分類模式則會輸出：

- per-fold confusion matrix
- total confusion matrix
- Accuracy / Macro-F1 / Balanced Accuracy 的 summary 圖

## 8. 匯總結果

如果某次訓練已經產生 `results.json`，可以再做 summary：

```bash
conda run -n nnMamba python regression/scripts/summarize_results.py regression/figures/<task>/<run_uuid>/results.json
```

這個 summary script 現在同時支援 regression 與 classification。

## 9. 你最常會改的地方

- 改任務切換： [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml) 或 [config.gold.yaml](/home/felix/Research/nnMamba/regression/config.gold.yaml)
- 改資料載入： [loader.py](/home/felix/Research/nnMamba/regression/data/loader.py)
- 改 model： [mamba_regressor.py](/home/felix/Research/nnMamba/regression/networks/mamba_regressor.py)
- 改訓練流程： [trainer.py](/home/felix/Research/nnMamba/regression/core/trainer.py)

## 10. 最小使用流程

如果你只想照順序跑一次 regression：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
```

如果你要跑 GOLD 四分類：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.gold.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.gold.yaml
```
