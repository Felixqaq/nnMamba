# CT Regression / GOLD 分類 使用說明

這個資料夾現在支援四種任務，而且共用同一套 CT 輸入 pipeline 與三個模型：

- `angle`：原本的 regression，從 CT 預測 PFT 塌陷角度
- `gold`：新的四分類，從 CT 預測 `GOLD 1 ~ 4`
- `angle_3class`：用 131° / 152° 門檻做三分類，從 CT 預測 `Emphysema/Abnormal`、`Intermediate`、`Normal`
- `angle_binary_extreme`：排除 132°-151° 灰區，只用 `AC <=131°` 與 `AC >=152°` 做二分類

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
- `by_angle_all_gold_augmented/` 是 GOLD 訓練用的實體增強資料集，由 `regression/scripts/generate_gold_augmented_dataset.py` 產生
- `by_angle_all_angle_3class_augmented/` 是 131° / 152° 三分類訓練用的實體增強資料集，由 `regression/scripts/generate_angle_3class_augmented_dataset.py` 產生
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

GOLD 若要重產實體增強資料集：

```bash
conda run -n nnMamba python regression/scripts/generate_gold_augmented_dataset.py --overwrite
```

這會建立 `by_angle_all_gold_augmented/`，保留原始 66 筆，再把 `GOLD 2/3/4` 補到每類 36 筆，輸出 `regression/datasets/generated/gold_manifest.augmented.json`。

131° / 152° 三分類若要重產實體增強資料集：

```bash
conda run -n nnMamba python regression/scripts/generate_angle_3class_augmented_dataset.py --overwrite
```

這會建立 `by_angle_all_angle_3class_augmented/`，保留原始 66 筆，再把 `<=131°` 和 `132-151°` 補到每類 47 筆，輸出 `regression/datasets/generated/angle_3class_manifest.augmented.json`。

## 4. 如何用 yaml 切換任務

你只要改這幾個欄位：

- `data.target_mode: angle | gold | angle_3class | angle_binary_extreme`
- `task: PFT_angle_regression | GOLD_stage_classification | Angle_3class_classification | Angle_extreme_binary_classification`
- `model.name: mamba | hybrid_mamba_attention | swinunetr`

`training.loss` 已經支援 `auto`：

- `angle` 時會自動用 regression loss
- `gold` 時會自動用 `cross_entropy`
- `angle_3class` 時會自動用 `cross_entropy`
- `angle_binary_extreme` 時會自動用 `cross_entropy`

已經附一份可直接使用的 GOLD 範例設定：

- [config.yaml](/home/felix/Research/nnMamba/regression/config.yaml)：原本 regression
- [config.hybrid.preset.yaml](/home/felix/Research/nnMamba/regression/config.hybrid.preset.yaml)：hybrid 的可選 preset
- [config.gold.yaml](/home/felix/Research/nnMamba/regression/config.gold.yaml)：GOLD 四分類，training fold 內把四個 GOLD class 都 virtual augmentation 補到 200，再做每 epoch 少類平衡
- [config.angle_3class.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.yaml)：131° / 152° 三分類，沿用實體增強資料集
- [config.angle_binary_extreme.yaml](/home/felix/Research/nnMamba/regression/config.angle_binary_extreme.yaml)：文獻式極端二分類，排除 `132-151°` 灰區，保留 `14/47` 筆
- [config.angle_binary_extreme.balanced_sampling.augmentation100.yaml](/home/felix/Research/nnMamba/regression/config.angle_binary_extreme.balanced_sampling.augmentation100.yaml)：極端二分類，training fold 內 virtual augmentation 到每類 100，再做每 epoch 少類平衡
- [docs/angle_binary_extreme_report.md](/home/felix/Research/nnMamba/regression/docs/angle_binary_extreme_report.md)：給教授看的文獻依據、分類規則、資料分布與建議實驗說明
- [config.angle_3class.balanced_sampling.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.yaml)：131° / 152° 三分類，使用原始資料並每個 epoch 隨機下採樣多數類
- [config.angle_3class.balanced_sampling.augmentation.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.augmentation.yaml)：131° / 152° 三分類，training fold 內把 class 0/1 virtual augmentation 補到 20，再做每 epoch 少類平衡
- [config.angle_3class.balanced_sampling.augmentation100.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.augmentation100.yaml)：131° / 152° 三分類，training fold 內把三個 class 都 virtual augmentation 補到 100，再做每 epoch 少類平衡
- [config.angle_3class.balanced_sampling.augmentation300.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.augmentation300.yaml)：131° / 152° 三分類，training fold 內把三個 class 都 virtual augmentation 補到 300，再做每 epoch 少類平衡
- [config.angle_3class.balanced_sampling.augmentation_x12.yaml](/home/felix/Research/nnMamba/regression/config.angle_3class.balanced_sampling.augmentation_x12.yaml)：131° / 152° 三分類，每個 epoch 先抽少類平衡 base set，再把每筆 base sample 展開成 12 個 train-time views

這些比較方法都有 `experiment.name`，新的 run id、`results.json` metadata、summary/confusion matrix 圖表標題都會顯示方法名稱。之後資料夾會像 `hybrid_mamba_attention_balanced_sampling_aug100_class_<timestamp>`，不用再只靠時間猜是哪個方法。

## 5. 開始訓練

進到 `regression/` 後執行：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.yaml
conda run -n nnMamba python train.py --config config.gold.yaml
conda run -n nnMamba python train.py --config config.angle_3class.yaml
conda run -n nnMamba python train.py --config config.angle_binary_extreme.yaml
conda run -n nnMamba python train.py --config config.angle_binary_extreme.balanced_sampling.augmentation100.yaml
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.yaml
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation.yaml
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation100.yaml
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation300.yaml
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation_x12.yaml
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
- `input_normalization`: CT 輸入正規化方式；GOLD 範例使用 `zscore`，避免 raw HU 在 AMP 訓練時造成數值不穩
- `target_normalization`: 只對 regression target 生效
- `balanced_sampling`: classification task 可啟用每個 epoch 重新隨機下採樣，多數類會抽到和該 fold 少數類一樣的數量
- `augmentation`: GOLD 範例使用原始 `by_angle_all/`，並在每個 training fold 內把四個 GOLD class 都 virtual augmentation 補到 200。
- validation/test 仍只使用原始病人 CT；virtual augmented copies 只會進 training fold，避免同病人資料洩漏。
- `early_stopping`: GOLD 範例預設用 validation Macro-F1；連續 6 次 evaluation 沒有至少 `0.005` 的進步就停止該 fold。

如果是 `angle_binary_extreme` 模式：

- class 0: `Abnormal/emphysema-like (AC <=131°)`
- 排除: `132° <= AC < 152°`
- class 1: `Normal-like (AC >=152°)`
- 這個設定來自 Topalovic et al. 的 Angle of Collapse (AC) 文獻：`AC < 131°` 是 heavy smokers 中預測 emphysema 的 high-specificity cut-off；但 sensitivity 不高，所以不能把所有 `>131°` 都解釋成 normal。
- 因此 `132-151°` 直接當灰區排除，留下 61 筆，分布為 `14/47`。
- `config.angle_binary_extreme.yaml` 是 baseline，不做 augmentation / balanced sampling。
- `config.angle_binary_extreme.balanced_sampling.augmentation100.yaml` 是正式比較用設定，會在每個 training fold 內把兩類都補到 100，再做少類平衡。

如果是 `angle_3class` 模式：

- class 0: `Emphysema/Abnormal (<=131°)`
- class 1: `Intermediate (132-151°)`
- class 2: `Normal (>=152°)`
- `config.angle_3class.yaml` 保留原本方法：使用 `by_angle_all_angle_3class_augmented/`，總樣本數為 141，三類各 47 筆，並保留 balanced CrossEntropy。
- `config.angle_3class.balanced_sampling.yaml` 是新增方法：使用原始 `by_angle_all/`，總樣本數為 66，三類分布為 14 / 5 / 47。
- 新增方法的 `balanced_sampling: true` 會在每個 train epoch 以該 fold 的少數類數量為基準，重新隨機抽取多數類樣本。
- `config.angle_3class.balanced_sampling.augmentation.yaml` 會在每個 training fold 內把 class 0/1 用 virtual augmented copies 補到 20，再做少類平衡；fold 1 會從 11/4/37 變成 20/20/37，每 epoch 抽 20/20/20。
- `config.angle_3class.balanced_sampling.augmentation100.yaml` 會在每個 training fold 內把 class 0/1/2 都補到 100，再做少類平衡；fold 1 會變成 100/100/100，每 epoch 抽 300 張。
- `config.angle_3class.balanced_sampling.augmentation300.yaml` 會在每個 training fold 內把 class 0/1/2 都補到 300，再做少類平衡；fold 1 會變成 300/300/300，每 epoch 抽 900 張。
- `config.angle_3class.balanced_sampling.augmentation_x12.yaml` 會在每個 epoch 先從 fold 內隨機抽少類平衡 base set；fold 1 會先抽成 4/4/4，再把每筆 base sample 展開成 12 個 views，所以每 epoch 會訓練 48/48/48。
- 新增方法關閉 class weights，避免「已平衡抽樣」後又重複加權少數類。

## 6. 評估模型

訓練完後，假設 run id 是 `<run_uuid>`：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.gold.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_binary_extreme.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_binary_extreme.balanced_sampling.augmentation100.yaml
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

`run_uuid` 會包含模型與方法名稱，例如：

```text
hybrid_mamba_attention_materialized_aug47_class_weights_2026-04-29_15:30:00
hybrid_mamba_attention_per_epoch_minority_undersampling_2026-04-29_15:30:00
hybrid_mamba_attention_balanced_sampling_aug20_class_2026-04-29_15:30:00
hybrid_mamba_attention_balanced_sampling_aug100_class_2026-04-29_15:30:00
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

如果你要跑 131° / 152° 三分類：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.yaml
```

如果你要跑 131° / 152° 三分類的每 epoch 隨機下採樣版本：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.balanced_sampling.yaml
```

如果你要跑 131° / 152° 三分類的 20/class virtual augmentation + 每 epoch 隨機下採樣版本：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.balanced_sampling.augmentation.yaml
```

如果你要跑 131° / 152° 三分類的 100/class virtual augmentation + 每 epoch 隨機下採樣版本：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation100.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.balanced_sampling.augmentation100.yaml
```

如果你要跑 131° / 152° 三分類的 300/class virtual augmentation + 每 epoch 隨機下採樣版本：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation300.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.balanced_sampling.augmentation300.yaml
```

如果你要跑 131° / 152° 三分類的「先每 epoch 少類平衡，再 x12 train-time augmentation」版本：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation_x12.yaml
conda run -n nnMamba python evaluate.py --uuid <run_uuid> --config config.angle_3class.balanced_sampling.augmentation_x12.yaml
```
