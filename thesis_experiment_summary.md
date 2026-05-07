# 論文題目發想用實驗整理

整理日期：2026-05-07

這份文件整理目前 repo 裡可追溯到的實驗結果，目的是幫助發想論文題目與主線。整體研究歷程可以看成三個階段：

1. 早期：在 `classification/` 做 3D CT / medical image binary classification。
2. 中期：轉到 `regression/`，先做 PFT angle 連續值回歸。
3. 近期：以同一套 regression pipeline 延伸出 angle-derived classification、GOLD 四分類、以及極端二分類。

目前最有論文主線感的方向不是單純「分類準不準」，而是：

> 使用 3D CT 影像與 Mamba / hybrid Mamba-attention 模型，預測與 COPD / 肺功能相關的連續角度、角度分級、以及 GOLD 嚴重度，並研究小樣本不平衡資料下 augmentation 與 balanced sampling 的影響。

## 目前可用資料與任務

| 階段 | 主要資料 / label | 任務 | 目標 |
| --- | --- | --- | --- |
| Classification | `Normal/`, `Abnormal/`, `classification/datasets/` | Binary classification | Normal vs Abnormal / COPD 相關分類 |
| Regression | `by_angle_all/`, `patient_angle_classification_by_group.json` | PFT angle regression | 從 CT 預測連續塌陷角度 |
| Angle 3-class | 同上 | 3-class classification | `<=131°`, `132-151°`, `>=152°` |
| Angle extreme binary | 同上 | Gray-zone excluded binary classification | 只分 `AC <=131°` 與 `AC >=152°`，排除灰區 |
| GOLD | `by_angle_all/`, `pft.json` | 4-class ordinal classification | GOLD 1, 2, 3, 4 COPD 嚴重度 |

## 1. 早期 Classification 實驗

早期工作集中在 `classification/`。此 pipeline 支援 `nnmamba`, `densenet`, `vit`, `crate`，任務設定支援：

- `NC_v_AD`
- `sMCI_v_pMCI`
- `Normal_v_COPD`
- `Normal_v_Abnormal`

目前 repo 中可直接讀到數字結果的 classification artifact 主要是 `Normal_v_Abnormal`，輸出位置在 repo root 的 `figures/Normal_v_Abnormal/`、`train_log/Normal_v_Abnormal/`、`weights/Normal_v_Abnormal/`。

### Normal vs Abnormal 可追溯結果

| Run | Model | Epochs | K-fold | AUC | Accuracy | Sensitivity | Specificity | 備註 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `nnMamba_2026-02-12_15:21:50_e2_soft` | nnMamba | 50 | 5 | 0.9486 | 0.9167 | 0.8571 | 1.0000 | specificity 很高 |
| `nnMamba_2026-02-12_15:32:33_e5` | nnMamba | 50 | 5 | 1.0000 | 0.9409 | 0.9019 | 1.0000 | 目前 classification artifact 裡最佳 |
| `nnMamba_2026-02-12_16:00:5_e1` | nnMamba | 50 | 5 | 0.9643 | 0.8491 | 0.7857 | 0.9500 | 分數較不穩 |
| `nnMamba_2026-03-20_17:04:42` | nnMamba | 50 | 5 | 0.9400 | 0.8709 | 0.8714 | 0.8800 | sensitivity / specificity 較平均 |

Classification 階段的意義：

- 已經建立 3D CT binary classification pipeline。
- 已經有 5-fold cross-validation、ROC、PR、confusion matrix、sensitivity / specificity 等圖表。
- 結果顯示 nnMamba 在 Normal vs Abnormal 上有可學訊號，但當時主軸仍是一般二分類。
- 後續轉向 regression 是合理延伸，因為肺功能 / COPD 嚴重度不只是 binary label，也可以用連續角度或 ordinal stage 表示。

## 2. PFT Angle Regression 實驗

Regression 階段先把任務改成從 CT 預測連續角度，任務名稱為 `PFT_angle_regression`。評估指標包含 MAE、RMSE、R2、Pearson。

### 代表性模型比較

| Run | Model | MAE | RMSE | R2 | Pearson | 解讀 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `nnMambaReg_2026-04-08_13:05:59` | nnMambaReg | 17.1180 | 23.9086 | 0.0314 | 0.3413 | 原始 regression baseline 中較好的 nnMambaReg |
| `mamba_2026-04-08_14:15:06` | mamba | 17.5599 | 25.0490 | -0.0606 | 0.3812 | Mamba baseline |
| `swinunetr_2026-04-08_14:21:42` | swinunetr | 19.0884 | 24.3769 | -0.0012 | 0.1898 | SwinUNETR 在此設定較弱 |
| `hybrid_mamba_attention_2026-04-09_13:58:58` | hybrid_mamba_attention | 15.8512 | 21.2647 | 0.1723 | 0.6699 | MAE / RMSE / Pearson 最有代表性 |
| `hybrid_mamba_attention_2026-04-09_14:05:42` | hybrid_mamba_attention | 16.4486 | 22.1694 | 0.1728 | 0.5741 | R2 略高，但 MAE 較差 |
| `hybrid_mamba_attention_2026-04-09_14:57:03` | hybrid_mamba_attention | 16.5739 | 23.0462 | 0.1040 | 0.4710 | 目前開發報告中使用的 hybrid run |

Regression 階段的觀察：

- `hybrid_mamba_attention` 明顯比 `mamba`, `swinunetr`, 早期 `nnMambaReg` 更有潛力。
- 最好的 MAE 約在 15.85 度，Pearson 可到 0.67。
- R2 仍偏低，代表連續值 regression 還沒有非常穩定；可能原因包括資料量小、angle label noise、CT 與單一角度之間關係不完全線性。
- 這也是後來改做 angle-derived classification 的動機：把連續角度轉成 clinically interpretable groups，可能比硬回歸一個數值更穩。

## 3. GOLD Stage 四分類實驗

GOLD 任務把 `pft.json` 的 `GOLD 1 ~ 4` 當作四分類 label。這其實更接近 ordinal classification，因為 GOLD 1 < GOLD 2 < GOLD 3 < GOLD 4 有疾病嚴重度順序。

### GOLD 代表性結果

| Run | Model / 方法 | Accuracy | Macro-F1 | Balanced Acc | Macro Recall | 備註 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `mamba_2026-04-16_15:35:18` | mamba baseline | 0.4259 | 0.3107 | 0.3674 | 0.3674 | baseline |
| `swinunetr_2026-04-16_15:43:22` | swinunetr baseline | 0.4444 | 0.3144 | 0.3472 | 0.3472 | baseline |
| `hybrid_mamba_attention_2026-04-22_13:38:53` | hybrid baseline/tuning | 0.5151 | 0.4639 | 0.5194 | 0.5194 | 早期 balanced 指標較好 |
| `hybrid_mamba_attention_2026-04-22_14:45:11` | hybrid baseline/tuning | 0.5758 | 0.4355 | 0.5000 | 0.5000 | accuracy 較高 |
| `hybrid_mamba_attention_gold_balanced_sampling_aug200_class_2026-05-06_15:04:13` | aug200/class + balanced sampling | 0.5606 | 0.4436 | 0.4611 | 0.4611 | 補到 200/class 後 macro 指標沒有上升 |
| `hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40` | aug36/class + balanced sampling | 0.5606 | 0.5323 | 0.5222 | 0.5222 | 目前 GOLD 最值得保留的結果 |

GOLD 階段的觀察：

- `aug36/class` 比 `aug200/class` 更好，尤其 Macro-F1 從 0.4436 提升到 0.5323。
- 對小樣本 GOLD 任務來說，補太多 virtual samples 可能只是反覆變形少數病人，未必增加真正資訊。
- 目前 GOLD 結果還不適合只看 Accuracy，因為類別不平衡會誤導判斷。
- 論文指標建議補上：Quadratic Weighted Kappa、class-index MAE、within-1 accuracy。這三個很適合 GOLD ordinal stage。

## 4. Angle 3-Class 實驗

Angle 3-class 使用文獻與資料分布定義三類：

| Class | 定義 | 原始數量 |
| --- | --- | ---: |
| 0 | Emphysema/Abnormal (`<=131°`) | 14 |
| 1 | Intermediate (`132-151°`) | 5 |
| 2 | Normal (`>=152°`) | 47 |

這個任務最能呈現「label design + imbalance handling」的研究價值，因為原始資料為 14 / 5 / 47，Normal class 明顯最多。

### Angle 3-Class 主要結果

| Run / 方法 | Accuracy | Macro-F1 | Balanced Acc | 解讀 |
| --- | ---: | ---: | ---: | --- |
| `2026-04-23_14:40:43` early baseline | 0.7121 | 0.2796 | 0.3333 | 高 accuracy 但接近 majority-class collapse |
| `2026-04-23_15:09:27` early improved run | 0.6846 | 0.5650 | 0.6022 | balanced 指標大幅改善 |
| `2026-04-29_13:02:22` 純少類平衡 | 0.6396 | 0.2910 | 0.3437 | class 1 很難學 |
| `2026-04-29_13:25:54` 原本實體增強 baseline | 0.6352 | 0.5042 | 0.5630 | 三類相對平衡 |
| `2026-04-29_14:06:11` aug20/class + balanced | 0.4132 | 0.3592 | 0.5037 | 少數類改善但 Normal 掉太多 |
| `balanced_sampling_aug50_class_2026-04-30_14:33:34` | 0.7275 | 0.5605 | 0.6193 | 先前最推薦的折衷點 |
| `balanced_sampling_aug75_class_2026-04-30_14:52:46` | 0.6967 | 0.4502 | 0.4570 | 補更多不一定更好 |
| `balanced_sampling_aug100_class_2026-05-05_10:31:00` | 0.7264 | 0.5106 | 0.5444 | 比 75 好，但不如 50/300 |
| `balanced_sampling_aug200_class_2026-05-05_12:37:26` | 0.7571 | 0.5099 | 0.5007 | accuracy 高，但 balanced 指標普通 |
| `balanced_sampling_aug300_class_2026-05-07_09:38:05` | 0.7593 | 0.5713 | 0.6319 | 目前 angle 3-class 最佳 |
| `balanced_sampling_augx12_epoch_2026-05-05_13:48:47` | 0.4231 | 0.3432 | 0.4770 | 每 epoch 多 view 展開效果不佳 |

Angle 3-class 階段的觀察：

- 不能只看 Accuracy，因為 Normal class 太多。
- `50/class` 曾經是最好折衷，Macro-F1 0.5605、Balanced Acc 0.6193，而且三類 recall 較平均。
- 最新 `300/class` 目前數字最高，Macro-F1 0.5713、Balanced Acc 0.6319，但需要小心解讀：class 1 原始樣本非常少，補到 300/class 可能有 over-augmentation 風險。
- 論文可以把 `50`, `100`, `200`, `300` 做成 augmentation intensity ablation，主張「小樣本醫學影像不是補越多越好，需要用 macro / balanced 指標檢查」。

## 5. Angle Extreme Binary 實驗

此任務是教授建議把中間類別拿掉後的版本。依據 Topalovic et al. 的 Angle of Collapse 觀念，只保留兩端較明確族群：

| 類別 | 條件 | 病人數 |
| --- | --- | ---: |
| class 0 | `AC <= 131°` abnormal / emphysema-like | 14 |
| exclude | `132° <= AC < 152°` gray zone | 5 |
| class 1 | `AC >= 152°` normal-like | 47 |

### Extreme Binary 結果

| Run / 方法 | Accuracy | Macro-F1 | Balanced Acc | 解讀 |
| --- | ---: | ---: | ---: | --- |
| `hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43` | 0.7705 | 0.4350 | 0.5000 | 全部猜 Normal-like，屬於 majority-class collapse |
| `hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44` | 0.8692 | 0.8177 | 0.8289 | balanced augmentation 後真正學到兩端差異 |

Extreme binary 的價值：

- 這是目前最漂亮的分類結果。
- 它有清楚文獻依據，且排除灰區能降低 label noise。
- 如果要找論文主線，這個任務可以當「clinical interpretable endpoint classification」。
- 但它不是完整 COPD staging，只能說明 CT 模型能分辨 angle 兩端明確族群。

## 目前最佳結果摘要

| 任務 | 目前最佳 / 最值得報告結果 | 指標摘要 |
| --- | --- | --- |
| Normal vs Abnormal classification | `nnMamba_2026-02-12_15:32:33_e5` | AUC 1.0000, Accuracy 0.9409 |
| PFT angle regression | `hybrid_mamba_attention_2026-04-09_13:58:58` | MAE 15.8512, RMSE 21.2647, Pearson 0.6699 |
| GOLD 4-class | `gold_balanced_sampling_aug36_class_2026-05-07_11:17:40` | Accuracy 0.5606, Macro-F1 0.5323, Balanced Acc 0.5222 |
| Angle 3-class | `balanced_sampling_aug300_class_2026-05-07_09:38:05` | Accuracy 0.7593, Macro-F1 0.5713, Balanced Acc 0.6319 |
| Angle extreme binary | `extreme_binary_balanced_aug100_class_2026-05-06_14:27:44` | Accuracy 0.8692, Macro-F1 0.8177, Balanced Acc 0.8289 |

## 可形成的論文故事線

### 主線 A：CT 影像預測 COPD 相關 phenotype

核心敘事：

> 3D CT 不只可以做 Normal / Abnormal classification，也可以預測與肺功能相關的塌陷角度、角度分級、以及 GOLD severity stage。Mamba 類模型適合處理 3D medical image 的長距離空間關係。

適合放的實驗：

- Normal vs Abnormal binary classification 作為早期 baseline。
- PFT angle regression 作為連續 phenotype prediction。
- Angle 3-class 與 extreme binary 作為 clinically interpretable label design。
- GOLD stage 作為 COPD severity staging。

### 主線 B：小樣本不平衡 COPD CT 分類的 augmentation / sampling 研究

核心敘事：

> 在小樣本與類別不平衡的 3D CT 任務中，overall accuracy 會被 majority class 誤導。透過 patient-level split、training-fold-only augmentation、balanced sampling，以及 macro / balanced 指標，可以更可靠地評估模型是否真的學到少數類。

適合放的實驗：

- Angle 3-class 的 `50/75/100/200/300 class` ablation。
- GOLD 的 `aug36/class` vs `aug200/class`。
- Extreme binary baseline collapse vs balanced aug100 的對比。

### 主線 C：從連續回歸到臨床分級的 label redesign

核心敘事：

> 單純預測連續角度在小樣本下不穩定，但將角度轉為文獻支持的臨床分組後，模型更容易學到可解釋的 CT phenotype。特別是排除灰區後的 extreme binary classification，能明顯改善 balanced 指標。

適合放的實驗：

- PFT angle regression 的 MAE / Pearson。
- Angle 3-class 的中間類困難。
- Extreme binary 排除灰區後的 Macro-F1 / Balanced Acc 提升。

## 論文題目草案

### 中文題目

1. 基於 3D CT 與 Hybrid Mamba-Attention 模型之 COPD 肺功能角度與嚴重度預測研究
2. 從塌陷角度回歸到 GOLD 分級：3D CT Mamba 模型於 COPD 表徵預測之研究
3. 小樣本不平衡 3D CT 資料下之肺部角度分類與 GOLD 嚴重度預測
4. 結合 Mamba 與注意力機制之 3D CT 肺功能表徵預測：以 Angle of Collapse 與 GOLD 分級為例
5. 基於文獻門檻之 COPD CT 影像角度分級與極端二分類研究

### 英文題目

1. 3D CT-Based Prediction of COPD-Related Pulmonary Phenotypes Using Hybrid Mamba-Attention Networks
2. From Angle Regression to GOLD Severity Classification: A 3D CT Mamba Framework for COPD Phenotype Prediction
3. Handling Class Imbalance in Small-Sample 3D CT Classification for COPD Angle and GOLD Stage Prediction
4. Clinically Interpretable Angle-of-Collapse Classification from 3D CT Using Hybrid Mamba-Attention Models
5. Regression-to-Classification Label Redesign for COPD-Related 3D CT Phenotype Learning

## 建議下一步

1. GOLD 補 ordinal 指標：Quadratic Weighted Kappa、class-index MAE、within-1 accuracy。
2. Angle 3-class 對 `50/class` 與 `300/class` 重跑不同 seed，確認 300/class 是否真的穩定。
3. Extreme binary 保留為強結果，但在論文中要清楚說它排除了 `132-151°` 灰區，所以不是全資料分類。
4. Regression 保留 hybrid best run，並用它說明「連續角度可學，但分類化 label 更穩」。
5. 最終主表建議同時放 Accuracy、Macro-F1、Balanced Accuracy，GOLD 額外放 ordinal metrics。

## 主要結果來源

- Classification results: `figures/Normal_v_Abnormal/*/results.json`
- Regression results: `regression/figures/PFT_angle_regression/*/results.json`
- Angle 3-class results: `regression/figures/Angle_3class_classification/*/results.json`
- Extreme binary results: `regression/figures/Angle_extreme_binary_classification/*/results.json`
- GOLD results: `regression/figures/GOLD_stage_classification/*/results.json`
- Existing notes: `TODO.md`, `4classTODO.md`, `DataAugment.md`, `regression/docs/*.md`
