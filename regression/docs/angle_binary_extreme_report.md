# Angle of Collapse 極端二分類實驗設計

## 目的

教授建議將角度三分類中的中間類別拿掉，因此本次新增一個 gray-zone excluded binary classification 實驗。目標不是把所有病例硬切成 abnormal / normal，而是只訓練模型辨識文獻上較明確的兩端族群。

## 文獻依據

Topalovic et al. 定義的是 Angle of Collapse (AC)。該研究指出：

- 有 emphysema 的受試者 AC 較低，平均約 `131° ± 14°`
- 沒有 emphysema 的受試者 AC 較高，平均約 `152° ± 10°`
- ROC 分析中，作者將 `AC < 131°` 視為 heavy smokers 中預測 emphysema 的 high-specificity cut-off
- 這個 cut-off 的 specificity / PPV 高，但 sensitivity 不高

因此，`AC < 131°` 比較適合解釋成 rule-in emphysema-like，而不是解釋成 `AC >= 131°` 就一定 normal。

Reference:

- Topalovic et al., Respiratory Research 2013: https://link.springer.com/article/10.1186/1465-9921-14-131
- PubMed: https://pubmed.ncbi.nlm.nih.gov/24251975/

## 本次分類規則

本次採用以下二分類規則：

| 類別 | 條件 | 說明 |
| --- | --- | --- |
| class 0 | `AC <= 131°` | Abnormal / emphysema-like |
| exclude | `132° <= AC < 152°` | 中間灰區，不進訓練、不進驗證 |
| class 1 | `AC >= 152°` | Normal-like |

這個設計保留文獻中最明確的 abnormal 端點，並用 no-emphysema group 的平均值附近作為 normal-like 端點。中間灰區不硬分，避免 label noise。

## 目前資料分布

原始資料共 `66` 位病人。依照上述規則切分後：

| 分組 | 病人數 |
| --- | ---: |
| `AC <= 131°` | 14 |
| `132° <= AC < 152°` | 5 |
| `AC >= 152°` | 47 |
| 實際進入二分類 | 61 |

因此此任務仍然不平衡，class distribution 為 `14 / 47`。

## 已新增的程式功能

新增 target mode:

```yaml
data:
  target_mode: angle_binary_extreme
```

此模式會自動：

1. 讀取原始 `by_angle_all/` CT 影像
2. 讀取 `patient_angle_classification_by_group.json` 中的 AC label
3. 將 `AC <= 131°` 標成 class 0
4. 將 `132° <= AC < 152°` 直接排除
5. 將 `AC >= 152°` 標成 class 1
6. 只用剩下 61 位病人做 5-fold patient-level stratified split

舊的 `166°` 二分類 config 已移除，避免與本次文獻式切法混淆。

## 建議實驗

### 1. Baseline

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_binary_extreme.yaml
```

設定：

- 不做 augmentation
- 不做 balanced sampling
- 用原始 `14 / 47` 分布訓練

用途：觀察模型在最原始設定下是否已有可學訊號。

Baseline 已完成一次訓練：

```text
run: hybrid_mamba_attention_angle_extreme_binary_baseline_2026-05-05_15:49:43
```

結果摘要：

| 指標 | 結果 |
| --- | ---: |
| Accuracy | 0.7705 |
| Macro-F1 | 0.4350 |
| Macro recall | 0.5000 |
| Balanced accuracy | 0.5000 |

Total confusion matrix：

| true \ pred | Abnormal/emphysema-like | Normal-like |
| --- | ---: | ---: |
| Abnormal/emphysema-like | 0 | 14 |
| Normal-like | 0 | 47 |

此 baseline 的 accuracy 看起來有 `77%`，但這剛好等於全部猜 Normal-like 的 majority-class accuracy (`47/61`)。模型沒有抓到任何 abnormal/emphysema-like 病人，class 0 recall 為 `0/14`。因此這次 baseline 應解讀為 majority-class collapse，而不是有效分類結果。

造成此結果的主要原因是 baseline 沒有 augmentation、沒有 balanced sampling、也沒有 class weighting；每個 training fold 只有約 11-12 位 abnormal/emphysema-like 病人，Normal-like 病人約 37-38 位。未加權 cross entropy 很容易學到只猜多數類的捷徑。

### 2. 正式比較設定

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_binary_extreme.balanced_sampling.augmentation100.yaml
```

設定：

- training fold 內做 virtual train-time augmentation
- 每類補到 100 samples
- 每個 epoch 做 balanced sampling
- validation fold 永遠只用原始病人 CT，不使用 augmented views

用途：處理 `14 / 47` 的 class imbalance，作為主要報告結果。

## 評估重點

因為類別仍不平衡，建議不要只看 accuracy。報告時優先看：

- Macro-F1
- Balanced Accuracy
- class 0 recall
- class 0 precision
- confusion matrix

若模型 class 0 recall 太低，代表模型仍抓不到 emphysema-like 端點；若 class 0 precision 太低，代表模型把 normal-like 誤判成 abnormal 太多。

## 結論描述建議

可以向教授說明：

本次二分類不是任意選擇一個中位數門檻，而是根據 Topalovic et al. 的 Angle of Collapse 文獻，將 `AC <= 131°` 視為 emphysema-like 的 high-specificity abnormal group。由於 `AC > 131°` 不能直接代表 normal，故排除 `132-151°` 的中間灰區，只保留 `AC >= 152°` 作為 normal-like group，以降低 label noise 並測試 CT 模型是否能分辨兩端明確族群。
