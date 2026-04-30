# Angle 三分類 100/Class 少類平衡與 Data Augmentation 實驗報告

報告日期：2026-04-30

## 實驗目的

本次實驗針對 CT 角度三分類任務，依照教授建議測試「少類平衡」策略，並進一步加入 data augmentation，觀察是否能改善少數類別辨識。

三分類定義如下：

| Class | 定義 | 原始數量 |
| --- | --- | ---: |
| 0 | Emphysema/Abnormal (<=131 deg) | 14 |
| 1 | Intermediate (132-151 deg) | 5 |
| 2 | Normal (>=152 deg) | 47 |

原始資料明顯不平衡，Normal 類占 47/66。因此若只看 overall accuracy，模型只要偏向 Normal 就可能得到不低的分數。本報告主要同時觀察 Accuracy、Macro-F1、Balanced Accuracy、Macro Recall 與 confusion matrix。

## 100/Class 方法流程

使用設定檔：

```text
regression/config.angle_3class.balanced_sampling.augmentation100.yaml
```

核心策略：

| 步驟 | 說明 |
| --- | --- |
| 1. 使用原始資料 | 使用 `by_angle_all/` 原始 66 位病人資料，不使用已寫出的 augmented dataset。 |
| 2. Patient-level 5-fold | 每個 fold 以病人為單位切分，validation/test 不會含有 training 病人的 augmented copy。 |
| 3. Training fold 內 augmentation | 只在 training fold 裡做 virtual augmentation，不作用於 validation/test。 |
| 4. 三類都補到 100 | 每個 training fold 內把 class 0、class 1、class 2 都補到 100 samples。 |
| 5. 每 epoch 少類平衡抽樣 | DataLoader 每個 epoch 重新用 `BalancedClassSampler` 抽樣，確保每 epoch 三類數量一致。 |
| 6. 不使用 class weights | 因為已經做 balanced sampling，關閉 class weights，避免少數類被重複補償。 |

以 fold 1 為例，原始 training fold 分布約為：

```text
class 0 / class 1 / class 2 = 11 / 4 / 37
```

100/class virtual augmentation 後變成：

```text
class 0 / class 1 / class 2 = 100 / 100 / 100
```

因此每個 epoch 會看到 300 筆 balanced training samples。這些 augmented samples 是 train-time virtual copies，每次被取出時會隨機套用保守的 3D CT augmentation。

使用的 augmentation 強度：

| 參數 | 數值 |
| --- | ---: |
| Rotation | 5 deg |
| Translation | 0.03 |
| Scale | 0.97 - 1.03 |
| Intensity scale | 0.98 - 1.02 |
| Intensity shift | -0.05 - 0.05 |
| Noise std | 0.02 |

## 訓練設定

正式報告採用較穩定的 `amp: false` 版本，避免 AMP mixed precision 在 100/class 設定下產生 non-finite loss。

| 項目 | 設定 |
| --- | --- |
| Model | `hybrid_mamba_attention` |
| Task | `Angle_3class_classification` |
| Epochs | 160 |
| K-folds | 5 |
| Learning rate | 0.0001 |
| Weight decay | 0.001 |
| Loss | CrossEntropyLoss (`loss: auto`) |
| Class weights | none |
| Balanced sampling | true |
| Train batch | `swin_batch_size: 8` |
| Eval batch | `swin_eval_batch_size: 8` |
| AMP | false |
| Early stopping | enabled, patience 6, min_delta 0.005 |

執行指令：

```bash
cd /home/felix/Research/nnMamba/regression
conda run -n nnMamba python train.py --config config.angle_3class.balanced_sampling.augmentation100.yaml
```

正式 100/class stable run：

```text
hybrid_mamba_attention_2026-04-29_15:24:35
```

輸出位置：

```text
regression/figures/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_15:24:35/results.json
```

訓練時間：

```text
1658.263 seconds = 27.6 minutes = 0.4606 hours
```

## 100/Class 結果

### Fold-wise results

| Fold | Best Epoch | Accuracy | Macro-F1 | Macro Recall | Balanced Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 50 | 0.6429 | 0.2609 | 0.3000 | 0.3000 |
| 2 | 10 | 0.6923 | 0.4889 | 0.6296 | 0.6296 |
| 3 | 35 | 0.8462 | 0.5380 | 0.5556 | 0.5556 |
| 4 | 55 | 0.6923 | 0.4000 | 0.4074 | 0.4074 |
| 5 | 20 | 0.9231 | 0.6508 | 0.6667 | 0.6667 |

### Mean results

| Metric | Mean | Std |
| --- | ---: | ---: |
| Accuracy | 0.7593 | 0.1067 |
| Macro-F1 | 0.4677 | 0.1314 |
| Macro Precision | 0.4415 | 0.1349 |
| Macro Recall | 0.5119 | 0.1382 |
| Balanced Accuracy | 0.5119 | 0.1382 |

### Total confusion matrix

五個 folds 合併後，validation/test 總分布仍是原始 66 位病人：

```text
true support = class 0 / class 1 / class 2 = 14 / 5 / 47
```

總 confusion matrix：

```text
[[ 5,  0,  9],
 [ 1,  1,  3],
 [ 2,  1, 44]]
```

Per-class recall：

| Class | Recall | 解讀 |
| --- | ---: | --- |
| 0 | 0.3571 | 14 位中抓到 5 位，仍有 9 位被判成 Normal。 |
| 1 | 0.2000 | 5 位中抓到 1 位，Intermediate 仍是最困難的類別。 |
| 2 | 0.9362 | 47 位 Normal 中抓到 44 位，Normal 辨識非常好。 |

模型預測分布：

```text
predicted class 0 / class 1 / class 2 = 8 / 2 / 56
```

這代表 100/class stable run 雖然 overall accuracy 高，但模型仍明顯偏向預測 Normal。它改善了 Normal 的穩定度，但對 class 0 和 class 1 的辨識仍不足。

## 和其他方法比較

| Run | 方法 | Mean Accuracy | Mean Macro-F1 | Mean Balanced Accuracy | 備註 |
| --- | --- | ---: | ---: | ---: | --- |
| `13:02:22` | 純少類平衡 | 0.6396 | 0.2911 | 0.3437 | class 1 recall = 0.0000 |
| `13:25:54` | 原本實體增強 baseline | 0.6352 | 0.5042 | 0.5630 | 三類相對較平衡 |
| `14:06:11` | 20/class augmentation + 少類平衡 | 0.4132 | 0.3592 | 0.5037 | class 1 recall = 0.6000，但 Normal recall 下降 |
| `15:24:35` | 100/class augmentation + 少類平衡，AMP off | 0.7593 | 0.4677 | 0.5119 | Accuracy 最高，但 class 0/1 recall 偏低 |

補充：曾先跑過一版 `100/class + AMP on`：

```text
hybrid_mamba_attention_2026-04-29_14:17:25
```

該 run 指標較高：

```text
Accuracy = 0.7879
Macro-F1 = 0.5861
Balanced Accuracy = 0.6200
```

但訓練過程出現多次：

```text
Non-finite loss encountered under AMP; retrying the batch in full precision.
```

因此不建議把 AMP-on run 當作正式主結果。正式報告建議使用 `15:24:35` 的 AMP-off stable run，因為數值穩定性較好。

## 結論

100/class augmentation + 少類平衡確實讓模型取得最高 overall accuracy：

```text
Accuracy: 0.6352 baseline -> 0.7593 100/class stable
```

但從 balanced 指標來看，100/class stable run 沒有完全超過原本實體增強 baseline：

```text
Macro-F1: 0.5042 baseline -> 0.4677 100/class stable
Balanced Accuracy: 0.5630 baseline -> 0.5119 100/class stable
```

主要原因是 100/class stable run 對 Normal 類非常好，但 class 0 和 class 1 recall 還偏低：

```text
class 0 recall = 0.3571
class 1 recall = 0.2000
class 2 recall = 0.9362
```

因此，100/class 的結論應該這樣向教授報告：

```text
把三類都 virtual augmentation 到 100 後，overall accuracy 明顯提升，
代表模型對主要類別 Normal 學得更穩定；但少數類 class 0/1 的 recall
仍不足，所以如果目標是三類平均辨識，仍需要調整 augmentation 或 sampling 策略。
```

## 建議下一步

1. 保留 `100/class + amp:false` 作為穩定版主結果，因為它避免 AMP non-finite loss。
2. 不建議只報 Accuracy，應同時報 Macro-F1、Balanced Accuracy 和 confusion matrix。
3. 若教授重視少數類辨識，下一步應針對 class 0/1 改善，而不是單純繼續把所有類補更多。
4. 可嘗試 `class 1` 更強的策略，例如 class 1 專用 augmentation、調整 class 1 sampling ratio，或比較 50/class、75/class、100/class 的趨勢。
5. 若要追求整體分數，可再重跑數次不同 seed，確認 100/class 的結果是否穩定。

## 報告重點摘要

可對教授簡短說明：

```text
本次依照建議使用少類平衡，並把三個 angle class 在 training fold 內
用 virtual augmentation 補到 100/class。validation/test 維持原始病人，
避免資料洩漏。穩定版關閉 AMP 後，5-fold mean accuracy 達 0.7593，
高於原本實體增強 baseline 的 0.6352；但 Macro-F1 為 0.4677，
低於 baseline 的 0.5042。confusion matrix 顯示 Normal recall 很高
(0.9362)，但 class 0 recall 只有 0.3571、class 1 recall 只有 0.2000。
因此 100/class 對整體 accuracy 有幫助，但少數類辨識仍需進一步改善。
```
