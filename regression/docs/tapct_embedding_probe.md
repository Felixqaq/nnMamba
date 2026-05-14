# TAP-CT Frozen Embedding Probe 實驗紀錄

本實驗使用 TAP-CT 作為 frozen 3D CT foundation encoder，先把每位病人的 CT 轉成 patient-level embedding，再用小型 sklearn 分類器做 angle-derived classification。目的不是重新訓練大型 3D 模型，而是測試 pretrained CT representation 在 66 位小樣本資料上，是否比從零訓練 3D model 更穩定。

## 給老師報告版摘要

這個實驗的核心問題是：在只有 66 位病人的小樣本 3D CT 資料下，是否可以不要從零訓練大型 3D deep model，而是先用已經在大量 CT 上預訓練好的 foundation model 抽特徵，再用很小的分類器完成 angle 相關分類。

本實驗採用 TAP-CT 作為 3D CT foundation model。TAP-CT encoder 在本實驗中不更新權重，只負責把每位病人的 CT volume 轉成一個固定長度的 embedding。接著用 Logistic regression、Linear SVM、Ridge classifier 等小模型去做兩個分類任務：

1. `angle_3class`：把 angle 分成低角度、中間灰區、高角度三類。
2. `angle_binary_extreme`：拿掉中間灰區，只分低角度與高角度兩端。

這個方法的意義是 low-resource transfer learning。因為本專案資料量太小，直接訓練 3D CNN / Mamba / Transformer 很容易 overfit；而 pretrained CT embedding 可能已經包含肺部結構、密度、形狀等資訊，所以只需要訓練小分類器就能測試這些 representation 是否對本任務有用。

## 完整實驗流程

整體流程可以分成六步：

| 步驟 | 做什麼 | 目的 |
| --- | --- | --- |
| 1 | 讀取每位病人的 NIfTI CT volume | 取得 3D CT 影像輸入 |
| 2 | 讀取 angle label，建立 3-class 與 extreme binary label | 把連續 angle 轉成臨床可解釋分類任務 |
| 3 | 使用 TAP-CT image processor 做前處理 | 讓 CT 符合 TAP-CT 預訓練模型的輸入格式 |
| 4 | 沿 axial direction 切成 overlapping 12-slice windows | TAP-CT 一次吃 12 張 slice，因此 full CT 要用 sliding window 處理 |
| 5 | 對每個 window 抽 TAP-CT embedding，再做 mean / std / max pooling | 把多個 window-level features 聚合成 single patient-level embedding |
| 6 | 用 sklearn 小分類器做 5-fold stratified CV | 評估 frozen TAP-CT representation 對分類任務是否有效 |

資料與 label 設定：

| 任務 | 使用樣本 | Class 定義 | 樣本數 |
| --- | ---: | --- | --- |
| `angle_3class` | 66 | class 0: `AC <= 131°`；class 1: `132° <= AC < 152°`；class 2: `AC >= 152°` | 14 / 5 / 47 |
| `angle_binary_extreme` | 61 | class 0: `AC <= 131°`；class 1: `AC >= 152°`；排除中間 5 筆灰區 | 14 / 47 |

目前抽出的 embedding 維度：

| TAP-CT model | Hidden feature 聚合後維度 | 說明 |
| --- | ---: | --- |
| TAP-CT-S | 1152 | 較小模型，速度較快、結果在 extreme binary 較好 |
| TAP-CT-B | 2304 | 較大模型，參數較多、3-class balanced accuracy 較好 |

## 為什麼要做 3-class 和 extreme binary

`angle_3class` 比較接近完整臨床分級，因為它保留低角度、中間灰區、高角度三種狀態。但它最困難，原因是 class 1 只有 5 筆，而且中間角度本來就可能是生理或標註上的 transition zone，不像兩端那麼明確。

`angle_binary_extreme` 是把中間灰區排除，只比較 `AC <= 131°` 和 `AC >= 152°` 兩端。這個任務比較容易，也比較能測試 CT representation 是否抓到明確 phenotype 差異。它的限制是任務變簡化了，不能宣稱完整解決三分類或 COPD staging，但很適合當作 clinically cleaner endpoint。

## 名詞解釋

| 名詞 | 解釋 | 在本實驗中的意思 |
| --- | --- | --- |
| CT volume | 由多張 CT slice 組成的 3D 影像 | 每位病人的完整 3D chest CT |
| NIfTI | 醫學影像常用格式，副檔名常見 `.nii` 或 `.nii.gz` | 本專案 CT 檔案的主要格式 |
| LPS orientation | 醫學影像座標方向，代表 Left-Posterior-Superior | 讀取後統一方向，避免左右或上下方向不一致 |
| Foundation model | 在大量資料上預訓練，可轉移到其他任務的大模型 | TAP-CT 是 3D CT foundation model |
| Pretraining | 在大資料上先訓練模型學一般特徵 | TAP-CT 已先學過 CT representation |
| Frozen encoder | 固定 encoder 權重，不在本資料上更新 | 本實驗只用 TAP-CT 抽特徵，不 fine-tune TAP-CT |
| Embedding | 模型把影像轉成的數值向量 | 每位病人最後會得到一個 patient-level vector |
| Window-level embedding | 每個 12-slice window 抽出的特徵 | 因為 TAP-CT 一次處理 12 slices |
| Patient-level embedding | 把同一病人多個 window features 聚合後的特徵 | 最後拿去訓練分類器的輸入 |
| Sliding window | 用固定大小視窗沿著影像逐段切取 | 例如 12 slices 一段、stride 6，讓 full CT 都被看過 |
| Stride | sliding window 每次往前移動的距離 | stride 6 代表相鄰 windows 重疊 6 slices |
| Pooling | 把多個 feature 向量合成一個向量 | 本實驗用 mean / std / max pooling |
| Probe | 用簡單模型測試 frozen embedding 是否含有任務資訊 | Logistic / SVM / Ridge 都是 probe classifier |
| Logistic regression | 線性分類器，輸出類別機率 | 小樣本常用 baseline |
| Linear SVM | 找最大分類間隔的線性分類器 | 對高維 embedding 常有不錯表現 |
| Ridge classifier | 加 L2 regularization 的線性分類器 | 小樣本、高維特徵下較穩 |
| Ordinal classification | 考慮類別有順序的分類 | angle class 0 < class 1 < class 2 |
| Angle ridge threshold | 先回歸 angle，再用固定門檻轉分類 | 預測連續角度後套 `131° / 152°` |
| 5-fold CV | 把資料切成 5 份，輪流用 1 份測試、4 份訓練 | 減少單一次 train/test split 的偶然性 |
| Stratified CV | 切 fold 時盡量保持各類別比例 | 對 class imbalance 特別重要 |
| Class imbalance | 各類樣本數差距很大 | 本資料 3-class 是 14 / 5 / 47 |
| Balanced class weight | 訓練時提高少數類權重 | 避免模型只猜多數類 |
| Accuracy | 全部樣本中預測正確比例 | 在不平衡資料上容易被多數類影響 |
| Macro-F1 | 各類 F1 平均，每一類權重相同 | 更能反映少數類是否被學到 |
| Balanced accuracy | 各類 recall 平均 | 適合不平衡分類任務 |
| Confusion matrix | 真實類別與預測類別的交叉表 | 可看出模型把哪一類誤判成哪一類 |
| Recall | 某類真實樣本中被正確抓到的比例 | class 1 recall 可看中間類有沒有被抓到 |
| fp16 / float32 | 模型計算的浮點精度 | TAP-CT-B 用 fp16 曾出現 NaN，因此正式用 float32 |
| NaN | Not a Number，代表數值計算異常 | 若 embedding 有 NaN，分類器結果會不可靠 |

## Probe 分類器的直覺例子

這幾個方法都可以理解成：TAP-CT 已經把每位病人的 CT 轉成一個高維 embedding，分類器只是在這個 embedding 空間中找規則，把不同 angle class 分開。

為了方便理解，可以先假設 embedding 只有兩個特徵：

- 特徵 A：肺部整體密度 / emphysema-like pattern。
- 特徵 B：airway 或局部結構形狀。

實際 TAP-CT embedding 不是 2 維，而是 TAP-CT-S 的 1152 維或 TAP-CT-B 的 2304 維；但概念一樣，只是分界線變成高維空間中的分界面。

### Logistic regression

Logistic regression 會學一個線性分數，再把這個分數轉成機率。以 extreme binary 為例，它可能學到：

```text
score = 0.8 * 特徵A + 0.3 * 特徵B - 1.2
```

如果 `score` 很高，模型就認為病人比較像 `AC >= 152°`；如果 `score` 很低，就比較像 `AC <= 131°`。Logistic regression 的特色是它會輸出「屬於某類的機率」，例如：

```text
P(normal-like) = 0.82
P(abnormal-like) = 0.18
```

報告時可以這樣講：

> Logistic regression 是最基本、最容易解釋的線性分類器。它把 TAP-CT embedding 加權後轉成類別機率，所以可以看成是在問：這位病人的 CT representation 有多像某一類。

在本實驗中它適合作為 baseline，因為如果連簡單線性模型都有不錯結果，代表 TAP-CT embedding 本身已經含有和 angle label 相關的訊號。

### Linear SVM

Linear SVM 也會找線性分界面，但它的重點不是輸出機率，而是找一個「離兩邊樣本都盡量遠」的分界面，也就是最大化 margin。

舉例來說，如果 abnormal-like 和 normal-like 病人在 embedding 空間中大致可以分成左右兩群，Logistic regression 會找一條能讓機率預測好的線；Linear SVM 則會特別在意靠近邊界的病人，讓分界線和兩邊最接近的樣本保持最大距離。

簡化例子：

```text
abnormal-like:  x x x x       |
normal-like:                 |       o o o o
```

Linear SVM 會希望中間這條分界線不要貼太近任何一邊，因為 margin 越大，通常泛化能力越好。

報告時可以這樣講：

> Linear SVM 適合高維 embedding，因為它不是試著學複雜曲線，而是在高維特徵空間裡找一個最大間隔的線性分界面。對小樣本來說，這種限制反而比較不容易 overfit。

在本實驗中，TAP-CT-B + Linear SVM 是 3-class balanced accuracy 最好的 TAP-CT probe，表示 TAP-CT-B 的高維 embedding 可能有一些線性可分的 angle pattern。

### Ridge classifier

Ridge classifier 也是線性分類器，但它會加上 L2 regularization。L2 regularization 的意思是限制模型不要把某些特徵權重放得太大，避免模型過度依賴少數幾個 feature。

舉例來說，如果某個 embedding feature 剛好在訓練資料中和 label 很相關，但其實只是小樣本偶然現象，沒有 regularization 的模型可能會給它很大的權重。Ridge classifier 會懲罰過大的權重，讓模型比較平均、比較穩。

簡化例子：

```text
沒有 regularization:
score = 8.5 * 特徵A + 0.1 * 特徵B

Ridge classifier:
score = 1.6 * 特徵A + 0.8 * 特徵B
```

第一個模型可能太依賴特徵 A；第二個模型比較保守。這在本專案很重要，因為樣本只有 66 筆，但 embedding 維度超過 1000 維，特徵數遠大於樣本數，很容易 overfit。

報告時可以這樣講：

> Ridge classifier 的重點是穩定。它一樣用線性分界面分類，但透過 L2 regularization 限制權重不要太極端，因此特別適合小樣本、高維 embedding 的設定。

在本實驗中，TAP-CT-S + Ridge classifier 是 extreme binary 最好的組合，表示在兩端明確族群中，較保守的線性分類器能穩定利用 TAP-CT-S embedding 的訊號。

### Ordinal classification

Ordinal classification 的重點是：類別之間不是完全獨立，而是有順序。以本實驗的 angle 3-class 來說：

```text
class 0: AC <= 131°
class 1: 132° <= AC < 152°
class 2: AC >= 152°
```

這三類不是像「貓、狗、車」那種沒有順序的類別，而是 angle 從小到大的三個區間，所以有：

```text
class 0 < class 1 < class 2
```

一般 3-class classifier 可能把 class 0、1、2 當成三個彼此獨立的類別。但 ordinal classification 會利用這個順序資訊，把問題拆成兩個 threshold decision：

1. 這個病人是否大於 class 0？也就是 `AC > 131°` 嗎？
2. 這個病人是否達到 class 2？也就是 `AC >= 152°` 嗎？

舉例來說，如果模型輸出：

```text
P(AC > 131°) = 0.90
P(AC >= 152°) = 0.30
```

代表它覺得這位病人很可能不是低角度 class 0，但也還不像高角度 class 2，所以最後會判成中間 class 1。

如果模型輸出：

```text
P(AC > 131°) = 0.95
P(AC >= 152°) = 0.88
```

代表它覺得這位病人已經超過第一個門檻，也超過第二個門檻，所以判成 class 2。

報告時可以這樣講：

> Ordinal classification 把 angle class 視為有順序的分級，而不是三個無關類別。它比較符合 angle 本身由小到大的連續性，因此理論上比普通三分類更合理。

在本實驗中，ordinal logistic probe 是用兩個 logistic regression 來模擬兩個 angle threshold。它的目標是讓模型更尊重 `class 0 < class 1 < class 2` 的順序。

### Angle ridge threshold

Angle ridge threshold 是另一種利用 angle 順序的方法。它不是直接訓練分類器預測 class，而是先訓練 Ridge regression 預測連續 angle，然後再用固定門檻把預測角度轉成 class。

流程是：

```text
TAP-CT embedding -> Ridge regression -> predicted angle -> 131° / 152° threshold -> class
```

舉例來說，如果模型預測某位病人的 angle 是：

```text
predicted angle = 124°
```

因為 `124° <= 131°`，所以轉成：

```text
class 0
```

如果預測：

```text
predicted angle = 140°
```

因為 `132° <= 140° < 152°`，所以轉成：

```text
class 1
```

如果預測：

```text
predicted angle = 165°
```

因為 `165° >= 152°`，所以轉成：

```text
class 2
```

這個方法的好處是它最直接保留 angle 的連續性。模型先學「角度大概是多少」，最後才依照文獻或實驗定義的門檻轉成分類。

但它也有缺點：如果 regression 本身誤差大，分類會受到影響。尤其在門檻附近很敏感，例如真實 angle 是 `151°`，模型預測成 `153°`，就會從 class 1 變成 class 2。

報告時可以這樣講：

> Angle ridge threshold 是 regression-to-classification 的方法。它先預測連續 angle，再用 `131° / 152°` 這兩個 clinically defined thresholds 轉成三分類。這個方法可解釋性很好，但會受連續角度預測誤差影響。

在本實驗中，它用來檢查 TAP-CT embedding 是否能支撐連續 angle estimation。如果 regression 預測不夠準，最後 threshold 後的 3-class 表現就會受限。

### 五種方法差異總結

| 方法 | 主要想法 | 優點 | 在本實驗中的角色 |
| --- | --- | --- | --- |
| Logistic regression | 線性分數轉成類別機率 | 好解釋、可看機率、baseline 清楚 | 檢查 embedding 是否已有基本線性訊號 |
| Linear SVM | 找最大 margin 的線性分界面 | 高維特徵常有效，對邊界樣本敏感 | 3-class 中 TAP-CT-B 表現最好 |
| Ridge classifier | 線性分類 + L2 regularization | 權重較穩，小樣本高維較不易 overfit | extreme binary 中 TAP-CT-S 表現最好 |
| Ordinal classification | 把 class 視為有順序的 threshold decisions | 符合 angle 分級的順序性 | 測試 `class 0 < class 1 < class 2` 是否有幫助 |
| Angle ridge threshold | 先預測連續 angle，再用固定門檻轉 class | 可解釋性高，連結 regression 與 classification | 測試 embedding 是否能支撐 angle regression |

## 報告時可以怎麼講結果

第一個重點是 extreme binary 表現最好。TAP-CT-S + Ridge classifier 達到：

- Accuracy: `0.8872`
- Macro-F1: `0.7971`
- Balanced Accuracy: `0.8111`

這代表在排除中間灰區後，TAP-CT 的 frozen CT representation 已經能分辨 angle 兩端的 phenotype，而且不需要訓練大型 3D model。

第二個重點是 3-class 仍然困難。TAP-CT-B + Linear SVM 的 3-class balanced accuracy 達到 `0.6363`，是目前 TAP-CT probe 中 3-class 較好的結果，但 Macro-F1 只有 `0.5202`。主要問題仍是中間類只有 5 筆，且中間角度本身是灰區，因此模型容易把它分到兩端。

第三個重點是這個實驗不是要完全取代原本 Hybrid Mamba-Attention，而是提供一個更適合小樣本的 foundation-model baseline。後續可以把 TAP-CT embedding 和原本模型的 feature 做 late fusion，測試 pretrained CT representation 是否能補足現有模型。

## 目前限制

1. 樣本數只有 66，尤其 3-class 的 class 1 只有 5 筆，因此 cross-validation variance 會很大。
2. TAP-CT encoder 目前完全 frozen，沒有做 fine-tuning，所以只測試 representation transfer，不測試 task-specific adaptation。
3. 目前 probe 的 hyperparameter 還沒有完整 grid search，例如 Logistic / SVM 的 `C`、Ridge 的 `alpha`。
4. Extreme binary 排除灰區後結果較好，但任務也變簡化，不能直接等同完整三分類表現。
5. 目前只用 CT image embedding，還沒有結合 PFT metadata 或原本 Hybrid Mamba-Attention feature。

## 環境

本實驗使用獨立 conda 環境 `tapct`，避免影響原本 `nnMamba` 訓練環境。

```bash
conda env create -f regression/environment.tapct.yml
conda activate tapct
```

第一次實驗使用的環境包含 Python 3.11、PyTorch 2.8.0 + CUDA 12.8、Transformers、SimpleITK、MONAI、scikit-learn，以及 matplotlib。

## YAML 一鍵訓練

最簡單的入口是使用 `regression/config.tapct_embedding_probe.yaml`。預設流程會使用 TAP-CT-B、`float32`、既有資料路徑，先抽 embedding，再訓練所有 sklearn probe，最後自動產生 `results.json`、`predictions.csv` 和 figures。

```bash
conda activate tapct
python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml
```

如果 embeddings 已經抽好，只想重新訓練 probe 和產圖：

```bash
python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml \
  --probe-only
```

如果只想抽 embeddings，不訓練 probe：

```bash
python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml \
  --extract-only
```

如果要先檢查實際會執行哪些 command：

```bash
python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml \
  --dry-run
```

常改的 YAML 欄位：

- `tapct.model_id`：`fomofo/tap-ct-b-3d` 或 `fomofo/tap-ct-s-3d`
- `tapct.dtype`：TAP-CT-B 建議 `float32`
- `embedding.output_dir`：embedding 輸出位置
- `embedding.force`：是否強制重抽 case-level embeddings
- `probe.output_dir`：probe 結果與 figures 輸出位置
- `probe.target`：`all`、`angle_3class`、`angle_binary_extreme`
- `probe.model`：`all`、`logistic`、`linear_svm`、`ridge_classifier`、`ordinal_logistic`、`angle_ridge_threshold`
- `probe.plots`：是否在訓練後自動產圖

## TAP-CT + Hybrid Mamba Late Fusion

Late fusion 是用來測試 pretrained TAP-CT representation 能不能補上原本
Hybrid Mamba-Attention 影像分支缺少的全局 CT 訊號。它不是把 TAP-CT
embedding 當成新的 3D image input，也不是 fine-tune TAP-CT encoder；它保留
原本 nnMamba/Hybrid 的 CT 影像分支，另外讀取已抽好的 frozen TAP-CT
patient-level embedding，最後把兩邊 feature 串接後交給同一個 classification head。

### 核心架構

```text
同一個 patient_id
├─ 原始 CT volume ──> Hybrid Mamba-Attention image branch ──> CT feature
└─ TAP-CT-B frozen patient embedding ──> LayerNorm + Linear projection ──> TAP feature

concat(CT feature, TAP feature) ──> MLP classification head ──> angle class logits
```

對應模型名稱：

```yaml
model:
  name: hybrid_mamba_tapct_fusion
```

對應實作檔案：

- `regression/networks/hybrid_mamba_tapct_fusion_regressor.py`
- `regression/config.angle_3class.tapct_late_fusion.augmentation100.yaml`
- `regression/data/loader.py`
- `regression/data/dataset.py`
- `regression/core/trainer.py`

### 從頭到尾的流程

#### 1. 先在 TAP-CT 環境抽 frozen TAP-CT-B-3D embedding

Late fusion 的 `B-3D aug100` 使用的是：

```yaml
tapct:
  model_id: fomofo/tap-ct-b-3d
  dtype: float32

embedding:
  output_dir: regression/embeddings/tapct_b_3d
  depth_window: 12
  depth_stride: 6
  pooling: mean_std_max
```

執行方式：

```bash
cd /home/felix/Research/nnMamba
conda run -n tapct python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml \
  --extract-only
```

輸出重點：

```text
regression/embeddings/tapct_b_3d/features.npz
regression/embeddings/tapct_b_3d/cases/*.npz
regression/embeddings/tapct_b_3d/metadata.csv
regression/embeddings/tapct_b_3d/extraction_config.json
```

`features.npz` 至少包含：

- `features`：shape 為 `66 x 2304`
- `patient_ids`：和 `features` 每一列對齊的病人 ID
- `angles`
- `angle_3class`
- `angle_binary_extreme`

`2304` 維的來源是 TAP-CT-B 每個 window 的 embedding 維度做
`mean + std + max` pooling 後串接而成。這一步只做 feature extraction，
不會訓練 TAP-CT。

#### 2. 用 nnMamba 環境跑 late fusion 訓練

`B-3D aug100` 的訓練設定是：

```bash
cd /home/felix/Research/nnMamba/regression

conda run -n nnMamba python train.py \
  --config config.angle_3class.tapct_late_fusion.augmentation100.yaml
```

主要 YAML 欄位：

```yaml
experiment:
  name: TAP-CT late fusion + aug100/class

model:
  name: hybrid_mamba_tapct_fusion
  num_classes: 3
  tapct_embedding_dim: 2304
  fusion_projection_dim: 128

data:
  target_mode: angle_3class
  tapct_features: ./embeddings/tapct_b_3d/features.npz
  balanced_sampling: true
  augmentation:
    enabled: true
    target_per_class: 100
    class_indices: [0, 1, 2]
```

#### 3. Data loader 如何把 embedding 抓進來

`RegressionLoaderHelper` 讀到 `data.tapct_features` 後會載入
`features.npz`，檢查：

- 檔案是否存在
- 是否同時包含 `features` 和 `patient_ids`
- `features.shape[0]` 是否等於 `patient_ids` 數量
- 是否有重複 patient ID
- 每個 manifest record 的 patient ID 是否都能在 TAP-CT features 找到

接著建立：

```python
tapct_embeddings[patient_id] = features[row_index]
```

`AngleRegressionDataset` 每次讀一位病人時會同時放入：

- `ct`：原始 CT volume，給 Hybrid Mamba-Attention image branch
- `label` / `target`：angle 3-class label
- `tapct_embedding`：同一個 patient ID 對應的 frozen TAP-CT-B embedding

`ToTensor` 會把 `tapct_embedding` 轉成 `torch.float32`。

#### 4. Aug100 在這裡代表什麼

`aug100/class` 是 train-fold 內的 virtual augmentation，不是重抽 TAP-CT
embedding。流程是：

1. 先做 patient-level stratified 5-fold split。
2. validation fold 只保留原始病人，不放 augmented copy。
3. train fold 依照 class 補到每類最多 100 個訓練 view。
4. CT branch 的 augmented copy 會套用 random affine / intensity / noise。
5. TAP-CT embedding 分支仍使用同一個病人的 frozen patient-level embedding。

也就是說，augmentation 只作用在 Hybrid CT image branch；TAP-CT-B embedding
是先前離線抽好的穩定 patient-level feature。

#### 5. Trainer 如何餵給模型

在 training/evaluation 時，trainer 看到 batch 裡有 `tapct_embedding`，就不再
只傳 CT tensor，而是傳一個 dict：

```python
{
  "ct": ct_tensor,
  "tapct_embedding": tapct_embedding_tensor,
}
```

`HybridMambaTapctFusionRegressor` 內部做：

1. `image_encoder.forward_features(ct)` 取得 Hybrid image feature。
2. `embedding_branch(tapct_embedding)` 做 `LayerNorm -> Linear(2304, 128) -> GELU -> Dropout`。
3. `torch.cat([image_features, embedding_features], dim=1)` 串接。
4. MLP head 輸出 3-class logits。
5. 用 cross entropy loss 訓練，early stopping 依 validation Macro-F1。

#### 6. 輸出與目前已跑完的結果

訓練完成後會沿用現有 regression pipeline 自動輸出：

- 每 fold 的 loss、accuracy、Macro-F1、Balanced Accuracy、Macro Recall 曲線
- 每 fold 的 confusion matrix
- 所有 folds 合併後的 `total_confusion_matrix.png`
- fold-wise `metric_boxplot.png`
- `results.json` 與 `fold*_predictions.json`

目前已完成的 run：

```text
regression/figures/Angle_3class_classification/
  hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20/
```

結果摘要：

| Method | Accuracy | Macro-F1 | Balanced Acc |
| --- | ---: | ---: | ---: |
| Late fusion B-3D aug100 | 0.82088 | 0.61406 | 0.63259 |

Total confusion matrix：

```text
[[11, 0, 3],
 [ 0, 1, 4],
 [ 3, 2, 42]]
```

重要限制：

- TAP-CT encoder 是 frozen，不會在 nnMamba 環境中訓練。
- 原本 MLP late fusion 不是 TAP-CT ABMIL；TAP-CT-only ABMIL 是另一個只吃 TAP-CT feature 的 head。
- 這不是把 2304 維 embedding reshape 成 3D image。
- Augmentation 不會改變 TAP-CT embedding，只會改變 Hybrid CT image branch 的輸入。
- `aug300` 那次中途停止在 fold3，所以沒有完整 `results.json` 和 total CM；完整比較請看 `aug100` run。

其他可直接訓練的 late-fusion config：

```bash
cd /home/felix/Research/nnMamba/regression

conda run -n nnMamba python train.py \
  --config config.angle_3class.tapct_late_fusion.yaml

conda run -n nnMamba python train.py \
  --config config.angle_binary_extreme.tapct_late_fusion.yaml
```

### ABMIL final head 版本

如果想把最後分類器從單純 MLP 改成 ABMIL，可以使用新的 hybrid fusion head：

```yaml
model:
  name: hybrid_mamba_tapct_abmil_fusion
  tapct_embedding_dim: 2304
  fusion_projection_dim: 128
  tapct_attention_dim: 128
  tapct_gated_attention: true
```

對應 config：

```bash
cd /home/felix/Research/nnMamba/regression

conda run -n nnMamba python train.py \
  --config config.angle_3class.tapct_abmil_fusion.augmentation100.yaml
```

這個版本不是把幾萬個 TAP-CT patch embeddings 丟進 ABMIL；它使用的是已經
pool 好的 patient-level TAP-CT embedding。差別在最後 fusion head：

```text
CT volume ──> Hybrid Mamba-Attention ──> CT feature ──> Linear projection ─┐
                                                                          ├─> gated ABMIL attention ─> classifier
TAP-CT-B frozen patient embedding ─────> TAP feature ──> Linear projection ┘
```

也就是把 `CT feature` 和 `TAP feature` 視為兩個 modality instances。ABMIL
attention 會學習每個病人在這兩個來源上的權重，再用加權後的 pooled feature
做三分類。這保留 late fusion 的優點，同時讓模型不必固定把 CT 和 TAP-CT
concat 後交給 MLP，而是可以依 case 動態調整哪個來源比較重要。

對應實作檔案：

- `regression/networks/hybrid_mamba_tapct_abmil_fusion_regressor.py`
- `regression/config.angle_3class.tapct_abmil_fusion.augmentation100.yaml`
- `regression/models.py`
- `regression/core/trainer.py`
- `regression/data/loader.py`

## Embedding 抽取流程

TAP-CT 接受的 3D crop shape 為 `(B, 1, 12, 224, 224)`。因此 extractor 的流程是：

1. 使用 SimpleITK 讀取每個 NIfTI CT。
2. 將 CT 方向轉成 LPS。
3. 套用 TAP-CT Hugging Face image processor。
4. 沿 axial direction 切出 overlapping 12-slice windows。
5. 對每個 window 抽 embedding。
6. 用 mean / std / max pooling 聚合成單一 patient-level vector。

### 每一步為什麼要這樣做

#### 1. 使用 SimpleITK 讀取每個 NIfTI CT

NIfTI 是醫學影像常用格式，不只包含 voxel intensity，還包含 spacing、orientation、origin 等 metadata。一般影像讀取工具通常只把它當成普通 3D array，但醫學影像不能只看 array，因為同樣的 `(x, y, z)` index 在不同檔案中可能代表不同身體方向或實際距離。

因此使用 SimpleITK 的原因是它比較適合處理醫學影像資料。它可以保留 CT 的空間資訊，後續才能正確做方向轉換與模型前處理。

舉例來說，如果一位病人的 CT slice thickness 是 1 mm，另一位是 5 mm，這兩個 volume 在 array 上都可能看起來是很多張切片，但實際空間厚度不同。SimpleITK 能讀到這些 metadata，讓我們知道這不是普通圖片堆疊。

#### 2. 將 CT 方向轉成 LPS

不同醫院、掃描儀或轉檔流程輸出的 CT orientation 可能不同。有些可能是 RAS，有些可能是 LPS。如果不統一方向，同一個 anatomical location 在不同病人資料中可能會被模型看成不同位置。

轉成 LPS 的目的，是讓所有 CT 使用同一個空間座標系。這樣模型看到的左右、前後、上下方向比較一致，減少非疾病因素造成的差異。

舉例來說，如果沒有統一方向，模型可能在某些病人看到「左肺」其實是 array 的右側，在另一些病人看到「左肺」卻是 array 的左側。這會讓模型學到混亂的 spatial pattern，尤其 3D CT 模型很依賴空間位置。

#### 3. 套用 TAP-CT Hugging Face image processor

TAP-CT 是預訓練模型，它在 pretraining 時使用固定的輸入格式與 normalization。若我們把原始 CT 直接丟進模型，資料分布可能和它訓練時看到的不一致，embedding 會變得不穩。

Hugging Face image processor 的作用是把我們的 CT window 轉成 TAP-CT 預期的格式，例如調整 intensity、resize 到模型需要的 spatial size、轉 tensor、補 channel dimension 等。

舉例來說，TAP-CT 預期輸入可能是 `(1, 12, 224, 224)` 的 3D patch。如果原始 CT 是 `(512, 512, 300)`，且 intensity range 是 Hounsfield unit，直接輸入會尺寸不符，也可能 normalization 不對。processor 就像是把資料翻譯成 TAP-CT 看得懂的語言。

#### 4. 沿 axial direction 切出 overlapping 12-slice windows

TAP-CT 一次接受 12 張 axial slices，不是直接吃完整 CT volume。可是每位病人的 CT 通常有數十到數百張 slices，所以不能只取其中 12 張，否則會漏掉很多肺部區域。

因此我們用 sliding window 沿 axial direction 切出多個 12-slice windows。`overlapping` 的意思是相鄰 windows 會有重疊，例如 window size 是 12、stride 是 6，第一個 window 看 slice 1-12，第二個看 slice 7-18，第三個看 slice 13-24。

這樣做有兩個好處：

1. 讓整個 CT volume 都有機會被 TAP-CT 看過。
2. 重疊可以減少邊界問題，避免重要結構剛好被切在兩個 window 中間而被忽略。

舉例來說，如果某個和 emphysema 或 airway collapse 相關的影像特徵剛好出現在 slice 10-16，只切非重疊 windows 時可能會被拆成兩段，模型每次只看到一半。使用 overlapping windows 後，至少有一個 window 比較完整地涵蓋這個區域。

#### 5. 對每個 window 抽 embedding

每個 12-slice window 都代表 CT 的一小段 3D 區域。TAP-CT encoder 會把這段影像轉成 embedding，也就是一個高維數值向量。這個向量不是人眼可直接解讀的影像，而是模型學到的 representation，可能包含肺部密度、紋理、形狀、airway、血管結構等資訊。

這一步的重點是：我們不在 66 筆資料上訓練大型 3D model，而是借用 TAP-CT 已經學好的 CT feature extractor。這比較適合小樣本任務，因為真正需要訓練的只剩後面的小分類器。

舉例來說，可以把 TAP-CT 想成已經讀過大量 CT 的醫學影像特徵抽取器。它看到一段 12-slice CT 後，不直接給出我們的 class label，而是給出「這段 CT 的影像特徵摘要」。後面的 Logistic regression / SVM 再學這些摘要和 angle label 的關係。

#### 6. 用 mean / std / max pooling 聚合成單一 patient-level vector

一位病人會被切成很多 windows，因此會得到很多 window-level embeddings。但我們的 label 是 patient-level label，也就是每位病人只有一個 angle class。分類器最後也需要每位病人一個固定長度向量，所以要把多個 window embeddings 聚合成一個 patient-level embedding。

本實驗使用 mean / std / max pooling：

- `mean pooling`：表示整體平均特徵，回答「這位病人的 CT 整體長什麼樣」。
- `std pooling`：表示不同位置的變異程度，回答「這位病人的 CT 各區域差異大不大」。
- `max pooling`：保留最強的局部訊號，回答「是否有某個區域出現很明顯的特徵」。

舉例來說，假設某位病人大部分肺部看起來接近正常，但有局部區域有明顯 abnormal pattern。只用 mean pooling 時，這個局部訊號可能被整體平均掉；加入 max pooling 後，模型仍有機會保留「某個 window 很異常」這個資訊。std pooling 則可以反映這位病人的 CT 是否呈現很不均勻的局部變化。

簡單說，這個聚合策略是在回答三個層次的問題：整體平均狀態、區域差異程度、最強局部異常。最後把這三種資訊串起來，形成單一 patient-level vector，再交給小分類器做 3-class 或 extreme binary。

#### 流程小例子

假設某位病人的 CT 有 120 張 axial slices，window size 是 12，stride 是 6：

1. 會切出大約 19 個 overlapping windows。
2. TAP-CT 會對每個 window 抽出一個 embedding。
3. 如果 TAP-CT-S 的單一 window embedding 是 384 維，19 個 windows 就會得到 19 個 384 維向量。
4. 對這 19 個向量做 mean / std / max pooling，會得到 `384 x 3 = 1152` 維 patient-level embedding。
5. 最後用這個 1152 維向量代表該病人的整體 CT，再訓練小分類器預測 angle class。

TAP-CT-S：

```bash
python regression/scripts/extract_tapct_embeddings.py \
  --model-id fomofo/tap-ct-s-3d \
  --sw-batch-size 4 \
  --output-dir regression/embeddings/tapct_s_3d
```

TAP-CT-B：

```bash
python regression/scripts/extract_tapct_embeddings.py \
  --model-id fomofo/tap-ct-b-3d \
  --sw-batch-size 1 \
  --dtype float32 \
  --output-dir regression/embeddings/tapct_b_3d
```

TAP-CT-B 建議使用 `float32`。第一次 smoke run 使用 fp16 時，部分 CT window 產生 NaN，因此正式 TAP-CT-B embedding 改用 `float32` 較穩定。

## Probe 訓練

目前 probe script 會在儲存 `results.json` 和 `predictions.csv` 後，自動產生報告用 figures。

TAP-CT-S：

```bash
python regression/scripts/train_embedding_probe.py \
  --features regression/embeddings/tapct_s_3d/features.npz
```

TAP-CT-B：

```bash
python regression/scripts/train_embedding_probe.py \
  --features regression/embeddings/tapct_b_3d/features.npz \
  --output-dir regression/figures/TAPCT_embedding_probes/tapct_b_3d_2026-05-13_14-06-00
```

如果只想跑數字、不想自動產圖，可以加上：

```bash
--no-plots
```

## Figure 產生

現在圖會在 probe 訓練時自動產生。若要從既有 `results.json` 重新產圖，可執行：

```bash
python regression/scripts/plot_embedding_probe_results.py \
  regression/figures/TAPCT_embedding_probes/2026-05-13_13-59-34/results.json

python regression/scripts/plot_embedding_probe_results.py \
  regression/figures/TAPCT_embedding_probes/tapct_b_3d_2026-05-13_14-06-00/results.json
```

每個 run 目錄會產生：

- `probe_metric_overview.png`
- `{target}_metric_comparison.png`
- `{target}_{probe}_confusion_matrix.png`
- `{target}_{probe}_class_recall.png`

例如：

- `angle_3class_linear_svm_confusion_matrix.png`
- `angle_3class_linear_svm_class_recall.png`
- `angle_binary_extreme_ridge_classifier_confusion_matrix.png`
- `angle_binary_extreme_metric_comparison.png`

## Probe 方法

目前 probe script 評估以下小分類器：

- Logistic regression with balanced class weights
- Linear SVM with balanced class weights
- Ridge classifier with balanced class weights
- Angle 3-class 的 ordinal two-threshold logistic probe
- Ridge angle regression，再用固定 `131° / 152°` threshold 轉成 3-class

這些 probe 都只訓練很小的 sklearn model，不更新 TAP-CT encoder 權重。

## 第一輪結果

TAP-CT-S 輸出：
`regression/figures/TAPCT_embedding_probes/2026-05-13_13-59-34/results.json`

| Target | Probe | Accuracy | Macro-F1 | Balanced Acc |
| --- | --- | ---: | ---: | ---: |
| Angle 3-class | Logistic | 0.7604 | 0.4870 | 0.5126 |
| Angle 3-class | Linear SVM | 0.6363 | 0.4472 | 0.4852 |
| Angle 3-class | Ridge classifier | 0.7901 | 0.5035 | 0.5267 |
| Angle 3-class | Ordinal logistic | 0.7747 | 0.4997 | 0.5193 |
| Angle 3-class | Angle ridge threshold | 0.6670 | 0.4567 | 0.5163 |
| Extreme binary | Logistic | 0.8705 | 0.7798 | 0.8000 |
| Extreme binary | Linear SVM | 0.7539 | 0.7027 | 0.7689 |
| Extreme binary | Ridge classifier | 0.8872 | 0.7971 | 0.8111 |

TAP-CT-B 輸出：
`regression/figures/TAPCT_embedding_probes/tapct_b_3d_2026-05-13_14-06-00/results.json`

| Target | Probe | Accuracy | Macro-F1 | Balanced Acc |
| --- | --- | ---: | ---: | ---: |
| Angle 3-class | Logistic | 0.7890 | 0.4776 | 0.5119 |
| Angle 3-class | Linear SVM | 0.6363 | 0.5202 | 0.6363 |
| Angle 3-class | Ridge classifier | 0.7582 | 0.4420 | 0.4711 |
| Angle 3-class | Ordinal logistic | 0.7582 | 0.4723 | 0.4970 |
| Angle 3-class | Angle ridge threshold | 0.6209 | 0.4397 | 0.4200 |
| Extreme binary | Logistic | 0.8538 | 0.7389 | 0.7678 |
| Extreme binary | Linear SVM | 0.7359 | 0.7154 | 0.8289 |
| Extreme binary | Ridge classifier | 0.8372 | 0.7042 | 0.7178 |

## 結果解讀

TAP-CT-S + Ridge classifier 是目前 frozen-feature extreme binary probe 中 Accuracy 和 Macro-F1 最好的組合：

- Accuracy: `0.8872`
- Macro-F1: `0.7971`
- Balanced Accuracy: `0.8111`

這個結果接近原本 end-to-end extreme binary deep model，而且不需要訓練 3D network、不需要 augmentation。這代表 TAP-CT 的 pretrained CT representation 對兩端明確 angle phenotype 有可用訊號。

TAP-CT-B + Linear SVM 是目前 3-class probe 中 Balanced Accuracy 最好的組合：

- Accuracy: `0.6363`
- Macro-F1: `0.5202`
- Balanced Accuracy: `0.6363`

其 combined confusion matrix 為：

```text
[[13, 1, 0],
 [ 1, 2, 2],
 [11, 9, 27]]
```

這個設定可以抓到 class 1 intermediate 的 `2/5`，和目前較好的 balanced 3-class runs 接近。不過它明顯過度預測低角度類別，導致 Normal recall 下降。因此 TAP-CT-B + Linear SVM 比較適合作為 high-recall / balanced baseline，還不適合作為最終分類器。

## 目前結論

第一輪 TAP-CT embedding probe 顯示：

1. Foundation embedding 對 extreme binary endpoint task 很有幫助。
2. 3-class 的 intermediate group 仍然是主要瓶頸。
3. 單純使用 frozen embedding 還沒有完全解決 class 1 樣本太少與灰區 label ambiguity 的問題。
4. 下一步不應該只增加 augmentation，而應該針對 probe tuning 或 hybrid strategy。

建議下一步：

- 對 Logistic / SVM / Ridge 做 `C`、`alpha` grid search。
- 對 extreme binary 做 threshold calibration，調整 sensitivity / specificity trade-off。
- 對 3-class 做 calibrated ordinal threshold，而不是固定 `0.5` threshold。
- 嘗試 TAP-CT embedding + 現有 hybrid Mamba feature 的 late fusion。
- 嘗試 TAP-CT embedding + class 1 專用 decision rule，觀察 intermediate recall 是否能穩定提升。

## ABMIL 分類頭

新增 paper-style frozen TAP-CT ABMIL head：

```bash
cd regression
conda activate nnMamba
python train.py --config config.angle_3class.tapct_abmil.yaml
```

目前 repo 內的 TAP-CT feature bundle 只保存每位病人的 pooled/CLS-like scan embedding，
所以 `config.angle_3class.tapct_abmil.yaml` 預設是 single-instance ABMIL：

```yaml
model:
  name: tapct_abmil
data:
  tapct_features: ./embeddings/tapct_s_3d/features.npz
  tapct_feature_key: features
  load_ct: false
```

如果要跑真正的多 instance ABMIL，先在 TAP-CT 環境重新抽 feature 並保存 window-level embeddings：

```bash
conda activate tapct
python regression/scripts/run_tapct_embedding_probe.py \
  --config regression/config.tapct_embedding_probe.yaml \
  --extract-only \
  --force
```

並在 extraction config 內把 `embedding.save_window_embeddings` 設成 `true`。之後把 ABMIL config 改成：

```yaml
data:
  tapct_feature_key: window_embeddings
```

注意：不要直接把五萬多個 patch tokens 硬丟進 ABMIL。這個實作優先支援 pooled/CLS feature 或 window-level instance bags，避免論文提到的 needle-in-a-haystack 問題。
