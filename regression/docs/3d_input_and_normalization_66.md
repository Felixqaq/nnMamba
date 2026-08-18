# 三維影像輸入、正規化做法、模型入口與 DicomToNii 輸出樣式（66 例）

本文件補齊論文第三章「三維影像輸入」小節所需素材：原始 CT 規格（矩陣大小、體素間距、切片張數、掃描範圍）、正規化（強度與幾何）流程與公式、**深度學習模型的程式入口與其影像正規化管線**、使用工具與版本、以及 `DicomToNii` 的實際輸出樣式。

影像規格數值由 `DicomToNii/dicom_metadata.csv` 與轉檔後 NIfTI 檔頭實測，模型端數值皆由 `nnMamba/regression/` 之程式碼與 config 實查，非估計值。

**資料對象統一為 66 例**：33 abnormal（COPD，醫院dataset）+ 21 normal（20260116）+ 12 外部 normal（20260421）= 33 Abnormal + 33 Normal。此即模型訓練所用之 `classification/datasets/normal_v_abnormal_66/` 世代（manifest 實查：`total = 66`、`unique_patients = 66`、`class_counts = {Abnormal: 33, Normal: 33}`）。

---

## 1. 原始 CT 影像規格（66 例）

### 1.1 全體實測統計

| 項目 | 最小 | 最大 | 平均 | 中位數 | IQR | 備註 |
|---|---:|---:|---:|---:|---|---|
| 軸向矩陣 | 512 × 512 | 1024 × 1024 | — | — | — | 512² 65 例、1024² 1 例（`5925853`） |
| 切片張數 (Z) | 63 | 386 | 133.0 | 101 | 95–112 | — |
| 像素間距 (in-plane, mm) | 0.435 | 0.963 | 0.627 | 0.609 | 0.565–0.652 | 各例 x = y |
| 切片間距 (mm) | 1.0 | 5.0 | 2.79 | 3.0 | 3.0–3.0 | 3.0 mm 52 例、1.0 mm 10 例、5.0 mm 3 例、2.965 mm 1 例 |
| 切片厚度 SliceThickness (mm) | 1.0 | 5.0 | — | 3.0 | — | 3.0 mm 53 例、1.0 mm 10 例、5.0 mm 3 例 |
| 重建視野 ReconstructionDiameter (mm) | 243.4 | 493.0 | 324.6 | 312.8 | 291.8–340.0 | 即 in-plane FOV |
| Z 軸涵蓋範圍 (mm) | 237 | 675 | 310 | 303 | 283–321 | = 切片張數 × 切片間距 |
| 單一體積體素數 | 16.5 M | 378.5 M | — | — | — | 最大者為 1024² × 361 |
| 壓縮後檔案大小 (MB) | 19.1 | 468.1 | 52.4 | 37.5 | 35.2–41.5 | `.nii.gz`，66 例合計 3.46 GB |

轉檔前後切片張數完全一致（DICOM `PrimarySeriesSlices` 與 NIfTI 第三維統計量相同），代表轉檔過程沒有掉片或補片。切片間距 2.965 mm 之單例（`E717248`）係重複切片所致，說明見第 6 節。

### 1.2 分組差異

| 子集 | 例數 | 類別 | 切片張數 | 中位數 | 像素間距 (mm) | 切片間距 (mm) |
|---|---:|---|---|---:|---|---|
| COPD（醫院dataset） | 33 | Abnormal | 63–320 | 102 | 0.475–0.793 | 1.0 / 3.0 / 5.0 |
| Normal（20260116） | 21 | Normal | 81–382 | 97 | 0.517–0.822 | 1.0 / 2.965 / 3.0 |
| Normal（20260421，外部批次） | 12 | Normal | 81–386 | 110 | 0.435–0.963 | 1.0 / 3.0 |

### 1.3 掃描與重建參數

（20260421 之 12 例係直接自 DICOM 檔頭讀取，未收錄於 `dicom_metadata.csv`。）

| 參數 | 分布 |
|---|---|
| Manufacturer | SIEMENS 41、Siemens Healthineers 25 |
| Scanner model | SOMATOM Definition AS+ 31、go.Top 27、Definition Flash 6、Sensation 16 1、SOMATOM Force 1 |
| Convolution kernel | Br60f 24、B60f 21、I70f 8、I50f 6、Br40f 3、B30f 2、B31f 1、Br59f 1 |
| KVP | 120 kVp 34、100 kVp 18、110 kVp 11、80 / 130 / 150 kVp 各 1 |
| BitsStored / RescaleIntercept | 12 bit & −1024 共 39 例；16 bit & −8192 共 27 例 |
| RescaleSlope | 1.0（66/66 例） |
| 預設顯示窗 (WindowCenter / WindowWidth) | −600 / 1200 共 62 例（肺窗）；其餘 4 例為 −610/1170 或縱膈窗 |
| 每位病人 series 數 | 5 ～ 9（平均 5.5） |

外部批次引入了三項院內主資料集未涵蓋的協定：**SOMATOM Force 掃描儀**、**150 kVp**、**1024 × 1024 重建矩陣**（`5925853`，Br59f、1.0 mm、低劑量方案）。此例的體素數（378.5 M）為其餘各例最大者的 3.8 倍，處理時須注意記憶體與時間成本。

論文可寫的一句話：

> 本研究之胸部 CT 影像取自單一醫學中心之 Siemens 多切面 CT（含一批外部收案），軸向矩陣為 512 × 512（1 例為 1024 × 1024），像素間距 0.435–0.963 mm，切片間距 1.0–5.0 mm（52/66 例為 3.0 mm），單一病人之影像張數介於 63 至 386 張，Z 軸涵蓋範圍 237–675 mm，可完整包含雙側肺野。

---

## 2. 正規化做法（前處理端）

本研究的正規化分成四個層級：**強度正規化（intensity）**、**幾何正規化（geometry）**、**特徵層正規化（feature-level）**，以及**深度學習模型輸入端正規化**（第 3 節另述）。

前處理端的核心設計原則是：影像層只做「物理量還原」，不做 z-score 或 min–max 縮放，因為肺氣腫（−950 HU）、血管與氣道分析全部依賴絕對 HU 值；縮放後閾值即失去臨床意義。**縮放只發生在模型輸入端，且不回寫入分析資料。**

### 2.1 強度正規化：DICOM 像素值 → Hounsfield Unit

每張切片逐片套用 DICOM 標準的線性轉換（`src/convert/dicom_reader.py:88-91`）：

$$\mathrm{HU}(x,y,z) = SV(x,y,z) \times \text{RescaleSlope} + \text{RescaleIntercept}$$

其中 $SV$ 為 DICOM 儲存像素值。本資料集 66 例的 RescaleSlope 皆為 1.0，RescaleIntercept 為 −1024（12-bit，39 例）或 −8192（16-bit Siemens Extended CT Scale，27 例）。**逐片而非整體套用**，因為同一 series 中不同切片的 rescale 參數不保證一致。

接著做下界截斷（`dicom_reader.py:96`）：

$$\mathrm{HU}' = \max(\mathrm{HU},\ -1024)$$

理由：Extended CT Scale 下界（−8192 附近）為掃描視野外的 padding 值，物理上不存在低於空氣（−1000 HU）的組織；若不截斷，這些值會嚴重扭曲後續 LAA% 的分母與直方圖統計。截斷後全部案例的 HU 下界一致為 −1024 HU。

轉換後以 `int16` 儲存，HU 為整數且範圍在 int16 內，屬無損表示；NIfTI 檔頭的 `scl_slope`/`scl_inter` 不啟用，讀取端取得的即是 HU，不需二次換算。

**前處理影像層不做強度縮放**，理由如上（保留絕對 HU）。僅在下列情境做視覺／模型專用轉換：

| 用途 | 轉換 | 位置 |
|---|---|---|
| 影像顯示、圖表製作 | 肺窗 WC = −600 HU、WW = 1200 HU（即顯示範圍 −1200 ～ 0 HU） | 沿用 DICOM 預設窗值（62/66 例） |
| 互動式檢視器 | 截斷至 [−1024, 1024] HU 後 min–max 線性映射至 [0, 255] 之 uint8 | `AeroPath/demo/src/utils.py:14-19` |
| 肺區 ROI 遮罩 | 肺區以外填入 −1000 HU（空氣值） | `COPDClassification/make_lung_roi.py:224` |
| **深度學習模型輸入** | **窗寬截斷 [−1000, 400] HU → 1/99 百分位截斷 → 逐體積 z-score** | **`nnMamba/regression/data/dataset.py:43-63`（見第 3 節）** |

### 2.2 幾何正規化：切片排序、間距重估與 affine 建構

1. **切片排序**：以 `ImagePositionPatient` 的 z 分量由小到大排序，失敗時退回 `InstanceNumber`（`dicom_reader.py:75-81`）。以空間座標排序可避免 InstanceNumber 與實際幾何不一致造成的切片錯位。

2. **切片間距重估**（`dicom_reader.py:119-127`）：不直接採用 `SliceThickness`，而是由首尾切片的空間位置反推平均間距

   $$\Delta z = \frac{\lVert \mathbf{p}_{N-1} - \mathbf{p}_{0} \rVert_2}{N - 1}, \qquad \hat{\mathbf{n}} = \frac{\mathbf{p}_{N-1} - \mathbf{p}_{0}}{\lVert \mathbf{p}_{N-1} - \mathbf{p}_{0} \rVert_2}$$

   其中 $\mathbf{p}_i$ 為第 $i$ 張切片的 `ImagePositionPatient`。此法可正確處理 slice gap 或 overlap；例如 `E717248` 由此得到 2.9647 mm 而非標稱的 3.0 mm，體積計算誤差因此可控制在 1% 以內。

3. **affine 矩陣**（`dicom_reader.py:129-137`）：以 `ImageOrientationPatient` 的列/行方向餘弦 $\mathbf{r}, \mathbf{c}$ 與 `PixelSpacing` $(s_r, s_c)$ 組成

   $$A = \begin{bmatrix} s_c\,\mathbf{r} & s_r\,\mathbf{c} & \Delta z\,\hat{\mathbf{n}} & \mathbf{p}_0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

   寫入 NIfTI 的 `srow_*`（`sform_code = aligned`），使體素索引可還原為病人座標，體積量化得以使用真實體素體積 $V_{\text{voxel}} = s_r \cdot s_c \cdot \Delta z$。

4. **量化分析不做等向性重取樣（no resampling）**：所有分割與定量分析皆在原生網格上進行。理由是內插會改變 HU 直方圖（影響 LAA%）並模糊小血管與遠端氣道；體素體積差異改以上式的實際 $V_{\text{voxel}}$ 加權吸收。已驗證分割輸出與來源 CT 完全共用同一網格（例：`1596038` 之 CT 與 airway mask 皆為 512 × 512 × 93、間距 0.5299 × 0.5299 × 3.0 mm）。
   **注意**：深度學習分支不同——CNN 需要固定張量尺寸，故在載入時做一次性 resize 至 112 × 136 × 112（見 3.3 節）。此為模型輸入專用，不回寫檔案、不影響量化指標。

5. **分割與 CT 的 Z 軸方向對齊**：3 位病人（`2094528`、`5630846`、`8244460`）的外部分割結果與 CT 切片順序相反，於 `copd_analyzer.py` 的 `_align_seg_to_ct()` 自動偵測並校正，避免肺氣腫指標配對錯誤。

### 2.3 特徵層正規化：z-score 標準化（量化特徵分支）

12 維定量特徵在送入分類網路前做 z-score 標準化：

$$z_j = \frac{x_j - \mu_j^{\text{train}}}{\sigma_j^{\text{train}}}$$

關鍵是 $\mu, \sigma$ **僅由每一折的訓練集估計**，再套用到驗證/測試集（`copd_classifier.py:294-297`、`666-668`），每折的 scaler 另存為 `fold_k_scaler.pkl` 供推論階段重用，以避免交叉驗證中的資料洩漏（data leakage）。

---

## 3. 深度學習模型：程式入口與影像正規化管線

本節對象為 `nnMamba/regression/` 之 3D CT 分類/迴歸管線（RQ1 Normal vs Abnormal 即由此執行）。

### 3.1 程式入口

| 入口 | 檔案 | 用途 |
|---|---|---|
| 訓練 | `regression/train.py`（`main()`，行 17-58） | 由 YAML config 驅動之 k-fold 訓練，唯一訓練入口 |
| 評估 | `regression/evaluate.py`（`main()`） | 以 run UUID 載入 `fold{k}_best_weight.pth` 重新評估 |
| 模型工廠 | `regression/models.py`（`MODEL_REGISTRY` 行 20-45、`build_model()` 行 48） | 依 config 之 `model.name` 字串實例化網路 |
| 資料管線 | `regression/data/loader.py`（`LoaderHelper`）→ `regression/data/dataset.py`（`AngleRegressionDataset`、`load_ct()`） | 建 manifest、切 fold、載入並正規化影像 |

執行方式：

```bash
cd regression
python train.py --config config.rq1.normal_v_abnormal.image.yaml     # 影像單模態
python train.py --config config.rq1.normal_v_abnormal.fusion.yaml    # 影像 + TAP-CT 融合
python train.py --config config.production.normal_v_abnormal.66.yaml # 5 成員多數決集成
python evaluate.py --uuid <RUN_UUID> --config <same config>
```

`train.py` 的控制流：`Config.from_yaml(args.config)` → 設定 `CUDA_VISIBLE_DEVICES` 與 torch runtime → 若 `ensemble.enabled` 走 `MajorityVotingEnsembleTrainer`，否則 `LoaderHelper(config)` + `build_model(config.model, output_dim=config.model_output_dim())` → `Trainer.train()`，回傳 run UUID，權重寫入 `weights/{task}/{uuid}/`。

**所有前處理與正規化參數皆來自 config 的 `data:` 區塊，沒有寫死在程式中**；RQ1 與 production 之 66 例設定完全一致：

```yaml
data:
  target_mode: normal_v_abnormal
  source_dir: ../classification/datasets/normal_v_abnormal_66   # 33 Abnormal + 33 Normal
  manifest: ./datasets/generated/rq1_nva66_manifest.image.json  # total=66, unique_patients=66
  image_size: [112, 136, 112]
  intensity_window: [-1000.0, 400.0]
  input_normalization: zscore
  target_normalization: none
```

預設值定義於 `regression/core/config.py:172-173`（`intensity_window = (-1000.0, 400.0)`、`input_normalization = "zscore"`）。

66 例世代由 `regression/scripts/build_nva66.py` 由 `patient_angle_classification_by_group.json` + `by_angle_all/` 以 symlink 建出，故不複製影像資料。類別索引為 **Abnormal = 0、Normal = 1**（manifest `class_names = ["Abnormal", "Normal"]`）。

### 3.2 模型架構入口（RQ1 主模型）

`model.name: hybrid_mamba_attention` → `HybridMambaAttentionRegressor`（`regression/networks/hybrid_mamba_attention_regressor.py:74`）：

- Stem：`Conv3d(1, 32, k=7, stride=4, pad=3)` → GroupNorm → GELU
- Stage1/2/3：`DownsampleStage`（殘差 Mamba 區塊），通道 32 → 32 → 64 → 128，stride 1/2/2
- Attention bridge：`HybridAttentionBlock` × `attn_layers`（深度可分離位置卷積 + MultiheadAttention + MLP，皆為 pre-norm）
- Head：四個階段各自 `AdaptiveAvgPool3d(1)` 後串接（32 + 64 + 128 + 128 = 352 維）→ 3 層 MLP → `num_classes`

融合模型 `hybrid_mamba_tapct_fusion` 則在上述影像分支外，另以 `LayerNorm(1152) → Linear → …` 的分支處理凍結的 TAP-CT 病人嵌入後串接（`hybrid_mamba_tapct_fusion_regressor.py:60-62`）。

### 3.3 模型輸入端正規化管線（本節重點）

單一體積的完整處理鏈在 `regression/data/dataset.py` 的 `load_ct()`（行 66-87）：

1. **載入**：`nib.load(path).get_fdata().astype(np.float32)`；若為 4D 取第 0 個 volume。此時數值即為 HU（第 2.1 節已保證）。
2. **幾何重取樣**（`_resize_volume`，行 30-40）：`skimage.transform.resize(volume, (112, 136, 112), order=1, preserve_range=True, anti_aliasing=True)`——三線性內插 + 反鋸齒，`preserve_range=True` 確保內插仍在 HU 尺度上進行。所有病人不論原始矩陣（512² 或 1024²）與切片張數（63–386）一律映射到同一張量尺寸，體素間距因此變成 **每例不同的隱含縮放**，屬本管線的已知簡化。
3. **強度窗截斷**（`_normalize_volume`，行 49-51）：

   $$x' = \mathrm{clip}(x,\ -1000,\ 400)\ \text{HU}$$

   下界 −1000 HU 為空氣，上界 400 HU 涵蓋軟組織與部分鈣化，捨去金屬與緻密骨的極端值。此窗同時解決第 6 節所述的 HU 上界離群問題。
4. **1/99 百分位再截斷**（行 56-58）：以該體積自身的 1% / 99% 百分位 $(l, h)$ 再夾一次，$h > l$ 時 $x'' = \mathrm{clip}(x', l, h)$，抑制殘餘離群體素對後續統計量的影響。
5. **逐體積 z-score**（行 59-63）：

   $$z = \frac{x'' - \mu_{\text{volume}}}{\sigma_{\text{volume}}}, \qquad \sigma_{\text{volume}} < 10^{-6} \text{ 時取 } \sigma = 1$$

   **$\mu, \sigma$ 由「該一份體積自身」估計，不使用訓練集或全資料集統計量。** 這是與 2.3 節表格特徵不同的設計：每例自我標準化，本質上不可能造成跨 fold 的資料洩漏，也讓不同掃描協定（kVp、kernel、劑量）造成的整體亮度/對比差異在輸入端就被吸收。
6. **加通道軸**：`np.expand_dims(volume, axis=0)` → `(1, 112, 136, 112)`，float32。

若 config 設 `input_normalization: none` 則跳過步驟 4–5，只保留窗截斷（行 52-53）。目前 66 例的所有 RQ 設定皆為 `zscore`。

**順序上的兩個關鍵細節**：(a) resize 在 windowing **之前**，故內插發生在原始 HU 值上；(b) z-score 在 augmentation **之前**，因此 config 內的強度增強參數（`intensity_scale_range: [0.95, 1.05]`、`intensity_shift_range: [-0.1, 0.1]`、`noise_std: 0.03`）單位是**標準化後的無因次單位**，而非 HU——`data/transforms.py` 的函式預設值（shift ±25.0、noise 8.0）才是 HU 尺度，兩者不可混用。

### 3.4 模型內部的正規化層

| 位置 | 層 | 理由 |
|---|---|---|
| 卷積主幹（stem、DownsampleStage、attention pre-norm） | `GroupNorm`（`networks/mamba_regressor.py:31-40`，`num_groups = min(8, C)`） | BatchNorm 在此小樣本 3D 任務上不穩定：eval 模式依賴 running statistics，導致 train/eval 行為不一致；GroupNorm 無此問題 |
| Attention 區塊的 token 路徑 | `LayerNorm`（token 與 MLP 各一，pre-norm） | 標準 Transformer 配置 |
| TAP-CT 嵌入分支（融合模型） | `LayerNorm(1152)` 於投影前 | 凍結外部嵌入與影像特徵尺度差異大，先標準化再串接 |

模型輸出端 `target_normalization: none`（分類任務不需要），迴歸任務才會啟用目標值正規化。

### 3.5 訓練期資料流（66 例）

- 5-fold 分層交叉驗證，`seed: 42`，**病人層級切分**（`loader.py` 明確排除增強副本進入驗證折，避免同一病人同時出現在 train/val）
- 訓練折啟用 `balance_then_augment: true` + `views_per_sample: 5`，即先類別平衡再每例展開 5 個增強視角；驗證/測試折**不做任何增強**，只走 3.3 節的確定性管線
- `epochs: 100`、`batch_size: 12`、`lr: 1e-4`、`weight_decay: 1e-3`、`cross_entropy`、`clip_grad_norm: 1.0`、early stopping 關閉

論文可寫的一句話：

> 量化分析階段影像層僅執行 HU 還原與下界截斷，不進行任何強度縮放，以保留肺氣腫閾值（−950 HU）所依賴的絕對衰減值，空間上亦不做等向性重取樣；深度學習分支則於資料載入時額外執行固定尺寸重取樣（112 × 136 × 112，三線性內插）、[−1000, 400] HU 窗截斷、1/99 百分位截斷與**逐體積** z-score 標準化，標準化統計量僅取自該筆影像本身，故不引入交叉驗證的資料洩漏。

---

## 4. 使用工具與版本

### 4.1 轉檔與量化管線

| 階段 | 工具 | 版本 | 用途 |
|---|---|---|---|
| DICOM 解析 | pydicom | 3.0.1 | 讀取 metadata 與像素資料、series 分組 |
| 壓縮解碼 | pylibjpeg / pylibjpeg-libjpeg / pylibjpeg-openjpeg / GDCM | 2.1.0 / 2.3.0 / 2.5.0 | JPEG Lossless、JPEG 2000 之 DICOM 解壓縮 |
| 陣列運算 | NumPy | 2.3.4 | HU 轉換、截斷、體積堆疊 |
| NIfTI 讀寫 | NiBabel | 5.3.2 | affine 建構與 `.nii.gz` 輸出 |
| 格式轉換 | SimpleITK | 2.5.3 | NIfTI ↔ NRRD（3D Slicer 推論伺服器介接） |
| 肺葉分割 | 3D Slicer 推論伺服器 + `lungs-v2.0.1` 模型 | — | 肺葉 label map 產生 |
| 氣道分割 | AeroPath（raidionicsrads pipeline，`CT_Airways`） | — | 氣道樹分割，`resample_first` + thresholding 重建 |
| 影像處理 | SciPy (`ndimage`) | 1.16.3 | ROI 距離轉換、形態學膨脹 |
| 特徵標準化與驗證 | scikit-learn | 1.7.2 | `StandardScaler`、`StratifiedKFold` |
| 12 維特徵分類器 | PyTorch | 2.9.1 (CUDA 12.6) | 12-12-… MLP 二元分類 |
| 執行環境 | Python | 3.11.14 | — |

### 4.2 nnMamba 3D 影像模型管線

| 階段 | 工具 | 用途 |
|---|---|---|
| NIfTI 載入 | NiBabel | `load_ct()` 讀取 HU 體積 |
| 幾何重取樣 | scikit-image `transform.resize` | 三線性內插至 112 × 136 × 112 |
| 強度正規化 | NumPy | 窗截斷、百分位截斷、逐體積 z-score |
| 網路與訓練 | PyTorch | `HybridMambaAttentionRegressor` / Mamba SSM 區塊 |
| SSM 核心 | `mamba-ssm` | `ResidualMambaBlock` 內之狀態空間運算 |
| 指標 | torchmetrics | Accuracy、AUROC 等 |
| 執行環境 | conda env `nnMamba` | — |

---

## 5. DicomToNii 輸出樣式

### 5.1 Series 自動挑選規則

每位病人有 5–9 個 series，程式自動挑出最適合肺部分析者（`src/convert/batch.py`）：

1. 僅保留 `Modality == CT`。
2. 切片數 ≥ 10（`MIN_SLICE_COUNT`），排除定位像與截圖。
3. 依 SeriesDescription 排除：`TOPOGRAM`、`TOPO`、`PROTOCOL`、`SCREEN CAPTURE`、`COR`/`CORONAL`、`SAG`/`SAGITTAL`，以及以 `SW `/`SF ` 開頭之重組影像。
4. 於剩餘候選中取切片數最多者（等同於最薄、涵蓋最完整之軸向肺窗影像）。

### 5.2 檔名規則

預設模式（每人一檔）：

```text
{PatientID}_{SeriesDescription}.nii.gz
```

`SeriesDescription` 中的 `/ \ : * ? " < > |` 一律以 `_` 取代。實例：

```text
1261736_LW AXI 3_3  B60f.nii.gz          # 原始 series 描述 "LW AXI 3/3  B60f"
1596038_Thorax Lung Br60 S2 3.00.nii.gz
5127217_Thorax 1_1 Br40 S3 1.00.nii.gz
9075311_Thorax  5.0  B30f.nii.gz
```

模型端的 patient ID 即取檔名第一個底線前的字串（`build_nva66.py` 的 `ct_files_by_pid()`），故此命名規則同時是資料集與標籤 JSON 的對應鍵。

`--all-series` 模式（保留全部合格 series，供 protocol robustness 實驗）改用協定標籤命名：

```text
{PatientID}_{ConvolutionKernel}_{SliceThickness}mm.nii.gz   # 例：1261736_B60f_3mm.nii.gz
```

`--group-by` 可選 `none`（全部平放）／`patient`（每人一資料夾）／`series`（病人/序列兩層）。

### 5.3 輸出檔案格式

- 容器：NIfTI-1，gzip 壓縮（`.nii.gz`）
- 資料型別：`int16`，值域即 HU（下界 −1024）
- `scl_slope` / `scl_inter`：未使用
- `sform_code = aligned`（2）、`qform_code = unknown`（0），方向資訊寫於 `srow_x/y/z`
- 維度 `dim = [3, 512, 512, N, 1, 1, 1, 1]`

實際檔頭範例（`1596038_Thorax Lung Br60 S2 3.00.nii.gz`）：

```text
dim         : [  3 512 512  93   1   1   1   1]
datatype    : int16              bitpix : 16
pixdim      : [1. 0.5299219 0.5299219 3. 1. 1. 1. 1.]
sform_code  : aligned
srow_x      : [  0.5299219  0.         0.       -134.33205 ]
srow_y      : [  0.         0.5299219  0.       -270.68704 ]
srow_z      : [  0.         0.         3.      -1166.477   ]
```

### 5.4 輸入／輸出資料夾對應（66 例）

| 輸入（原始 DICOM） | 輸出（NIfTI/HU） | 例數 | 模型類別 |
|---|---|---:|---|
| `醫院dataset` | `醫院dataset_nii_hu` | 33 | Abnormal（class 0） |
| `醫院NormalDataset20260116` | `醫院NormalDataset20260116_nii_hu` | 21 | Normal（class 1） |
| `醫院NormalDataset20260421` | `醫院NormalDataset20260421_nii_hu` | 12 | Normal（class 1） |
| — | `classification/datasets/normal_v_abnormal_66/{Abnormal,Normal}/` | 33 / 33 | 模型實際讀取之 symlink 世代 |

執行指令：

```bash
# 轉檔
python -m src.convert.batch -d 醫院dataset -o 醫院dataset_nii_hu
python -m src.convert.batch -d 醫院dataset -o dataset_grouped -g patient --all-series

# 建 66 例模型世代（symlink）
cd nnMamba/regression && python scripts/build_nva66.py
```

---

## 6. 已知限制（建議寫入論文限制或以註腳說明）

1. **HU 上界未截斷（轉檔端）**：轉檔程式僅截斷下界（−1024 HU），未設上界。院內 54 例中有 6 例出現 > 3071 HU 的體素（最大 19216 HU，出現於 `1596038`），推測為金屬植入物或 Extended CT Scale 之高值端。受影響體素比例極低（最高 0.006%，且皆位於骨骼/金屬區，不在肺實質），不影響 LAA% 與血管、氣道指標；若需嚴謹一致，可補上 `np.clip(img3d, -1024, 3071)`。深度學習分支不受此問題影響，因其輸入端已套用 [−1000, 400] HU 窗（3.3 節步驟 3）。
2. **座標系標記**：affine 直接沿用 DICOM 的 LPS 方向餘弦寫入 NIfTI（NIfTI 定義為 RAS），故 nibabel 讀出的方向碼為 `RAS` 但實際左右／前後與真 RAS 互為鏡像。因所有 mask 與 CT 共用同一網格，體積與 HU 統計不受影響；惟若論文需報告「左肺 vs 右肺」分側結果，須先確認左右標籤未互換。
3. **量化分析未做等向性重取樣**：3.0 mm 與 1.0 mm 切片並存（52 : 10，另有 5.0 mm 3 例），厚切片會低估遠端小氣道與細小血管，屬本研究之協定異質性來源之一，已於資料異質性分析章節討論。
4. **深度學習端的體素間距未保留（重取樣尺度不一致）**：3.3 節步驟 2 將所有體積以 `skimage.transform.resize` 壓到固定張量尺寸，無裁切、無填補，純內插。輸出第 $k$ 片取自原始座標 $(k+0.5)\,N/112 - 0.5$，故每例的縮放比例 $N/112$ 皆不同：

   | 案例 | 原始 shape | 原 z 間距 | Z 方向 factor | resize 後等效 z 間距 |
   |---|---|---:|---:|---:|
   | `9075311` | 512² × 63 | 5.00 mm | 0.562（上取樣） | 2.812 mm |
   | 典型例 | 512² × 101 | 3.00 mm | 0.902（上取樣） | 2.705 mm |
   | `1814107` | 512² × 386 | 1.00 mm | 3.446（下取樣） | 3.446 mm |
   | `5925853` | 1024² × 361 | 1.00 mm | 3.223（下取樣，in-plane 達 9.14） | 3.223 mm |

   下取樣時 `anti_aliasing=True` 會先施加 $\sigma = (\text{factor}-1)/2$ 的高斯模糊（`1814107` 為 σ = 1.223），故被跳過的切片是被抹進鄰片而非單純丟棄，但一樣不可逆；上取樣時（如 `9075311`）輸出第 0、1 片皆落在原始第 0 張附近（座標 0.219 / 0.344），等於內插出不存在的切片。

   66 例實測之 resize 後等效體素邊長：X 2.173–4.402 mm（極差 2.03 倍）、Y 1.790–3.625 mm（2.03 倍）、Z 2.116–6.027 mm（**2.85 倍**）。**同一顆 10 mm 病灶在張量中佔 1.66 ～ 4.73 個體素。** 模型看到的尺寸取決於掃描涵蓋範圍而非解剖尺寸。

   **且此失真與標籤相關**：Abnormal 之 Z 涵蓋中位數 312 mm、Normal 288 mm，Mann-Whitney $p = 0.0082$；in-plane FOV 則無差異（$p = 0.97$）。因此 (a) Z 涵蓋範圍本身所帶的訊號（COPD 過度充氣使肺野變長之合理生理解釋）被 resize 抹除；(b) 抹除後殘留的是與標籤相關的 z 方向縮放差異，模型有可能以紋理頻率的系統性差異作為捷徑，而非學習肺氣腫本身。此為 RQ1 結果詮釋時應揭露的混淆因子。

   若後續要求尺度一致性，應改為先重取樣到固定體素間距（如 1.5 mm 等向）再裁切／填補到固定尺寸，使物理尺度成為模型可見的資訊。
5. **外部批次的協定外推**：1024 × 1024 / 150 kVp / SOMATOM Force 各僅 1 例，模型無法從單一樣本學到這些協定的不變性；其在交叉驗證中落入哪一折會影響該折表現，屬 66 例規模下無法消除的變異來源。
