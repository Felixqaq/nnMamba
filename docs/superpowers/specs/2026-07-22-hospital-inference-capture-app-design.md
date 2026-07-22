# 醫院推論 + 資料擷取 App — 設計 Spec

**日期:** 2026-07-22
**狀態:** 設計已核可,待實作計畫
**作者:** felixchang

## 1. 目的

一個能帶去醫院、跟臨床醫生 demo Normal-vs-Abnormal(COPD 相關)CT 分類器的工具,並且**在臨床使用的同時順便長大訓練資料集**:醫生對某位病人的 CT 跑推論時,同一份 CT + 病人號碼被擷取進 staging 區,供日後補標籤。

- **現階段:** Gradio 本地 web app,供當面 demo(兩個月開發期)。
- **未來:** 可能搬到醫院自架服務 / 雲端。架構讓這件事之後做起來便宜,但現在不蓋。

### 交付模式 —— 兩個 repo

程式碼**要交給醫院的人**,所以不能夾帶整個 nnMamba 研究碼。系統拆成兩個 repo:

- **`~/Research/copd-ct-app/`**(新,面向醫院,自足)—— Gradio app + 推論/擷取核心。不依賴 nnMamba。這是醫院實際跑的東西。醫院端只跑推論 + 擷取,永不重訓或 backfill。
- **nnMamba / 研究端**(不交出)—— production 重訓、label backfill(會動 dataset)、發版打包。留在研究者這邊。

**為什麼這樣化解前處理漂移風險:** 醫院不重訓,所以前處理是**凍結的、隨每次 checkpoint 發版一起送**。每個 release = 5 個 checkpoint + 訓練當時用的那份前處理,綁成一包。醫院端不可能漂移(他們永不改前處理)。唯一可能漂移的地方是研究端「訓練 → 打包」之間,而那個在**發版時**由 `package_release.py` 的逐位元一致性測試擋掉。漂移是發版時的事(研究者的責任),不是執行時的事(醫院的責任)。

**資料如何流回:** 擷取的 CT 累積在醫院機器的 `staging/`。研究者定期把 `staging/` 收回自己的機器,跑 `label_backfill`(比對 PFT、入庫),累積夠了再重訓、打包新 release 給醫院。

## 2. 已定案的決策

| 主題 | 決策 |
|---|---|
| Repo 切分 | 新的自足醫院 repo `~/Research/copd-ct-app/`(app + 擷取);訓練 + backfill + 打包留研究端,不交出 |
| 前處理一致性 | 凍結並隨每次 checkpoint release 一起送;漂移在發版時由 `package_release.py` 一致性測試擋,絕不在醫院端發生 |
| App 型態 | Gradio 本地 web app,跑在有 NVIDIA GPU 的機器(localhost) |
| CT 輸入 | 從 PACS 匯出的 DICOM series(一個資料夾 / zip) |
| 擷取時的真值標籤 | 當下拿不到。App 只存 CT + 病人號;標籤日後離線補 |
| 病人號碼儲存 | **真實病人號當檔名**(沿用現有 `{patient_id}_*.nii.gz`)。去識別化做成 config 開關,預設關,程式接口保留 |
| 部署的模型 | 5-member soft-vote ensemble,**用全部資料重訓**(無 held-out)—— production checkpoint,不是 CV 的 fold 模型 |
| 重訓時機 | Human-in-the-loop、週期性。腳本訓練「執行當下 dataset 裡的資料」;何時跑由研究者決定 |
| 模型換版 | App 啟動時載 `models/current/`;換新版 = 重指 `models/current` + 重啟 app |
| Disclaimer | UI 顯示「研究用途、非診斷」,config 開關,**預設開** |
| 雲端準備 | core 維持無狀態/config 驅動、儲存存取收斂到單一模組。現在不蓋雲端基礎設施。PHI 上雲是法規題,日後跟 IRB/醫院決定 |

### 硬限制
模型用 `mamba_ssm`(CUDA-only 的 selective-scan kernel)。**NVIDIA GPU 是必要條件** —— 不可能用 CPU / AMD / Intel GPU 部署。實測佔用:1.15M 參數、每個 checkpoint ~5.8MB(5 個共 ~29MB)、推論 VRAM ~0.1GB、每 volume 每 member 的 GPU forward ~6ms。

### 誠實效能註記
Production checkpoint 用全部資料訓練,所以**沒有乾淨的 held-out 數字**。描述預期效能時要引用 nested-CV 參考值(~0.73 Acc / 0.80 AUC),絕不用 best-epoch 的 0.833/0.879。

## 3. 架構

兩個 repo。醫院 repo 自足(Gradio app + 推論/擷取核心,加一份凍結的前處理模組和一包 checkpoint release)。研究端負責重訓、backfill、發版打包,且不修改 `regression/` 訓練碼。

### 醫院 repo —— `~/Research/copd-ct-app/`(交出)
```
copd-ct-app/
├── app.py                 # Gradio UI 外殼(薄,~50 行)
├── core/
│   ├── api.py             # predict_and_capture(dicom_dir, *, capture=True) -> PredictionResult(唯一對外入口)
│   ├── dicom_io.py        # DICOM series -> NIfTI(SimpleITK)
│   ├── preprocess.py      # 凍結副本:clip[-1000,400] + resize[112,136,112] + zscore(隨 release 一起送)
│   ├── ensemble.py        # load 5 個 checkpoint -> soft-vote -> 機率
│   ├── gradcam.py         # 取一個代表 member 的 Grad-CAM 熱圖
│   └── staging.py         # 所有 staging 讀寫;儲存後端可抽換(現在本地,未來物件儲存)
├── models/
│   └── current/           # 打包的 release:member_1.pth ... member_5.pth、metrics.json、PREPROCESS_HASH
├── staging/
│   ├── incoming/{patient_id}_{YYYYMMDD_HHMMSS}.nii.gz
│   └── capture_log.jsonl
├── config.yaml            # 模型目錄、staging 路徑、image_size、disclaimer + 去識別化開關
├── environment.yml / Dockerfile
└── tests/                 # 含「前處理對凍結 hash」的一致性測試
```

### 研究端 —— nnMamba / 私有(不交出)
```
scripts/
├── train_production_ensemble.py  # 用全部資料重訓 5 個 seed-diverse member -> release/<date>/
├── label_backfill.py             # 收回的 staging -> 比對 PFT -> 補標籤 -> 搬進 dataset
└── package_release.py            # 逐位元比對(凍結前處理 == 訓練前處理),
                                  #   通過才打包 5 checkpoints + 凍結前處理 -> 給 copd-ct-app 的一包 release
```

`label_backfill.py` 與重訓**直接重用**真正的 `regression/data/dataset.py` 前處理和 `regression/data/manifest.py` label 規則(同 repo,不漂移)。醫院 repo 的 `core/preprocess.py` 是凍結副本,它與訓練前處理相等這件事由 `package_release.py` 在發版時斷言。

### 設計原則
- **core/ 與 UI 完全解耦。** `app.py` 只呼叫 `predict_and_capture`。未來的 FastAPI/CLI/桌面前端包同一個入口,core 不動。
- **`PredictionResult` 是純資料**(機率、預測類別、patient_id、staging 路徑、可選 Grad-CAM 圖)—— 不含任何 UI 概念。
- **前處理必須與訓練逐位元一致。** 研究端重用 `regression/data/dataset.py`;醫院端用凍結副本 + hash 把關。絕不放任兩份各自漂移。
- **推論與擷取解耦。** 擷取包在 try/except;擷取失敗只記 log,絕不擋醫生看到預測。
- **儲存存取集中在 `staging.py`**,好讓後端(現在本地 FS、未來物件儲存)一處抽換。

### 資料流
醫生上傳 DICOM → `dicom_io` → NIfTI → `preprocess` → `ensemble` → 顯示機率給醫生 → 同一份 NIfTI 交給 `staging`(檔名 = 真實病人號)+ 在 `capture_log.jsonl` append 一行。日後 `label_backfill.py` 讀 staging + PFT 指派標籤、搬進 dataset。

## 4. 推論流程(§2 細節)

**① DICOM → NIfTI(`dicom_io.py`)** —— SimpleITK 讀資料夾。要處理:
- HU 校正(`RescaleSlope`/`RescaleIntercept`)—— 驗證有套,否則 clip 範圍就錯。
- slice 排序 / spacing / orientation —— 依 `ImagePositionPatient` 排序,統一到訓練方向(RAS)。
- 一個資料夾多個 series —— 依 `SeriesInstanceUID` 分組,選正確那個(或讓醫生選)。
- 從 DICOM 檔頭抽 `PatientID` 當檔名。

**② 前處理(`preprocess.py`)** —— clip `[-1000, 400]` → resize `[112,136,112]`(skimage,同參數)→ z-score。與 `regression/data/dataset.py` 對齊。任何不一致都會讓分布飄移、預測靜默變差。

**③ Ensemble(`ensemble.py`)** —— 啟動時把 5 個 checkpoint 從 `models/current/` 載到 GPU(~0.1GB VRAM,常駐)。每例對 5 個 model forward → softmax 機率平均(soft-vote)→ Abnormal 機率。

**④ 呈現** —— Abnormal 機率(0–100% 機率條)、預測類別、disclaimer(config 開關,預設開)。Grad-CAM 熱圖見下方 §8。

## 5. 擷取流程與 Staging(§3 細節)

擷取在跟預測同一個 request 內發生。擷取失敗絕不能影響醫生看到預測。

```
staging/
├── incoming/{patient_id}_{YYYYMMDD_HHMMSS}.nii.gz   # 轉好的 NIfTI,真實病人號
└── capture_log.jsonl                                # 每次擷取一行,append-only
```

`capture_log.jsonl` 每筆:`patient_id`、nii 檔名、擷取時間、模型預測(機率/類別)、來源 `SeriesInstanceUID`、`label: null`(待補)。

行為:
- 擷取包在 try/except:失敗記一行 error,醫生照樣看到預測。
- 重複病人:用時間戳區分、兩筆都留(同病人可能多次 CT),不覆蓋。
- 存**轉好的原始解析度 NIfTI**(不是 resize 後的),日後改用不同 `image_size` 重訓不會被鎖死。
- 去識別化開關(預設關、真實號):開啟時 `staging.py` 把檔名換成研究代碼、strip DICOM PHI 檔頭、另存對照表。程式接口現在就留好,不啟用。
- **所有 staging 讀寫走 `staging.py`**;儲存後端可抽換。

## 6. 資料成長迴圈(重訓)

Human-in-the-loop、非自動,且跨兩個 repo 邊界:

```
① 醫院用 copd-ct-app 跑目前的 release(models/current/)
② 醫生用 app -> 新 CT 進醫院機器的 staging/incoming/(label=null)
③ 研究者定期把 staging/ 收回自己的機器
④ 研究端:label_backfill.py 比對 PFT -> 有真值的資料補標籤 -> 搬進 dataset
⑤ 累積夠了 -> train_production_ensemble.py 用放大後的 dataset 重訓
   -> release/<date>/(不動醫院)
⑥ package_release.py:逐位元比對(凍結前處理 == 訓練前處理),
   打包 5 checkpoints + 凍結前處理 -> 一包 release
⑦ 把 release 送到醫院 repo;在醫院機器 promote models/current + 重啟 app
-> 回到 ①(dataset 更大、模型更新)
```

- `train_production_ensemble.py`(研究端)訓練「**執行當下 dataset 資料夾裡的全部資料**」,不切 fold;5 個 member 只差種子(init + augmentation 取樣)。沿用 `config.normal_v_abnormal.imageonly.aug5.ensemble.yaml`(160ep、5× aug、early-stop)。輸出到 `release/<date>/`,附 `metrics.json`(nested-CV 參考值、訓練資料筆數)。**不動醫院。**
- 不自動/即時的原因:新資料要先等 PFT 真值(幾天~幾週);加 3~5 例不值得重訓(等 ~+20~30);換版該經人審,不無人監督。
- `label_backfill.py` 負責 收回的 staging → dataset;`train_production_ensemble.py` 只讀 dataset,不碰 staging。職責分開。

### 換版(醫院機器)
- App 只讀 `models/current/`;不需知道日期。
- 新 release 由 `package_release.py` 打包、以帶日期的目錄送過去;promote 前不上線。
- Promote 是醫院機器上的明確動作:看過 `metrics.json` 新舊差異 → 把 `models/current` 指向新 release → 重啟 app(啟動時載 checkpoint,不做熱重載)。
- 回退:把 `models/current` 指回舊 release、重啟。舊 release 都留著。

## 7. Label Backfill(§4 細節)

離線、冪等、可重複執行的腳本,把 staging 擷取對上 PFT 真值。

**輸入:** `staging/capture_log.jsonl` + `staging/incoming/*.nii.gz`;PFT 來源(`pft.json` / `GOLD_2026_classification.json`,patient_id → PFT 指標)。

**每筆 pending 擷取:**
1. 用 `patient_id` 查 PFT 來源。
2. 查不到 → PFT 還沒出 → 留在 staging、跳過(冪等,下次重試)。
3. 查得到 → 用 FEV1/FVC ≥ 70% 判 Normal / < 70% 判 Abnormal,重用 `regression/data/manifest.py` 的 `normal_v_abnormal_label`(順便存 GOLD 1–4 供未來多分類)。
4. 把 NIfTI 搬到 `classification/datasets/normal_v_abnormal_XX/{Normal|Abnormal}/{patient_id}_*.nii.gz`。
5. 把標籤/角度/GOLD 寫進 labels JSON(沿用現有 `patient_angle_classification_by_group.json` 結構)。
6. 在 `capture_log.jsonl` 標記該筆已補標籤 + 搬去哪。

**安全:**
- 冪等 + 預設 `--dry-run`(印出會搬什麼 + 判成什麼),`--commit` 才動作。重跑會跳過已入庫的。
- 衝突檢查:patient_id 已在 dataset → 報出來讓人決定(同病人新 CT 還是重複),不靜默覆蓋。
- PFT 落灰帶 / 指標缺漏 → 列進 `needs_review`,不猜。
- 搬檔前驗證 NIfTI 讀得開、shape 合理;壞檔不入庫。
- 產出 backfill 報告(新增 N、各類筆數、還在等 PFT、需人工複核)。

## 8. 可解釋性 / 錯誤處理 / 打包 / 測試(§5–§8)

**Grad-CAM** —— 重用 `regression/test_gradcam.py` 邏輯,對一個代表 member 產熱圖,疊在 CT 切面上。config 開關,預設開。只做一個 member(5 個太重)。

**錯誤處理 / 輸入檢查** —— 擋在 core 入口:非 DICOM / 空資料夾 / 多 series 未選 / HU 異常 / shape 不合理 → 回清楚的錯誤給前端,不吐 traceback,壞資料絕不進模型或 staging。

**打包** —— 醫院 repo 送 `environment.yml`(torch cu124 + mamba-ssm + SimpleITK + gradio)+ README;建議 **Dockerfile(CUDA base)**,因為 mamba-ssm 常編譯失敗;啟動一行 `python app.py`。研究端經 `package_release.py` 送模型更新,它在打包前斷言凍結的 `core/preprocess.py` 與訓練前處理相符(逐位元 / hash),比對失敗就擋下 release。

**測試** —— 每個模組可單獨測(本專案無 pytest;用 `nnMamba` conda env inline runner):DICOM 轉檔(小 fixture series)、前處理與訓練逐位元一致、ensemble soft-vote 數值、staging 冪等 + 失敗解耦、backfill 冪等 / dry-run / 衝突。UI 不寫自動測試(手動 demo 驗)。

## 9. 範圍外(YAGNI)

容器化/K8s/autoscaling、真的接物件儲存 SDK、多使用者登入 / API 金鑰、雲端部署腳本、熱重載、桌面 GUI。接口留乾淨,日後可加而不用重寫;實作延後。
