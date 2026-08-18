# RQ 1/2/3 — 兩主模型 × 66 病人 對齊實驗總覽

本文件整理論文的三個研究問題(RQ),每個 RQ 用**兩個主模型**在**同一份 66 病人世代**、
**完全對齊的訓練協定**下比較。設計依據:
[docs/superpowers/specs/2026-07-14-rq123-two-model-matrix-design.md](../../docs/superpowers/specs/2026-07-14-rq123-two-model-matrix-design.md)。

## 研究問題

- **RQ1｜COPD 正常 vs 異常**：CT 影像能否區分臨床標記的正常與異常肺?(`normal_v_abnormal`)
- **RQ2｜塌陷角 (collapse angle)**：以三種標的形式評估——
  - RQ2a 三分類 `angle_3class`(≤131° / 132–151° / ≥152°)
  - RQ2b 極端二分類 `angle_binary_extreme`(排除中間灰區)
  - RQ2c 角度數值迴歸 `angle`
- **RQ3｜OI**：OI 阻塞指數的氣腫二分類(`oi_emphysema`,門檻 OI=3)。

## 兩個主模型

| 模型 | `model.name` | 說明 |
|------|--------------|------|
| Mamba + Attention(image-only) | `hybrid_mamba_attention` | CT 影像編碼器,無外部 embedding |
| TAP-CT Late Fusion | `hybrid_mamba_tapct_fusion` | 同一 CT 編碼器 + 凍結 TAP-CT-S-3D(1152 維)patient embedding 後段串接 |

## 實驗矩陣(2 模型 × 5 任務 = 10 個 config)

| RQ | 任務 | image-only config | fusion config | 病人數 |
|----|------|-------------------|---------------|--------|
| RQ1 | normal_v_abnormal | `config.rq1.normal_v_abnormal.image.yaml` | `config.rq1.normal_v_abnormal.fusion.yaml` | 66 (33/33) |
| RQ2a | angle_3class | `config.rq2a.angle_3class.image.yaml` | `config.rq2a.angle_3class.fusion.yaml` | 66 |
| RQ2b | angle_binary_extreme | `config.rq2b.angle_binary.image.yaml` | `config.rq2b.angle_binary.fusion.yaml` | 61(排除灰區) |
| RQ2c | angle(迴歸) | `config.rq2c.angle_reg.image.yaml` | `config.rq2c.angle_reg.fusion.yaml` | 66 |
| RQ3 | oi_emphysema | `config.rq3.oi_emphysema.image.yaml` | `config.rq3.oi_emphysema.fusion.yaml` | 66 |

## 統一世代 (66 病人)

- **單一來源**：`by_angle_all`(66 個 CT,覆蓋全部 66 病人)。
- RQ1 由 `scripts/build_nva66.py` 依 `patient_angle_classification_by_group.json` 的臨床組別
  (abnormal_group=33、normal_group=33)建立 `classification/datasets/normal_v_abnormal_66/`
  的 `Abnormal/`(33)+`Normal/`(33)symlink。
- Fusion 一律用 `embeddings/tapct_s_3d/features.npz`(1152 維,涵蓋 66)。
- **merlin 淘汰**(只有 54 病人)。
- 例外:`angle_binary_extreme` 依定義排除中間灰區 → 61 病人子集(任務本質,非缺陷)。

## 對齊訓練協定(全 10 個一致)

| 項目 | 值 |
|------|----|
| seed | 42 |
| k_folds | 5(patient-level stratified) |
| epochs | 100 |
| early stopping | **關閉**(固定預算、最終 epoch 評估,避免用 test fold 選停點的樂觀偏差) |
| augmentation | 5x(`balance_then_augment` + `views_per_sample=5`;迴歸任務不套用) |
| balanced_sampling | true(分類) |
| image_size | [112, 136, 112] |
| intensity_window | [-1000, 400] |
| input_normalization | zscore |
| 決策 | argmax,不調閾值 |

**兩模型唯一差異**:`model.name`、`data.tapct_features`(null vs tapct_s_3d)。
任務本質差異(允許):分類用 `cross_entropy`;迴歸用 `num_classes=1 / loss=auto / target_normalization=zscore`。

## 執行方式

```bash
cd regression
conda run -n nnMamba python scripts/run_all_rq.py   # 依序跑 10 個,可續跑、失敗不中斷
```

進度與 UUID 記錄於 `regression/train_log/run_all_rq/summary.json`;各實驗 log 於同目錄。

## 結果(5-fold 交叉驗證,2026-07-14 完成)

指標彙總各 fold 的 test 表現(pipeline 報 Accuracy / Macro-F1 / Sensitivity / Specificity;
迴歸報 MAE / RMSE / R² / Pearson,未計算 AUC)。下表列分類 **Acc(mean±std)/ Macro-F1**,
迴歸列 **MAE / R²**。**粗體 = 該 RQ 兩模型中較佳者**。

### 分類任務(RQ1、RQ2a、RQ2b、RQ3)

Acc / Sensitivity / Specificity 皆為 mean±std(5-fold);多分類(RQ2a)的 Sens/Spec 為 macro 平均。
**粗體 = 該 RQ 兩模型中 Accuracy 較佳者。**

| RQ | 任務 | 模型 | Accuracy | Sensitivity | Specificity | Macro-F1 |
|----|------|------|----------|-------------|-------------|----------|
| RQ1 | normal_v_abnormal | image | **0.879±0.078** | 0.848±0.091 | 0.914±0.114 | 0.878 |
| RQ1 | normal_v_abnormal | fusion | 0.866±0.107 | 0.848±0.091 | 0.886±0.140 | 0.866 |
| RQ2a | angle_3class | image | 0.518±0.253 | 0.510±0.137 | 0.741±0.121 | 0.352 |
| RQ2a | angle_3class | fusion | **0.774±0.042** | 0.550±0.079 | 0.830±0.056 | 0.532 |
| RQ2b | angle_binary_extreme | image | **0.885±0.086** | 0.867±0.163 | 0.891±0.122 | 0.852 |
| RQ2b | angle_binary_extreme | fusion | 0.871±0.080 | 0.867±0.163 | 0.871±0.106 | 0.841 |
| RQ3 | oi_emphysema | image | 0.803±0.036 | 0.767±0.065 | 0.843±0.106 | 0.802 |
| RQ3 | oi_emphysema | fusion | **0.834±0.088** | 0.848±0.106 | 0.819±0.149 | 0.832 |

> 註:RQ1、RQ2b 兩模型 Sensitivity 連 mean±std 都相同(RQ1 皆 0.848±0.091、RQ2b 皆 0.867±0.163),
> **並非貼錯**,而是每折陽性類別樣本少(RQ2b 每折 3 例、RQ1 每折 6~7 例)、sensitivity 只能取幾個離散值,
> 兩模型剛好落在**同一組值的集合**所致的小樣本巧合(逐折值:RQ1 image `[.714,1.0,.857,.833,.833]` vs
> fusion `[.714,.857,1.0,.833,.833]`,僅 fold1↔fold2 對調;RQ2b 亦僅一折對調)。**實際漏判的 fold 與個案
> 兩模型並不相同**——例如 RQ1 fold1 image 零漏判、fusion 漏 1,fold2 反之(見各 `results.json` 混淆矩陣),
> 故非「集中於同一批個案」的共同錯誤模式。

### 迴歸任務(RQ2c 塌陷角角度值)

**計算口徑:pooled(合併全 66 例 held-out 預測後計一組指標)。** RMSE、R²、Pearson
必須同口徑,否則會自相矛盾——早期版本四欄皆取**逐折平均**,而逐折 R² 是用各折自己的
局部變異(~13 樣本)正規化,跨折平均後不再與逐折平均 RMSE 單調,導致「RMSE 較低卻 R² 較低」
的數學上不可能組合。pooled 口徑下 SST 唯一,RMSE↔R² 恢復單調、內部自洽。

| RQ | 任務 | 模型 | MAE (°) | RMSE (°) | R² | Pearson r |
|----|------|------|---------|----------|-----|-----------|
| RQ2c | angle | image | 15.49 | 23.55 | -0.004 | 0.376 |
| RQ2c | angle | fusion | **14.60** | **21.00** | **0.202** | **0.509** |

> 逐折 mean±std(供離散度參考,**非**主表口徑):image MAE 15.48±1.09、RMSE 23.43±2.41、
> R² -0.003±0.156、Pearson 0.422±0.068;fusion MAE 14.61±2.17、RMSE 20.72±3.59、
> R² 0.174±0.299、Pearson 0.525±0.209。N=66、每折 ~13 例使逐折 R² 極不穩(fusion 折間 -0.13~+0.66),
> 故主表採 pooled;若需區間,建議報 bootstrap 95% CI 而非 ±std。

### 觀察

- **TAP-CT late fusion 在角度/OI 相關任務明顯有幫助**:RQ2a 三分類 image 幾乎失效(Acc 0.52、Macro-F1 0.35),
  fusion 拉到 Acc 0.77;RQ2c 迴歸 image 的 R² 幾乎 0(-0.004),fusion 升到 0.202(pooled);RQ3 OI fusion 也優於 image。
- **正異常(RQ1)與極端二分類(RQ2b)image-only 已很強**(Acc 0.88–0.89),fusion 略降 —— 與先前
  54-case 的發現一致(這兩個任務的 CT 影像訊號已足夠,frozen embedding 反而稀釋)。
- 每個 run 的完整 per-fold 指標與混淆矩陣見 `figures/<task>/<uuid>/results.json`;UUID 對照見
  `train_log/run_all_rq/summary.json`。
