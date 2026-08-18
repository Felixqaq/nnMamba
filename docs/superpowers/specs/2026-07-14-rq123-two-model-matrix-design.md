# RQ 1/2/3 — 兩主模型 × 66 病人 對齊實驗矩陣 設計文件

日期：2026-07-14
狀態：已與使用者確認,待實作

## 1. 目的

把目前散落的 regression 實驗重構成一個乾淨、可寫進論文的 **RQ 1/2/3** 結構。
每個 RQ 用**兩個主模型**評估,全部在**同一份 66 病人世代**、**完全對齊的訓練協定**下執行,
讓「兩模型的唯一差異 = 有無 TAP-CT late fusion」這個宣稱在論文中站得住腳。

> 註：使用者原始構想含第三個方法「量化標記」,經確認**本次不做、不納入範圍**。

## 2. 研究問題 (RQ)

| RQ | 主題 | 任務 (target_mode) |
|----|------|--------------------|
| **RQ1** | COPD 正常 vs 異常 | `normal_v_abnormal` |
| **RQ2** | 塌陷角 (collapse angle) | `angle_3class`、`angle_binary_extreme`、`angle`(迴歸) |
| **RQ3** | OI | `oi_emphysema`(二分類) |

## 3. 兩個主模型

| 模型 | `model.name` | 說明 |
|------|--------------|------|
| Mamba + Attention (image-only) | `hybrid_mamba_attention` | CT 影像編碼器,無外部 embedding |
| TAP-CT Late Fusion | `hybrid_mamba_tapct_fusion` | 同一 CT 編碼器,後段串接凍結的 TAP-CT-S-3D patient embedding |

兩者在同一任務內**除了上述兩行之外完全相同**。

## 4. 統一世代 (66 病人)

- **單一資料來源**：`../by_angle_all`(剛好 66 個 NIfTI,覆蓋全部 66 個病人 ID)。
- Label 由 `target_mode` + `../patient_angle_classification_by_group.json` 決定,不靠資料夾名。
- Fusion embedding 一律用 `./embeddings/tapct_s_3d/features.npz`(1152 維,涵蓋 66 病人)。
- **淘汰 merlin**(只有 54 病人,無法對齊 66)。

### 世代例外(任務本質造成,非缺陷)
- `angle_binary_extreme`(RQ2b)依定義**排除中間灰區病人**,使用 66 的固定子集。
  其餘 4 個任務(RQ1、RQ2a、RQ2c、RQ3)使用滿 66 病人。

## 5. 對齊訓練協定(全 10 個 config 一致)

固定協定(方案 A：固定 epoch、關早停),理由是根除專案既有的「用 test fold 選 epoch / 早停」
造成的樂觀偏差(見專案記憶 S209/S210),並讓兩模型絕對可比。

```
seed: 42
k_folds: 5
epochs: 100
early_stopping.enabled: false        # 固定預算,最終 epoch 評估,不用 test fold 選停點
augmentation.enabled: true
augmentation.balance_then_augment: true
augmentation.views_per_sample: 5
balanced_sampling: true
image_size: [112, 136, 112]
intensity_window: [-1000.0, 400.0]
input_normalization: zscore
gradcam.enabled: false               # 只為速度,不影響指標
決策: argmax,不做 train-threshold 調整
```

### 任務本質差異(允許,不算破壞對齊)
- 分類任務:`loss = cross_entropy`,`target_normalization = none`,`num_classes` 依任務
  (正異常=2、angle_3class=3、angle_binary_extreme=2、oi_emphysema=2)。
- 迴歸任務(angle 值):`num_classes = 1`,`loss = auto`,`target_normalization = zscore`。
- batch size 依任務記憶體需求設定(迴歸體積大者較小),但同一任務兩模型相同。

### 兩模型唯一差異
| 欄位 | image-only | fusion |
|------|-----------|--------|
| `model.name` | `hybrid_mamba_attention` | `hybrid_mamba_tapct_fusion` |
| `data.tapct_features` | `null` | `./embeddings/tapct_s_3d/features.npz` |
| `model.tapct_embedding_dim` | (不使用) | `1152` |

## 6. 交付物

### 6.1 十個對齊 config(舊 config 一律保留不動,當探索紀錄)
| 檔名 | target_mode | 病人數 |
|------|-------------|--------|
| `config.rq1.normal_v_abnormal.image.yaml` / `.fusion.yaml` | normal_v_abnormal | 66 |
| `config.rq2a.angle_3class.image.yaml` / `.fusion.yaml` | angle_3class | 66 |
| `config.rq2b.angle_binary.image.yaml` / `.fusion.yaml` | angle_binary_extreme | 66 子集(排除灰區) |
| `config.rq2c.angle_reg.image.yaml` / `.fusion.yaml` | angle | 66 |
| `config.rq3.oi_emphysema.image.yaml` / `.fusion.yaml` | oi_emphysema | 66 |

### 6.2 66 病人 manifest
- 同一任務的 image 與 fusion **共用同一份 manifest / 同一 fold 分割**;fusion 僅多設 `tapct_features`。
- **RQ1 需新建** 66 病人 `normal_v_abnormal` manifest(由 `by_angle_all` + labels_json 重建;
  目前的 `.cmp.json` 只有 54)。
- RQ2/RQ3 的 by_angle_all manifest 若已涵蓋 66 則沿用,否則以相同規則重生。

### 6.3 驅動腳本 `regression/scripts/run_all_rq.py`
- 依序執行 10 個 config:`python train.py --config <path>`(cwd = `regression/`,conda env `nnMamba`)。
- 每個 config 獨立 log 檔;記錄回傳 UUID。
- **可續跑**:已完成者(輸出/checkpoint 已存在)跳過。
- **失敗不中斷**:記錄失敗、繼續下一個,最後印出總結(成功/失敗/UUID 對照表)。

### 6.4 RQ 組織文件 `regression/docs/rq_overview.md`
- 三個 RQ 的敘述、2 模型矩陣、66 病人世代、對齊協定說明。
- 結果表(image vs fusion,各 RQ/任務),欄位:Acc / AUC / 其他;跑完回填,未跑標 TBD。

## 7. 執行順序(高層)
1. 建立/驗證 66 病人 manifest(先 RQ1 新建)。
2. 產出 10 個對齊 config。
3. 寫 `run_all_rq.py` 並以單一 config smoke 驗證進入點正確。
4. 背景依序跑完 10 個。
5. 回填 `rq_overview.md` 結果表。

## 8. 明確不做 (YAGNI)
- 不做「量化標記」方法。
- 不刪除既有舊 config / manifest(只新增、保留歷史)。
- 不引入新的 augmentation 種類、不改模型架構、不動 merlin/abmil/attention-fusion 等探索分支。
