# 3D Binary Model Accuracy Summary

這份摘要整理 workspace 內可找到的二分類結果，並以每個模型家族的最高 `mean_accuracy` 作為比較基準。

口徑說明：
- `mamba` 對應 `nnmamba`
- `mamba+attention` 對應 `hybrid_mamba_attention`
- `tapct` 目前 workspace 內沒有獨立的 TAP-CT 深度二分類 run，因此改用 TAP-CT embedding probe 的最高 `mean_accuracy`
- `tapct_late_fusion` 對應 `hybrid_mamba_tapct_fusion`

## 最高準確率

| 模型 | 最佳 run | 任務 / 來源 | mean_accuracy | 備註 |
| --- | --- | --- | --- | --- |
| mamba | [nnMamba_2026-02-12_15:32:33](runs/classification/Normal_v_Abnormal/nnMamba_2026-02-12_15_32_33.md) | Normal_v_Abnormal | 0.94094 | 目前 binary 任務中最高的 nnMamba 結果 |
| tapct_late_fusion | [hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25](runs/regression/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17_32_25.md) | Angle_extreme_binary_classification | 0.91923 | binary late fusion 最高 |
| tapct | [autoplot_smoke](runs/tapct_embedding_probes/autoplot_smoke.md) | TAP-CT embedding probe / angle_binary_extreme | 0.88718 | 以 TAP-CT probe 的最高 accuracy 代表 |
| mamba+attention | [hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14:27:44](runs/regression/Angle_extreme_binary_classification/hybrid_mamba_attention_extreme_binary_balanced_aug100_class_2026-05-06_14_27_44.md) | Angle_extreme_binary_classification | 0.86923 | binary attention 最高 |

## 結論

若只看 `mean_accuracy`，四個模型的排名是：

1. `mamba` - `0.94094`
2. `tapct_late_fusion` - `0.91923`
3. `tapct` - `0.88718`
4. `mamba+attention` - `0.86923`

如果你要，我也可以把這份整理再改成「只保留 binary task 的深度模型」版本，或補上 `mean_balanced_accuracy` / `mean_macro_f1` 的對照表。

## Normal_v_Abnormal (figures/Normal_v_Abnormal)

以下統整僅來自 figures/Normal_v_Abnormal 內有 results.json 的 run。

| run | mean_accuracy | std_accuracy | mean_auc | std_auc | mean_sensitivity | mean_specificity |
| --- | --- | --- | --- | --- | --- | --- |
| [nnMamba_2026-02-12_15:32:33_e5](figures/Normal_v_Abnormal/nnMamba_2026-02-12_15:32:33_e5/results.json) | 0.94094 | 0.04222 | 1.00000 | 0.00000 | 0.90190 | 1.00000 |
| [nnMamba_2026-02-12_15:21:50_e2_soft](figures/Normal_v_Abnormal/nnMamba_2026-02-12_15:21:50_e2_soft/results.json) | 0.91667 | 0.12910 | 0.94857 | 0.10286 | 0.85714 | 1.00000 |
| [nnMamba_2026-03-20_17:04:42](figures/Normal_v_Abnormal/nnMamba_2026-03-20_17:04:42/results.json) | 0.87091 | 0.16826 | 0.94000 | 0.12000 | 0.87143 | 0.88000 |
| [nnMamba_2026-02-12_16:00:5_e1](figures/Normal_v_Abnormal/nnMamba_2026-02-12_16:00:5_e1/results.json) | 0.84909 | 0.16897 | 0.96429 | 0.07143 | 0.78571 | 0.95000 |

最佳 mean_accuracy 是 nnMamba_2026-02-12_15:32:33_e5 (0.94094)。

## RQ2: 三維影像模型二分類結果 - 建議放的實驗數據

若 RQ2 要對外呈現模型效能與穩定度，建議至少放：

- 5-fold mean accuracy + std (主指標，包含變異度)
- mean AUC + std (對 threshold 不敏感的補充)
- mean sensitivity / specificity (臨床解讀必要)
- 合併 confusion matrix (可用 total_cm.png)
- 每個 fold 的樣本數或 class 分布 (避免只看高分但樣本偏斜)

如需要，我可以再把每個 run 的 fold-level accuracy / AUC 表格也補上。