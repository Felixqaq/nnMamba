# Best Results For Paper Tables

這裡只列每個 task 依目前主要指標排序後的最佳候選；完整排名請看 `results/` 與 `04_all_results_master_table.md`。

| task | run | model | experiment | primary_metric | primary_value | mean_mae | mean_rmse | mean_r2 | mean_pearson | mean_auc | mean_accuracy | mean_macro_f1 | mean_bal_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Angle_3class_classification | [hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18:21:20](runs/regression/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug100_class_2026-05-13_18_21_20.md) | hybrid_mamba_tapct_fusion | TAP-CT late fusion + aug100/class | mean_macro_f1 | 0.61406 |  |  |  |  |  | 0.82088 | 0.61406 | 0.63259 |
| Angle_extreme_binary_classification | [hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17:32:25](runs/regression/Angle_extreme_binary_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_extreme_binary_aug100_class_2026-05-13_17_32_25.md) | hybrid_mamba_tapct_fusion | TAP-CT late fusion extreme binary + aug100/class | mean_macro_f1 | 0.8763 |  |  |  |  |  | 0.91923 | 0.8763 | 0.87778 |
| GOLD_stage_classification | [hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:17:40](runs/regression/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11_17_40.md) | hybrid_mamba_attention | GOLD balanced sampling + aug36/class | mean_macro_f1 | 0.53231 |  |  |  |  |  | 0.56061 | 0.53231 | 0.52222 |
| Normal_v_Abnormal | [nnMamba_2026-02-12_15:32:33](runs/classification/Normal_v_Abnormal/nnMamba_2026-02-12_15_32_33.md) | nnmamba |  | mean_auc | 1 |  |  |  |  | 1 | 0.94094 |  |  |
| PFT_angle_regression | [hybrid_mamba_attention_2026-04-09_13:58:58](runs/regression/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13_58_58.md) | hybrid_mamba_attention |  | mean_mae | 15.85124 | 15.85124 | 21.26467 | 0.17234 | 0.66989 |  |  |  |  |
