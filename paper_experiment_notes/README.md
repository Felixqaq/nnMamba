# Paper Experiment Notes

Generated on `2026-05-15T12:00:18` from `/home/felix/Research/nnMamba`.

這個資料夾是為論文寫作整理的實驗資料庫。重點是 `classification/` 與 `regression/`，包含方法、設定、資料集 manifest、所有 `results.json` 的結果表、每個 run 的完整結果內容與 artifact 路徑。

## Start Here

- [01_methods_and_pipeline.md](01_methods_and_pipeline.md): Method 章節可直接用的管線、資料、模型與 metric 事實。
- [02_config_catalog.md](02_config_catalog.md): 所有 YAML config 的索引；完整 YAML 也拆成單篇。
- [03_dataset_and_manifests.md](03_dataset_and_manifests.md): 資料來源、manifest、label JSON 與完整 manifest appendix。
- [04_all_results_master_table.md](04_all_results_master_table.md): 所有 training `results.json` 的總表。
- [05_best_results_for_paper.md](05_best_results_for_paper.md): 每個 task 目前最適合放進論文表格的最佳候選。
- [06_tapct_embedding_probe_summary.md](06_tapct_embedding_probe_summary.md): TAP-CT embedding probe 的攤平比較表。
- [07_artifact_inventory.md](07_artifact_inventory.md): 所有 figures/train_log/weights run directory 索引，包含 incomplete/result-less runs。
- [08_reproducibility_checklist.md](08_reproducibility_checklist.md): 重跑與放入論文前的確認清單。
- [09_log_only_or_incomplete_runs.md](09_log_only_or_incomplete_runs.md): 有 log/weight/figure 但缺正式 `results.json` 的早期或未完成 run。
- [10_project_codebase_map.md](10_project_codebase_map.md): 整個 repo 的程式碼地圖，區分原始 nnMamba stack、classification、regression 與 artifact。
- [11_coverage_audit.md](11_coverage_audit.md): 這份筆記庫目前已覆蓋與仍缺少的資訊稽核，避免把 local COPD 實驗和 upstream nnMamba 發表結果混在一起。

## Result Pages By Task

- [Angle_3class_classification](results/Angle_3class_classification.md)
- [Angle_extreme_binary_classification](results/Angle_extreme_binary_classification.md)
- [GOLD_stage_classification](results/GOLD_stage_classification.md)
- [Normal_v_Abnormal](results/Normal_v_Abnormal.md)
- [PFT_angle_regression](results/PFT_angle_regression.md)

## Config Pages

- [classification/config.yaml](configs/classification_config.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation100.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation100.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation200.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation200.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation300.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation300.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation50.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation50.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation75.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation75.yaml.md)
- [regression/config.angle_3class.balanced_sampling.augmentation_x12.yaml](configs/regression_config.angle_3class.balanced_sampling.augmentation_x12.yaml.md)
- [regression/config.angle_3class.balanced_sampling.yaml](configs/regression_config.angle_3class.balanced_sampling.yaml.md)
- [regression/config.angle_3class.tapct_abmil.s25d.yaml](configs/regression_config.angle_3class.tapct_abmil.s25d.yaml.md)
- [regression/config.angle_3class.tapct_abmil.yaml](configs/regression_config.angle_3class.tapct_abmil.yaml.md)
- [regression/config.angle_3class.tapct_abmil_fusion.augmentation100.yaml](configs/regression_config.angle_3class.tapct_abmil_fusion.augmentation100.yaml.md)
- [regression/config.angle_3class.tapct_attention_fusion.augmentation100.yaml](configs/regression_config.angle_3class.tapct_attention_fusion.augmentation100.yaml.md)
- [regression/config.angle_3class.tapct_late_fusion.augmentation100.yaml](configs/regression_config.angle_3class.tapct_late_fusion.augmentation100.yaml.md)
- [regression/config.angle_3class.tapct_late_fusion.yaml](configs/regression_config.angle_3class.tapct_late_fusion.yaml.md)
- [regression/config.angle_3class.tapct_s25d_late_fusion.augmentation100.yaml](configs/regression_config.angle_3class.tapct_s25d_late_fusion.augmentation100.yaml.md)
- [regression/config.angle_3class.yaml](configs/regression_config.angle_3class.yaml.md)
- [regression/config.angle_binary_extreme.balanced_sampling.augmentation100.yaml](configs/regression_config.angle_binary_extreme.balanced_sampling.augmentation100.yaml.md)
- [regression/config.angle_binary_extreme.tapct_late_fusion.yaml](configs/regression_config.angle_binary_extreme.tapct_late_fusion.yaml.md)
- [regression/config.angle_binary_extreme.yaml](configs/regression_config.angle_binary_extreme.yaml.md)
- [regression/config.gold.balanced_sampling.augmentation36.yaml](configs/regression_config.gold.balanced_sampling.augmentation36.yaml.md)
- [regression/config.gold.yaml](configs/regression_config.gold.yaml.md)
- [regression/config.hybrid.preset.yaml](configs/regression_config.hybrid.preset.yaml.md)
- [regression/config.smoke.yaml](configs/regression_config.smoke.yaml.md)
- [regression/config.tapct_embedding_probe.yaml](configs/regression_config.tapct_embedding_probe.yaml.md)
- [regression/config.tapct_s_25d_embedding_probe.yaml](configs/regression_config.tapct_s_25d_embedding_probe.yaml.md)
- [regression/config.yaml](configs/regression_config.yaml.md)
- [regression/environment.tapct.yml](configs/regression_environment.tapct.yml.md)

## Coverage

- Training/probe `results.json` files scanned: `57`。
- Standard training run pages written: `52`。
- TAP-CT probe rows flattened: `33`。
- Config files documented: `28`。
- Generated manifest/label JSON files summarized: `22`。
- Whole-project scope audit added: root-level `nnMamba*.py` 與 `nnunet/` 已在 code map/audit 中標出；逐檔完整 artifact 索引仍以本地 COPD `classification/`、`regression/` 實驗為主。

## Important Note

每個 run 頁已嵌入完整 `results.json`。Prediction/error CSV/JSON、figure、log、weight 檔案因為數量多且常含逐病人資料，統一以路徑、大小、row count/key summary 索引；需要追單一病人錯分或 residual 時，從 run 頁的 Artifact Index 直接打開原檔。
