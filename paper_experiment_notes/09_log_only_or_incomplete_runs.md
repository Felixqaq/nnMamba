# Log-Only Or Incomplete Runs

這裡列出有 `train_log` 但沒有對應 `results.json` 的 run。這些數字只從 log 文字粗略解析，不能取代正式 `results.json`，但可用來追早期實驗脈絡。

| source | task | uuid | log_files | parsed_metric_lines | best_epoch | best_auc | best_acc | best_mae | best_rmse | log_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | Normal_v_Abnormal | nnMamba_2026-02-02_12:02:55 | 1 | 0 |  |  |  |  |  | train_log/Normal_v_Abnormal/nnMamba_2026-02-02_12:02:55 |
| classification | Normal_v_Abnormal | nnMamba_2026-02-02_17:51:20 | 1 | 0 |  |  |  |  |  | train_log/Normal_v_Abnormal/nnMamba_2026-02-02_17:51:20 |
| classification | Normal_v_Abnormal | nnMamba_2026-02-02_19:03:19 | 1 | 50 | 15 | 1.0 | 0.96923 |  |  | train_log/Normal_v_Abnormal/nnMamba_2026-02-02_19:03:19 |
| classification | Normal_v_Abnormal | nnMamba_2026-02-12_15:04:19 | 1 | 50 | 5 | 1.0 | 0.9 |  |  | train_log/Normal_v_Abnormal/nnMamba_2026-02-12_15:04:19 |
| classification | Normal_v_Abnormal | nnMamba_2026-02-12_16:00:50 | 1 | 50 | 5 | 1.0 | 0.8 |  |  | train_log/Normal_v_Abnormal/nnMamba_2026-02-12_16:00:50 |
| regression | Angle_3class_classification | hybrid_mamba_attention_2026-04-29_14:53:29 | 4 | 0 |  |  |  |  |  | regression/train_log/Angle_3class_classification/hybrid_mamba_attention_2026-04-29_14:53:29 |
| regression | Angle_3class_classification | hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-05_14:43:51 | 2 | 0 |  |  |  |  |  | regression/train_log/Angle_3class_classification/hybrid_mamba_attention_balanced_sampling_aug300_class_2026-05-05_14:43:51 |
| regression | Angle_3class_classification | hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug300_class_2026-05-13_17:50:49 | 3 | 0 |  |  |  |  |  | regression/train_log/Angle_3class_classification/hybrid_mamba_tapct_fusion_tap_ct_late_fusion_aug300_class_2026-05-13_17:50:49 |
| regression | GOLD_stage_classification | hybrid_mamba_attention_2026-04-22_12:40:42 | 2 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_12:40:42 |
| regression | GOLD_stage_classification | hybrid_mamba_attention_2026-04-22_14:29:24 | 1 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_2026-04-22_14:29:24 |
| regression | GOLD_stage_classification | hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:13:06 | 1 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/hybrid_mamba_attention_gold_balanced_sampling_aug36_class_2026-05-07_11:13:06 |
| regression | GOLD_stage_classification | mamba_2026-04-16_14:36:52 | 1 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/mamba_2026-04-16_14:36:52 |
| regression | GOLD_stage_classification | mamba_2026-04-16_14:40:01 | 5 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/mamba_2026-04-16_14:40:01 |
| regression | GOLD_stage_classification | swinunetr_2026-04-16_14:35:09 | 1 | 0 |  |  |  |  |  | regression/train_log/GOLD_stage_classification/swinunetr_2026-04-16_14:35:09 |
| regression | PFT_angle_regression | hybrid_mamba_attention_2026-04-09_13:48:45 | 3 | 0 | 80 |  |  | 15.5806 | 20.3474 | regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:48:45 |
| regression | PFT_angle_regression | hybrid_mamba_attention_2026-04-09_14:25:55 | 1 | 0 | 30 |  |  | 17.6948 | 23.9768 | regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:25:55 |
| regression | PFT_angle_regression | hybrid_mamba_attention_2026-04-09_14:34:41 | 1 | 0 | 25 |  |  | 17.7839 | 21.5617 | regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:34:41 |
| regression | PFT_angle_regression | hybrid_mamba_attention_2026-04-09_14:36:41 | 3 | 0 | 35 |  |  | 11.7892 | 15.2336 | regression/train_log/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_14:36:41 |
| regression | PFT_angle_regression | mamba_2026-04-08_14:09:01 | 2 | 0 | 40 |  |  | 17.1303 | 24.9892 | regression/train_log/PFT_angle_regression/mamba_2026-04-08_14:09:01 |
| regression | PFT_angle_regression | mamba_2026-04-08_14:12:00 | 1 | 0 | 10 |  |  | 19.538 | 23.6502 | regression/train_log/PFT_angle_regression/mamba_2026-04-08_14:12:00 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-01_18:20:29 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-01_18:20:29 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-01_18:21:15 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-01_18:21:15 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-01_18:22:48 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-01_18:22:48 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-01_18:24:11 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-01_18:24:11 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-01_18:24:49 | 1 | 0 | 10 |  |  | 20.8701 | 25.0407 | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-01_18:24:49 |
| regression | PFT_angle_regression | nnMambaReg_2026-04-08_12:51:27 | 1 | 0 | 40 |  |  | 14.4669 | 20.8584 | regression/train_log/PFT_angle_regression/nnMambaReg_2026-04-08_12:51:27 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_13:44:02 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_13:44:02 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_13:48:42 | 1 | 0 | 25 |  |  | 15.9946 | 18.4997 | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_13:48:42 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_13:52:45 | 1 | 0 | 25 |  |  | 14.0455 | 19.3694 | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_13:52:45 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_13:56:18 | 1 | 0 | 5 |  |  | 19.2184 | 23.6567 | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_13:56:18 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_13:58:42 | 1 | 0 | 5 |  |  | 19.5265 | 23.6258 | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_13:58:42 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_14:01:06 | 1 | 0 |  |  |  |  |  | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_14:01:06 |
| regression | PFT_angle_regression | swinunetr_2026-04-08_14:02:18 | 2 | 0 | 15 |  |  | 18.1755 | 24.8995 | regression/train_log/PFT_angle_regression/swinunetr_2026-04-08_14:02:18 |
