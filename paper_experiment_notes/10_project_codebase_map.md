# Project Codebase Map

這份文件補上 `paper_experiment_notes` 原本最缺的全專案視角。現有 run/result/config 筆記已經把本地 COPD 實驗寫得很細；這裡回答另一個問題：repo 裡每一塊程式碼在做什麼，哪些是這次論文最直接會用到的 pipeline，哪些是原始 nnMamba 發表版 stack。

## One-Page Scope

| area | role in repo | main entry points / anchors | paper use |
| --- | --- | --- | --- |
| repo root | 原始 nnMamba 介紹、核心 model definitions、CT label JSON 與本地資料根目錄 | `README.md`, `nnMamba.py`, `nnMamba4cls.py`, `patient_angle_classification_by_group.json`, `pft.json` | 交代 nnMamba 背景與 label 來源；不要把 upstream 發表結果直接當成本地 COPD 實驗結果 |
| `classification/` | 3D CT binary classification pipeline | `train.py`, `evaluate.py`, `models.py`, `core/`, `data/`, `networks/` | 本地 `Normal_v_Abnormal` baseline 與早期 classification 實驗 |
| `regression/` | 目前論文主體的 unified CT pipeline | `train.py`, `evaluate.py`, `models.py`, `core/`, `data/`, `networks/`, `scripts/` | PFT angle regression、Angle 3-class、Angle extreme binary、GOLD、TAP-CT probe/fusion |
| `nnunet/` | 原始 segmentation 與 landmark-detection stack，含 nnUNet planning/training/inference variants | `network_architecture/nnMamba.py`, `training/network_training/`, `run/`, `inference/`, `experiment_planning/` | 說明 repo 仍含 segmentation/landmark 實作；本地 COPD artifact 表不是從這裡生成 |
| output/artifact dirs | 訓練結果、圖、權重、logs、embedding metadata | `figures/`, `train_log/`, `weights/`, `regression/figures/`, `regression/train_log/`, `regression/weights/`, `regression/embeddings/` | 結果表、run pages、可重現性追蹤 |

## Root-Level nnMamba Definitions

| file | what it contains | note for writing |
| --- | --- | --- |
| `nnMamba.py` | `nnMambaSeg` segmentation network，3D residual encoder-decoder、Mamba blocks、channel attention scaling、skip connections | 對應原始 nnMamba dense prediction side |
| `nnMamba4cls.py` | `nnMambaEncoder` classification network，3D Conv stem、Mamba stem feature path、多尺度 pooled features、MLP classifier | 對應原始 classification model definition |
| `nnunet/network_architecture/nnMamba.py` | nnUNet-compatible `nnMambaSeg` with deep-supervision outputs plus related variants | segmentation 訓練 stack 內實際可掛入 trainer 的 architecture |
| `README.md` | 原始 repo 對 MICCSS/CSS、BraTS、AMOS、ADNI、landmark result tables 的說明 | 可當背景來源，但它列的是 upstream nnMamba paper result context |

## Local Thesis Pipelines

### `classification/`

| layer | files | responsibility |
| --- | --- | --- |
| entry | `classification/train.py`, `classification/evaluate.py` | 啟動 train/eval，讀 YAML config |
| config / registry | `classification/config.yaml`, `classification/core/config.py`, `classification/models.py` | task、hyperparameters、model selection |
| data | `classification/data/dataset.py`, `loader.py`, `transforms.py` | CT/NIfTI loading、resize、DataLoader、task labels |
| train/eval | `classification/core/trainer.py`, `evaluator.py`, `checkpoints.py`, `visualizer.py` | fold training、threshold evaluation、checkpointing、plots |
| networks | `classification/networks/ssm_nnMamba.py`, `conv_Densenet121.py`, `tr_ViT.py`, `tr_crate.py` | nnMamba 與比較模型 |
| scripts | `classification/scripts/` | COPD dataset setup、shape checks、result summary helpers |

這條 pipeline 目前筆記中最可直接引用的本地結果是 `Normal_v_Abnormal`。任務 enum 也保留 `NC_v_AD`, `sMCI_v_pMCI`, `Normal_v_COPD`，但是否放進論文主表要看本地 artifact 是否完整。

### `regression/`

| layer | files | responsibility |
| --- | --- | --- |
| entry | `regression/train.py`, `regression/evaluate.py` | unified regression/classification train/eval |
| config / registry | `regression/config*.yaml`, `regression/core/config.py`, `regression/models.py` | target mode、model family、loss、augmentation、TAP-CT paths |
| data | `regression/data/manifest.py`, `dataset.py`, `loader.py`, `transforms.py` | manifest building、patient-level split、HU preprocessing、virtual augmentation、balanced sampling |
| train/eval | `regression/core/trainer.py`, `evaluator.py`, `checkpoints.py`, `visualizer.py`, `runtime.py` | cross-validation、best metric selection、metric/report outputs |
| networks | `regression/networks/` | Mamba regressor、Hybrid Mamba-Attention、SwinUNETR、TAP-CT late/attention/ABMIL fusion、TAP-CT-only ABMIL |
| scripts | `regression/scripts/` | manifest checks, dataset plots, augmentation generation, result summary, model graph export, TAP-CT embedding extraction/probes |
| tests | `regression/test_*.py` | target-mode, augmentation, early-stopping, hybrid/fusion behavior checks |

這條 pipeline 名稱雖然叫 `regression`，實際支援四種 `target_mode`：

| target mode | paper task | primary output |
| --- | --- | --- |
| `angle` | PFT / Angle of Collapse regression | MAE, RMSE, R2, Pearson |
| `gold` | GOLD stage classification | accuracy, macro-F1, balanced accuracy, confusion matrix |
| `angle_3class` | low/intermediate/high angle classification | macro-F1, balanced accuracy, confusion matrix |
| `angle_binary_extreme` | gray-zone excluded extreme angle binary classification | macro-F1, balanced accuracy, class-0 recall |

## Model Families In The Current Paper Notes

| family | appears in | thesis role |
| --- | --- | --- |
| `nnmamba` / `nnMambaEncoder` | original root classification code and `classification/` | early local 3D CT classifier baseline |
| `MambaAngleRegressor` | `regression/networks/mamba_regressor.py` | CT-only regression baseline |
| `HybridMambaAttentionRegressor` | `regression/networks/hybrid_mamba_attention_regressor.py` | strongest CT-only family in current PFT/angle/GOLD experiments |
| `SwinUNETRV2AngleRegressor` | `regression/networks/swinunetr_v2_regressor.py` | transformer-style comparison |
| TAP-CT probe models | `regression/scripts/extract_tapct_embeddings.py`, `train_embedding_probe.py`, `run_tapct_embedding_probe.py` | frozen foundation embedding low-resource baseline |
| TAP-CT fusion families | `regression/networks/hybrid_mamba_tapct_*` | CT model plus frozen TAP-CT feature fusion ablations |
| `TapctABMILClassifier` | `regression/networks/tapct_abmil_classifier.py` | TAP-CT-only bag-level classifier |

## Data And Artifact Flow

### Classification Flow

1. Read task/config from `classification/config.yaml`.
2. Load CT samples from task-specific folders through `classification/data/`.
3. Resize volumes and run k-fold training.
4. Select checkpoints by validation AUC.
5. Write root-level `weights/`, `train_log/`, and `figures/` artifacts.
6. Summaries enter `04_all_results_master_table.md`, task pages, and per-run pages.

### Regression / COPD Flow

1. Read labels from `patient_angle_classification_by_group.json` and `pft.json`.
2. Build or read generated manifests under `regression/datasets/generated/`.
3. Use patient-level fold logic before virtual augmentation/balanced sampling.
4. Train CT-only, TAP-CT-only, or CT plus TAP-CT fusion model family.
5. Write task-scoped artifacts under `regression/weights/`, `regression/train_log/`, and `regression/figures/`.
6. Results, configs, datasets, run pages, and incomplete logs are indexed in this folder.

### TAP-CT Probe Flow

1. Extract frozen TAP-CT embeddings into `regression/embeddings/`.
2. Aggregate window-level embeddings to patient-level features.
3. Train small sklearn probes for `angle_3class` and `angle_binary_extreme`.
4. Write probe result JSON and plots under `regression/figures/TAPCT_embedding_probes/`.
5. Read the flattened comparison in [06_tapct_embedding_probe_summary.md](06_tapct_embedding_probe_summary.md).

## `nnunet/` Stack In This Repo

`nnunet/` is large because it carries the dense-prediction framework around nnMamba, not just one model file. Its code is organized around:

| subarea | examples | responsibility |
| --- | --- | --- |
| planning / preprocessing | `experiment_planning/`, `preprocessing/` | nnUNet plan generation, dataset analysis, spacing/cropping preprocessing |
| architecture | `network_architecture/` | nnMamba and generic nnUNet network definitions |
| training | `training/network_training/`, `training/data_augmentation/`, `training/loss_functions/` | trainers, augmentation, losses, optimizer variants |
| inference / evaluation | `inference/`, `evaluation/`, `postprocessing/` | prediction export, ensembles, metrics, postprocessing |
| dataset conversion | `dataset_conversion/` | challenge/task conversion scripts |

The current local scan found `291` Python files under `nnunet/`. This is why the paper notes summarize that stack here instead of copying a 291-file tree into the COPD experiment notebook.

## Do Not Mix These Result Sources

| result source | safe interpretation |
| --- | --- |
| root `README.md` BraTS / AMOS / ADNI / landmark tables | 原始 nnMamba 發表版背景與 benchmark context |
| `paper_experiment_notes/results/`, `runs/`, `04_all_results_master_table.md` | 這個 workspace 目前可追溯的本地 `classification/` 與 `regression/` 實驗證據 |
| `07_artifact_inventory.md` rows without matching `results.json` | 只能追早期實驗脈絡，不應直接作 final quantitative table |

## Where To Read Next

- Method facts: [01_methods_and_pipeline.md](01_methods_and_pipeline.md)
- Configs: [02_config_catalog.md](02_config_catalog.md)
- Dataset manifests: [03_dataset_and_manifests.md](03_dataset_and_manifests.md)
- Result master table: [04_all_results_master_table.md](04_all_results_master_table.md)
- Coverage verdict: [11_coverage_audit.md](11_coverage_audit.md)
- Classification/regression source inventory: [appendices/source_file_inventory.md](appendices/source_file_inventory.md)
