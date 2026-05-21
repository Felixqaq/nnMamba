# Paper Notes Coverage Audit

Audit date: `2026-05-21`

## Verdict

這個資料夾在本次補件前，已經把本地 `classification/` 與 `regression/` 實驗整理得相當完整；但還不能說「整個專案的所有資訊都已經在 md 中」，因為：

1. 原本的 source inventory 明確只逐檔索引 `classification/` 和 `regression/`。
2. root-level `nnMamba.py`, `nnMamba4cls.py` 與大型 `nnunet/` segmentation/landmark stack 沒有一份專案級 code map。
3. root `README.md` 的 upstream nnMamba benchmark tables 和本地 COPD 實驗 artifact 來源不同，原本筆記沒有把這個邊界說得夠醒目。
4. 權重、CT/NIfTI、TAP-CT `.npz`、大量 figure、逐病人 prediction/error 檔案不適合整份嵌入 Markdown，目前只做索引或摘要。

本次已新增 [10_project_codebase_map.md](10_project_codebase_map.md) 補上全專案程式地圖，並把這份 audit 接進 `README.md` 與 `00_index.md`。所以現在的狀態是：

> 論文寫作最需要的本地實驗、設定、資料 manifest、result JSON、artifact index、以及整個 repo 的程式碼分層都已經有 md 導航；但這不是整個 repo 所有原始檔與二進位 artifact 的全文備份。

## Audit Basis

本次比對了：

- `paper_experiment_notes/` 的 index、methods、configs、dataset manifests、result tables、best-result table、TAP-CT summary、artifact inventory、run pages、source inventory。
- repo root 的 `README.md`, `thesis_experiment_summary.md`, `nnMamba.py`, `nnMamba4cls.py`。
- `classification/` docs, config, core/data/network entry points。
- `regression/` docs, config family, core/data/network/scripts/tests entry points。
- `nnunet/` 的 architecture/training/inference/planning/evaluation structure。

## Coverage Matrix

| information type | status in this notes folder | where to read |
| --- | --- | --- |
| Local classification methods and metrics | covered | [01_methods_and_pipeline.md](01_methods_and_pipeline.md) |
| Local regression/classification unified pipeline | covered | [01_methods_and_pipeline.md](01_methods_and_pipeline.md), [10_project_codebase_map.md](10_project_codebase_map.md) |
| Regression target definitions: angle, GOLD, angle 3-class, extreme binary | covered | [01_methods_and_pipeline.md](01_methods_and_pipeline.md), [03_dataset_and_manifests.md](03_dataset_and_manifests.md) |
| YAML configs used by current experiments | covered with full config pages | [02_config_catalog.md](02_config_catalog.md), `configs/` |
| Generated manifests and label JSON summaries | covered, many full JSON appendices embedded | [03_dataset_and_manifests.md](03_dataset_and_manifests.md) |
| Finished local result JSON tables | covered | [04_all_results_master_table.md](04_all_results_master_table.md), `results/`, `runs/` |
| Best current paper-table candidates | covered | [05_best_results_for_paper.md](05_best_results_for_paper.md) |
| TAP-CT embedding probe summaries | covered | [06_tapct_embedding_probe_summary.md](06_tapct_embedding_probe_summary.md) |
| Logs/weights/figures with or without finished result JSON | covered as inventory | [07_artifact_inventory.md](07_artifact_inventory.md), [09_log_only_or_incomplete_runs.md](09_log_only_or_incomplete_runs.md) |
| Reproducibility commands and paper-table checklist | covered | [08_reproducibility_checklist.md](08_reproducibility_checklist.md) |
| Classification/regression source-file inventory | covered | [appendices/source_file_inventory.md](appendices/source_file_inventory.md) |
| Root model definitions and repo-level stack boundaries | covered after this audit | [10_project_codebase_map.md](10_project_codebase_map.md) |
| `nnunet/` segmentation and landmark code organization | summarized after this audit | [10_project_codebase_map.md](10_project_codebase_map.md) |
| Upstream nnMamba published benchmark tables in root README | identified, not duplicated into local result tables | root `README.md`, [10_project_codebase_map.md](10_project_codebase_map.md) |
| Every `nnunet/` source file copied into Markdown | intentionally not covered | use source tree; code map summarizes the stack |
| Binary weights, raw CT volumes, embeddings, full patient-level prediction files | intentionally not embedded | use artifact paths from run pages and inventories |

## What The Existing Notes Already Prove

The current notebook already records these local artifact counts:

| item | count |
| --- | ---: |
| training/probe `results.json` files scanned | 57 |
| standard training run pages written | 52 |
| flattened TAP-CT probe rows | 33 |
| documented YAML/YML configs | 28 |
| generated manifest/label JSON files summarized | 22 |

The current project scan also found:

| source area | scan observation |
| --- | --- |
| `classification/` | compact maintained classifier pipeline with source/config/docs under one folder |
| `regression/` | current COPD experiment pipeline with CT-only, TAP-CT probe, and TAP-CT fusion variants |
| `nnunet/` | `291` Python files in the dense-prediction framework stack |

## Remaining Gaps That Matter For A Thesis

| gap | why it matters | recommended handling |
| --- | --- | --- |
| Which upstream nnMamba benchmark results are only background | avoids mixing BraTS/AMOS/ADNI/landmark benchmarks with local COPD experiments | cite/use root README or original paper context separately from local result tables |
| Raw dataset provenance beyond local folder/manifest facts | thesis methods may need scanner, cohort, IRB, inclusion/exclusion, acquisition protocol | add human-verified cohort/protocol text when available |
| Final experiment selection rationale | too many exploratory runs can blur the story | use [05_best_results_for_paper.md](05_best_results_for_paper.md) plus a hand-written Methods/Results narrative |
| Patient-level error analysis and representative cases | paper discussion may need false positive/false negative examples | open prediction/error files from run pages rather than embedding all patient data in notes |
| Re-run stability across seeds for key small-sample results | current best row may be unstable | document repeated-seed runs if they become part of the final claim |

## Thesis-Safe Reading Order

1. Start with [10_project_codebase_map.md](10_project_codebase_map.md) to know which code path produced which evidence.
2. Use [01_methods_and_pipeline.md](01_methods_and_pipeline.md), [02_config_catalog.md](02_config_catalog.md), and [03_dataset_and_manifests.md](03_dataset_and_manifests.md) for Methods.
3. Use [05_best_results_for_paper.md](05_best_results_for_paper.md) first for Results, then verify each chosen row through its linked run page and `results.json`.
4. Use [07_artifact_inventory.md](07_artifact_inventory.md) and [09_log_only_or_incomplete_runs.md](09_log_only_or_incomplete_runs.md) only for experiment history unless the run has formal result evidence.
