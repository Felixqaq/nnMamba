# Dataset And Label Manifest Summary

這份文件記錄論文方法與結果表會用到的資料來源、manifest、label JSON 與實體增強資料夾。

## Raw / Materialized CT Directories
| directory | n_nii_gz | group_counts |
| --- | --- | --- |
| by_angle_all | 66 | abnormal_low_angle=35, normal_high_angle=31 |
| by_angle_all_angle_3class_augmented | 141 | abnormal_low_angle=110, normal_high_angle=31 |
| by_angle_all_gold_augmented | 144 | abnormal_low_angle=107, normal_high_angle=37 |

## JSON Manifests And Labels
| json | n | counts | class_counts | gold_stage_counts | size |
| --- | --- | --- | --- | --- | --- |
| [regression/datasets/generated/angle_3class_augmented_dataset_summary.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_augmented_dataset_summary.json) |  |  |  |  | 29.5 KB |
| [regression/datasets/generated/angle_3class_manifest.augmented.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.augmented.json) | 141 | {<br>  "total": 141,<br>  "unique_patients": 66,<br>  "low_angle_group": 110,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 47,<br>  "Intermediate (132-151 deg)": 47,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 45,<br>  "GOLD 2 (Moderate)": 32,<br>  "GOLD 3 (Severe)": 43,<br>  "GOLD 4 (Very Severe)": 21<br>} | 57.1 KB |
| [regression/datasets/generated/angle_3class_manifest.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_abmil.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_abmil.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_abmil_fusion_aug100.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_abmil_fusion_aug100.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_abmil_s25d.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_abmil_s25d.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_attention_fusion_aug100.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_attention_fusion_aug100.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_late_fusion.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_late_fusion.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_late_fusion_aug100.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_late_fusion_aug100.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_3class_manifest.tapct_s25d_late_fusion_aug100.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.tapct_s25d_late_fusion_aug100.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "Emphysema/Abnormal (<=131 deg)": 14,<br>  "Intermediate (132-151 deg)": 5,<br>  "Normal (>=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 25.0 KB |
| [regression/datasets/generated/angle_binary_extreme_manifest.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_binary_extreme_manifest.json) | 61 | {<br>  "total": 61,<br>  "unique_patients": 61,<br>  "low_angle_group": 30,<br>  "high_angle_group": 31<br>} | {<br>  "Abnormal/emphysema-like (AC <=131 deg)": 14,<br>  "Normal-like (AC >=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 35,<br>  "GOLD 2 (Moderate)": 9,<br>  "GOLD 3 (Severe)": 11,<br>  "GOLD 4 (Very Severe)": 6<br>} | 23.6 KB |
| [regression/datasets/generated/angle_binary_extreme_manifest.tapct_late_fusion.json](/home/felix/Research/nnMamba/regression/datasets/generated/angle_binary_extreme_manifest.tapct_late_fusion.json) | 61 | {<br>  "total": 61,<br>  "unique_patients": 61,<br>  "low_angle_group": 30,<br>  "high_angle_group": 31<br>} | {<br>  "Abnormal/emphysema-like (AC <=131 deg)": 14,<br>  "Normal-like (AC >=152 deg)": 47<br>} | {<br>  "GOLD 1 (Mild)": 35,<br>  "GOLD 2 (Moderate)": 9,<br>  "GOLD 3 (Severe)": 11,<br>  "GOLD 4 (Very Severe)": 6<br>} | 23.6 KB |
| [regression/datasets/generated/gold_augmented_dataset_summary.json](/home/felix/Research/nnMamba/regression/datasets/generated/gold_augmented_dataset_summary.json) |  |  |  |  | 29.4 KB |
| [regression/datasets/generated/gold_manifest.aug36.json](/home/felix/Research/nnMamba/regression/datasets/generated/gold_manifest.aug36.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 24.6 KB |
| [regression/datasets/generated/gold_manifest.augmented.json](/home/felix/Research/nnMamba/regression/datasets/generated/gold_manifest.augmented.json) | 144 | {<br>  "total": 144,<br>  "unique_patients": 66,<br>  "low_angle_group": 107,<br>  "high_angle_group": 37<br>} |  | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 36,<br>  "GOLD 3 (Severe)": 36,<br>  "GOLD 4 (Very Severe)": 36<br>} | 46.7 KB |
| [regression/datasets/generated/gold_manifest.json](/home/felix/Research/nnMamba/regression/datasets/generated/gold_manifest.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 24.6 KB |
| [regression/datasets/generated/regression_manifest.hybrid.json](/home/felix/Research/nnMamba/regression/datasets/generated/regression_manifest.hybrid.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} |  | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 20.4 KB |
| [regression/datasets/generated/regression_manifest.json](/home/felix/Research/nnMamba/regression/datasets/generated/regression_manifest.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} |  | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 20.4 KB |
| [regression/datasets/generated/regression_manifest.smoke.json](/home/felix/Research/nnMamba/regression/datasets/generated/regression_manifest.smoke.json) | 66 | {<br>  "total": 66,<br>  "unique_patients": 66,<br>  "low_angle_group": 35,<br>  "high_angle_group": 31<br>} |  | {<br>  "GOLD 1 (Mild)": 36,<br>  "GOLD 2 (Moderate)": 11,<br>  "GOLD 3 (Severe)": 13,<br>  "GOLD 4 (Very Severe)": 6<br>} | 20.4 KB |
| [patient_angle_classification_by_group.json](/home/felix/Research/nnMamba/patient_angle_classification_by_group.json) | 66 |  |  |  | 2.1 KB |
| [pft.json](/home/felix/Research/nnMamba/pft.json) | 66 |  | {<br>  "GOLD 1 (輕度)": 36,<br>  "GOLD 2 (中度)": 11,<br>  "GOLD 3 (重度)": 13,<br>  "GOLD 4 (極重度)": 6<br>} |  | 14.7 KB |
| [regression/datasets/patient_angles_simple.json](/home/felix/Research/nnMamba/regression/datasets/patient_angles_simple.json) | 54 |  |  |  | 1.0 KB |

## Full Manifest/Label JSON Appendix
以下嵌入所有 generated manifest 與 label JSON 的完整內容，方便論文寫作時不用切回原檔。
### regression/datasets/generated/angle_3class_augmented_dataset_summary.json
```json
{
  "source_root": "/home/felix/Research/nnMamba/by_angle_all",
  "output_root": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented",
  "manifest": "/home/felix/Research/nnMamba/regression/datasets/generated/angle_3class_manifest.augmented.json",
  "target_count_per_class": 47,
  "source_class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "augmented_class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 47,
    "Intermediate (132-151 deg)": 47,
    "Normal (>=152 deg)": 47
  },
  "total_records": 141,
  "unique_patients": 66,
  "generated_count": 75,
  "generated_records": [
    {
      "patient_id": "1261736",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "1687031",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1800944",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "2588424",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug001_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "4372708",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4372708_aug001_LW-insp.  AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "6887256",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/6887256_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "8404129",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8404129_aug001_Chest C-  5.0  B31f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/B213449_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C435832",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/C435832_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D132855",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/D132855_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E353272_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E558113_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "E647833",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E647833_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "1687031",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1800944",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "2588424",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug002_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "4372708",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4372708_aug002_LW-insp.  AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "6887256",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/6887256_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "8404129",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8404129_aug002_Chest C-  5.0  B31f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/B213449_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C435832",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/C435832_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D132855",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/D132855_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E353272_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E558113_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "E647833",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/E647833_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "1687031",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug003_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1800944",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug003_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "2588424",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug003_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug001_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug001_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug001_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug001_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug002_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug002_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug002_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug002_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug003_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug003_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug003_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug003_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug003_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug004_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug004_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug004_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug004_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug004_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug005_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug005_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug005_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug005_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug005_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug006_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug006_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug006_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug006_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug006_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug007_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug007_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug007_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug007_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug007_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug008_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug008_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug008_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "5630846",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug008_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug008_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "4204917",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug009_LW AXI 3_3  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug009_Thorax Lung Br60 S2 3.00.nii.gz"
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.augmented.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all_angle_3class_augmented",
  "counts": {
    "total": 141,
    "unique_patients": 66,
    "low_angle_group": 110,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 45,
    "GOLD 2 (Moderate)": 32,
    "GOLD 3 (Severe)": 43,
    "GOLD 4 (Very Severe)": 21
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 47,
    "Intermediate (132-151 deg)": 47,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1261736_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1687031_aug003_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/1800944_aug003_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2588424_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug001_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug002_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/2991621_aug003_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug001_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug002_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug003_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug004_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug005_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug006_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug007_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug008_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4204917_aug009_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4372708_aug001_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4372708_aug002_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug003_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug004_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug005_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug006_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug007_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug008_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4796667_aug009_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug001_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug002_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug003_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug004_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug005_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug006_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug007_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5127217_aug008_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug001_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug002_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug003_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug004_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug005_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug006_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug007_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/5630846_aug008_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/6887256_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/6887256_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8404129_aug001_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8404129_aug002_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug001_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug002_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug003_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug004_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug005_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug006_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug007_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/8704416_aug008_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/B213449_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/B213449_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C435832_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C435832_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/D132855_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/D132855_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E353272_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E353272_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E558113_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E558113_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E647833_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all_angle_3class_augmented/abnormal_low_angle/E647833_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all_angle_3class_augmented/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_abmil.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_abmil_fusion_aug100.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_abmil_s25d.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_attention_fusion_aug100.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_late_fusion.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_late_fusion_aug100.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_3class_manifest.tapct_s25d_late_fusion_aug100.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Emphysema/Abnormal (<=131 deg)": 14,
    "Intermediate (132-151 deg)": 5,
    "Normal (>=152 deg)": 47
  },
  "class_names": [
    "Emphysema/Abnormal (<=131 deg)",
    "Intermediate (132-151 deg)",
    "Normal (>=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Intermediate (132-151 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Emphysema/Abnormal (<=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "Normal (>=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_binary_extreme_manifest.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 61,
    "unique_patients": 61,
    "low_angle_group": 30,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 35,
    "GOLD 2 (Moderate)": 9,
    "GOLD 3 (Severe)": 11,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Abnormal/emphysema-like (AC <=131 deg)": 14,
    "Normal-like (AC >=152 deg)": 47
  },
  "class_names": [
    "Abnormal/emphysema-like (AC <=131 deg)",
    "Normal-like (AC >=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/angle_binary_extreme_manifest.tapct_late_fusion.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 61,
    "unique_patients": 61,
    "low_angle_group": 30,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 35,
    "GOLD 2 (Moderate)": 9,
    "GOLD 3 (Severe)": 11,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "Abnormal/emphysema-like (AC <=131 deg)": 14,
    "Normal-like (AC >=152 deg)": 47
  },
  "class_names": [
    "Abnormal/emphysema-like (AC <=131 deg)",
    "Normal-like (AC >=152 deg)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "Abnormal/emphysema-like (AC <=131 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "Normal-like (AC >=152 deg)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/gold_augmented_dataset_summary.json
```json
{
  "source_root": "/home/felix/Research/nnMamba/by_angle_all",
  "output_root": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented",
  "manifest": "/home/felix/Research/nnMamba/regression/datasets/generated/gold_manifest.augmented.json",
  "target_count_per_gold_stage": 36,
  "source_gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "augmented_gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 36,
    "GOLD 3 (Severe)": 36,
    "GOLD 4 (Very Severe)": 36
  },
  "total_records": 144,
  "unique_patients": 66,
  "generated_count": 78,
  "generated_records": [
    {
      "patient_id": "4372708",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug001_LW-insp.  AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "4710629",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug001_LW  AXI 3.0  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/5127217_aug001_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "8126939",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8126939_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "9529629",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/9529629_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "A762364",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/A762364_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D132855",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/D132855_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D550510",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/D550510_aug001_Thorax CM Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "9075311",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/9075311_aug001_Thorax  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "A613117",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/A613117_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "4372708",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug002_LW-insp.  AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "4710629",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug002_LW  AXI 3.0  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "5127217",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/5127217_aug002_Thorax 1_1 Br40 S3 1.00.nii.gz"
    },
    {
      "patient_id": "8126939",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8126939_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "9529629",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/9529629_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "A762364",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/A762364_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D132855",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/D132855_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "D550510",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/D550510_aug002_Thorax CM Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "9075311",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/9075311_aug002_Thorax  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "A613117",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/A613117_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "4372708",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug003_LW-insp.  AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "4710629",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug003_LW  AXI 3.0  I70f  2.nii.gz"
    },
    {
      "patient_id": "4796667",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug003_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1687031",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1687031_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1800944",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1800944_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "3647457",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/3647457_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "5630846",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/5630846_aug001_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "6887256",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/6887256_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "8404129",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8404129_aug001_Chest C-  5.0  B31f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8704416_aug001_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "C435832",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C435832_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C586742",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C586742_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C905524",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C905524_aug001_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz"
    },
    {
      "patient_id": "E647833",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E647833_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "8009284",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/8009284_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "E771850",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/normal_high_angle/E771850_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1687031",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1687031_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1800944",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1800944_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "3647457",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/3647457_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "5630846",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/5630846_aug002_Aorta C+  5.0  B30f.nii.gz"
    },
    {
      "patient_id": "6887256",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/6887256_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "8404129",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8404129_aug002_Chest C-  5.0  B31f.nii.gz"
    },
    {
      "patient_id": "8704416",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/8704416_aug002_Thorax Lung Br60 S3 3.00.nii.gz"
    },
    {
      "patient_id": "C435832",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C435832_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C586742",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C586742_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "C905524",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/C905524_aug002_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2588424",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug001_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug001_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug001_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2588424",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug002_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug002_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug002_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2588424",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug003_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug003_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug003_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug004_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2588424",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug004_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug004_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug004_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug004_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug004_Thorax Lung Br60 S2 3.00.nii.gz"
    },
    {
      "patient_id": "1261736",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug005_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2588424",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug005_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "2991621",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug005_LW  AXI 3.0  B60f.nii.gz"
    },
    {
      "patient_id": "B213449",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug005_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E353272",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug005_LW AXI 3_3  B60f.nii.gz"
    },
    {
      "patient_id": "E558113",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "source_path": "/home/felix/Research/nnMamba/by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "output_path": "/home/felix/Research/nnMamba/by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug005_Thorax Lung Br60 S2 3.00.nii.gz"
    }
  ]
}
```
### regression/datasets/generated/gold_manifest.aug36.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/gold_manifest.augmented.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all_gold_augmented",
  "counts": {
    "total": 144,
    "unique_patients": 66,
    "low_angle_group": 107,
    "high_angle_group": 37
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 36,
    "GOLD 3 (Severe)": 36,
    "GOLD 4 (Very Severe)": 36
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug004_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1261736",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1261736_aug005_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1687031_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1687031_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1800944_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/1800944_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug004_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2588424_aug005_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug001_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug002_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug003_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug004_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/2991621_aug005_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/3647457_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/3647457_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug001_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug002_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4372708_aug003_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug001_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug002_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4710629_aug003_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4796667_aug003_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5127217_aug001_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5127217_aug002_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5630846_aug001_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/5630846_aug002_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/6887256_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/6887256_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8126939_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8126939_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8404129_aug001_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8404129_aug002_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8704416_aug001_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/8704416_aug002_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/9529629_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/9529629_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/A762364_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/A762364_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug004_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/B213449_aug005_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C435832_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C435832_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C586742_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C586742_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C905524_aug001_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/C905524_aug002_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D132855_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D132855_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D550510_aug001_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/D550510_aug002_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug003_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug004_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E353272_aug005_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug002_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug003_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug004_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E558113_aug005_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all_gold_augmented/abnormal_low_angle/E647833_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/8009284_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/9075311_aug001_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/9075311_aug002_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/A613117_aug001_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/A613117_aug002_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/E771850_aug001_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all_gold_augmented/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/gold_manifest.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "class_index": 3,
      "class_label": "GOLD 4 (Very Severe)",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "class_index": 1,
      "class_label": "GOLD 2 (Moderate)",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "class_index": 2,
      "class_label": "GOLD 3 (Severe)",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "class_index": 0,
      "class_label": "GOLD 1 (Mild)",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/regression_manifest.hybrid.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/regression_manifest.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### regression/datasets/generated/regression_manifest.smoke.json
```json
{
  "source_json": "../patient_angle_classification_by_group.json",
  "data_root": "../by_angle_all",
  "counts": {
    "total": 66,
    "unique_patients": 66,
    "low_angle_group": 35,
    "high_angle_group": 31
  },
  "missing_from_source": [],
  "extra_in_source_not_in_json": [],
  "missing_gold_labels": [],
  "gold_stage_counts": {
    "GOLD 1 (Mild)": 36,
    "GOLD 2 (Moderate)": 11,
    "GOLD 3 (Severe)": 13,
    "GOLD 4 (Very Severe)": 6
  },
  "class_names": [
    "GOLD 1 (Mild)",
    "GOLD 2 (Moderate)",
    "GOLD 3 (Severe)",
    "GOLD 4 (Very Severe)"
  ],
  "records": [
    {
      "patient_id": "1261736",
      "path": "../by_angle_all/abnormal_low_angle/1261736_LW AXI 3_3  B60f.nii.gz",
      "angle": 114.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 26.0
    },
    {
      "patient_id": "1604378",
      "path": "../by_angle_all/abnormal_low_angle/1604378_Insp. Phase  Lung  Br60  S3  3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1687031",
      "path": "../by_angle_all/abnormal_low_angle/1687031_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 112.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "1800944",
      "path": "../by_angle_all/abnormal_low_angle/1800944_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 32.0
    },
    {
      "patient_id": "2094528",
      "path": "../by_angle_all/abnormal_low_angle/2094528_LUNG AX 1_1 LW.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2221276",
      "path": "../by_angle_all/abnormal_low_angle/2221276_LW AXI 3_3  B60f.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "2588424",
      "path": "../by_angle_all/abnormal_low_angle/2588424_LW AXI 3_3  B60f.nii.gz",
      "angle": 105.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "2991621",
      "path": "../by_angle_all/abnormal_low_angle/2991621_LW  AXI 3.0  B60f.nii.gz",
      "angle": 117.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 27.0
    },
    {
      "patient_id": "3097765",
      "path": "../by_angle_all/abnormal_low_angle/3097765_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 95.0
    },
    {
      "patient_id": "3647457",
      "path": "../by_angle_all/abnormal_low_angle/3647457_LW AXI 3_3  B60f.nii.gz",
      "angle": 155.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 42.0
    },
    {
      "patient_id": "4204917",
      "path": "../by_angle_all/abnormal_low_angle/4204917_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 140.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 84.0
    },
    {
      "patient_id": "4372708",
      "path": "../by_angle_all/abnormal_low_angle/4372708_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 51.0
    },
    {
      "patient_id": "4710629",
      "path": "../by_angle_all/abnormal_low_angle/4710629_LW  AXI 3.0  I70f  2.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 73.0
    },
    {
      "patient_id": "4796667",
      "path": "../by_angle_all/abnormal_low_angle/4796667_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 136.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "4996166",
      "path": "../by_angle_all/abnormal_low_angle/4996166_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 166.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "5127217",
      "path": "../by_angle_all/abnormal_low_angle/5127217_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 72.0
    },
    {
      "patient_id": "5630846",
      "path": "../by_angle_all/abnormal_low_angle/5630846_Aorta C+  5.0  B30f.nii.gz",
      "angle": 143.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "6887256",
      "path": "../by_angle_all/abnormal_low_angle/6887256_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 36.0
    },
    {
      "patient_id": "8126939",
      "path": "../by_angle_all/abnormal_low_angle/8126939_LW AXI 3_3  B60f.nii.gz",
      "angle": 158.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 61.0
    },
    {
      "patient_id": "8404129",
      "path": "../by_angle_all/abnormal_low_angle/8404129_Chest C-  5.0  B31f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 39.0
    },
    {
      "patient_id": "8704416",
      "path": "../by_angle_all/abnormal_low_angle/8704416_Thorax Lung Br60 S3 3.00.nii.gz",
      "angle": 142.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 34.0
    },
    {
      "patient_id": "9529629",
      "path": "../by_angle_all/abnormal_low_angle/9529629_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 65.0
    },
    {
      "patient_id": "A762364",
      "path": "../by_angle_all/abnormal_low_angle/A762364_LW AXI 3_3  B60f.nii.gz",
      "angle": 159.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 58.0
    },
    {
      "patient_id": "B213449",
      "path": "../by_angle_all/abnormal_low_angle/B213449_LW AXI 3_3  B60f.nii.gz",
      "angle": 108.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 19.0
    },
    {
      "patient_id": "C041635",
      "path": "../by_angle_all/abnormal_low_angle/C041635_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 161.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "C081146",
      "path": "../by_angle_all/abnormal_low_angle/C081146_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 164.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "C435832",
      "path": "../by_angle_all/abnormal_low_angle/C435832_LW AXI 3_3  B60f.nii.gz",
      "angle": 126.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "C543831",
      "path": "../by_angle_all/abnormal_low_angle/C543831_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 163.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 80.0
    },
    {
      "patient_id": "C586742",
      "path": "../by_angle_all/abnormal_low_angle/C586742_LW AXI 3_3  B60f.nii.gz",
      "angle": 165.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 33.0
    },
    {
      "patient_id": "C905524",
      "path": "../by_angle_all/abnormal_low_angle/C905524_Thorax_No CM  3_3  Br60  S2  3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 41.0
    },
    {
      "patient_id": "D132855",
      "path": "../by_angle_all/abnormal_low_angle/D132855_LW AXI 3_3  B60f.nii.gz",
      "angle": 130.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "D550510",
      "path": "../by_angle_all/abnormal_low_angle/D550510_Thorax CM Lung Br60 S3 3.00.nii.gz",
      "angle": 160.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 60.0
    },
    {
      "patient_id": "E353272",
      "path": "../by_angle_all/abnormal_low_angle/E353272_LW AXI 3_3  B60f.nii.gz",
      "angle": 107.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 20.0
    },
    {
      "patient_id": "E558113",
      "path": "../by_angle_all/abnormal_low_angle/E558113_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 113.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 3,
      "gold_stage_label": "GOLD 4 (Very Severe)",
      "post_fev1_percent_predicted": 25.0
    },
    {
      "patient_id": "E647833",
      "path": "../by_angle_all/abnormal_low_angle/E647833_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 116.0,
      "source_group": "abnormal_low_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 44.0
    },
    {
      "patient_id": "0781915",
      "path": "../by_angle_all/normal_high_angle/0781915_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 105.0
    },
    {
      "patient_id": "1596038",
      "path": "../by_angle_all/normal_high_angle/1596038_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 87.0
    },
    {
      "patient_id": "1663485",
      "path": "../by_angle_all/normal_high_angle/1663485_LW  AXI 3.0  B60f.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 116.0
    },
    {
      "patient_id": "1746380",
      "path": "../by_angle_all/normal_high_angle/1746380_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 92.0
    },
    {
      "patient_id": "1814107",
      "path": "../by_angle_all/normal_high_angle/1814107_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "2256243",
      "path": "../by_angle_all/normal_high_angle/2256243_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 111.0
    },
    {
      "patient_id": "2291134",
      "path": "../by_angle_all/normal_high_angle/2291134_LW-insp.  AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "2500824",
      "path": "../by_angle_all/normal_high_angle/2500824_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 100.0
    },
    {
      "patient_id": "2860903",
      "path": "../by_angle_all/normal_high_angle/2860903_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 175.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "3635301",
      "path": "../by_angle_all/normal_high_angle/3635301_Thorax Br60 S3 3.00.nii.gz",
      "angle": 169.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4205212",
      "path": "../by_angle_all/normal_high_angle/4205212_~LUNG AXL 1_1 LW.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    },
    {
      "patient_id": "4230847",
      "path": "../by_angle_all/normal_high_angle/4230847_LW AXI 3_3  B60f.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "4302294",
      "path": "../by_angle_all/normal_high_angle/4302294_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 104.0
    },
    {
      "patient_id": "5046455",
      "path": "../by_angle_all/normal_high_angle/5046455_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 174.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "5390303",
      "path": "../by_angle_all/normal_high_angle/5390303_LW AXI 3_3  I70f  2.nii.gz",
      "angle": 167.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 85.0
    },
    {
      "patient_id": "5925853",
      "path": "../by_angle_all/normal_high_angle/5925853_LungLowDose  1.0  Br59  2 LW.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 107.0
    },
    {
      "patient_id": "6212308",
      "path": "../by_angle_all/normal_high_angle/6212308_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "6312603",
      "path": "../by_angle_all/normal_high_angle/6312603_LW AXI 3_3  B60f.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 91.0
    },
    {
      "patient_id": "6757504",
      "path": "../by_angle_all/normal_high_angle/6757504_~LUNG AX 1_1 LW.nii.gz",
      "angle": 171.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 138.0
    },
    {
      "patient_id": "6858508",
      "path": "../by_angle_all/normal_high_angle/6858508_LW AXI 3_3  B60f.nii.gz",
      "angle": 179.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 86.0
    },
    {
      "patient_id": "7871759",
      "path": "../by_angle_all/normal_high_angle/7871759_~LUNG AX 1_1 LW.nii.gz",
      "angle": 176.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 99.0
    },
    {
      "patient_id": "8009284",
      "path": "../by_angle_all/normal_high_angle/8009284_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 31.0
    },
    {
      "patient_id": "8244460",
      "path": "../by_angle_all/normal_high_angle/8244460_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 96.0
    },
    {
      "patient_id": "8332556",
      "path": "../by_angle_all/normal_high_angle/8332556_Thorax 1_1 Br40 S3 1.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 114.0
    },
    {
      "patient_id": "9075311",
      "path": "../by_angle_all/normal_high_angle/9075311_Thorax  5.0  B30f.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 62.0
    },
    {
      "patient_id": "A267542",
      "path": "../by_angle_all/normal_high_angle/A267542_LW  AXI 3.0  B60f.nii.gz",
      "angle": 178.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 94.0
    },
    {
      "patient_id": "A613117",
      "path": "../by_angle_all/normal_high_angle/A613117_LW AXI 3_3  B60f.nii.gz",
      "angle": 168.0,
      "source_group": "normal_high_angle",
      "gold_stage": 1,
      "gold_stage_label": "GOLD 2 (Moderate)",
      "post_fev1_percent_predicted": 66.0
    },
    {
      "patient_id": "A754735",
      "path": "../by_angle_all/normal_high_angle/A754735_LW AXI 3_3  B60f.nii.gz",
      "angle": 170.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 110.0
    },
    {
      "patient_id": "E717248",
      "path": "../by_angle_all/normal_high_angle/E717248_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 172.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 83.0
    },
    {
      "patient_id": "E771850",
      "path": "../by_angle_all/normal_high_angle/E771850_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 173.0,
      "source_group": "normal_high_angle",
      "gold_stage": 2,
      "gold_stage_label": "GOLD 3 (Severe)",
      "post_fev1_percent_predicted": 48.0
    },
    {
      "patient_id": "E797258",
      "path": "../by_angle_all/normal_high_angle/E797258_Thorax Lung Br60 S2 3.00.nii.gz",
      "angle": 177.0,
      "source_group": "normal_high_angle",
      "gold_stage": 0,
      "gold_stage_label": "GOLD 1 (Mild)",
      "post_fev1_percent_predicted": 93.0
    }
  ]
}
```
### patient_angle_classification_by_group.json
```json
{
  "abnormal_group_33": {
    "total": 33,
    "statistics": {
      "mean": 141.0,
      "median": 142,
      "min": 105,
      "max": 177
    },
    "by_angle": {
      "low_angle": {
        "1261736": 114,
        "1687031": 112,
        "1800944": 105,
        "2588424": 105,
        "2991621": 117,
        "4372708": 126,
        "4796667": 136,
        "5127217": 142,
        "6887256": 108,
        "8404129": 130,
        "8704416": 142,
        "B213449": 108,
        "C435832": 126,
        "D132855": 130,
        "E353272": 107,
        "E558113": 113,
        "E647833": 116
      },
      "high_angle": {
        "3647457": 155,
        "4710629": 159,
        "5630846": 143,
        "8009284": 177,
        "8126939": 158,
        "9075311": 173,
        "9529629": 165,
        "A613117": 168,
        "A762364": 159,
        "C041635": 161,
        "C543831": 163,
        "C586742": 165,
        "C905524": 160,
        "D550510": 160,
        "E771850": 173,
        "E797258": 177
      }
    }
  },
  "normal_group_21": {
    "total": 33,
    "statistics": {
      "mean": 171.45454545454547,
      "median": 173,
      "min": 140,
      "max": 179
    },
    "by_angle": {
      "low_angle": {
        "1596038": 173,
        "2094528": 165,
        "2221276": 166,
        "3097765": 165,
        "3635301": 169,
        "4204917": 140,
        "5390303": 167,
        "8244460": 170,
        "8332556": 173,
        "A754735": 170,
        "C081146": 164,
        "E717248": 172,
        "1604378": 166,
        "1663485": 171,
        "1814107": 171,
        "2256243": 171,
        "4996166": 166,
        "5925853": 173,
        "6757504": 171
      },
      "high_angle": {
        "1746380": 174,
        "2500824": 179,
        "2860903": 175,
        "4302294": 179,
        "5046455": 174,
        "6212308": 178,
        "6312603": 177,
        "6858508": 179,
        "7871759": 176,
        "0781915": 176,
        "2291134": 176,
        "4205212": 178,
        "4230847": 176,
        "A267542": 178
      }
    }
  }
}
```
### pft.json
```json
{
  "GOLD 1 (輕度)": [
    {
      "patient_id": "C041635",
      "group": "abnormal",
      "post_fev1_percent_predicted": 87
    },
    {
      "patient_id": "C543831",
      "group": "abnormal",
      "post_fev1_percent_predicted": 80
    },
    {
      "patient_id": "E797258",
      "group": "abnormal",
      "post_fev1_percent_predicted": 93
    },
    {
      "patient_id": "1596038",
      "group": "normal",
      "post_fev1_percent_predicted": 87
    },
    {
      "patient_id": "1746380",
      "group": "normal",
      "post_fev1_percent_predicted": 92
    },
    {
      "patient_id": "2094528",
      "group": "normal",
      "post_fev1_percent_predicted": 85
    },
    {
      "patient_id": "2221276",
      "group": "normal",
      "post_fev1_percent_predicted": 85
    },
    {
      "patient_id": "2500824",
      "group": "normal",
      "post_fev1_percent_predicted": 100
    },
    {
      "patient_id": "2860903",
      "group": "normal",
      "post_fev1_percent_predicted": 83
    },
    {
      "patient_id": "3097765",
      "group": "normal",
      "post_fev1_percent_predicted": 95
    },
    {
      "patient_id": "3635301",
      "group": "normal",
      "post_fev1_percent_predicted": 93
    },
    {
      "patient_id": "4204917",
      "group": "normal",
      "post_fev1_percent_predicted": 84
    },
    {
      "patient_id": "4302294",
      "group": "normal",
      "post_fev1_percent_predicted": 104
    },
    {
      "patient_id": "5046455",
      "group": "normal",
      "post_fev1_percent_predicted": 91
    },
    {
      "patient_id": "5390303",
      "group": "normal",
      "post_fev1_percent_predicted": 85
    },
    {
      "patient_id": "6212308",
      "group": "normal",
      "post_fev1_percent_predicted": 114
    },
    {
      "patient_id": "6312603",
      "group": "normal",
      "post_fev1_percent_predicted": 91
    },
    {
      "patient_id": "6858508",
      "group": "normal",
      "post_fev1_percent_predicted": 86
    },
    {
      "patient_id": "7871759",
      "group": "normal",
      "post_fev1_percent_predicted": 99
    },
    {
      "patient_id": "8244460",
      "group": "normal",
      "post_fev1_percent_predicted": 96
    },
    {
      "patient_id": "8332556",
      "group": "normal",
      "post_fev1_percent_predicted": 114
    },
    {
      "patient_id": "A754735",
      "group": "normal",
      "post_fev1_percent_predicted": 110
    },
    {
      "patient_id": "C081146",
      "group": "normal",
      "post_fev1_percent_predicted": 99
    },
    {
      "patient_id": "E717248",
      "group": "normal",
      "post_fev1_percent_predicted": 83
    },
    {
      "patient_id": "0781915",
      "group": "normal",
      "post_fev1_percent_predicted": 105
    },
    {
      "patient_id": "1604378",
      "group": "normal",
      "post_fev1_percent_predicted": 87
    },
    {
      "patient_id": "1663485",
      "group": "normal",
      "post_fev1_percent_predicted": 116
    },
    {
      "patient_id": "1814107",
      "group": "normal",
      "post_fev1_percent_predicted": 96
    },
    {
      "patient_id": "2256243",
      "group": "normal",
      "post_fev1_percent_predicted": 111
    },
    {
      "patient_id": "2291134",
      "group": "normal",
      "post_fev1_percent_predicted": 110
    },
    {
      "patient_id": "4205212",
      "group": "normal",
      "post_fev1_percent_predicted": 93
    },
    {
      "patient_id": "4230847",
      "group": "normal",
      "post_fev1_percent_predicted": 94
    },
    {
      "patient_id": "4996166",
      "group": "normal",
      "post_fev1_percent_predicted": 110
    },
    {
      "patient_id": "5925853",
      "group": "normal",
      "post_fev1_percent_predicted": 107
    },
    {
      "patient_id": "6757504",
      "group": "normal",
      "post_fev1_percent_predicted": 138
    },
    {
      "patient_id": "A267542",
      "group": "normal",
      "post_fev1_percent_predicted": 94
    }
  ],
  "GOLD 2 (中度)": [
    {
      "patient_id": "4372708",
      "group": "abnormal",
      "post_fev1_percent_predicted": 51
    },
    {
      "patient_id": "4710629",
      "group": "abnormal",
      "post_fev1_percent_predicted": 73
    },
    {
      "patient_id": "4796667",
      "group": "abnormal",
      "post_fev1_percent_predicted": 61
    },
    {
      "patient_id": "5127217",
      "group": "abnormal",
      "post_fev1_percent_predicted": 72
    },
    {
      "patient_id": "8126939",
      "group": "abnormal",
      "post_fev1_percent_predicted": 61
    },
    {
      "patient_id": "9075311",
      "group": "abnormal",
      "post_fev1_percent_predicted": 62
    },
    {
      "patient_id": "9529629",
      "group": "abnormal",
      "post_fev1_percent_predicted": 65
    },
    {
      "patient_id": "A613117",
      "group": "abnormal",
      "post_fev1_percent_predicted": 66
    },
    {
      "patient_id": "A762364",
      "group": "abnormal",
      "post_fev1_percent_predicted": 58
    },
    {
      "patient_id": "D132855",
      "group": "abnormal",
      "post_fev1_percent_predicted": 62
    },
    {
      "patient_id": "D550510",
      "group": "abnormal",
      "post_fev1_percent_predicted": 60
    }
  ],
  "GOLD 3 (重度)": [
    {
      "patient_id": "1687031",
      "group": "abnormal",
      "post_fev1_percent_predicted": 36
    },
    {
      "patient_id": "1800944",
      "group": "abnormal",
      "post_fev1_percent_predicted": 32
    },
    {
      "patient_id": "3647457",
      "group": "abnormal",
      "post_fev1_percent_predicted": 42
    },
    {
      "patient_id": "5630846",
      "group": "abnormal",
      "post_fev1_percent_predicted": 44
    },
    {
      "patient_id": "6887256",
      "group": "abnormal",
      "post_fev1_percent_predicted": 36
    },
    {
      "patient_id": "8009284",
      "group": "abnormal",
      "post_fev1_percent_predicted": 31
    },
    {
      "patient_id": "8404129",
      "group": "abnormal",
      "post_fev1_percent_predicted": 39
    },
    {
      "patient_id": "8704416",
      "group": "abnormal",
      "post_fev1_percent_predicted": 34
    },
    {
      "patient_id": "C435832",
      "group": "abnormal",
      "post_fev1_percent_predicted": 44
    },
    {
      "patient_id": "C586742",
      "group": "abnormal",
      "post_fev1_percent_predicted": 33
    },
    {
      "patient_id": "C905524",
      "group": "abnormal",
      "post_fev1_percent_predicted": 41
    },
    {
      "patient_id": "E647833",
      "group": "abnormal",
      "post_fev1_percent_predicted": 44
    },
    {
      "patient_id": "E771850",
      "group": "abnormal",
      "post_fev1_percent_predicted": 48
    }
  ],
  "GOLD 4 (極重度)": [
    {
      "patient_id": "1261736",
      "group": "abnormal",
      "post_fev1_percent_predicted": 26
    },
    {
      "patient_id": "2588424",
      "group": "abnormal",
      "post_fev1_percent_predicted": 27
    },
    {
      "patient_id": "2991621",
      "group": "abnormal",
      "post_fev1_percent_predicted": 27
    },
    {
      "patient_id": "B213449",
      "group": "abnormal",
      "post_fev1_percent_predicted": 19
    },
    {
      "patient_id": "E353272",
      "group": "abnormal",
      "post_fev1_percent_predicted": 20
    },
    {
      "patient_id": "E558113",
      "group": "abnormal",
      "post_fev1_percent_predicted": 25
    }
  ]
}
```
### regression/datasets/patient_angles_simple.json
```json
{
  "1261736": 114,
  "1687031": 112,
  "1800944": 105,
  "2588424": 105,
  "2991621": 117,
  "3647457": 155,
  "4372708": 126,
  "4710629": 159,
  "4796667": 136,
  "5127217": 142,
  "5630846": 143,
  "6887256": 108,
  "8009284": 177,
  "8126939": 158,
  "8404129": 130,
  "8704416": 142,
  "9075311": 173,
  "9529629": 165,
  "A613117": 168,
  "A762364": 159,
  "B213449": 108,
  "C041635": 161,
  "C435832": 126,
  "C543831": 163,
  "C586742": 165,
  "C905524": 160,
  "D132855": 130,
  "D550510": 160,
  "E353272": 107,
  "E558113": 113,
  "E647833": 116,
  "E771850": 173,
  "E797258": 177,
  "1596038": 173,
  "1746380": 174,
  "2094528": 165,
  "2221276": 166,
  "2500824": 179,
  "2860903": 175,
  "3097765": 165,
  "3635301": 169,
  "4204917": 140,
  "4302294": 179,
  "5046455": 174,
  "5390303": 167,
  "6212308": 178,
  "6312603": 177,
  "6858508": 179,
  "7871759": 176,
  "8244460": 170,
  "8332556": 173,
  "A754735": 170,
  "C081146": 164,
  "E717248": 172
}
```
