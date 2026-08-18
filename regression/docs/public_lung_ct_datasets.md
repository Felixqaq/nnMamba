# 公開肺部 CT 資料集 — 擴充訓練資料筆記

> 目的：現有 Normal vs Abnormal cohort 只有 54 例（33 Abnormal / 21 Normal），
> 資料量是效能瓶頸。此文件整理「可下載、能加進訓練集」的公開胸腔 CT 資料集。
> 建立日期：2026-07-09。

## ⚠️ 最關鍵前提：標籤語意要對得上

本專案的 **Abnormal 是用 PFT／塌陷角度（Angle of Collapse）／OI 定義的氣流阻塞**，
不是放射報告上的「emphysema」。外部資料集的標籤（多半是報告抽取的 emphysema/no-finding）
**未必等於**本專案的標籤定義。直接把外部 emphysema 當成本專案的 Abnormal 混入訓練，
會把模型帶去學「不同的任務」→ 反而變差。

## 資料集清單（依對本專案的實用度排序）

| # | 資料集 | 內容 / 標籤 | 格式 | 下載方式 | 用途 |
|---|--------|-----------|------|---------|------|
| 1 | **CT-RATE** | 25,692 顆胸腔 CT，18 種異常標註**含 emphysema**，附報告 | **NIfTI（現成）** | HuggingFace：`load_dataset("ibrahimhamamci/CT-RATE")`，CC-BY 4.0 | **最推薦**：已是 NIfTI、有氣腫欄位、量超大 |
| 2 | **RAD-ChestCT**（Duke） | 3,630 顆（1,344 有 83 種異常標註，含 emphysema） | NIfTI / npz | 公開下載（TCIA / Zenodo） | CT-RATE 的外部驗證集，補異常樣本 |
| 3 | **SPIE-AAPM Lung CT Challenge** | 50 顆「正常/近正常」胸腔 CT | DICOM | TCIA 直接下載 | 補 **Normal** 類的乾淨來源 |
| 4 | **HEALTHY-TOTAL-BODY-CTS** | 健康受試者 CT | DICOM | TCIA | 補 Normal（total-body，需裁胸腔） |
| 5 | **LIDC-IDRI** | 1,010 例胸腔 CT（結節為主） | DICOM（124 GB） | TCIA | 量大但偏結節、肺結構多正常 |
| 6 | **COPDGene / NLST** | COPD/氣腫**含 PFT**，金標準 | DICOM | 需申請 data-use（非一鍵） | 語意**最接近本任務**，但門檻高 |
| 7 | **OSIC Pulmonary Fibrosis** | 肺纖維化 + FVC | DICOM | Kaggle 一鍵 | 纖維化 ≠ 氣腫，語意不同 |

## 建議做法（避開標籤語意坑）

先拿 **CT-RATE（#1）**：已是 NIfTI、有 emphysema 標籤、一行就能拉。
但**不要**直接把它的 emphysema 當本專案 Abnormal 混入訓練。兩條較穩的路：

- **路線 A（最安全）— 當預訓練**：用 CT-RATE 大量資料做 self-supervised／分類預訓練，
  再用本專案的 54 例 fine-tune。小資料最有效的用法，直接解「54 例太少」。
- **路線 B — 擴充 Normal 類**：公開的「正常胸腔 CT」很多（#3 #4）。目前 33 異常 / 21 正常，
  正常偏少；補乾淨 Normal 進去、異常維持本專案定義，label 語意不會亂。

## 加進去前必做：統一預處理

否則不同掃描儀／重建核造成 domain shift，反而更糟。

- DICOM → NIfTI：`dcm2niix`（CT-RATE 已是 NIfTI，可略過）
- resample 到 `[112, 136, 112]`
- HU window `[-1000, 400]`
- z-score normalization
- 放入 pipeline 的 `Normal/` `Abnormal/` 資料夾結構（folder name = label）

## 待辦 / 下一步

1. **探勘 CT-RATE metadata**：確認 label CSV 欄位、怎麼篩 emphysema / no-finding。
2. **寫下載 + 預處理腳本**：從 CT-RATE 拉指定數量，自動轉成 `Normal/` `Abnormal/`
   結構並 resample。

## 來源

- [CT-RATE (Hugging Face)](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
- [RAD-ChestCT / Multiple Abnormality Prediction (arXiv)](https://arxiv.org/pdf/2002.04752)
- [HEALTHY-TOTAL-BODY-CTS (TCIA)](https://www.cancerimagingarchive.net/collection/healthy-total-body-cts/)
- [LIDC-IDRI (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/)
- [Radiomics dataset from healthy adults (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12925589/)
- [Medical Imaging Datasets 清單 (GitHub)](https://github.com/m-aryayi/Medical-Imaging-Datasets)
