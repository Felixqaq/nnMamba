# Methods And Pipeline Notes

這份文件把論文 Method 章節會用到的固定事實集中在一起。細節來源主要是 `classification/` 與 `regression/` 兩套訓練程式、dataset loader、model registry、evaluation utilities 與 YAML config。

## Shared CT Input

- 3D CT/NIfTI 讀入後轉成 channel-first tensor，主要尺寸是 `1 x 112 x 136 x 112`。
- `classification/data/dataset.py` 會把輸入 resize 到 `TARGET_SHAPE = (112, 136, 112)`。
- `regression/data/dataset.py` 使用相同預設尺寸，並可套用 HU window clipping 與 per-volume z-score normalization。
- regression 系列常用 HU window 為 `[-1000, 400]`；是否使用 z-score 由各 config 的 `data.input_normalization` 決定。

## Classification Pipeline

- 入口: `classification/train.py`。
- 任務 enum: `NC_v_AD`, `sMCI_v_pMCI`, `Normal_v_COPD`, `Normal_v_Abnormal`。
- 目前主要結果集中在 `Normal_v_Abnormal`。
- 模型 registry: `nnmamba`, `densenet`, `vit`, `crate`。
- Loss: `BCEWithLogitsLoss`。
- Optimizer: `AdamW`。
- Cross validation: `StratifiedKFold`，seed 來自 config，預設 `42`。
- Thresholding: 每次 evaluation 先在 train split 以 Youden's J 找最佳 threshold，再套到 validation/test split。
- Best checkpoint selection: validation AUC。
- Metrics: AUC, accuracy, sensitivity/recall, specificity；每個 fold 另存錯分案例 JSON。

## Regression / Classification Unified Pipeline

- 入口: `regression/train.py`。
- 這個資料夾名稱是 regression，但現在支援四種 target mode: `angle`, `gold`, `angle_3class`, `angle_binary_extreme`。
- `angle` 是角度迴歸；其他三個都是 classification。
- Data split:
  - regression: 角度用 quantile bins 進行 stratified k-fold；若無法分層則 fallback 到 KFold。
  - classification: 以原始病人為單位做 patient-level split，materialized/virtual augmented copies 只進 training fold，validation 只保留原始病人，避免 leakage。
- Loss:
  - `training.loss: auto` 在 classification 解析成 `cross_entropy`。
  - `training.loss: auto` 在 regression 解析成 `smooth_l1`。
- Optimizer: AdamW；CUDA 可用時嘗試 fused AdamW。
- Scheduler: ReduceLROnPlateau。
- AMP: 由 `training.amp` 控制，CUDA 下預設啟用。
- Gradient clipping: `training.clip_grad_norm`，常見值 `1.0`。
- Best checkpoint selection:
  - classification: validation macro-F1。
  - regression: validation MAE 最低。
- Metrics:
  - regression: MAE, RMSE, R2, Pearson, mean error, MSE。
  - classification: accuracy, macro-F1, macro precision, macro recall, balanced accuracy, confusion matrix。

## Target Definitions

- `angle`: continuous PFT/Angle of Collapse target。
- `gold`: GOLD 1-4, normalized in code as `GOLD 1 (Mild)`, `GOLD 2 (Moderate)`, `GOLD 3 (Severe)`, `GOLD 4 (Very Severe)`。
- `angle_3class`:
  - class 0: `Emphysema/Abnormal (<=131 deg)`
  - class 1: `Intermediate (132-151 deg)`
  - class 2: `Normal (>=152 deg)`
- `angle_binary_extreme`:
  - class 0: `Abnormal/emphysema-like (AC <=131 deg)`
  - class 1: `Normal-like (AC >=152 deg)`
  - 132-151 deg gray zone is excluded。

## Augmentation And Balancing

- `balanced_sampling: true` enables epoch-level random undersampling to the fold-local minority class count for classification tasks。
- `augmentation.target_per_class` can add virtual augmented training copies within each fold before sampling。
- `augmentation.balance_then_augment: true` first samples a balanced base set, then emits multiple train-time views per selected sample。
- Random CT augmentation includes small 3D affine changes, intensity scaling/shift, and Gaussian noise.
- Validation/test folds do not receive train-time augmentation.

## Model Families

- `MambaAngleRegressor`: 3D Conv stem, residual Mamba blocks over flattened spatial tokens, multi-scale pooled features, MLP head。
- `HybridMambaAttentionRegressor`: Mamba stages plus global multi-head attention bridge over final-stage spatial tokens。
- `SwinUNETRV2AngleRegressor`: MONAI SwinTransformer encoder, pooled hidden states, MLP head。
- `HybridMambaTapctFusionRegressor`: Hybrid Mamba-Attention CT features concatenated with frozen TAP-CT embedding projection。
- `HybridMambaTapctABMILFusionRegressor`: CT and TAP-CT are projected as modality instances and pooled with gated ABMIL attention。
- `HybridMambaTapctAttentionFusionRegressor`: branch-level attention reweights CT and TAP-CT features before concatenation。
- `TapctABMILClassifier`: TAP-CT-only multiple-instance classifier over frozen embedding bags.
