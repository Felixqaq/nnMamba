# TODO

## 目前狀態

- 現在主力模型是 `HybridMambaAttentionRegressor`
- 模型位置: `regression/networks/hybrid_mamba_attention_regressor.py`
- 主設定位置: `regression/config.hybrid.yaml`
- 目前最佳 tuning 設定來自 `attn_drop_0p00`
- 最佳設定檔: `regression/figures/PFT_angle_regression/tuning_runs/2026-04-09_13-51-31/best_config.yaml`

## 目前最佳結果

- hybrid baseline 結果: `regression/figures/PFT_angle_regression/hybrid_mamba_attention_2026-04-09_13:23:43/results.json`
- 後續 tuning 最佳候選: `attn_drop_0p00`
- 最佳分數:
  - `MAE = 15.85124`
  - `RMSE = 21.26467`
  - `R2 = 0.17234`
- 結論:
  - hybrid 比原本 `mamba` 和 `swinunetr` 更好
  - `attn_dropout = 0.0` 目前優於 `0.1`
  - 目前最值得繼續追的是 `hybrid + 更強 encoder`

## 目前架構判讀

- 現在是 `encoder-only regression model`，不是 segmentation 架構
- 流程:
  - `3D CT`
  - `stem conv`
  - `stage1 / stage2 / stage3`
  - `high-level attention bridge`
  - `multi-scale global pooling`
  - `MLP regression head`
  - `輸出 1 個角度`
- 沒有 decoder，因為任務是 `global scalar regression`，不是 `dense voxel prediction`
- 所以下一版重點不是加 decoder，而是把 `encoder backbone` 換成更強的 3D Mamba backbone

## 已確認的有用結論

- 目前這版 hybrid 是 `MambaVision-inspired`，但不是論文復刻
- 真正對這個 repo 最有用的是:
  - `Mamba backbone`
  - `高層 attention bridge`
  - `multi-scale pooling`
- 下一步最該強化的是:
  - `3D local modeling`
  - `3D scan strategy`
  - `multi-scale feature extraction`
  - `global/local feature balance`

## 值得參考的最新 backbone

### 1. 3D Multi-scale Mamba + 3D DWConv + Tri-scan

- Source: https://arxiv.org/abs/2503.19308
- 最適合你目前 repo 的方向
- 可借的重點:
  - `3D DWConv`
  - `Tri-scan`
  - `multi-scale Mamba`
- 這篇比較像是在把 `3D Mamba backbone 本身做對`
- 優先度: `最高`

### 2. HybridMamba

- Source: https://papers.miccai.org/miccai-2025/0426-Paper2815.html
- 更醫學影像導向
- 可借的重點:
  - `SoMamba`
  - `LoMamba`
  - `FGM / FFT gated mechanism`
- 這篇比較像是在補:
  - `global + local`
  - `spatial + frequency`
- 對肺部形狀、局部邊界、異常區域可能更有幫助
- 優先度: `中高`

### 3. MambaVision

- Source: https://arxiv.org/abs/2407.08083
- 目前 hybrid 設計最接近它
- 可借的重點:
  - `high-level attention on top of Mamba`
  - `hybrid Mamba-Transformer design`
- 比較像通用 vision backbone，不是最貼 3D medical，但設計哲學有用
- 優先度: `中`

### 4. 其他值得後續補看的 backbone

- `SegResMamba`
  - Source: https://openreview.net/forum?id=zQLrITbcxJ
  - 重點: `Convolution Mamba Mixed Block (CMMB)` + `Tri-oriented Mamba (ToM)`
- `GroupMamba`
  - Source: https://openaccess.thecvf.com/content/CVPR2025/html/Shaker_GroupMamba_Efficient_Group-Based_Visual_State_Space_Model_CVPR_2025_paper.html
  - 重點: `4-direction grouped scanning`
- `TinyViM`
  - Source: https://arxiv.org/abs/2411.17473
  - 重點: `frequency decoupling`
- `RAVLT / RALA`
  - Source: https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Breaking_the_Low-Rank_Dilemma_of_Linear_Attention_CVPR_2025_paper.html
  - 重點: `更強的 linear attention`

## 下一版 backbone 實作優先順序

### A. 最優先

- 把現在的 encoder 往 `3D DWConv + Tri-scan + multi-scale` 升級
- 保留目前的 `encoder-only + pooling + regression head`
- 不需要加 decoder

### B. 第二階段

- 在 encoder 裡加 `local/global Mamba` 分工
- 先試簡化版 `LoMamba + SoMamba`
- 若有明確收益，再考慮 `FGM / frequency branch`

### C. 第三階段

- 若 high-level attention 還想再升級
- 再考慮:
  - `RALA / linear attention`
  - 更像 `MambaVision` 的後層 hybrid block

## 實作建議

- 先不要大改 regression head
- 先動這些檔案:
  - `regression/networks/mamba_regressor.py`
  - `regression/networks/hybrid_mamba_attention_regressor.py`
  - `regression/models.py`
  - `regression/core/config.py`
- 先做一個小步版本:
  - `3D DWConv + Tri-scan`
- 第二步再做:
  - `multi-scale Mamba block`
- 第三步才做:
  - `LoMamba / FGM / frequency gating`

## 實驗 TODO

- 用目前最佳 `config.hybrid.yaml` 再跑 `3 seeds`
- 比較:
  - `baseline hybrid`
  - `hybrid + 3D DWConv`
  - `hybrid + Tri-scan`
  - `hybrid + 3D DWConv + Tri-scan`
- 主要觀察:
  - `MAE`
  - `RMSE`
  - `R2`
  - 穩定性
  - 是否容易出現 `Non-finite loss`

## 重要提醒

- 不要直接拿 segmentation paper 的 decoder 進 regression
- 要借的是 `encoder block / scan strategy / local-global fusion`
- 現在最合理的路線是:
  - `先學 3D DWConv + Tri-scan + multi-scale`
  - `再往 HybridMamba 的 LoMamba + FGM 走`
