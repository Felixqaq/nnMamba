# COPD 分類專案使用指南

## 📁 資料集結構

請按照以下結構準備你的胸部 CT 資料：

```
classification/
└── datasets/
    ├── copd_train/          # 訓練集
    │   ├── Normal/          # 正常影像 (NIfTI 格式: .nii 或 .nii.gz)
    │   │   ├── patient001.nii.gz
    │   │   ├── patient002.nii.gz
    │   │   └── ...
    │   └── COPD/            # COPD 患者影像
    │       ├── patient101.nii.gz
    │       ├── patient102.nii.gz
    │       └── ...
    └── copd_test/           # 測試集
        ├── Normal/
        │   └── ...
        └── COPD/
            └── ...
```

## ⚙️ 環境準備

確保已激活 conda 環境並安裝必要的套件：

```bash
conda activate nnMamba
```

如果還未安裝所有依賴，請執行：
```bash
pip install torchmetrics pandas nibabel scikit-image monai matplotlib
```

## 🚀 使用方式

### 方法一：使用專用腳本（推薦）

```bash
cd classification
python train_copd.py
```

### 方法二：修改原始 train_model.py

在 `train_model.py` 的 `main()` 函數中添加：

```python
def main():
    setup_seed(42)

    # COPD 分類任務
    ld_helper = LoaderHelper(task=Task.Normal_v_COPD)
    model_uuid = train_camull(ld_helper, epochs=100, model_name='nnmamba')
    evaluate_model(DEVICE, model_uuid, ld_helper)
```

然後在文件末尾改為：
```python
main()
# eval()  # 註解掉評估模式
```

## 🔧 模型選擇

你可以選擇不同的架構（在 `train_copd.py` 或 `train_model.py` 中修改 `model_name`）：

- `'nnmamba'` - nnMamba 架構（推薦，效能最佳）
- `'densenet'` - DenseNet121
- `'vit'` - Vision Transformer
- `'crate'` - CRATE

## 📊 訓練參數調整

在訓練腳本中可以調整：

- `epochs`: 訓練輪數（默認 100）
- `batch_size`: 批次大小（在 loader_helper.py 中，默認 2）
- `learning_rate`: 學習率（在 train_model.py 的 train_loop 中，默認 0.0001）
- GPU 設定：修改 `os.environ["CUDA_VISIBLE_DEVICES"]` 的值

## 📈 查看訓練結果

訓練過程中會自動保存：

- **模型權重**: `../weights/Normal_v_COPD/{UUID}/best_weight.pth`
- **訓練日誌**: `../train_log/{UUID}.txt`
- **訓練曲線**: `../figures/{UUID}eva.png`

## 🔍 評估已訓練的模型

如果你已有訓練好的模型，想要評估：

```python
from train_copd import evaluate_copd_model

# 替換為你的模型 UUID
model_uuid = "nnMamba_2026-01-02_10:30:45"
evaluate_copd_model(model_uuid)
```

## 💡 資料預處理建議

1. **影像格式**: 確保所有 CT 影像為 NIfTI 格式（.nii 或 .nii.gz）
2. **影像大小**: 程式會自動處理，但建議預處理為相近的尺寸
3. **正規化**: 程式會進行 min-max 正規化
4. **資料增強**: 如需啟用資料增強，可在 `data_declaration.py` 的 `get_mri()` 函數中取消註解相關程式碼

## 📝 注意事項

1. 確保訓練集和測試集分開，避免資料洩漏
2. 建議每個類別至少有 50-100 個樣本
3. GPU 記憶體不足時，可降低 batch_size（在 loader_helper.py 中修改）
4. 訓練時間視資料量和 GPU 效能而定，100 epochs 可能需要數小時

## 🐛 常見問題

**Q: 出現記憶體不足錯誤？**
A: 減少 batch_size 或使用較小的模型（如 DenseNet）

**Q: 準確率不理想？**
A: 嘗試增加訓練輪數、調整學習率，或啟用資料增強

**Q: 讀取影像錯誤？**
A: 確認影像格式為 NIfTI，且檔案完整未損壞
