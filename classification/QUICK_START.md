# 🏥 nnMamba COPD 分類專案 - 快速開始指南

## ✅ 環境狀態
- **Conda 環境**: nnMamba
- **Python**: 3.10.19
- **PyTorch**: 2.5.1+cu124
- **CUDA**: 可用 ✓
- **資料集**: 已設置完成 ✓

## 📊 資料集摘要
```
訓練集: 97 個樣本
  ├── Normal: 26 個
  └── COPD: 71 個

測試集: 25 個樣本
  ├── Normal: 7 個
  └── COPD: 18 個
```

## 🚀 開始訓練

### 方法1: 使用專用腳本（推薦）

```bash
cd /home/felix/Research/nnMamba/classification
conda activate nnMamba
python train_copd.py
```

### 方法2: 自訂訓練參數

編輯 `train_copd.py` 修改以下參數：

```python
# 模型架構選擇
model_name = 'nnmamba'  # 可選: 'nnmamba', 'densenet', 'vit', 'crate'

# 訓練參數
epochs = 100            # 訓練輪數
k_folds = 1             # 交叉驗證折數

# GPU 設定
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 選擇 GPU 編號
```

## 📁 專案結構

```
classification/
├── datasets/
│   ├── copd_train/          # 訓練集
│   │   ├── Normal/          # 26 個正常影像
│   │   └── COPD/            # 71 個 COPD 影像
│   └── copd_test/           # 測試集  
│       ├── Normal/          # 7 個正常影像
│       └── COPD/            # 18 個 COPD 影像
│
├── train_copd.py            # 訓練腳本
├── test_dataset.py          # 測試資料載入
├── setup_copd_dataset.py    # 資料集設置腳本
│
└── networks/
    ├── ssm_nnMamba.py       # nnMamba 架構
    ├── conv_Densenet121.py  # DenseNet 架構
    ├── tr_ViT.py            # ViT 架構
    └── tr_crate.py          # CRATE 架構
```

## 📈 訓練過程

訓練過程中會自動：

1. **保存模型權重**
   - 路徑: `../weights/Normal_v_COPD/{UUID}/`
   - 最佳模型: `best_weight.pth`
   - 每10輪: `fold_1_epoch{N}_weights-{timestamp}.pth`

2. **記錄訓練日誌**
   - 路徑: `../train_log/{UUID}.txt`
   - 包含每輪的 loss 和評估指標

3. **生成訓練曲線**
   - 路徑: `../figures/{UUID}eva.png`
   - 顯示訓練和驗證曲線

## 📊 評估指標

模型會輸出以下指標：
- **Accuracy** (準確率)
- **Sensitivity** (敏感度/召回率)
- **Specificity** (特異性)
- **AUC** (ROC曲線下面積)

## 🔧 常用命令

### 測試資料集載入
```bash
python test_dataset.py
```

### 檢查影像尺寸
```bash
python check_image_sizes.py
```

### 重新設置資料集
```bash
python setup_copd_dataset.py
```

### 評估已訓練的模型
```bash
# 編輯 train_copd.py，修改最後幾行：
# model_uuid = "nnMamba_2026-01-02_xx:xx:xx"  # 替換為實際 UUID
# evaluate_copd_model(model_uuid)
python train_copd.py
```

## 💡 訓練建議

1. **初次訓練**: 使用默認設定（nnMamba, 100 epochs）
2. **記憶體不足**: 減少 batch_size（在 loader_helper.py 中）
3. **提升效能**: 嘗試啟用資料增強（在 data_declaration.py 的 get_mri() 中）
4. **過擬合**: 增加 dropout rate（在模型定義中）

## 🐛 疑難排解

### GPU 記憶體不足
修改 `loader_helper.py`:
```python
train_dl = DataLoader(train_ds, batch_size=1, ...)  # 改為 1
```

### 訓練太慢
修改 `loader_helper.py`:
```python
train_dl = DataLoader(train_ds, ..., num_workers=4)  # 增加 workers
```

### 查看 GPU 使用狀況
```bash
watch -n 1 nvidia-smi
```

## 📞 快速參考

- 影像尺寸: 自動 resize 至 (112, 136, 112)
- Batch size: 2
- 學習率: 0.0001
- 優化器: AdamW
- 損失函數: BCEWithLogitsLoss
- 標籤: Normal=0, COPD=1

## 🎯 預期結果

根據 nnMamba 在 ADNI 資料集上的表現：
- Accuracy: ~89%
- AUC: ~95%

實際結果會因資料集而異，建議訓練至少 50-100 epochs。

---

## 🚀 開始訓練！

```bash
cd /home/felix/Research/nnMamba/classification
conda activate nnMamba
python train_copd.py
```

訓練時間估計: 約 2-4 小時（取決於 GPU）
