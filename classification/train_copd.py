'''訓練和評估 COPD 分類模型的範例腳本'''
import sys
sys.path.append("./")
import os
import torch

from data_declaration import Task
from loader_helper import LoaderHelper
from train_model import train_camull, evaluate_model, build_arch, setup_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 選擇使用的 GPU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

if not torch.cuda.is_available():
    print("\n⚠️  警告: 未檢測到 CUDA GPU！")
    print("nnMamba 模型需要 GPU 支援。")
    print("\n解決方案:")
    print("1. 確認 NVIDIA GPU 已正確安裝")
    print("2. 檢查 CUDA 驅動: nvidia-smi")
    print("3. 或改用其他模型: model_name='densenet' 或 'vit'")
    print("\n繼續使用 CPU 可能會失敗...\n")
    # sys.exit(1)  # 取消註解以強制停止


def train_copd_model():
    """訓練 Normal vs COPD 分類模型"""
    setup_seed(42)
    
    # 創建 LoaderHelper 並設定為 COPD 任務
    ld_helper = LoaderHelper(task=Task.Normal_v_COPD)
    
    # 訓練模型 (可選擇不同的架構: 'nnmamba', 'densenet', 'vit', 'crate')
    model_name = 'nnmamba'  # 使用 nnMamba 架構
    model_uuid = train_camull(
        ld_helper, 
        k_folds=1,          # 折數 (設為1表示不做交叉驗證)
        model=None,         # 從頭訓練
        epochs=50,          # 50 輪足夠（原本 100）
        model_name=model_name
    )
    
    print(f"\n訓練完成！模型 UUID: {model_uuid}")
    print(f"最佳權重保存在: ../weights/Normal_v_COPD/{model_uuid}/best_weight.pth")
    
    # 評估模型
    print("\n開始評估模型...")
    evaluate_model(DEVICE, model_uuid, ld_helper)
    
    return model_uuid


def evaluate_copd_model(model_uuid):
    """評估已訓練好的 COPD 分類模型"""
    setup_seed(42)
    
    ld_helper = LoaderHelper(task=Task.Normal_v_COPD)
    evaluate_model(DEVICE, model_uuid, ld_helper)


if __name__ == "__main__":
    # 訓練新模型
    model_uuid = train_copd_model()
    
    # 如果要評估已有的模型，請註解上面一行並取消註解下面兩行：
    # model_uuid = "nnMamba_2026-01-02_xx:xx:xx"  # 替換為實際的 UUID
    # evaluate_copd_model(model_uuid)
