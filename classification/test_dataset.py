'''測試 COPD 資料集載入'''
import sys
sys.path.append("./")
import torch
from data_declaration import Task
from loader_helper import LoaderHelper

def test_dataset():
    """測試資料集是否能正常載入"""
    print("🔍 測試 COPD 資料集載入...")
    print("-" * 50)
    
    try:
        # 創建 LoaderHelper
        ld_helper = LoaderHelper(task=Task.Normal_v_COPD)
        
        print(f"✅ Task: {ld_helper.get_task_string()}")
        print(f"✅ Labels: {ld_helper.labels}")
        
        # 獲取訓練集和測試集的 DataLoader
        train_dl = ld_helper.get_train_dl(0, shuffle=False)
        test_dl = ld_helper.get_test_dl(0, shuffle=False)
        
        print(f"\n📊 訓練集:")
        print(f"  - Batch 數量: {len(train_dl)}")
        print(f"  - 總樣本數: {len(train_dl.dataset)}")
        
        print(f"\n📊 測試集:")
        print(f"  - Batch 數量: {len(test_dl)}")
        print(f"  - 總樣本數: {len(test_dl.dataset)}")
        
        # 測試載入一個 batch
        print(f"\n🔬 測試載入第一個 batch...")
        sample_batch = next(iter(train_dl))
        
        print(f"  - 影像形狀: {sample_batch['mri'].shape}")
        print(f"  - 標籤形狀: {sample_batch['label'].shape}")
        print(f"  - 標籤值: {sample_batch['label'].squeeze().tolist()}")
        print(f"  - 影像數值範圍: [{sample_batch['mri'].min():.2f}, {sample_batch['mri'].max():.2f}]")
        
        # 統計標籤分布
        print(f"\n📈 訓練集標籤分布:")
        label_counts = {0: 0, 1: 0}
        for batch in train_dl:
            labels = batch['label'].squeeze().tolist()
            if isinstance(labels, list):
                for label in labels:
                    label_counts[int(label)] += 1
            else:
                label_counts[int(labels)] += 1
        
        print(f"  - Normal (0): {label_counts[0]} 個")
        print(f"  - COPD (1): {label_counts[1]} 個")
        
        print(f"\n✅ 資料集載入測試成功！")
        print(f"\n💡 現在可以開始訓練模型了:")
        print(f"   python train_copd.py")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
