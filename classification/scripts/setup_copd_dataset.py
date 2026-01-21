'''設置 COPD 資料集的腳本 - 將資料分為訓練集和測試集'''
import os
import shutil
from pathlib import Path
import random

def setup_dataset(normal_dir, abnormal_dir, output_dir, train_ratio=0.8):
    """
    將 Normal 和 Abnormal 資料分為訓練集和測試集
    
    Args:
        normal_dir: Normal 資料夾路徑
        abnormal_dir: Abnormal (COPD) 資料夾路徑  
        output_dir: 輸出資料集目錄
        train_ratio: 訓練集比例 (默認 0.8)
    """
    random.seed(42)
    
    # 創建目錄結構
    train_normal = Path(output_dir) / "copd_train" / "Normal"
    train_copd = Path(output_dir) / "copd_train" / "COPD"
    test_normal = Path(output_dir) / "copd_test" / "Normal"
    test_copd = Path(output_dir) / "copd_test" / "COPD"
    
    for dir_path in [train_normal, train_copd, test_normal, test_copd]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 處理 Normal 資料
    normal_files = []
    for pattern in ["*.nii", "*.nii.gz"]:
        normal_files.extend(list(Path(normal_dir).glob(pattern)))
    normal_files = sorted(normal_files)
    random.shuffle(normal_files)
    
    train_size_normal = int(len(normal_files) * train_ratio)
    train_normal_files = normal_files[:train_size_normal]
    test_normal_files = normal_files[train_size_normal:]
    
    print(f"Normal 資料: 總共 {len(normal_files)} 個")
    print(f"  - 訓練集: {len(train_normal_files)} 個")
    print(f"  - 測試集: {len(test_normal_files)} 個")
    
    # 複製 Normal 訓練集
    for i, src_file in enumerate(train_normal_files, 1):
        # 保留完整副檔名 (.nii 或 .nii.gz)
        ext = ''.join(src_file.suffixes)
        dst_file = train_normal / f"normal_{i:04d}{ext}"
        shutil.copy2(src_file, dst_file)
    
    # 複製 Normal 測試集
    for i, src_file in enumerate(test_normal_files, 1):
        ext = ''.join(src_file.suffixes)
        dst_file = test_normal / f"normal_{i:04d}{ext}"
        shutil.copy2(src_file, dst_file)
    
    # 處理 Abnormal (COPD) 資料
    abnormal_files = []
    for pattern in ["*.nii", "*.nii.gz"]:
        abnormal_files.extend(list(Path(abnormal_dir).glob(pattern)))
    abnormal_files = sorted(abnormal_files)
    random.shuffle(abnormal_files)
    
    train_size_abnormal = int(len(abnormal_files) * train_ratio)
    train_abnormal_files = abnormal_files[:train_size_abnormal]
    test_abnormal_files = abnormal_files[train_size_abnormal:]
    
    print(f"\nCOPD 資料: 總共 {len(abnormal_files)} 個")
    print(f"  - 訓練集: {len(train_abnormal_files)} 個")
    print(f"  - 測試集: {len(test_abnormal_files)} 個")
    
    # 複製 COPD 訓練集
    for i, src_file in enumerate(train_abnormal_files, 1):
        ext = ''.join(src_file.suffixes)
        dst_file = train_copd / f"copd_{i:04d}{ext}"
        shutil.copy2(src_file, dst_file)
    
    # 複製 COPD 測試集
    for i, src_file in enumerate(test_abnormal_files, 1):
        ext = ''.join(src_file.suffixes)
        dst_file = test_copd / f"copd_{i:04d}{ext}"
        shutil.copy2(src_file, dst_file)
    
    print(f"\n✅ 資料集設置完成！")
    print(f"\n資料集結構:")
    print(f"  {output_dir}/copd_train/")
    print(f"    ├── Normal/  ({len(train_normal_files)} 個檔案)")
    print(f"    └── COPD/    ({len(train_abnormal_files)} 個檔案)")
    print(f"  {output_dir}/copd_test/")
    print(f"    ├── Normal/  ({len(test_normal_files)} 個檔案)")
    print(f"    └── COPD/    ({len(test_abnormal_files)} 個檔案)")
    
    print(f"\n總訓練樣本: {len(train_normal_files) + len(train_abnormal_files)}")
    print(f"總測試樣本: {len(test_normal_files) + len(test_abnormal_files)}")


if __name__ == "__main__":
    # 設定路徑
    normal_dir = "../Normal"
    abnormal_dir = "../Abnormal"
    output_dir = "./datasets"
    
    print("開始設置 COPD 資料集...")
    print(f"Normal 資料來源: {normal_dir}")
    print(f"Abnormal 資料來源: {abnormal_dir}")
    print(f"輸出目錄: {output_dir}")
    print("-" * 50)
    
    setup_dataset(normal_dir, abnormal_dir, output_dir, train_ratio=0.8)
