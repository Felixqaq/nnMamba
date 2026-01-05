'''檢查資料集中影像的尺寸'''
import nibabel as nib
from pathlib import Path
import numpy as np

def check_image_sizes(dataset_dir):
    """檢查資料集中所有影像的尺寸"""
    dataset_path = Path(dataset_dir)
    
    all_shapes = []
    
    for split in ['copd_train', 'copd_test']:
        for label in ['Normal', 'COPD']:
            folder = dataset_path / split / label
            if folder.exists():
                files = list(folder.glob("*.nii*"))
                print(f"\n{split}/{label}: {len(files)} 個檔案")
                
                for i, file in enumerate(files[:5]):  # 只檢查前5個
                    img = nib.load(str(file))
                    shape = img.shape
                    all_shapes.append(shape)
                    print(f"  {file.name}: {shape}")
                    
                if len(files) > 5:
                    print(f"  ... (還有 {len(files)-5} 個檔案)")
    
    if all_shapes:
        print(f"\n📊 統計:")
        shapes_array = np.array(all_shapes)
        print(f"  最小尺寸: {shapes_array.min(axis=0)}")
        print(f"  最大尺寸: {shapes_array.max(axis=0)}")
        print(f"  平均尺寸: {shapes_array.mean(axis=0).astype(int)}")
        
        unique_shapes = set(map(tuple, all_shapes))
        print(f"\n  不同的尺寸: {len(unique_shapes)} 種")
        for shape in list(unique_shapes)[:10]:
            count = all_shapes.count(shape)
            print(f"    {shape}: {count} 個")

if __name__ == "__main__":
    check_image_sizes("./datasets")
