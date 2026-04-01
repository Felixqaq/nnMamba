import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 讀取數據
with open('patient_angle_classification_by_group.json', 'r') as f:
    data = json.load(f)

# 提取異常和正常病人的角度數據
abnormal_angles = []
normal_angles = []

# 異常病人
abnormal_data = data['abnormal_group_33']['by_angle']
for group in abnormal_data.values():
    abnormal_angles.extend(group.values())

# 正常病人
normal_data = data['normal_group_21']['by_angle']
for group in normal_data.values():
    normal_angles.extend(group.values())

# 創建圖表
plt.figure(figsize=(12, 6))

# 設置bins和範圍
bins = np.arange(100, 185, 5)  # 5度為一個bin

# 繪製堆疊直方圖
plt.hist([abnormal_angles, normal_angles], bins=bins, alpha=0.8, label=[f'Abnormal (n={len(abnormal_angles)})', f'Normal (n={len(normal_angles)})'], 
         color=['red', 'blue'], edgecolor='black', stacked=True)

# 設置標籤和標題
plt.xlabel('Angle (degrees)', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.title('Patient Angle Distribution Histogram', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(axis='y', alpha=0.3)

# 調整佈局
plt.tight_layout()

# 保存和顯示
plt.savefig('patient_angle_histogram.png', dpi=300, bbox_inches='tight')
print(f"Abnormal patients: {len(abnormal_angles)}, Angle range: {min(abnormal_angles)}°-{max(abnormal_angles)}°")
print(f"Normal patients: {len(normal_angles)}, Angle range: {min(normal_angles)}°-{max(normal_angles)}°")
print("Figure saved as: patient_angle_histogram.png")
plt.show()
