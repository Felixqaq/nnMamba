import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np

# Set font
rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 讀取JSON數據
with open('patient_angle_classification_by_group.json', 'r') as f:
    data = json.load(f)

# 提取數據
abnormal_group = data['abnormal_group_33']
normal_group = data['normal_group_21']

# 計算第一組（≤166°）和第二組（>166°）
group1_data = []
group2_data = []

# 計算低angle的患者在第一二組中的分佈
for patient_id, angle in abnormal_group['by_angle']['low_angle'].items():
    if angle <= 166:
        group1_data.append(('abnormal', angle))
    else:
        group2_data.append(('abnormal', angle))

for patient_id, angle in abnormal_group['by_angle']['high_angle'].items():
    if angle <= 166:
        group1_data.append(('abnormal', angle))
    else:
        group2_data.append(('abnormal', angle))

for patient_id, angle in normal_group['by_angle']['low_angle'].items():
    if angle <= 166:
        group1_data.append(('normal', angle))
    else:
        group2_data.append(('normal', angle))

for patient_id, angle in normal_group['by_angle']['high_angle'].items():
    if angle <= 166:
        group1_data.append(('normal', angle))
    else:
        group2_data.append(('normal', angle))

# 計算統計信息
def calculate_stats(data):
    total = len(data)
    abnormal_count = sum(1 for status, _ in data if status == 'abnormal')
    normal_count = sum(1 for status, _ in data if status == 'normal')
    angles = [angle for _, angle in data]
    
    if angles:
        mean_angle = np.mean(angles)
        min_angle = min(angles)
        max_angle = max(angles)
    else:
        mean_angle = min_angle = max_angle = 0
    
    return {
        'total': total,
        'abnormal': abnormal_count,
        'normal': normal_count,
        'mean': mean_angle,
        'min': min_angle,
        'max': max_angle
    }

stats1 = calculate_stats(group1_data)
stats2 = calculate_stats(group2_data)

# 創建表格
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# 表格數據
table_data = [
    ['Metric', 'Group 1 (≤166°)', 'Group 2 (>166°)'],
    ['Total Patients', str(stats1['total']), str(stats2['total'])],
    ['Angle Range', f"{stats1['min']:.0f}° - {stats1['max']:.0f}°", 
     f"{stats2['min']:.0f}° - {stats2['max']:.0f}°"],
    ['Mean Angle', f"{stats1['mean']:.1f}°", f"{stats2['mean']:.1f}°"],
    ['Abnormal Patients', str(stats1['abnormal']), str(stats2['abnormal'])],
    ['Normal Patients', str(stats1['normal']), str(stats2['normal'])]
]

# 創建表格
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.35, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 設置表頭樣式
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#2b2b2b')
    cell.set_text_props(weight='bold', color='white')

# 設置其他行的顏色
for i in range(1, 6):
    for j in range(3):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f0f0f0')
        else:
            cell.set_facecolor('#ffffff')
        cell.set_edgecolor('#cccccc')

plt.title('Patient Angle Group Statistics', fontsize=14, fontweight='bold', pad=20)
plt.savefig('patient_angle_table_en.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Table saved as patient_angle_table_en.png")
plt.show()
