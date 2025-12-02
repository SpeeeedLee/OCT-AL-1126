import numpy as np
import matplotlib.pyplot as plt

# 四組實驗數據
data = {
    'No Augmentation': [0.4824, 0.5333, 0.4824, 0.4863, 0.5216, 0.5176, 0.4941, 0.5098, 0.4980, 0.4784],
    '(2x) Horizontal Flip': [0.5176, 0.5608, 0.5373, 0.5098, 0.5294, 0.5412, 0.5529, 0.5294, 0.5294, 0.5451],
    '(2x) Vertical Flip': [0.5137, 0.5451, 0.5529, 0.5216, 0.5294, 0.5294, 0.5490, 0.4980, 0.5294, 0.5451],
    '(3x) Horizontal + Vertical Flip': [0.5373, 0.5569, 0.6039, 0.5608, 0.5294, 0.5255, 0.5490, 0.5569, 0.5686, 0.5451],
    '(4x) Horizontal, Vertical,\nHorizontal + Vertical Flip': [0.5333, 0.5529, 0.5608, 0.5647, 0.5451, 0.5569, 0.5294, 0.5608, 0.5647]
}

# 計算每組的mean和std
labels = list(data.keys())
means = [np.mean(data[label]) for label in labels]
stds = [np.std(data[label], ddof=1) for label in labels]  # 使用樣本標準差

# 打印統計結果
print("統計結果:")
print("-" * 60)
for label, mean, std in zip(labels, means, stds):
    print(f"{label.replace(chr(10), ' ')}")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std:  {std:.4f}")
    print()

# 創建柱狀圖
fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(labels))
bars = ax.bar(x_pos, means, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.2)

# 添加error bars (黑色)
ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', 
            capsize=5, capthick=2, linewidth=2)

# 設置圖表標籤和標題
# ax.set_xlabel('Data Augmentation Method', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=28, labelpad=10)
# ax.set_title('Effect of Data Augmentation on Model Performance', fontsize=14, fontweight='bold')

# 修改x軸標籤為更簡潔的版本
x_labels_short = [
    'No Aug',
    'Horizontal\n(2×)',
    'Vertical\n(2×)',
    'H + V\n(3×)',
    'H + V + HV\n(4×)'
]
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels_short, fontsize=22, ha='center')

# 放大y軸刻度標籤
ax.tick_params(axis='y', labelsize=22)

# 設置y軸範圍，讓差異更明顯
ax.set_ylim([0.46, 0.58])

# 添加網格線
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 在每個柱子上方顯示數值
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.003, f'{mean:.4f}', 
            ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.savefig('./classification/figs/data_aug/fig_imagent_portion5.png', dpi=300, bbox_inches='tight')
print("圖表已保存至 ./classification/figs/data_aug/fig_imagent_portion5.png")
plt.close()

print("\n圖表創建完成!")