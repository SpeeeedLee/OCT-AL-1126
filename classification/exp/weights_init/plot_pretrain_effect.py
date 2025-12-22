import numpy as np
import matplotlib.pyplot as plt

# 數據結構：每個預訓練方法有兩組數據（Linear Evaluation 和 Full Finetune）
## 這邊都是選 5e-5這個lr!
data = {
    'Random Init.': {
        'Linear Evaluation': [0.4784],  # 填入你的數據
        'Full Finetune': [0.3451, 0.4196, 0.4431, 0.3686, 0.451]       # 填入你的數據
    },
    'ImageNet Pretrain': {
        'Linear Evaluation': [0.4863],  # 填入你的數據
        'Full Finetune': [0.6, 0.5412, 0.5725, 0.5529, 0.5608]       # 填入你的數據
    },
    'OCT Pretrain (SimCLR)': {
        'Linear Evaluation': [0.5922],  # 填入你的數據
        'Full Finetune': [0.5961, 0.5961, 0.6, 0.6039, 0.6078]       # 填入你的數據
    }
}

# 計算每組的 mean 和 std
pretrain_methods = list(data.keys())
eval_methods = ['Linear Evaluation', 'Full Finetune']

means = {method: [] for method in eval_methods}
stds = {method: [] for method in eval_methods}

# 打印統計結果
print("統計結果:")
print("=" * 80)
for pretrain in pretrain_methods:
    print(f"\n{pretrain}")
    print("-" * 80)
    for eval_method in eval_methods:
        results = data[pretrain][eval_method]
        if len(results) > 0:
            mean = np.mean(results)
            # 修改這裡：如果只有一個數據點，std 設為 0
            std = np.std(results, ddof=1) if len(results) > 1 else 0.0
            means[eval_method].append(mean)
            stds[eval_method].append(std)
            print(f"  {eval_method}:")
            print(f"    Mean: {mean:.4f}")
            print(f"    Std:  {std:.4f}")
            print(f"    N:    {len(results)}")
        else:
            means[eval_method].append(0)
            stds[eval_method].append(0)
            print(f"  {eval_method}: No data")
print()

# 創建分組柱狀圖
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(pretrain_methods))
width = 0.35  # 每個柱子的寬度

# 定義顏色
colors = {
    'Linear Evaluation': '#5DA5DA',  # 藍色
    'Full Finetune': '#FAA43A'       # 橙色
}

# 繪製柱狀圖
bars = {}
for i, eval_method in enumerate(eval_methods):
    offset = width * (i - 0.5)
    bars[eval_method] = ax.bar(
        x + offset, 
        means[eval_method], 
        width, 
        label=eval_method,
        alpha=0.8, 
        color=colors[eval_method], 
        edgecolor='black', 
        linewidth=1.2
    )
    
    # 添加 error bars
    ax.errorbar(
        x + offset, 
        means[eval_method], 
        yerr=stds[eval_method], 
        fmt='none', 
        ecolor='black', 
        capsize=5, 
        capthick=2, 
        linewidth=2
    )
    
    # 在柱子上方顯示數值
    for j, (mean, std) in enumerate(zip(means[eval_method], stds[eval_method])):
        if mean > 0:  # 只有當有數據時才顯示
            # 修改這裡：如果 std 是 NaN 或 0，就不加 std
            if np.isnan(std) or std == 0:
                y_position = mean + 0.005
            else:
                y_position = mean + std + 0.005
            
            ax.text(
                j + offset, 
                y_position,  # 使用計算後的位置
                f'{mean:.4f}', 
                ha='center', 
                va='bottom', 
                fontsize=14,
                # fontweight='bold'
            )

# 設置圖表標籤和標題
ax.set_ylabel('Accuracy', fontsize=28, labelpad=10)
ax.set_title('Effect of Weights Initialization\n(5% labeled training data)', 
             fontsize=27, pad=15,
            #    fontweight='bold')
            )

# 設置 x 軸
ax.set_xticks(x)
ax.set_xticklabels(pretrain_methods, fontsize=22, ha='center')

# 放大 y 軸刻度標籤
ax.tick_params(axis='y', labelsize=20)

# 設置 y 軸範圍（根據需要調整）
# ax.set_ylim([0.45, 0.62])  # 取消註釋並調整範圍
ax.set_ylim([0.35, 0.64])  # 取消註釋並調整範圍

# 添加圖例
ax.legend(loc='lower right', fontsize=18, frameon=True, 
          edgecolor='black', fancybox=False, shadow=True)

# 添加網格線
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()

# 保存圖表
output_path = './classification/exp/weights_init/pretrain_effect.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"圖表已保存至 {output_path}")

# # 也保存 PDF 版本以獲得更好的質量
# pdf_path = output_path.replace('.png', '.pdf')
# plt.savefig(pdf_path, bbox_inches='tight')
# print(f"PDF 版本已保存至 {pdf_path}")

plt.show()

print("\n圖表創建完成!")