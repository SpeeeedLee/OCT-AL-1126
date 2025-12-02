import matplotlib.pyplot as plt
import numpy as np

# ==================== 在這裡修改數據 ====================

# X 軸標籤（training epochs）
epochs = [10, 25, 50, 100]

# 不同 batch size 的數據
data = {
    '32':  [[0.5725, 0.5765, 0.5647], [0.5843, 0.5804, 0.5961], [0.5922, 0.5765, 0.5882], [0.6000, 0.5765, 0.5961]],
    '64':  [[0.5882, 0.5686, 0.5686], [0.5725, 0.5961, 0.6157], [0.6235, 0.5843, 0.6196], [0.6745, 0.6431, 0.6549]],
    '128': [[0.6275, 0.6039, 0.6510], [0.6745, 0.6588, 0.6588], [0.6667, 0.6588, 0.6549], [0.6706, 0.6431, 0.6627]],
    '256': [[0.6471, 0.6549, 0.6392], [0.6784, 0.6510, 0.6588], [0.6353, 0.6627, 0.6784], [0.6824, 0.6471, 0.6706]],
}

# 圖表設置
title = ''
x_label = 'Pretraining epochs'
y_label = 'Downstream Acc.'
legend_title = 'Batch size'

# 自定義顏色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ==================== 字體大小設置 ====================
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 30
FONT_SIZE_TICKS = 24
FONT_SIZE_LEGEND = 18
FONT_SIZE_LEGEND_TITLE = 20

# ========================================================

def plot_grouped_bar_chart():
    # # 設置全局字體（可選：使用適合論文的字體）
    # plt.rcParams['font.family'] = 'serif'  # 使用 serif 字體
    # plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    
    # 設置圖表大小
    plt.figure(figsize=(12, 7))  # 稍微調高一點以容納更大的字體
    
    # 計算柱子的位置
    batch_sizes = list(data.keys())
    n_groups = len(epochs)
    n_bars = len(batch_sizes)
    bar_width = 0.8 / n_bars
    
    # X 軸位置
    x = np.arange(n_groups)
    
    # 為每個 batch size 繪製柱子
    for i, (batch_size, runs_per_epoch) in enumerate(data.items()):
        # 計算每個 epoch 的平均值和標準差
        means = []
        stds = []
        
        for runs in runs_per_epoch:
            runs_array = np.array(runs)
            means.append(np.mean(runs_array))
            stds.append(np.std(runs_array, ddof=1))
        
        # 計算柱子的 x 位置偏移
        offset = (i - n_bars / 2) * bar_width + bar_width / 2
        
        # 繪製柱狀圖
        bars = plt.bar(
            x + offset, 
            means, 
            bar_width, 
            label=batch_size,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor='black',  # 添加黑色邊框使柱子更清晰
            linewidth=0.5
        )
        
        # 添加誤差線（標準差）
        plt.errorbar(
            x + offset, 
            means, 
            yerr=stds,
            fmt='none',
            ecolor='black',
            capsize=6,  # 稍微加大
            capthick=2,
            elinewidth=2,
            alpha=0.8
        )
    
    # 設置 Y 軸範圍（在畫水平線之前設置，以便獲取 xlim）
    all_means = []
    all_stds = []
    for runs_per_epoch in data.values():
        for runs in runs_per_epoch:
            runs_array = np.array(runs)
            all_means.append(np.mean(runs_array))
            all_stds.append(np.std(runs_array, ddof=1))
    
    # 考慮水平線的位置
    y_min = 0.35
    y_max = 0.70
    plt.ylim(y_min, y_max)

    # ==================== 添加水平線和標籤 ====================
    # 獲取當前的 x 軸範圍
    xlim = plt.xlim()
    x_range = xlim[1] - xlim[0]
    
    # 水平線 1: Random Init.
    y1 = 0.3752
    # 畫左段線（從最左到中間偏右前）
    plt.plot([xlim[0], xlim[0] + x_range * 0.70], [y1, y1], 
             color='black', linestyle='--', linewidth=2, alpha=0.8)
    # 畫右段線（從文字後到最右）
    plt.plot([xlim[0] + x_range * 0.95, xlim[1]], [y1, y1], 
             color='black', linestyle='--', linewidth=2, alpha=0.8)
    # 添加文字標籤
    plt.text(xlim[0] + x_range * 0.83, y1, 'Random Init.', 
             fontsize=20, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', 
                        edgecolor='black', alpha=0.9)
            )
    
    # 水平線 2: ImageNet Pretrain
    y2 = 0.5490
    # 畫左段線（從最左到中間偏右前）
    plt.plot([xlim[0], xlim[0] + x_range * 0.67], [y2, y2], 
             color='black', linestyle='--', linewidth=2, alpha=0.8)
    # 畫右段線（從文字後到最右）
    plt.plot([xlim[0] + x_range * 0.985, xlim[1]], [y2, y2], 
             color='black', linestyle='--', linewidth=2, alpha=0.8)
    # 添加文字標籤
    plt.text(xlim[0] + x_range * 0.83, y2, 'ImageNet Pretrain', 
             fontsize=20, va='center', ha='center',
             bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', 
                        edgecolor='black', alpha=0.9)
            )
    
    # ===================================================

    # 設置 X 軸標籤
    plt.xticks(x, epochs, fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    
    # 設置標籤和標題
    plt.xlabel(x_label, fontsize=FONT_SIZE_LABEL, labelpad=10)
    plt.ylabel(y_label, fontsize=FONT_SIZE_LABEL, labelpad=10)
    if title:
        plt.title(title, fontsize=FONT_SIZE_TITLE)
    
    # 設置圖例（只顯示 batch size）
    plt.legend(
        title=legend_title, 
        fontsize=FONT_SIZE_LEGEND, 
        title_fontsize=FONT_SIZE_LEGEND_TITLE,
        loc='lower left',
    )
    
    # 添加網格
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    
    # 添加邊框
    ax = plt.gca()
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 調整佈局
    plt.tight_layout()
    
    # 保存圖片
    plt.savefig('./classification/figs/weights_init/simclr.png', dpi=300, bbox_inches='tight')
    # plt.savefig('./classification/model/simclr/fig.pdf', bbox_inches='tight')  # 也保存 PDF 格式供論文使用
    print("圖表已保存為 ./classification/figs/weights_init/simclr.png")
    
    # # 顯示圖表
    # plt.show()

if __name__ == '__main__':
    plot_grouped_bar_chart()