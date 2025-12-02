#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_data(json_path):
    """載入JSON檔案並計算每個portion的均值和標準差"""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist!")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {}
    for portion_str, values in data.items():
        portion = float(portion_str)
        arr = np.array(values, dtype=float)
        stats[portion] = (arr.mean(), arr.std())
    
    print(f"Loaded {json_path}: {len(stats)} portions")
    return stats

def plot_comparison(data_paths, portion_range=(0, 100), output_path='./plot/comparison.png', x_margin=2.0):
    """
    繪製三條折線圖比較
    
    Args:
        data_paths: dict, 包含三個方法的JSON路徑
            {
                'random_init': 'path/to/random.json',
                'imagenet': 'path/to/imagenet.json',
                'simclr': 'path/to/simclr.json'
            }
        portion_range: tuple, 要顯示的portion範圍 (min, max)
        output_path: str, 輸出圖片路徑
        x_margin: float, x軸左右的空白邊距
    """
    # 載入數據
    print("Loading data...")
    random_stats = load_json_data(data_paths['random_init'])
    imagenet_stats = load_json_data(data_paths['imagenet'])
    simclr_stats = load_json_data(data_paths['simclr'])
    
    # 過濾數據範圍
    def filter_by_range(stats_dict, min_p, max_p):
        return {k: v for k, v in stats_dict.items() if min_p <= k <= max_p}
    
    random_stats = filter_by_range(random_stats, portion_range[0], portion_range[1])
    imagenet_stats = filter_by_range(imagenet_stats, portion_range[0], portion_range[1])
    simclr_stats = filter_by_range(simclr_stats, portion_range[0], portion_range[1])
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 配置：顏色、標記、線型
    configs = {
        'Random Init.': {
            'data': random_stats,
            'color': '#E74C3C',
            'marker': 's',
            'linestyle': '-'
        },
        'ImageNet Pretrain': {
            'data': imagenet_stats,
            'color': '#3498DB',
            'marker': '^',
            'linestyle': '-'
        },
        'OCT SimCLR Pretrain': {
            'data': simclr_stats,
            'color': '#2ECC71',
            'marker': 'o',
            'linestyle': '-'
        }
    }
    
    # 繪製折線
    for label, config in configs.items():
        stats = config['data']
        if stats:
            portions = sorted(stats.keys())
            means = [stats[p][0] for p in portions]
            stds = [stats[p][1] for p in portions]
            
            ax.plot(portions, means, 
                   marker=config['marker'],
                   label=label,
                   color=config['color'],
                   linewidth=2.5,
                   markersize=12,
                   linestyle=config['linestyle'])
            
            ax.fill_between(portions, 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=config['color'],
                          alpha=0.15)
    
    # 設置圖表屬性
    ax.set_xlabel('Training Data Label Portion (%)', fontsize=30, labelpad=10)
    ax.set_ylabel('Accuracy', fontsize=30, labelpad=10)
    ax.tick_params(axis='both', labelsize=28, width=2, length=8)
    ax.grid(axis='both', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # 設置x軸範圍（增加左右空白）和刻度
    ax.set_xlim(portion_range[0] - x_margin, portion_range[1] + x_margin)
    x_ticks = np.arange(portion_range[0], portion_range[1] + 1, 5)
    ax.set_xticks(x_ticks)
    
    # 增粗邊框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加圖例
    ax.legend(fontsize=22, framealpha=0.9, loc='lower right')
    
    # 調整佈局
    fig.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved figure to {output_path}")
    
    # 顯示統計信息
    print("\n=== Statistics Summary ===")
    for label, config in configs.items():
        stats = config['data']
        if stats:
            portions = sorted(stats.keys())
            final_mean, final_std = stats[portions[-1]]
            print(f"{label}: {len(portions)} portions, final: {final_mean:.4f} ± {final_std:.4f}")

def main():
    """主函數"""
    # ============ 在這裡設置三組數據的JSON路徑 ============
    data_paths = {
        'random_init': './exp_results/classification_hard/cold_start_no_pretrained/random42_lr5e-05_bs16.json',
        'imagenet': './exp_results/classification_hard/cold_start/random42_lr5e-05_bs16.json',
        'simclr': './exp_results/classification_hard/cold_start_pretrained_simclr/random42_lr5e-05_bs16.json'
    }
    
    # 設置參數
    portion_range = (5, 50.0)  # 可以修改這個範圍
    output_path = './classification//figs/weights_init/plot_across_label_portion.png'
    x_margin = 2.0  # 調整這個值來控制左右空白的大小（例如：1.0, 2.0, 3.0）
    
    # 繪製圖表
    plot_comparison(data_paths, portion_range=portion_range, output_path=output_path, x_margin=x_margin)

if __name__ == "__main__":
    main()