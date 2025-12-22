#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_json_data_by_prefix(data_dir, prefix=None):
    """
    讀取指定目錄下特定前綴的json檔案，合併相同key的數據
    返回 {portion: [values], ...} 格式的字典
    """
    combined_data = defaultdict(list)
    
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist!")
        return {}
    
    # 根據前綴過濾檔案
    if prefix:
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f.startswith(prefix)]
    else:
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Warning: No JSON files found with prefix '{prefix}' in {data_dir}")
        return {}
    
    print(f"Loading data from {data_dir} with prefix '{prefix}':")
    print(f"Found {len(json_files)} JSON files")
    
    for fn in sorted(json_files):
        full_path = os.path.join(data_dir, fn)
        if os.path.getsize(full_path) == 0:
            print(f"  Skipping empty file: {fn}")
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  Reading file: {fn}")
            
            # 將數據合併到combined_data中
            for portion_str, values in data.items():
                try:
                    portion = float(portion_str)
                    # 處理嵌套的JSON結構 - 提取'acc'值
                    if isinstance(values, dict) and 'acc' in values:
                        combined_data[portion].append(values['acc'])
                        print(f"    Portion {portion}%: acc = {values['acc']:.4f}")
                    elif isinstance(values, list):
                        combined_data[portion].extend(values)
                        print(f"    Portion {portion}%: added {len(values)} values")
                    elif isinstance(values, (int, float)):
                        combined_data[portion].append(values)
                        print(f"    Portion {portion}%: value = {values:.4f}")
                except ValueError:
                    print(f"    Skipping invalid portion: {portion_str}")
                    continue
                    
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read {fn}: {e}")
            continue
    
    print(f"Loaded portions: {sorted(combined_data.keys())}")
    for portion in sorted(combined_data.keys()):
        print(f"  Portion {portion}%: {len(combined_data[portion])} samples")
    
    return dict(combined_data)

def load_json_data(data_dir, file_pattern=None):
    """
    讀取指定目錄下的所有json檔案，合併相同key的數據
    返回 {portion: [values], ...} 格式的字典
    """
    return load_json_data_by_prefix(data_dir, prefix=None)

def calculate_stats(data_dict):
    """
    計算每個portion的均值和標準差
    返回 {portion: (mean, std), ...}
    """
    stats = {}
    for portion, values in data_dict.items():
        if values:
            arr = np.array(values, dtype=float)
            stats[portion] = (arr.mean(), arr.std())
        else:
            stats[portion] = (0.0, 0.0)
    return stats

def get_full_performance(data_dir, target_key=100.0):
    """
    從指定目錄獲取Full performance (key=100.0)
    返回 (mean, std)
    """
    combined_data = load_json_data(data_dir)
    
    if target_key in combined_data:
        values = combined_data[target_key]
        arr = np.array(values, dtype=float)
        return arr.mean(), arr.std()
    else:
        print(f"Warning: Key {target_key} not found in {data_dir}")
        return None, None

def plot_active_learning_results(portion_range=(0, 100), output_path='./plot/active_learning_results.png'):
    """
    繪製Active Learning結果圖
    
    Args:
        portion_range: tuple, 要顯示的portion範圍 (min, max)
        output_path: str, 輸出圖片路徑
    """
    
    # 數據路徑配置
    paths = {
        'full_imagenet': './paper_exp/method_0/results/classification_hard/cold_start',
        'random_imagenet': './paper_exp/method_0/results/classification_hard/cold_start',
        
        'random_simclr_oct': './paper_exp/method_0/results/classification_hard/cold_start_pretrained_simclr',
        'full_simclr_oct': './paper_exp/method_0/results/classification_hard/cold_start_pretrained_simclr',
        
        # Active Learning methods
        'al_base_dir': './paper_exp/method_1/results/iterative_pretrained_simclr/classification_hard'
    }
    
    # 加載基礎數據
    print("Loading Full (ImageNet) performance...")
    full_imagenet_mean, full_imagenet_std = get_full_performance(paths['full_imagenet'], 100.0)
    
    print("\nLoading Full (SimCLR-OCT) performance...")
    full_simclr_mean, full_simclr_std = get_full_performance(paths['full_simclr_oct'], 100.0)
    
    print("\nLoading Random Select (SimCLR-OCT) data...")
    random_simclr_data = load_json_data(paths['random_simclr_oct'])
    random_simclr_stats = calculate_stats(random_simclr_data)
    
    print("\nLoading Random Select (ImageNet) data...")
    random_imagenet_data = load_json_data(paths['random_imagenet'])
    random_imagenet_stats = calculate_stats(random_imagenet_data)
    
    # 加載Active Learning方法數據
    al_methods = {
        'coreset': ('Coreset', 'diversity'),
        'badge': ('Badge', 'hybrid'),
        'entropy': ('Entropy', 'uncertainty'), 
        'margin': ('Margin', 'uncertainty'),
        'conf': ('Confidence', 'uncertainty'),
    }
    
    al_stats = {}
    for prefix, (display_name, category) in al_methods.items():
        print(f"\nLoading {display_name} data...")
        al_data = load_json_data_by_prefix(paths['al_base_dir'], prefix)
        al_stats[display_name] = calculate_stats(al_data)
    
    # 過濾數據範圍
    def filter_by_range(stats_dict, min_portion, max_portion):
        return {k: v for k, v in stats_dict.items() if min_portion <= k <= max_portion}
    
    random_simclr_stats = filter_by_range(random_simclr_stats, portion_range[0], portion_range[1])
    random_imagenet_stats = filter_by_range(random_imagenet_stats, portion_range[0], portion_range[1])
    
    for method_name in al_stats:
        al_stats[method_name] = filter_by_range(al_stats[method_name], portion_range[0], portion_range[1])
    
    # 創建圖表 - 增大尺寸以適合paper
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.ylim(0.455, 0.935)
    
    # 顏色配置 - SimCLR方法使用鮮明對比色，ImageNet使用黑色
    colors = {
        # SimCLR Pretrained - 鮮明對比色
        'SimCLR_Full': 'black',
        'SimCLR_Random': 'black',
        'Confidence': '#E74C3C',        # 亮紅色
        'Entropy': '#3498DB',           # 藍色
        'Margin': '#2ECC71',            # 綠色
        'Coreset': '#F39C12',           # 橙色
        'Badge': '#9B59B6',             # 紫色
        
        # ImageNet Pretrained - 黑色
        'ImageNet_Full': 'black',
        'ImageNet_Random': 'black',
    }
    
    # 線寬配置 - 增加線寬以便於paper使用
    linewidths = {
        'SimCLR_Full': 2.5,             # SimCLR Full 一般實線
        'SimCLR_Random': 2.5,
        'Confidence': 2.5,
        'Entropy': 2.5,
        'Margin': 2.5,
        'Coreset': 2.5,
        'Badge': 2.5,
        'ImageNet_Full': 2.5,           # ImageNet Full 虛線
        'ImageNet_Random': 2.5,
    }
    
    # 標記符號配置 - SimCLR用圓形，ImageNet用三角形
    markers_config = {
        'SimCLR_Random': 'o',           # 圓形
        'ImageNet_Random': '^',         # 三角形
        'Confidence': 'o',              # 圓形
        'Entropy': 'o',                 # 圓形
        'Margin': 'o',                  # 圓形
        'Coreset': 'o',                 # 圓形
        'Badge': 'o',                   # 圓形
    }
    
    # 繪製水平參考線 - SimCLR Full實線，ImageNet Full虛線
    if full_imagenet_mean is not None:
        ax.axhline(y=full_imagenet_mean, color=colors['ImageNet_Full'], linestyle='--', 
                  linewidth=linewidths['ImageNet_Full'], 
                  label='ImageNet Pretrained - Full')
    
    if full_simclr_mean is not None:
        ax.axhline(y=full_simclr_mean, color=colors['SimCLR_Full'], linestyle='-', 
                  linewidth=linewidths['SimCLR_Full'], 
                  label='SimCLR Pretrained - Full')
    
    # 繪製Random Select折線 - 都用黑色虛線但不同標記和線寬
    if random_simclr_stats:
        portions = sorted(random_simclr_stats.keys())
        means = [random_simclr_stats[p][0] for p in portions]
        stds = [random_simclr_stats[p][1] for p in portions]
        
        ax.plot(portions, means, marker=markers_config['SimCLR_Random'], 
               label='SimCLR Pretrained - Random', 
               color=colors['SimCLR_Random'], 
               linewidth=linewidths['SimCLR_Random'], 
               markersize=12, linestyle='-')  # 增大marker size
        ax.fill_between(portions, 
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       color=colors['SimCLR_Random'], alpha=0.2)
    
    if random_imagenet_stats:
        portions = sorted(random_imagenet_stats.keys())
        means = [random_imagenet_stats[p][0] for p in portions]
        stds = [random_imagenet_stats[p][1] for p in portions]
        
        ax.plot(portions, means, marker=markers_config['ImageNet_Random'], 
               label='ImageNet Pretrained - Random', 
               color=colors['ImageNet_Random'], 
               linewidth=linewidths['ImageNet_Random'], 
               markersize=12, linestyle='--')  # 增大marker size
        ax.fill_between(portions, 
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       color=colors['ImageNet_Random'], alpha=0.2)
    
    # 繪製Active Learning方法折線 - 都屬於SimCLR Pretrained，用鮮明對比色和圓形標記
    for method_name, stats in al_stats.items():
        if stats:
            portions = sorted(stats.keys())
            means = [stats[p][0] for p in portions]
            stds = [stats[p][1] for p in portions]
            
            ax.plot(portions, means, 
                   marker=markers_config[method_name], 
                   label=f'SimCLR Pretrained - {method_name}', 
                   color=colors[method_name], 
                   linewidth=linewidths[method_name], 
                   markersize=12,  # 增大marker size
                   linestyle='-')
            
            ax.fill_between(portions, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           color=colors[method_name], 
                           alpha=0.15)
    
    # 設置圖表屬性 - 增大字體以適合paper
    ax.set_xlabel('Training Data Label Portion (%)', fontsize=28)
    ax.set_ylabel('Accuracy', fontsize=28)
    ax.tick_params(axis='both', labelsize=28, width=2, length=8)  # 增大刻度字體和刻度線
    ax.grid(axis='both', linestyle='--', alpha=0.5, linewidth=1.5)  # 增粗網格線，同時顯示x和y軸
    
    # 設置x軸範圍和刻度
    ax.set_xlim(portion_range[0], portion_range[1])
    
    # 設置x軸刻度為0, 5, 10, 15, 20, ... 等間距
    x_ticks = np.arange(0, portion_range[1] + 1, 5)
    ax.set_xticks(x_ticks)
    
    # 增粗邊框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加層次化圖例 - 修改為簡化標題和左對齊
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # SimCLR 組
    legend_elements.append(Line2D([0], [0], color='none', label='SimCLR'))
    
    # Full 和 Random (基礎方法) - 增大字體
    if full_simclr_mean is not None:
        legend_elements.append(Line2D([0], [0], color=colors['SimCLR_Full'], linestyle='-', 
                                     linewidth=linewidths['SimCLR_Full'], label='  Full'))
    if random_simclr_stats:
        legend_elements.append(Line2D([0], [0], color=colors['SimCLR_Random'], 
                                     linewidth=linewidths['SimCLR_Random'], linestyle='-',
                                     marker=markers_config['SimCLR_Random'], markersize=12, 
                                     label='  Random'))
    
    # Active Learning 方法 - 增大字體
    al_method_order = ['Confidence', 'Entropy', 'Margin', 'Coreset', 'Badge']
    for method_name in al_method_order:
        if method_name in al_stats and al_stats[method_name]:
            legend_elements.append(Line2D([0], [0], color=colors[method_name], 
                                         linewidth=linewidths[method_name],
                                         marker=markers_config[method_name], markersize=12, 
                                         label=f'  {method_name}'))
    
    # 添加空行分隔
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # ImageNet 組
    legend_elements.append(Line2D([0], [0], color='none', label='ImageNet'))
    
    if full_imagenet_mean is not None:
        legend_elements.append(Line2D([0], [0], color=colors['ImageNet_Full'], linestyle='--', 
                                     linewidth=linewidths['ImageNet_Full'], label='  Full'))
    if random_imagenet_stats:
        legend_elements.append(Line2D([0], [0], color=colors['ImageNet_Random'], 
                                     linewidth=linewidths['ImageNet_Random'], linestyle='--',
                                     marker=markers_config['ImageNet_Random'], markersize=12,
                                     label='  Random'))
    
    # 創建圖例 - 增大字體並左對齊
    legend = ax.legend(handles=legend_elements, fontsize=20, framealpha=0.9, 
                      loc='lower right', handlelength=2.5, markerscale=1.2,
                      columnspacing=0.5, labelspacing=0.3)
    
    # 設置分組標題為粗體並左對齊
    for text in legend.get_texts():
        if text.get_text() in ['SimCLR', 'ImageNet']:
            text.set_weight('bold')
            text.set_ha('left')  # 左對齊
    
    # 調整佈局
    fig.tight_layout()
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存圖片
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\nSaved figure to {output_path}")
    
    # 顯示統計信息
    print("\n=== Statistics Summary ===")
    if full_imagenet_mean is not None:
        print(f"Full (ImageNet): {full_imagenet_mean:.4f} ± {full_imagenet_std:.4f}")
    if full_simclr_mean is not None:
        print(f"Full (SimCLR-OCT): {full_simclr_mean:.4f} ± {full_simclr_std:.4f}")
    
    if random_simclr_stats:
        portions = sorted(random_simclr_stats.keys())
        final_mean, final_std = random_simclr_stats[portions[-1]]
        print(f"Random Select (SimCLR-OCT): {len(portions)} portions, final: {final_mean:.4f} ± {final_std:.4f}")
    
    if random_imagenet_stats:
        portions = sorted(random_imagenet_stats.keys())
        final_mean, final_std = random_imagenet_stats[portions[-1]]
        print(f"Random Select (ImageNet): {len(portions)} portions, final: {final_mean:.4f} ± {final_std:.4f}")
    
    # 顯示AL方法統計
    print("\n=== Active Learning Methods ===")
    for method_name, stats in al_stats.items():
        if stats:
            portions = sorted(stats.keys())
            if portions:
                final_mean, final_std = stats[portions[-1]]
                print(f"{method_name}: {len(portions)} portions, final: {final_mean:.4f} ± {final_std:.4f}")
            else:
                print(f"{method_name}: No data loaded")

def main():
    """
    主函數 - 可以自定義portion範圍和輸出路徑
    """
    # 設置參數
    portion_range = (0, 50.0)  # 可以修改這個範圍，例如 (10, 50)
    output_path = './paper_exp/plot/AL_results/hard.png'
    
    # 繪製圖表
    plot_active_learning_results(portion_range=portion_range, output_path=output_path)

if __name__ == "__main__":
    main()