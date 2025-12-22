#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_json_data(json_path, aug_key='aug4', lr='best'):
    """
    載入JSON檔案並計算每個portion的均值和標準差
    
    Args:
        json_path: JSON檔案路徑
        aug_key: 要使用的augmentation key（如 'aug4', 'aug2_horizontal' 等）
        lr: learning rate選擇
            - 'best': 對每個portion選擇mean最高的lr
            - 具體值（如 '5e-05'）: 只使用該lr的結果
    
    Returns:
        stats: dict, {portion: (mean, std, lr_used)}
    """
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist!")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 檢查 aug_key 是否存在
    if aug_key not in data:
        print(f"Warning: aug_key '{aug_key}' not found in {json_path}")
        print(f"Available keys: {list(data.keys())}")
        return {}
    
    aug_data = data[aug_key]
    stats = {}
    
    for portion_str, lr_dict in aug_data.items():
        portion = float(portion_str)
        
        # 特殊處理：portion=5 強制使用 lr='5e-05'
        if portion == 5.0:
            target_lr = '5e-05'
            if target_lr in lr_dict:
                arr = np.array(lr_dict[target_lr], dtype=float)
                mean = arr.mean()
                std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                stats[portion] = (mean, std, target_lr)
                print(f"  Portion {portion}% forced to use lr={target_lr}")
            else:
                print(f"Warning: lr '{target_lr}' not found for portion {portion_str}, skipping")
            continue
        
        if lr == 'best':
            # 對每個lr計算mean，選擇最好的
            best_mean = -1
            best_std = 0
            best_lr = None
            
            for lr_key, values in lr_dict.items():
                arr = np.array(values, dtype=float)
                mean = arr.mean()
                std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                
                if mean > best_mean:
                    best_mean = mean
                    best_std = std
                    best_lr = lr_key
            
            if best_lr is not None:
                stats[portion] = (best_mean, best_std, best_lr)
        else:
            # 使用指定的lr
            if lr not in lr_dict:
                print(f"Warning: lr '{lr}' not found for portion {portion_str} in {json_path}")
                continue
            
            arr = np.array(lr_dict[lr], dtype=float)
            mean = arr.mean()
            std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            stats[portion] = (mean, std, lr)
    
    print(f"Loaded {json_path} (aug_key={aug_key}, lr={lr}): {len(stats)} portions")
    return stats


def plot_comparison(data_paths, aug_key='aug4', lr='best', portion_range=(0, 100), 
                   output_path='./plot/comparison.png', x_margin=2.0, show_lr_legend=True,
                   show_100_baseline=True):
    """
    繪製三條折線圖比較
    
    Args:
        data_paths: dict, 包含三個方法的JSON路徑
            {
                'random_init': 'path/to/random.json',
                'imagenet': 'path/to/imagenet.json',
                'simclr': 'path/to/simclr.json'
            }
        aug_key: str, augmentation配置key（default: 'aug4'）
        lr: str, learning rate選擇（default: 'best'）
            - 'best': 每個portion選最好的lr
            - 具體值: 只使用該lr
        portion_range: tuple, 要顯示的portion範圍 (min, max)
        output_path: str, 輸出圖片路徑
        x_margin: float, x軸左右的空白邊距
        show_lr_legend: bool, 是否在圖例中顯示使用的lr
        show_100_baseline: bool, 是否顯示 portion=100% 的水平基準線
    """
    # 載入數據
    print("Loading data...")
    print(f"Aug key: {aug_key}")
    print(f"Learning rate: {lr}")
    print("-" * 60)
    
    # 載入完整數據（包含 portion=100）
    random_stats_full = load_json_data(data_paths['random_init'], aug_key, lr)
    imagenet_stats_full = load_json_data(data_paths['imagenet'], aug_key, lr)
    simclr_stats_full = load_json_data(data_paths['simclr'], aug_key, lr)
    
    # 過濾數據範圍（用於繪製折線）
    def filter_by_range(stats_dict, min_p, max_p):
        return {k: v for k, v in stats_dict.items() if min_p <= k <= max_p}
    
    random_stats = filter_by_range(random_stats_full, portion_range[0], portion_range[1])
    imagenet_stats = filter_by_range(imagenet_stats_full, portion_range[0], portion_range[1])
    simclr_stats = filter_by_range(simclr_stats_full, portion_range[0], portion_range[1])
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 配置：顏色、標記、線型
    configs = {
        'Random Init.': {
            'data': random_stats,
            'data_full': random_stats_full,
            'color': '#E74C3C',
            'marker': 's',
            'linestyle': '-'
        },
        'ImageNet Pretrain': {
            'data': imagenet_stats,
            'data_full': imagenet_stats_full,
            'color': '#3498DB',
            'marker': '^',
            'linestyle': '-'
        },
        'OCT SimCLR Pretrain': {
            'data': simclr_stats,
            'data_full': simclr_stats_full,
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
            lrs_used = [stats[p][2] for p in portions]
            
            # 決定圖例標籤
            if lr == 'best' and show_lr_legend:
                # 顯示使用的lr範圍
                unique_lrs = sorted(set(lrs_used), key=lambda x: float(x))
                if len(unique_lrs) == 1:
                    legend_label = f"{label} (w/ 4x Aug, lr={unique_lrs[0]})"
                else:
                    # legend_label = f"{label} (lr=best)"
                    legend_label = f"{label} (w/ 4x Aug)"
            else:
                legend_label = f"{label} (w/ 4x Aug)"
            
            ax.plot(portions, means, 
                   marker=config['marker'],
                   label=legend_label,
                   color=config['color'],
                   linewidth=4,
                   markersize=15,
                   linestyle=config['linestyle'])
            
            ax.fill_between(portions, 
                          np.array(means) - np.array(stds),
                          np.array(means) + np.array(stds),
                          color=config['color'],
                          alpha=0.15)
            
            # 如果是 'best' 模式，在每個點上標註使用的lr（可選）
            if lr == 'best' and False:  # 設為 True 來啟用標註
                for portion, mean, lr_used in zip(portions, means, lrs_used):
                    ax.annotate(lr_used, 
                              (portion, mean),
                              textcoords="offset points",
                              xytext=(0, 10),
                              ha='center',
                              fontsize=8,
                              color=config['color'])
    
    # 繪製 portion=100% 的水平基準線（只畫 ImageNet，固定使用 no_aug）
    if show_100_baseline:
        print("\n=== Drawing 100% Baseline (w/o Aug) ===")
        
        # 只載入 ImageNet 的 no_aug 數據
        imagenet_noaug = load_json_data(data_paths['imagenet'], aug_key='no_aug', lr=lr)
        
        # 只對 ImageNet 繪製水平線
        if imagenet_noaug and 100.0 in imagenet_noaug:
            baseline_mean = imagenet_noaug[100.0][0]
            baseline_std = imagenet_noaug[100.0][1]
            baseline_lr = imagenet_noaug[100.0][2]
            
            # 繪製水平虛線（使用 ImageNet 的顏色）
            ax.axhline(y=baseline_mean, 
                    #   color='#3498DB',  # ImageNet 的藍色
                      color='black',
                      linestyle='--', 
                      linewidth=6.0,
                      alpha=0.9,
                      label=f"ImageNet Pretrain (100%, w/o Aug)")
            
            print(f"ImageNet Pretrain: 100% w/o Aug accuracy = {baseline_mean:.4f} ± {baseline_std:.4f} (lr={baseline_lr})")
        else:
            print(f"Warning: portion=100.0 not found in no_aug for ImageNet")
    
    # 設置圖表屬性
    ax.set_xlabel('Training Data Label Portion (%)', fontsize=35, labelpad=10)
    ax.set_ylabel('Test Accuracy', fontsize=35, labelpad=10)
    ax.tick_params(axis='both', labelsize=32, width=2, length=8)
    ax.grid(axis='both', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # 設置x軸範圍（增加左右空白）和刻度
    ax.set_xlim(portion_range[0] - x_margin, portion_range[1] + x_margin)
    x_ticks = np.arange(portion_range[0], portion_range[1] + 1, 5)
    ax.set_xticks(x_ticks)
    
    # 增粗邊框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # 添加圖例
    ax.legend(fontsize=25, framealpha=0.9, loc='lower right')
    
    # 添加標題說明配置
    title_parts = []
    if aug_key != 'aug4':
        title_parts.append(f"Aug: {aug_key}")
    if lr != 'best':
        title_parts.append(f"LR: {lr}")
    
    if title_parts:
        ax.set_title(', '.join(title_parts), fontsize=20, pad=15)
    
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
            print(f"\n{label}:")
            print(f"  Portions: {len(portions)}")
            
            if lr == 'best':
                # 顯示每個portion使用的lr
                print(f"  Learning rates used:")
                for p in portions:
                    mean, std, lr_used = stats[p]
                    print(f"    {p:5.1f}%: lr={lr_used}, acc={mean:.4f}±{std:.4f}")
            else:
                # 只顯示最後一個
                final_mean, final_std, final_lr = stats[portions[-1]]
                print(f"  Final: {final_mean:.4f} ± {final_std:.4f}")


def main():
    """主函數"""
    # ============ 在這裡設置三組數據的JSON路徑 ============
    data_paths = {
        'random_init': './exp_results/classification_hard/cold_start_random/random42_bs16.json',
        'imagenet': './exp_results/classification_hard/cold_start_imagenet/random42_bs16.json',
        'simclr': './exp_results/classification_hard/cold_start_simclr/random42_bs16.json'
    }
    
    # ============ 設置參數 ============
    aug_key = 'aug4'  # 可選: 'aug4', 'aug2_horizontal', 'aug2_vertical', 'aug3', 'no_aug' 等
    lr = 'best'       # 可選: 'best' 或具體值如 '5e-05', '0.0001' 等
    # lr = '0.0001'

    portion_range = (5, 55.0)  # portion 範圍
    output_path = './classification/exp/weights_init/plot_across_label_portion.png'
    x_margin = 2.0  # x軸左右空白
    show_lr_legend = False  # 是否在圖例中顯示lr信息
    show_100_baseline = True  # 是否顯示 portion=100% 的水平基準線
    
    # ============ 繪製圖表 ============
    plot_comparison(
        data_paths, 
        aug_key=aug_key,
        lr=lr,
        portion_range=portion_range, 
        output_path=output_path, 
        x_margin=x_margin,
        show_lr_legend=show_lr_legend,
        show_100_baseline=show_100_baseline
    )


if __name__ == "__main__":
    main()