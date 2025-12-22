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
            else:
                print(f"Warning: lr '{target_lr}' not found for portion {portion_str}, skipping")
            continue
        
        if lr == 'best':
            # 對每個lr計算mean，選擇最好的
            best_mean = -1
            best_std = 0
            best_lr = None
            
            for lr_key, values in lr_dict.items():
                # 處理三種格式：{"acc": [...]}, [value1, value2, ...], 或單一數值
                if isinstance(values, dict) and 'acc' in values:
                    # 格式: {"acc": [value1, value2, ...], ...}
                    acc_data = values['acc']
                    if isinstance(acc_data, list):
                        arr = np.array(acc_data, dtype=float)
                        mean = arr.mean()
                        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                    else:
                        mean = float(acc_data)
                        std = 0.0
                elif isinstance(values, list):
                    # 格式: [value1, value2, ...]
                    arr = np.array(values, dtype=float)
                    mean = arr.mean()
                    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                else:
                    # 格式: 單一數值
                    mean = float(values)
                    std = 0.0
                
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
            
            values = lr_dict[lr]
            # 處理三種格式
            if isinstance(values, dict) and 'acc' in values:
                # 格式: {"acc": [value1, value2, ...], ...}
                acc_data = values['acc']
                if isinstance(acc_data, list):
                    arr = np.array(acc_data, dtype=float)
                    mean = arr.mean()
                    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
                else:
                    mean = float(acc_data)
                    std = 0.0
            elif isinstance(values, list):
                arr = np.array(values, dtype=float)
                mean = arr.mean()
                std = arr.std(ddof=1) if len(arr) > 1 else 0.0
            else:
                mean = float(values)
                std = 0.0
            
            stats[portion] = (mean, std, lr)
    
    print(f"Loaded {json_path} (aug_key={aug_key}, lr={lr}): {len(stats)} portions")
    return stats


def plot_with_active_learning(base_methods, al_methods, aug_key='aug4', lr='best',
                               portion_range=(0, 100), output_path='./plot/al_comparison.png',
                               x_margin=2.0, show_100_baseline=True):
    """
    繪製包含 Active Learning 方法的比較圖
    
    Args:
        base_methods: dict, 基礎方法的JSON路徑
            {
                'random_init': 'path/to/random.json',
                'imagenet': 'path/to/imagenet.json',
                'simclr': 'path/to/simclr.json'
            }
        al_methods: dict, Active Learning 方法的JSON路徑
            {
                'confidence': 'path/to/confidence.json',
                'entropy': 'path/to/entropy.json',
                'margin': 'path/to/margin.json',
                'coreset': 'path/to/coreset.json',
                'badge': 'path/to/badge.json'
            }
        aug_key: str, augmentation配置key
        lr: str, learning rate選擇
        portion_range: tuple, 要顯示的portion範圍
        output_path: str, 輸出圖片路徑
        x_margin: float, x軸左右的空白邊距
        show_100_baseline: bool, 是否顯示 portion=100% 的水平基準線
    """
    print("=" * 80)
    print("Loading data for Active Learning comparison...")
    print(f"Aug key: {aug_key}")
    print(f"Learning rate: {lr}")
    print("=" * 80)
    
    # ========== 載入基礎方法數據 ==========
    print("\n[Base Methods]")
    base_stats = {}
    for method_name, json_path in base_methods.items():
        stats_full = load_json_data(json_path, aug_key, lr)
        base_stats[method_name] = {k: v for k, v in stats_full.items() 
                                   if portion_range[0] <= k <= portion_range[1]}
    
    # 載入 100% baseline (ImageNet, no_aug)
    imagenet_100_stats = None
    if show_100_baseline and 'imagenet' in base_methods:
        imagenet_noaug = load_json_data(base_methods['imagenet'], aug_key='no_aug', lr=lr)
        if 100.0 in imagenet_noaug:
            imagenet_100_stats = imagenet_noaug[100.0]
            print(f"\nImageNet 100% (w/o Aug): {imagenet_100_stats[0]:.4f} ± {imagenet_100_stats[1]:.4f}")
    
    # ========== 載入 Active Learning 方法數據 ==========
    print("\n[Active Learning Methods]")
    al_stats = {}
    for method_name, json_path in al_methods.items():
        stats_full = load_json_data(json_path, aug_key, lr)
        al_stats[method_name] = {k: v for k, v in stats_full.items() 
                                if portion_range[0] <= k <= portion_range[1]}
    
    # ========== 所有 AL 方法的 5% 使用 SimCLR 的數據 ==========
    if 'simclr' in base_stats and 5.0 in base_stats['simclr']:
        simclr_5_mean, simclr_5_std, simclr_5_lr = base_stats['simclr'][5.0]
        print(f"\n[Using SimCLR 5% for all AL methods]")
        print(f"SimCLR 5%: {simclr_5_mean:.4f} ± {simclr_5_std:.4f} (lr={simclr_5_lr})")
        
        for method_name in al_stats.keys():
            if al_stats[method_name]:  # 如果該方法有數據
                # 替換 5% 的數據為 SimCLR 的數據
                al_stats[method_name][5.0] = (simclr_5_mean, simclr_5_std, simclr_5_lr)
                print(f"  → Updated {method_name} 5%")
    
    # ========== 創建圖表 ==========
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 顏色配置
    # 基礎方法：灰黑色系
    gray_colors = {
        'random_init': '#4A4A4A',      # 深灰
        'imagenet': '#2C2C2C',         # 更深灰
        'simclr': '#696969',           # 中灰
        'baseline': '#000000'          # 全黑 (目標 baseline)
    }
    
    # AL 方法：按類型分組
    al_colors = {
        # Uncertainty-based (藍色系)
        'confidence': '#2E86DE',       # 亮藍
        'entropy': '#54A0FF',          # 天藍
        'margin': '#1E3799',           # 深藍
        
        # Diversity-based (橙色系)
        'coreset': '#EE5A24',          # 橙紅
        
        # Hybrid (紫色系)
        'badge': '#9C27B0',            # 紫色
    }
    
    # 標記配置：不同基礎方法使用不同標記
    base_markers = {
        'random_init': 's',  # 方形
        'imagenet': '^',     # 三角形
        'simclr': 'o'        # 圓形
    }
    al_marker = 'o'    # AL 方法使用圓形
    
    # ========== 繪製 100% baseline ==========
    if imagenet_100_stats is not None:
        ax.axhline(y=imagenet_100_stats[0], 
                  color=gray_colors['baseline'],
                  linestyle='--', 
                  linewidth=6.0,
                  alpha=0.8,
                  label='ImageNet (100%, w/o Aug)')
    
    # ========== 繪製基礎方法折線 ==========
    base_labels = {
        'random_init': 'Random Init.',
        'imagenet': 'ImageNet Pretrain',
        'simclr': 'OCT SimCLR Pretrain'
    }
    
    for method_name in ['random_init', 'imagenet', 'simclr']:
        if method_name not in base_stats or not base_stats[method_name]:
            continue
            
        stats = base_stats[method_name]
        portions = sorted(stats.keys())
        means = [stats[p][0] for p in portions]
        stds = [stats[p][1] for p in portions]
        
        ax.plot(portions, means,
               marker=base_markers[method_name],
               label=base_labels[method_name],
               color=gray_colors[method_name],
               linewidth=4,
               markersize=15,
               linestyle='-',
               alpha=0.7)
        
        ax.fill_between(portions,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       color=gray_colors[method_name],
                       alpha=0.1)
    
    # ========== 繪製 Active Learning 方法 ==========
    al_labels = {
        'confidence': 'Confidence',
        'entropy': 'Entropy',
        'margin': 'Margin',
        'coreset': 'Coreset',
        'badge': 'Badge'
    }
    
    al_order = ['confidence', 'entropy', 'margin', 'coreset', 'badge']
    
    for method_name in al_order:
        if method_name not in al_stats or not al_stats[method_name]:
            continue
            
        stats = al_stats[method_name]
        portions = sorted(stats.keys())
        means = [stats[p][0] for p in portions]
        stds = [stats[p][1] for p in portions]
        
        ax.plot(portions, means,
               marker=al_marker,
               label=al_labels[method_name],
               color=al_colors[method_name],
               linewidth=4,
               markersize=15,
               linestyle='-')
        
        ax.fill_between(portions,
                       np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       color=al_colors[method_name],
                       alpha=0.15)
    
    # ========== 設置圖表屬性 ==========
    ax.set_xlabel('Training Data Label Portion (%)', fontsize=35, labelpad=10)
    ax.set_ylabel('Test Accuracy', fontsize=35, labelpad=10)
    ax.tick_params(axis='both', labelsize=32, width=2, length=8)
    ax.grid(axis='both', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # 設置x軸範圍和刻度
    ax.set_xlim(portion_range[0] - x_margin, portion_range[1] + x_margin)
    x_ticks = np.arange(portion_range[0], portion_range[1] + 1, 5)
    ax.set_xticks(x_ticks)
    
    # 增粗邊框
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # ========== 創建分組圖例（移到圖外右邊）==========
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    
    legend_elements = []
    
    # 基礎方法組 - 使用粗體標題
    legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                 label='Baseline Methods', 
                                 linestyle=''))
    
    if imagenet_100_stats is not None:
        legend_elements.append(Line2D([0], [0], 
                                     color=gray_colors['baseline'],
                                     linestyle='--',
                                     linewidth=3.5,
                                     label='ImageNet (100%, w/o Aug)'))
    
    for method_name in ['random_init', 'imagenet', 'simclr']:
        if method_name in base_stats and base_stats[method_name]:
            legend_elements.append(Line2D([0], [0],
                                         color=gray_colors[method_name],
                                         marker=base_markers[method_name],
                                         linewidth=2.5,
                                         markersize=10,
                                         label=f'{base_labels[method_name]}'))
    
    # 分隔線
    legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                 label='', linestyle=''))
    
    # Active Learning 組 - 使用粗體標題
    legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                 label='Active Learning', 
                                 linestyle=''))
    
    # Uncertainty-based - 改進排版
    legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                 label='Uncertainty-based', 
                                 linestyle=''))
    for method_name in ['confidence', 'entropy', 'margin']:
        if method_name in al_stats and al_stats[method_name]:
            legend_elements.append(Line2D([0], [0],
                                         color=al_colors[method_name],
                                         marker=al_marker,
                                         linewidth=2.5,
                                         markersize=10,
                                         label=f'  {al_labels[method_name]}'))
    
    # Diversity-based
    if 'coreset' in al_stats and al_stats['coreset']:
        legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                     label='Diversity-based', 
                                     linestyle=''))
        legend_elements.append(Line2D([0], [0],
                                     color=al_colors['coreset'],
                                     marker=al_marker,
                                     linewidth=2.5,
                                     markersize=10,
                                     label=f'  {al_labels["coreset"]}'))
    
    # Hybrid
    if 'badge' in al_stats and al_stats['badge']:
        legend_elements.append(Line2D([0], [0], color='none', marker='none',
                                     label='Hybrid', 
                                     linestyle=''))
        legend_elements.append(Line2D([0], [0],
                                     color=al_colors['badge'],
                                     marker=al_marker,
                                     linewidth=2.5,
                                     markersize=10,
                                     label=f'  {al_labels["badge"]}'))
    
    # 添加圖例到圖外右側
    legend = ax.legend(handles=legend_elements, 
                      fontsize=20,
                      framealpha=0.95,
                      loc='center left',
                      bbox_to_anchor=(1.02, 0.5),  # 移到圖外右邊
                      handlelength=2.5,
                      labelspacing=0.5,
                      handletextpad=0.8,
                      borderpad=1.0)
    
    # 設置分組標題為粗體和斜體
    for i, text in enumerate(legend.get_texts()):
        label = text.get_text()
        # 主標題：Baseline Methods, Active Learning
        if label in ['Baseline Methods', 'Active Learning']:
            text.set_weight('bold')
            text.set_fontsize(22)
        # 子標題：Uncertainty-based, Diversity-based, Hybrid
        elif label in ['Uncertainty-based', 'Diversity-based', 'Hybrid']:
            text.set_style('italic')
            text.set_fontsize(18)
    
    # 調整佈局以容納圖例
    fig.tight_layout()
    
    # 保存圖片（使用 bbox_inches='tight' 確保圖例不被裁剪）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n{'=' * 80}")
    print(f"Saved figure to {output_path}")
    print(f"{'=' * 80}")
    
    # ========== 顯示統計信息 ==========
    print("\n=== Statistics Summary ===\n")
    
    print("[Baseline Methods]")
    if imagenet_100_stats is not None:
        print(f"ImageNet 100% (w/o Aug): {imagenet_100_stats[0]:.4f} ± {imagenet_100_stats[1]:.4f}")
    
    for method_name in ['random_init', 'imagenet', 'simclr']:
        if method_name in base_stats and base_stats[method_name]:
            stats = base_stats[method_name]
            portions = sorted(stats.keys())
            if portions:
                final_p = portions[-1]
                final_mean, final_std = stats[final_p][0], stats[final_p][1]
                print(f"{base_labels[method_name]} ({final_p}%): {final_mean:.4f} ± {final_std:.4f}")
    
    print("\n[Active Learning Methods]")
    for method_name in al_order:
        if method_name in al_stats and al_stats[method_name]:
            stats = al_stats[method_name]
            portions = sorted(stats.keys())
            if portions:
                final_p = portions[-1]
                final_mean, final_std = stats[final_p][0], stats[final_p][1]
                print(f"{al_labels[method_name]} ({final_p}%): {final_mean:.4f} ± {final_std:.4f}")


def main():
    """主函數"""
    # ============ 設置基礎方法的JSON路徑 ============
    base_methods = {
        'random_init': './exp_results/classification_hard/cold_start_random/random42_bs16.json',
        'imagenet': './exp_results/classification_hard/cold_start_imagenet/random42_bs16.json',
        'simclr': './exp_results/classification_hard/cold_start_simclr/random42_bs16.json'
    }
    
    # ============ 設置 Active Learning 方法的JSON路徑 ============
    al_methods = {
        'confidence': './exp_results/classification_hard/AL_simclr/conf_seed42_bs16.json',
        'entropy': './exp_results/classification_hard/AL_simclr/entropy_seed42_bs16.json',
        'margin': './exp_results/classification_hard/AL_simclr/margin_seed42_bs16.json',
        'coreset': './exp_results/classification_hard/AL_simclr/coreset_seed42_bs16.json',
        'badge': './exp_results/classification_hard/AL_simclr/badge_seed42_bs16.json'
    }
    
    # ============ 設置參數 ============
    aug_key = 'aug4'
    lr = 'best'
    portion_range = (5, 60.0)
    output_path = './classification/exp/AL/main.png'
    x_margin = 2.0
    show_100_baseline = True
    
    # ============ 繪製圖表 ============
    plot_with_active_learning(
        base_methods=base_methods,
        al_methods=al_methods,
        aug_key=aug_key,
        lr=lr,
        portion_range=portion_range,
        output_path=output_path,
        x_margin=x_margin,
        show_100_baseline=show_100_baseline
    )


if __name__ == "__main__":
    main()