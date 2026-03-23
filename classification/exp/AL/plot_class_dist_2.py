#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from collections import Counter

# ==========================================
# 1. 基礎數據載入與處理函數
# ==========================================

def load_dataset_labels(data_dir):
    """載入數據集並獲取所有樣本的標籤"""
    dataset_path = os.path.join(data_dir, 'train')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return [], []

    train_dataset = datasets.ImageFolder(dataset_path)
    labels = train_dataset.targets
    class_names = train_dataset.classes
    
    print(f"Total training samples: {len(labels)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution: {Counter(labels)}")
    
    return labels, class_names


def extract_labeled_indices(json_path, aug_key='aug4', portion=25.0, lr='best'):
    """從 JSON 文件中提取指定 portion 的 labeled_idx"""
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist!")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if aug_key not in data:
        print(f"Warning: aug_key '{aug_key}' not found in {json_path}")
        return []
    
    portion_key = str(float(portion))
    if portion_key not in data[aug_key]:
        print(f"Warning: portion {portion_key} not found in {json_path}")
        return []
    
    lr_dict = data[aug_key][portion_key]
    
    # 如果 lr='best'，選擇 acc 最高的 lr
    if lr == 'best':
        best_lr = None
        best_mean_acc = -1
        
        for lr_key, values in lr_dict.items():
            if isinstance(values, dict) and 'acc' in values:
                acc_list = values['acc']
                if isinstance(acc_list, list):
                    mean_acc = np.mean(acc_list)
                else:
                    mean_acc = float(acc_list)
                
                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    best_lr = lr_key
        
        if best_lr is None:
            print(f"Warning: No valid lr found for portion {portion_key}")
            return []
        
        lr = best_lr
        print(f"Selected best lr: {lr} (mean acc: {best_mean_acc:.4f})")
    
    # 提取 labeled_idx
    if lr not in lr_dict:
        print(f"Warning: lr '{lr}' not found for portion {portion_key}")
        return []
    
    values = lr_dict[lr]
    
    if isinstance(values, dict) and 'labeled_idx' in values:
        labeled_idx_list = values['labeled_idx']
        print(f"Found {len(labeled_idx_list)} runs with labeled indices")
        return labeled_idx_list
    else:
        print(f"Warning: 'labeled_idx' not found in {json_path}")
        return []


def compute_class_distribution(labeled_indices, all_labels, num_classes):
    """計算給定 indices 的類別分布"""
    distribution = np.zeros(num_classes, dtype=int)
    for idx in labeled_indices:
        label = all_labels[idx]
        distribution[label] += 1
    return distribution


def generate_random_sampling(num_total, portion, num_runs, all_labels, num_classes, seed=42):
    """生成隨機採樣的類別分布"""
    target_num = round(num_total * portion / 100)
    distributions = []
    
    random.seed(seed)
    
    for run in range(num_runs):
        indices = random.sample(range(num_total), target_num)
        distribution = compute_class_distribution(indices, all_labels, num_classes)
        distributions.append(distribution)
    
    distributions = np.array(distributions)
    print(f"Random sampling: {num_runs} runs, {target_num} samples each")
    
    return distributions


# ==========================================
# 2. 核心繪圖函數 (百分比差異 + 指定 Legend)
# ==========================================

def plot_class_distribution_difference_pct(data_dict, class_names, output_path, 
                                     figsize=(14, 8), title=None):
    """
    繪製相對於 Random 的百分比差異圖 ((Method - Random)/Random * 100%)
    並使用特定的 Legend (Uncertainty, Diversity, Hybrid)
    """
    if 'Random' not in data_dict:
        print("Error: 'Random' data is required for difference plotting.")
        return

    # 類別名稱簡化映射
    name_mapping = {
        'Seborrhoeic keratosis': 'SK',
        'Solar lentigo': 'Lentigo'
    }
    display_names = [name_mapping.get(name, name) for name in class_names]
    num_classes = len(class_names)
    
    # 1. 計算 Random 的平均值作為基準
    random_distributions = data_dict['Random']
    random_mean = np.mean(random_distributions, axis=0)
    
    # 避免除以 0 (使用極小值替代 0)
    safe_random_mean = np.where(random_mean == 0, 1e-9, random_mean)
    
    # 2. 準備其他方法的統計數據 (計算百分比差異)
    methods_to_plot = [m for m in data_dict.keys() if m != 'Random']
    
    # 定義固定的排序順序 (如果存在的話)：Margin -> Coreset -> Badge
    desired_order = ['Margin', 'Coreset', 'Badge']
    methods_to_plot = sorted(methods_to_plot, key=lambda x: desired_order.index(x) if x in desired_order else 999)
    num_methods_to_plot = len(methods_to_plot)
    
    stats = {}
    for method_name in methods_to_plot:
        distributions = data_dict[method_name]
        
        raw_mean = np.mean(distributions, axis=0)
        # 原始標準差
        raw_std = np.std(distributions, axis=0, ddof=1) if len(distributions) > 1 else np.zeros(num_classes)
        
        # === 計算百分比差異 ===
        # Formula: (Method - Random) / Random * 100
        pct_diff_mean = ((raw_mean - random_mean) / safe_random_mean) * 100
        
        # === 將標準差轉換為百分比尺度 ===
        # Formula: raw_std / Random * 100
        pct_std = (raw_std / safe_random_mean) * 100
        
        stats[method_name] = {'diff': pct_diff_mean, 'std': pct_std}

    # 3. 排序邏輯：依照「Random 採樣數量的多寡」排序 (Majority -> Minority)
    sorted_indices = np.argsort(random_mean)[::-1]
    sorted_class_names = [display_names[i] for i in sorted_indices]
    
    print(f"\nClass order (by Random sample count - Majority to Minority):")
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {display_names[idx]}: Random Count={random_mean[idx]:.1f}")

    # 4. 開始繪圖
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(num_classes)
    width = 0.8 / max(num_methods_to_plot, 1)
    
    # === 設定 Legend 顯示名稱 ===
    method_label_mapping = {
        'Margin': 'Uncertainty',
        'Coreset': 'Diversity',
        'Badge': 'Hybrid'
    }
    
    # === 設定顏色 ===
    color_mapping = {
        'Margin': '#1E3799',      # 深藍 (Uncertainty)
        'Coreset': '#EE5A24',     # 橙紅 (Diversity)
        'Badge': '#9C27B0',       # 紫色 (Hybrid)
    }
    
    for i, method_name in enumerate(methods_to_plot):
        offset = (i - num_methods_to_plot/2 + 0.5) * width
        
        pct_diff = stats[method_name]['diff'][sorted_indices]
        pct_std = stats[method_name]['std'][sorted_indices]
        
        color = color_mapping.get(method_name, '#666666')
        display_label = method_label_mapping.get(method_name, method_name)
        
        ax.bar(x + offset, pct_diff, width, 
               label=display_label,
               color=color,
               yerr=pct_std,
               capsize=5,
               alpha=0.85,
               edgecolor='white',
               linewidth=0.5)

    # 添加 y=0 的基準線
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    
    # 設置標籤
    ax.set_ylabel('Relative Difference from Random (%)', fontsize=20, labelpad=10)
    
    if title:
        ax.set_title(title, fontsize=24, pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_class_names, fontsize=18, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=16)
    
    # 網格
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)
    
    # 圖例
    ax.legend(fontsize=18, loc='best', framealpha=0.95)
    
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    
    print(f"\n{'=' * 80}")
    print(f"Saved PERCENTAGE DIFFERENCE plot to {output_path}")
    print(f"{'=' * 80}")


# ==========================================
# 3. 主流程控制
# ==========================================

def analyze_al_selection(data_dir, al_methods, portion=25.0, aug_key='aug4', 
                        lr='best', output_path='./plot/al_diff_pct.png',
                        num_random_runs=5, random_seed=42):
    
    print("=" * 80)
    print("Analyzing Active Learning Sample Selection (PERCENTAGE DIFFERENCE MODE)")
    print("=" * 80)
    
    all_labels, class_names = load_dataset_labels(data_dir)
    if len(all_labels) == 0:
        return

    num_classes = len(class_names)
    num_total = len(all_labels)
    
    data_dict = {}
    
    # 1. Random Baseline
    print("\n[Random Sampling - Baseline]")
    random_distributions = generate_random_sampling(
        num_total, portion, num_random_runs, all_labels, num_classes, random_seed
    )
    data_dict['Random'] = random_distributions
    
    # 2. AL Methods
    print("\n[Active Learning Methods]")
    for method_name, json_path in al_methods.items():
        print(f"\n[{method_name}]")
        labeled_idx_list = extract_labeled_indices(json_path, aug_key, portion, lr)
        
        if not labeled_idx_list:
            continue
            
        distributions = []
        for labeled_indices in labeled_idx_list:
            distribution = compute_class_distribution(labeled_indices, all_labels, num_classes)
            distributions.append(distribution)
        
        data_dict[method_name] = np.array(distributions)
    
    # 3. Plotting
    if len(data_dict) > 1:
        title = f"AL Methods vs Random Selection (at {portion}% Labeled)"
        plot_class_distribution_difference_pct(
            data_dict, class_names, output_path, 
            figsize=(14, 8), title=title
        )
    else:
        print("\nWarning: Not enough data to plot difference.")


def main():
    # ============ 設置數據目錄 ============
    data_dir = './ds/classification/seven_class'
    
    # ============ 設置 Active Learning 方法 ============
    # 依照需求啟用 Margin, Coreset, Badge
    al_methods = {
        # 'Confidence': './classification/exp_results/classification_hard/AL_simclr/conf_seed42_bs16.json',
        # 'Entropy': './classification/exp_results/classification_hard/AL_simclr/entropy_seed42_bs16.json',
        'Margin': './classification/exp_results/classification_hard/AL_simclr/margin_seed42_bs16.json',
        'Coreset': './classification/exp_results/classification_hard/AL_simclr/coreset_seed42_bs16.json',
        'Badge': './classification/exp_results/classification_hard/AL_simclr/badge_seed42_bs16.json'
    }
    
    # ============ 設置參數 ============
    portion = 25.0
    aug_key = 'aug4'
    lr = 'best'
    
    # 輸出檔名
    output_path = './classification/exp/AL/class_distribution_pct_diff_3methods.png'
    
    num_random_runs = 5
    random_seed = 42
    
    analyze_al_selection(
        data_dir=data_dir,
        al_methods=al_methods,
        portion=portion,
        aug_key=aug_key,
        lr=lr,
        output_path=output_path,
        num_random_runs=num_random_runs,
        random_seed=random_seed
    )

if __name__ == "__main__":
    main()