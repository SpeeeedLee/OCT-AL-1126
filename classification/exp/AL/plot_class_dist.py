#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from collections import Counter


def load_dataset_labels(data_dir):
    """
    載入數據集並獲取所有樣本的標籤
    
    Args:
        data_dir: 數據目錄路徑
    
    Returns:
        labels: list, 所有樣本的標籤
        class_names: list, 類別名稱
    """
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    labels = train_dataset.targets
    class_names = train_dataset.classes
    
    print(f"Total training samples: {len(labels)}")
    print(f"Classes: {class_names}")
    print(f"Class distribution: {Counter(labels)}")
    
    return labels, class_names


def extract_labeled_indices(json_path, aug_key='aug4', portion=25.0, lr='best'):
    """
    從 JSON 文件中提取指定 portion 的 labeled_idx
    
    Args:
        json_path: JSON 文件路徑
        aug_key: augmentation key
        portion: 數據比例
        lr: learning rate (如果是 'best' 則選擇最佳的)
    
    Returns:
        labeled_idx_list: list of lists, 每次實驗的 labeled indices
    """
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
    """
    計算給定 indices 的類別分布
    
    Args:
        labeled_indices: list, 被選中的樣本索引
        all_labels: list, 所有樣本的標籤
        num_classes: int, 類別數量
    
    Returns:
        distribution: np.array, 每個類別的樣本數量
    """
    distribution = np.zeros(num_classes, dtype=int)
    
    for idx in labeled_indices:
        label = all_labels[idx]
        distribution[label] += 1
    
    return distribution


def generate_random_sampling(num_total, portion, num_runs, all_labels, num_classes, seed=42):
    """
    生成隨機採樣的類別分布
    
    Args:
        num_total: int, 總樣本數
        portion: float, 採樣比例 (%)
        num_runs: int, 採樣次數
        all_labels: list, 所有樣本的標籤
        num_classes: int, 類別數量
        seed: int, 隨機種子
    
    Returns:
        distributions: np.array, shape (num_runs, num_classes)
    """
    target_num = round(num_total * portion / 100)
    distributions = []
    
    random.seed(seed)
    
    for run in range(num_runs):
        # 隨機採樣
        indices = random.sample(range(num_total), target_num)
        distribution = compute_class_distribution(indices, all_labels, num_classes)
        distributions.append(distribution)
    
    distributions = np.array(distributions)
    print(f"Random sampling: {num_runs} runs, {target_num} samples each")
    
    return distributions


def plot_class_distribution_comparison(data_dict, class_names, output_path, 
                                       figsize=(14, 8), title=None):
    """
    繪製多個方法的類別分布比較圖
    
    Args:
        data_dict: dict, {method_name: distributions_array (shape: num_runs x num_classes)}
        class_names: list, 類別名稱
        output_path: str, 輸出圖片路徑
        figsize: tuple, 圖片大小
        title: str, 圖表標題
    """
    # 類別名稱簡化映射
    name_mapping = {
        'Seborrhoeic keratosis': 'SK',
        'Solar lentigo': 'Lentigo'
    }
    
    # 應用名稱映射
    display_names = [name_mapping.get(name, name) for name in class_names]
    
    num_classes = len(class_names)
    num_methods = len(data_dict)
    
    # 計算 mean 和 std
    stats = {}
    for method_name, distributions in data_dict.items():
        mean = np.mean(distributions, axis=0)
        std = np.std(distributions, axis=0, ddof=1) if len(distributions) > 1 else np.zeros(num_classes)
        stats[method_name] = {'mean': mean, 'std': std}
    
    # 計算所有方法的平均值，用於排序
    all_means = np.array([stats[method]['mean'] for method in data_dict.keys()])
    overall_mean = np.mean(all_means, axis=0)
    
    # 按照平均樣本數從多到少排序
    sorted_indices = np.argsort(overall_mean)[::-1]  # 降序排列
    sorted_class_names = [display_names[i] for i in sorted_indices]
    
    print(f"\nClass order (by average sample count):")
    for i, idx in enumerate(sorted_indices):
        print(f"  {i+1}. {display_names[idx]}: {overall_mean[idx]:.1f}")
    
    # 設置圖表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 設置 bar 的位置
    x = np.arange(num_classes)
    width = 0.8 / num_methods  # bar 寬度
    
    # 方法名稱映射（用於 legend）
    method_label_mapping = {
        'Random': 'Random Sampling',
        'Margin': 'Uncertainty AL',
        'Coreset': 'Diversity AL',
        'Badge': 'Hybrid AL',
        'Confidence': 'Uncertainty AL',
        'Entropy': 'Uncertainty AL'
    }
    
    # 顏色配置
    color_mapping = {
        'Random': '#808080',      # 灰色（避免和藍色太像）
        'Margin': '#1E3799',      # 深藍 - Uncertainty
        'Confidence': '#2E86DE',  # 亮藍 - Uncertainty
        'Entropy': '#54A0FF',     # 天藍 - Uncertainty
        'Coreset': '#EE5A24',     # 橙紅 - Diversity
        'Badge': '#9C27B0',       # 紫色 - Hybrid
    }
    
    # 繪製每個方法的 bars（使用排序後的順序）
    method_names = list(data_dict.keys())
    for i, method_name in enumerate(method_names):
        offset = (i - num_methods/2 + 0.5) * width
        mean = stats[method_name]['mean'][sorted_indices]  # 使用排序後的索引
        std = stats[method_name]['std'][sorted_indices]    # 使用排序後的索引
        
        # 獲取顯示標籤和顏色
        display_label = method_label_mapping.get(method_name, method_name)
        color = color_mapping.get(method_name, '#666666')
        
        ax.bar(x + offset, mean, width, 
               label=display_label,
               color=color,
               yerr=std,
               capsize=5,
               alpha=0.8)
    
    # 設置標籤和標題
    # ax.set_xlabel('Class', fontsize=24, labelpad=10)
    ax.set_ylabel('Number of Selected Samples', fontsize=24, labelpad=10)
    if title:
        ax.set_title(title, fontsize=28, pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_class_names, fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=20)
    
    # 添加網格
    ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_axisbelow(True)
    
    # 添加圖例
    ax.legend(fontsize=18, loc='best', framealpha=0.95)
    
    # 調整佈局
    fig.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n{'=' * 80}")
    print(f"Saved figure to {output_path}")
    print(f"{'=' * 80}")
    
    # 打印統計信息（按排序後的順序）
    print("\n=== Class Distribution Statistics (sorted by sample count) ===\n")
    for method_name in method_names:
        print(f"[{method_name}]")
        mean = stats[method_name]['mean']
        std = stats[method_name]['std']
        for idx in sorted_indices:
            display_name = display_names[idx]
            print(f"  {display_name:15s}: {mean[idx]:6.1f} ± {std[idx]:5.1f}")
        print()


def analyze_al_selection(data_dir, al_methods, portion=25.0, aug_key='aug4', 
                        lr='best', output_path='./plot/al_class_distribution.png',
                        include_random=True, num_random_runs=5, random_seed=42):
    """
    分析 Active Learning 方法的樣本選擇類別分布
    
    Args:
        data_dir: str, 數據目錄
        al_methods: dict, {method_name: json_path}
        portion: float, 數據比例 (%)
        aug_key: str, augmentation key
        lr: str, learning rate
        output_path: str, 輸出圖片路徑
        include_random: bool, 是否包含隨機採樣作為對比
        num_random_runs: int, 隨機採樣次數
        random_seed: int, 隨機種子
    """
    print("=" * 80)
    print("Analyzing Active Learning Sample Selection")
    print(f"Portion: {portion}%")
    print(f"Aug key: {aug_key}")
    print(f"Learning rate: {lr}")
    print("=" * 80)
    
    # 載入數據集標籤
    print("\n[Loading Dataset]")
    all_labels, class_names = load_dataset_labels(data_dir)
    num_classes = len(class_names)
    num_total = len(all_labels)
    
    # 收集所有方法的分布數據
    data_dict = {}
    
    # 隨機採樣
    if include_random:
        print("\n[Random Sampling]")
        random_distributions = generate_random_sampling(
            num_total, portion, num_random_runs, all_labels, num_classes, random_seed
        )
        data_dict['Random'] = random_distributions
    
    # Active Learning 方法
    print("\n[Active Learning Methods]")
    for method_name, json_path in al_methods.items():
        print(f"\n[{method_name}]")
        labeled_idx_list = extract_labeled_indices(json_path, aug_key, portion, lr)
        
        if not labeled_idx_list:
            print(f"Warning: No data found for {method_name}, skipping...")
            continue
        
        # 計算每次運行的類別分布
        distributions = []
        for run_idx, labeled_indices in enumerate(labeled_idx_list):
            distribution = compute_class_distribution(labeled_indices, all_labels, num_classes)
            distributions.append(distribution)
            print(f"  Run {run_idx + 1}: {len(labeled_indices)} samples, distribution: {distribution}")
        
        distributions = np.array(distributions)
        data_dict[method_name] = distributions
    
    # 繪製比較圖
    if data_dict:
        title = f"Class Distribution at {portion}% Labeled Training Data"
        plot_class_distribution_comparison(
            data_dict, class_names, output_path, 
            figsize=(14, 8), title=title
        )
    else:
        print("\nWarning: No data to plot!")


def main():
    """主函數"""
    # ============ 設置數據目錄 ============
    data_dir = './ds/classification/seven_class'
    
    # ============ 設置 Active Learning 方法的JSON路徑 ============
    al_methods = {
        # 'Confidence': './exp_results/classification_hard/AL_simclr/conf_seed42_bs16.json',
        # 'Entropy': './exp_results/classification_hard/AL_simclr/entropy_seed42_bs16.json',
        'Margin': './exp_results/classification_hard/AL_simclr/margin_seed42_bs16.json',
        'Coreset': './exp_results/classification_hard/AL_simclr/coreset_seed42_bs16.json',
        'Badge': './exp_results/classification_hard/AL_simclr/badge_seed42_bs16.json'
    }
    
    # ============ 設置參數 ============
    portion = 25.0  # 默認看 25% 的情況
    # portion = 30.0  # 默認看 25% 的情況
    aug_key = 'aug4'
    lr = 'best'
    output_path = './classification/exp/AL/class_distribution.png'
    include_random = True
    num_random_runs = 5
    random_seed = 42
    
    # ============ 分析並繪圖 ============
    analyze_al_selection(
        data_dir=data_dir,
        al_methods=al_methods,
        portion=portion,
        aug_key=aug_key,
        lr=lr,
        output_path=output_path,
        include_random=include_random,
        num_random_runs=num_random_runs,
        random_seed=random_seed
    )


if __name__ == "__main__":
    main()