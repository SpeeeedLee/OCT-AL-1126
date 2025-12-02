
import os
import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

def conf(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    # https://github.com/TooTouch/Active_Learning-Uncertainty_Sampling/blob/main/query_strategies/least_confidence.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 載入 full train dataset（保持原始排序）
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                        transform=transform)

    # 根據 unlabel_data_idx 建立子集 DataLoader
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=4)

    model.eval()
    model.to(device)

    all_probs = []

    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="inference for conf..."):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)          # [N_unlabeled, num_classes]
    max_conf = all_probs.max(dim=1).values           # [N_unlabeled]

    # ---- 新增：計算不確定度並組成 dict ------------------------------------
    uncertainty_scores = 1.0 - max_conf              # 信心愈低 → 不確定度愈高
    uncertainty_dict = {
        int(idx): float(uncertainty_scores[i])       # 轉成 Python float 方便 JSON
        for i, idx in enumerate(unlabel_data_idx)
    }
    # ---------------------------------------------------------------------

    # 取出最不確定的 num_data_to_label 筆（confidence 最小）
    sorted_indices = torch.argsort(max_conf)[:num_data_to_label]
    to_label_data_idx = [unlabel_data_idx[i] for i in sorted_indices]

    return to_label_data_idx, uncertainty_dict


def entropy(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    """
    Entropy Sampling 主动学习策略
    
    选择预测熵最大的样本进行标注。熵越大表示模型越不确定。
    
    参数:
        model: 已训练的分类模型
        data_dir: 数据目录路径
        unlabel_data_idx: 未标记数据的索引列表
        num_data_to_label: 需要选择的样本数量
        device: 计算设备 (cuda/cpu)
    
    返回:
        to_label_data_idx: 选中样本在全数据集中的索引列表
    
    熵计算公式:
        H(p) = -∑(p_i * log(p_i)) for i in classes
        其中 p_i 是第 i 个类别的预测概率
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 载入数据
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    model.to(device)
    
    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="inference for entropy..."):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [N_unlabeled, num_classes]
    
    # 计算每个样本的预测熵
    # H(p) = -∑(p_i * log(p_i))
    # 添加小的常数避免 log(0)
    epsilon = 1e-10
    log_probs = torch.log(all_probs + epsilon)
    entropy_values = -torch.sum(all_probs * log_probs, dim=1)  # [N_unlabeled]
    
    # 选择熵最大的样本（最不确定）
    sorted_indices = torch.argsort(entropy_values, descending=True)[:num_data_to_label]
    
    # 映射回原始数据集索引
    to_label_data_idx = [unlabel_data_idx[i] for i in sorted_indices]
    
    return to_label_data_idx


def margin(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    """
    Margin Sampling 主动学习策略
    
    选择前两个最高预测概率差距最小的样本。差距越小表示模型越不确定。
    
    参数同 entropy 函数
    
    返回:
        to_label_data_idx: 选中样本在全数据集中的索引列表
        margin_dict: 字典，key为样本在全数据集中的索引，value为对应的margin值
                    已按margin从小到大排序（从最不确定到最确定）
    
    Margin 计算公式:
        Margin = P(y_1|x) - P(y_2|x)
        其中 y_1, y_2 是概率最高的两个类别
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 载入数据
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    model.to(device)
    
    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="inference for margin..."):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [N_unlabeled, num_classes]
    
    # 计算每个样本的 margin
    # 获取前两个最高概率
    top2_probs = torch.topk(all_probs, 2, dim=1).values  # [N_unlabeled, 2]
    margins = top2_probs[:, 0] - top2_probs[:, 1]  # [N_unlabeled]
    
    # 按 margin 从小到大排序所有样本
    sorted_indices = torch.argsort(margins)
    
    # 创建排序后的 margin 字典（从最不确定到最确定）
    margin_dict = {}
    for i, sorted_idx in enumerate(sorted_indices):
        original_idx = unlabel_data_idx[sorted_idx]
        margin_value = margins[sorted_idx].item()
        margin_dict[original_idx] = margin_value
    
    # 选择 margin 最小的样本（最不确定）
    selected_indices = sorted_indices[:num_data_to_label]
    
    # 映射回原始数据集索引
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    return to_label_data_idx, margin_dict


def random_w_statistics(model, data_dir, unlabel_data_idx, num_data_to_label, device, random_seed=42):
    """
    隨機採樣主動學習策略 (帶統計功能)
    
    隨機選擇指定數量的樣本，但同時計算這些樣本的預測統計信息。
    
    參數:
        model: 訓練好的模型
        data_dir: 數據集目錄路徑
        unlabel_data_idx: 未標注樣本在全數據集中的索引列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        random_seed: 隨機種子，用於確保結果可重現 (預設: 42)
    
    返回:
        to_label_data_idx: 隨機選中樣本在全數據集中的索引列表
        margin_dict: 字典，key為選中樣本在全數據集中的索引，value為對應的margin值
        sorted_pairs: 列表，統計選中樣本的前兩個最高概率類別組合分布
                     格式：[("class_name1 and class_name2", count), ...]
                     按count從大到小排序
    """
    import os
    import random
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from tqdm import tqdm
    
    # 設置隨機種子
    random.seed(random_seed)
    
    # 隨機選擇樣本
    to_label_data_idx = random.sample(unlabel_data_idx, num_data_to_label)
    
    # 以下部分用於計算統計信息，不影響隨機選擇的結果
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 載入數據
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    
    # 只對選中的樣本進行預測計算統計信息
    selected_dataset = Subset(full_dataset, to_label_data_idx)
    selected_loader = DataLoader(selected_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 獲取類別名稱
    class_names = full_dataset.classes
    
    model.eval()
    model.to(device)
    
    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(selected_loader, desc="Computing statistics for selected samples..."):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [num_data_to_label, num_classes]
    
    # 計算每個選中樣本的 margin
    # 獲取前兩個最高概率及其對應的類別索引
    top2_values_indices = torch.topk(all_probs, 2, dim=1)
    top2_probs = top2_values_indices.values  # [num_data_to_label, 2]
    top2_indices = top2_values_indices.indices  # [num_data_to_label, 2]
    
    margins = top2_probs[:, 0] - top2_probs[:, 1]  # [num_data_to_label]
    
    # 創建 margin 字典
    margin_dict = {}
    for i, sample_idx in enumerate(to_label_data_idx):
        margin_value = margins[i].item()
        margin_dict[sample_idx] = margin_value
    
    # 統計選中樣本的前兩個最高概率類別組合
    class_pair_stats = {}
    for i in range(len(to_label_data_idx)):
        # 獲取該樣本的前兩個最高概率類別
        top1_class_idx = top2_indices[i, 0].item()
        top2_class_idx = top2_indices[i, 1].item()
        
        # 獲取類別名稱
        top1_class_name = class_names[top1_class_idx]
        top2_class_name = class_names[top2_class_idx]
        
        # 創建類別對的鍵（按字母順序排序確保一致性）
        sorted_classes = sorted([top1_class_name, top2_class_name])
        class_pair = f"{sorted_classes[0]} and {sorted_classes[1]}"
        
        # 統計
        if class_pair in class_pair_stats:
            class_pair_stats[class_pair] += 1
        else:
            class_pair_stats[class_pair] = 1
    
    # 按count從大到小排序class_pair_stats
    sorted_pairs = sorted(class_pair_stats.items(), key=lambda x: x[1], reverse=True)
    
    return to_label_data_idx, margin_dict, sorted_pairs

def margin_w_statistics(model, data_dir, unlabel_data_idx, num_data_to_label, device):
    """
    Margin Sampling 主动学习策略 (带统计功能)
    
    选择前两个最高预测概率差距最小的样本。差距越小表示模型越不确定。
    
    参数:
        model: 训练好的模型
        data_dir: 数据集目录路径
        unlabel_data_idx: 未标注样本在全数据集中的索引列表
        num_data_to_label: 需要选择的样本数量
        device: 计算设备 (cuda/cpu)
    
    返回:
        to_label_data_idx: 选中样本在全数据集中的索引列表
        margin_dict: 字典，key为样本在全数据集中的索引，value为对应的margin值
                    已按margin从小到大排序（从最不确定到最确定）
        sorted_pairs: 列表，统计选中样本的前两个最高概率类别组合分布
                     格式：[("class_name1 and class_name2", count), ...]
                     按count从大到小排序
    
    Margin 计算公式:
        Margin = P(y_1|x) - P(y_2|x)
        其中 y_1, y_2 是概率最高的两个类别
    """
    import os
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from tqdm import tqdm
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # 载入数据
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    unlabel_dataset = Subset(full_dataset, unlabel_data_idx)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 获取类别名称
    class_names = full_dataset.classes
    
    model.eval()
    model.to(device)
    
    all_probs = []
    with torch.no_grad():
        for inputs, _ in tqdm(unlabel_loader, desc="inference for margin..."):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [N_unlabeled, num_classes]
    
    # 计算每个样本的 margin
    # 获取前两个最高概率及其对应的类别索引
    top2_values_indices = torch.topk(all_probs, 2, dim=1)
    top2_probs = top2_values_indices.values  # [N_unlabeled, 2]
    top2_indices = top2_values_indices.indices  # [N_unlabeled, 2]
    
    margins = top2_probs[:, 0] - top2_probs[:, 1]  # [N_unlabeled]
    
    # 按 margin 从小到大排序所有样本
    sorted_indices = torch.argsort(margins)
    
    # 创建排序后的 margin 字典（从最不确定到最确定）
    margin_dict = {}
    for i, sorted_idx in enumerate(sorted_indices):
        original_idx = unlabel_data_idx[sorted_idx]
        margin_value = margins[sorted_idx].item()
        margin_dict[original_idx] = margin_value
    
    # 选择 margin 最小的样本（最不确定）
    selected_indices = sorted_indices[:num_data_to_label]
    
    # 映射回原始数据集索引
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    # 统计选中样本的前两个最高概率类别组合
    class_pair_stats = {}
    for selected_idx in selected_indices:
        # 获取该样本的前两个最高概率类别
        top1_class_idx = top2_indices[selected_idx, 0].item()
        top2_class_idx = top2_indices[selected_idx, 1].item()
        
        # 获取类别名称
        top1_class_name = class_names[top1_class_idx]
        top2_class_name = class_names[top2_class_idx]
        
        # 创建类别对的键（按字母顺序排序确保一致性）
        # 将两个类别名按字母顺序排序，避免重复组合
        sorted_classes = sorted([top1_class_name, top2_class_name])
        class_pair = f"{sorted_classes[0]} and {sorted_classes[1]}"
        
        # 统计
        if class_pair in class_pair_stats:
            class_pair_stats[class_pair] += 1
        else:
            class_pair_stats[class_pair] = 1
    
    # 按value大到小排序class_pair_stats，并转换为列表格式保持顺序
    sorted_pairs = sorted(class_pair_stats.items(), key=lambda x: x[1], reverse=True)
    
    return to_label_data_idx, margin_dict, sorted_pairs