import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def mean_entropy(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, width=384, height=512):
    """
    Mean Entropy Sampling for Semantic Segmentation
    
    計算整個影像的平均 entropy，選擇平均 entropy 最高的影像。
    Entropy 越高表示模型越不確定。
    
    Args:
        model: 訓練好的 U-Net 模型
        opath: 影像路徑
        gpath_cell: cell mask 路徑（這裡不使用，但保持介面一致）
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        entropy_dict: 字典，key為樣本檔名，value為對應的平均entropy值
    
    Entropy 計算公式 (Binary Segmentation):
        H(p) = -[p*log(p) + (1-p)*log(1-p)]
        其中 p 是預測為 nuclei 的機率 (sigmoid 輸出)
    """
    from segmentation.utils.data import o_data
    
    model.eval()
    model.to(device)
    
    all_entropies = []
    epsilon = 1e-10  # 避免 log(0)
    
    print(f"Computing mean entropy for {len(unlabel_data_idx)} unlabeled images...")
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="Mean Entropy"):
            # 載入單張影像
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            # 模型預測（已經過 sigmoid，輸出為機率）
            OUTPUT = model(INPUT)  # [1, 1, H, W]
            prob = OUTPUT.squeeze()  # [H, W]
            
            # 計算 binary entropy
            # H(p) = -[p*log(p) + (1-p)*log(1-p)]
            prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
            entropy = -(prob_clamped * torch.log(prob_clamped) + 
                       (1 - prob_clamped) * torch.log(1 - prob_clamped))
            
            # 計算整張影像的平均 entropy
            mean_entropy_value = entropy.mean().item()
            all_entropies.append(mean_entropy_value)
    
    # 轉換為 numpy array
    all_entropies = np.array(all_entropies)
    
    # 選擇平均 entropy 最高的樣本（最不確定）
    sorted_indices = np.argsort(all_entropies)[::-1]  # 降序排列
    selected_indices = sorted_indices[:num_data_to_label]
    
    # 映射回原始檔名
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    # 創建 entropy 字典（按 entropy 從高到低排序）
    entropy_dict = {}
    for i in sorted_indices:
        img_name = unlabel_data_idx[i]
        entropy_dict[img_name] = float(all_entropies[i])
    
    return to_label_data_idx, entropy_dict


def nuclei_entropy(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, width=384, height=512):
    """
    Nuclei-focused Entropy Sampling for Semantic Segmentation
    
    只計算預測為 nuclei 區域（sigmoid > 0.5）的平均 entropy。
    適合關注於模型對 nuclei 區域預測的不確定性。
    
    Args:
        model: 訓練好的 U-Net 模型
        opath: 影像路徑
        gpath_cell: cell mask 路徑（這裡不使用，但保持介面一致）
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        entropy_dict: 字典，key為樣本檔名，value為對應的 nuclei 區域平均entropy值
    
    Note:
        如果某張影像沒有預測為 nuclei 的 pixels (prob > 0.5)，
        則使用整張影像的平均 entropy 作為替代。
    """
    from segmentation.utils.data import o_data
    
    model.eval()
    model.to(device)
    
    all_entropies = []
    epsilon = 1e-10  # 避免 log(0)
    
    print(f"Computing nuclei-focused entropy for {len(unlabel_data_idx)} unlabeled images...")
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="Nuclei Entropy"):
            # 載入單張影像
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            # 模型預測（已經過 sigmoid，輸出為機率）
            OUTPUT = model(INPUT)  # [1, 1, H, W]
            prob = OUTPUT.squeeze()  # [H, W]
            
            # 計算 binary entropy
            prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
            entropy = -(prob_clamped * torch.log(prob_clamped) + 
                       (1 - prob_clamped) * torch.log(1 - prob_clamped))
            
            # 找出預測為 nuclei 的 pixels (prob > 0.5)
            nuclei_mask = prob > 0.5
            
            # 計算 nuclei 區域的平均 entropy
            if nuclei_mask.sum() > 0:
                # 只計算預測為 nuclei 區域的 entropy
                nuclei_entropy_value = entropy[nuclei_mask].mean().item()
            else:
                # 如果沒有 nuclei pixels，使用整張影像的平均 entropy
                nuclei_entropy_value = entropy.mean().item()
            
            all_entropies.append(nuclei_entropy_value)
    
    # 轉換為 numpy array
    all_entropies = np.array(all_entropies)
    
    # 選擇 nuclei entropy 最高的樣本（最不確定）
    sorted_indices = np.argsort(all_entropies)[::-1]  # 降序排列
    selected_indices = sorted_indices[:num_data_to_label]
    
    # 映射回原始檔名
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    # 創建 entropy 字典（按 entropy 從高到低排序）
    entropy_dict = {}
    for i in sorted_indices:
        img_name = unlabel_data_idx[i]
        entropy_dict[img_name] = float(all_entropies[i])
    
    return to_label_data_idx, entropy_dict
