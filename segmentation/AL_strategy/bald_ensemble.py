import os
import torch
import numpy as np
from tqdm import tqdm


def bald_ensemble_mean(models, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, 
                       width=384, height=512):
    """
    BALD Ensemble - Mean Strategy
    
    使用多個獨立訓練的模型進行 ensemble，計算整張影像的 BALD score。
    
    BALD Formula (Ensemble):
        I[y; θ | x] = H[E_θ[p(y|x,θ)]] - E_θ[H[p(y|x,θ)]]
        
    簡單說：
        BALD = Entropy(平均預測) - 平均(每個模型的Entropy)
        
    直觀理解：
        - 第一項高 = 平均起來不確定（總不確定性）
        - 第二項高 = 每個模型都不確定（偶然不確定性）
        - 相減 = 模型之間意見分歧（認識不確定性）
    
    Args:
        models: 訓練好的模型列表 [model1, model2, model3, ...]
        opath: 影像路徑
        gpath_cell: cell mask 路徑（保持介面一致）
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        bald_dict: 字典，包含 BALD scores 和其他資訊
    """
    from segmentation.utils.data import o_data
    
    n_models = len(models)
    print(f"\n{'='*80}")
    print(f"BALD ENSEMBLE - MEAN STRATEGY")
    print(f"{'='*80}")
    print(f"Number of models in ensemble: {n_models}")
    print(f"Processing {len(unlabel_data_idx)} unlabeled images...")
    print(f"{'='*80}\n")
    
    # Set all models to eval mode
    for model in models:
        model.eval()
        model.to(device)
    
    all_bald_scores = []
    all_h_mean_scores = []  # Total uncertainty
    all_mean_h_scores = []  # Aleatoric uncertainty
    epsilon = 1e-7
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="BALD Ensemble (Mean)"):
            # 載入單張影像
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            # Ensemble: 每個模型各預測一次
            probs_list = []
            for model in models:
                OUTPUT = model(INPUT)  # [1, 1, H, W]
                prob = OUTPUT.squeeze()  # [H, W]
                probs_list.append(prob.cpu())
            
            # Stack all predictions: [n_models, H, W]
            probs_stack = torch.stack(probs_list, dim=0)
            
            # === Step 1: 計算平均預測 ===
            mean_prob = probs_stack.mean(dim=0)  # [H, W]
            
            # === Step 2: 計算 H(E[p]) - 平均預測的 Entropy (Total Uncertainty) ===
            mean_prob_clamped = torch.clamp(mean_prob, epsilon, 1 - epsilon)
            H_mean = -(mean_prob_clamped * torch.log(mean_prob_clamped) + 
                      (1 - mean_prob_clamped) * torch.log(1 - mean_prob_clamped))
            H_mean = torch.nan_to_num(H_mean, nan=0.0)
            H_mean_score = H_mean.mean().item()
            
            # === Step 3: 計算 E[H(p)] - 每個模型 Entropy 的平均 (Aleatoric Uncertainty) ===
            H_individual_list = []
            for prob in probs_list:
                prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
                H_individual = -(prob_clamped * torch.log(prob_clamped) + 
                               (1 - prob_clamped) * torch.log(1 - prob_clamped))
                H_individual = torch.nan_to_num(H_individual, nan=0.0)
                H_individual_list.append(H_individual)
            
            H_individual_stack = torch.stack(H_individual_list, dim=0)  # [n_models, H, W]
            mean_H = H_individual_stack.mean(dim=0)  # [H, W]
            mean_H_score = mean_H.mean().item()
            
            # === Step 4: BALD Score = H(E[p]) - E[H(p)] (Epistemic Uncertainty) ===
            bald_map = H_mean - mean_H  # [H, W]
            bald_score = bald_map.mean().item()
            
            # 確保不是 NaN
            if np.isnan(bald_score):
                bald_score = 0.0
            if np.isnan(H_mean_score):
                H_mean_score = 0.0
            if np.isnan(mean_H_score):
                mean_H_score = 0.0
            
            all_bald_scores.append(bald_score)
            all_h_mean_scores.append(H_mean_score)
            all_mean_h_scores.append(mean_H_score)
    
    # 轉換為 numpy array
    all_bald_scores = np.array(all_bald_scores)
    all_h_mean_scores = np.array(all_h_mean_scores)
    all_mean_h_scores = np.array(all_mean_h_scores)
    
    # 選擇 BALD score 最高的樣本（認識不確定性最高 = 模型意見最分歧）
    sorted_indices = np.argsort(all_bald_scores)[::-1]  # 降序排列
    selected_indices = sorted_indices[:num_data_to_label]
    
    # 映射回原始檔名
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    # 創建詳細的資訊字典
    bald_dict = {
        "strategy": "bald_ensemble_mean",
        "n_models": n_models,
        "selected_files": to_label_data_idx,
        "scores": {}
    }
    
    for i in sorted_indices:
        img_name = unlabel_data_idx[i]
        bald_dict["scores"][img_name] = {
            "bald": float(all_bald_scores[i]),           # 認識不確定性
            "total_uncertainty": float(all_h_mean_scores[i]),  # 總不確定性
            "aleatoric": float(all_mean_h_scores[i])     # 偶然不確定性
        }
    
    print(f"\n{'='*80}")
    print(f"SELECTION SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Selected {len(to_label_data_idx)} samples with highest BALD scores")
    print(f"  BALD (Epistemic):  [{all_bald_scores[selected_indices].min():.6f}, {all_bald_scores[selected_indices].max():.6f}]")
    print(f"  Total Uncertainty: [{all_h_mean_scores[selected_indices].min():.6f}, {all_h_mean_scores[selected_indices].max():.6f}]")
    print(f"  Aleatoric:         [{all_mean_h_scores[selected_indices].min():.6f}, {all_mean_h_scores[selected_indices].max():.6f}]")
    print(f"{'='*80}\n")
    
    return to_label_data_idx, bald_dict


def bald_ensemble_nuclei(models, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, 
                         width=384, height=512):
    """
    BALD Ensemble - Nuclei-focused Strategy
    
    使用多個獨立訓練的模型進行 ensemble，只計算預測為 nuclei 區域的 BALD score。
    
    Args:
        models: 訓練好的模型列表
        opath: 影像路徑
        gpath_cell: cell mask 路徑
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 需要選擇的樣本數量
        device: 計算設備
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        bald_dict: 字典，包含 BALD scores 和其他資訊
    """
    from segmentation.utils.data import o_data
    
    n_models = len(models)
    print(f"\n{'='*80}")
    print(f"BALD ENSEMBLE - NUCLEI STRATEGY")
    print(f"{'='*80}")
    print(f"Number of models in ensemble: {n_models}")
    print(f"Processing {len(unlabel_data_idx)} unlabeled images...")
    print(f"{'='*80}\n")
    
    # Set all models to eval mode
    for model in models:
        model.eval()
        model.to(device)
    
    all_bald_scores = []
    epsilon = 1e-7
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="BALD Ensemble (Nuclei)"):
            # 載入單張影像
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            # Ensemble predictions
            probs_list = []
            for model in models:
                OUTPUT = model(INPUT)
                prob = OUTPUT.squeeze()
                probs_list.append(prob.cpu())
            
            probs_stack = torch.stack(probs_list, dim=0)
            mean_prob = probs_stack.mean(dim=0)
            
            # === 計算 H(E[p]) ===
            mean_prob_clamped = torch.clamp(mean_prob, epsilon, 1 - epsilon)
            H_mean = -(mean_prob_clamped * torch.log(mean_prob_clamped) + 
                      (1 - mean_prob_clamped) * torch.log(1 - mean_prob_clamped))
            H_mean = torch.nan_to_num(H_mean, nan=0.0)
            
            # === 計算 E[H(p)] ===
            H_individual_list = []
            for prob in probs_list:
                prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
                H_individual = -(prob_clamped * torch.log(prob_clamped) + 
                               (1 - prob_clamped) * torch.log(1 - prob_clamped))
                H_individual = torch.nan_to_num(H_individual, nan=0.0)
                H_individual_list.append(H_individual)
            
            H_individual_stack = torch.stack(H_individual_list, dim=0)
            mean_H = H_individual_stack.mean(dim=0)
            
            # === BALD Score ===
            bald_map = H_mean - mean_H
            
            # 找出預測為 nuclei 的區域（根據平均預測）
            nuclei_mask = mean_prob > 0.5
            
            # 計算 nuclei 區域的平均 BALD score
            if nuclei_mask.sum() > 0:
                bald_score = bald_map[nuclei_mask].mean().item()
            else:
                bald_score = bald_map.mean().item()
            
            if np.isnan(bald_score):
                bald_score = 0.0
            
            all_bald_scores.append(bald_score)
    
    all_bald_scores = np.array(all_bald_scores)
    sorted_indices = np.argsort(all_bald_scores)[::-1]
    selected_indices = sorted_indices[:num_data_to_label]
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    bald_dict = {
        "strategy": "bald_ensemble_nuclei",
        "n_models": n_models,
        "selected_files": to_label_data_idx,
        "scores": {unlabel_data_idx[i]: float(all_bald_scores[i]) for i in sorted_indices}
    }
    
    print(f"\n{'='*80}")
    print(f"✓ Selected {len(to_label_data_idx)} samples with highest nuclei BALD scores")
    print(f"  BALD score range: [{all_bald_scores[selected_indices].min():.6f}, {all_bald_scores[selected_indices].max():.6f}]")
    print(f"{'='*80}\n")
    
    return to_label_data_idx, bald_dict


def variance_ensemble(models, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, 
                      width=384, height=512):
    """
    Variance Ensemble Strategy (Simpler Alternative)
    
    直接計算多個模型預測的 variance。
    比 BALD 簡單，但也是有效的不確定性估計方法。
    
    Variance 高 = 模型意見分歧
    """
    from segmentation.utils.data import o_data
    
    n_models = len(models)
    print(f"\nVariance Ensemble with {n_models} models...")
    
    for model in models:
        model.eval()
        model.to(device)
    
    all_variances = []
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="Variance Ensemble"):
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            probs_list = []
            for model in models:
                OUTPUT = model(INPUT)
                prob = OUTPUT.squeeze()
                probs_list.append(prob.cpu())
            
            probs_stack = torch.stack(probs_list, dim=0)
            variance_map = probs_stack.var(dim=0)
            variance_score = variance_map.mean().item()
            
            if np.isnan(variance_score):
                variance_score = 0.0
            
            all_variances.append(variance_score)
    
    all_variances = np.array(all_variances)
    sorted_indices = np.argsort(all_variances)[::-1]
    selected_indices = sorted_indices[:num_data_to_label]
    to_label_data_idx = [unlabel_data_idx[i] for i in selected_indices]
    
    variance_dict = {
        "strategy": "variance_ensemble",
        "n_models": n_models,
        "scores": {unlabel_data_idx[i]: float(all_variances[i]) for i in sorted_indices}
    }
    
    print(f"✓ Selected {len(to_label_data_idx)} samples with highest variance")
    print(f"  Variance range: [{all_variances[selected_indices].min():.6f}, {all_variances[selected_indices].max():.6f}]")
    
    return to_label_data_idx, variance_dict