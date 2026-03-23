import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


def mean_entropy_clustering(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, width=384, height=512):
    """
    Mean Entropy + Clustering Hybrid Strategy
    
    步驟：
    1. 用 mean entropy 選出 10x 最不確定的影像
    2. 提取這些影像的 U-Net embeddings (512維)
    3. 用 KMeans++ 做 clustering，分成 num_data_to_label 個 clusters
    4. 從每個 cluster 選出最接近 centroid 的影像
    
    Args:
        model: 訓練好的 U-Net 模型
        opath: 影像路徑
        gpath_cell: cell mask 路徑（保持介面一致）
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 最終需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        info_dict: 包含 entropy 和 embedding 資訊的字典
    """
    from segmentation.utils.data import o_data
    
    model.eval()
    model.to(device)
    
    # ===== Step 1: 用 mean entropy 選出 4x 候選 =====
    candidate_num = min(num_data_to_label * 4, len(unlabel_data_idx))
    print(f"\nStep 1: Computing mean entropy for {len(unlabel_data_idx)} unlabeled images...")
    print(f"Will select top {candidate_num} uncertain images (4x of final {num_data_to_label})")
    
    all_entropies = []
    epsilon = 1e-10
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="Mean Entropy"):
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            OUTPUT = model(INPUT)  # [1, 1, H, W]
            prob = OUTPUT.squeeze()  # [H, W]
            
            prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
            entropy = -(prob_clamped * torch.log(prob_clamped) + 
                       (1 - prob_clamped) * torch.log(1 - prob_clamped))
            
            mean_entropy_value = entropy.mean().item()
            all_entropies.append(mean_entropy_value)
    
    all_entropies = np.array(all_entropies)
    
    # 選擇 entropy 最高的 10x 樣本
    sorted_indices = np.argsort(all_entropies)[::-1]  # 降序
    candidate_indices = sorted_indices[:candidate_num]
    candidate_files = [unlabel_data_idx[i] for i in candidate_indices]
    
    print(f"✓ Selected {len(candidate_files)} candidates with highest entropy")
    print(f"  Entropy range: [{all_entropies[candidate_indices].min():.4f}, {all_entropies[candidate_indices].max():.4f}]")
    
    # ===== Step 2: 提取 U-Net embeddings (512維) =====
    print(f"\nStep 2: Extracting 512-dim embeddings from U-Net bottleneck...")
    
    embeddings = []
    
    # 需要 hook 來提取 bottleneck features
    bottleneck_features = []
    
    def hook_fn(module, input, output):
        # 提取 bottleneck 的 feature
        # 假設 bottleneck 是 [batch, 512, H/32, W/32]
        # 做 global average pooling 得到 [batch, 512]
        feat = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        feat = feat.view(feat.size(0), -1)  # [batch, 512]
        bottleneck_features.append(feat.cpu())
    
    # 註冊 hook 到 U-Net 的 bottleneck (最深層)
    # 根據你的 U-Net 結構調整，這裡假設是 model.Conv5
    # 如果結構不同，需要修改這裡
    try:
        hook_handle = model.Conv5.register_forward_hook(hook_fn)
    except AttributeError:
        print("Warning: Cannot find Conv5 in model, trying alternative...")
        # 嘗試其他可能的 bottleneck 位置
        try:
            hook_handle = model.Maxpool4.register_forward_hook(hook_fn)
        except:
            raise AttributeError("Cannot find bottleneck layer in U-Net model")
    
    with torch.no_grad():
        for img_name in tqdm(candidate_files, desc="Extracting Embeddings"):
            bottleneck_features.clear()
            
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            _ = model(INPUT)  # Forward pass，hook 會自動提取 features
            
            if len(bottleneck_features) > 0:
                embeddings.append(bottleneck_features[0].numpy())
    
    hook_handle.remove()  # 移除 hook
    
    embeddings = np.vstack(embeddings)  # [candidate_num, 512]
    print(f"✓ Extracted embeddings shape: {embeddings.shape}")
    
    # ===== Step 3: KMeans++ Clustering =====
    print(f"\nStep 3: Performing KMeans++ clustering into {num_data_to_label} clusters...")
    
    kmeans = KMeans(
        n_clusters=num_data_to_label,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_  # [num_data_to_label, 512]
    
    print(f"✓ Clustering completed")
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")
    
    # ===== Step 4: 從每個 cluster 選出最接近 centroid 的影像 =====
    print(f"\nStep 4: Selecting one sample from each cluster (closest to centroid)...")
    
    selected_indices = []
    
    for cluster_id in range(num_data_to_label):
        # 找出屬於這個 cluster 的所有樣本
        cluster_mask = (cluster_labels == cluster_id)
        cluster_embeddings = embeddings[cluster_mask]
        cluster_files_idx = np.where(cluster_mask)[0]
        
        # 計算與 centroid 的距離
        centroid = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # 選擇距離最近的
        closest_idx = cluster_files_idx[np.argmin(distances)]
        selected_indices.append(closest_idx)
    
    # 映射回原始檔名
    to_label_data_idx = [candidate_files[i] for i in selected_indices]
    
    print(f"✓ Selected {len(to_label_data_idx)} diverse samples")
    
    # 創建資訊字典
    info_dict = {
        "strategy": "mean_entropy_clustering",
        "candidate_num": candidate_num,
        "final_num": num_data_to_label,
        "selected_files": to_label_data_idx,
        "entropy_scores": {candidate_files[i]: float(all_entropies[candidate_indices[i]]) 
                          for i in selected_indices}
    }
    
    return to_label_data_idx, info_dict


def nuclei_entropy_clustering(model, opath, gpath_cell, unlabel_data_idx, num_data_to_label, device, width=384, height=512):
    """
    Nuclei Entropy + Clustering Hybrid Strategy
    
    步驟：
    1. 用 nuclei entropy 選出 10x 最不確定的影像
    2. 提取這些影像的 U-Net embeddings (512維)
    3. 用 KMeans++ 做 clustering，分成 num_data_to_label 個 clusters
    4. 從每個 cluster 選出最接近 centroid 的影像
    
    Args:
        model: 訓練好的 U-Net 模型
        opath: 影像路徑
        gpath_cell: cell mask 路徑（保持介面一致）
        unlabel_data_idx: 未標記資料的檔名列表
        num_data_to_label: 最終需要選擇的樣本數量
        device: 計算設備 (cuda/cpu)
        width: 影像寬度
        height: 影像高度
    
    Returns:
        to_label_data_idx: 選中樣本的檔名列表
        info_dict: 包含 entropy 和 embedding 資訊的字典
    """
    from segmentation.utils.data import o_data
    
    model.eval()
    model.to(device)
    
    # ===== Step 1: 用 nuclei entropy 選出 4x 候選 =====
    candidate_num = min(num_data_to_label * 4, len(unlabel_data_idx))
    print(f"\nStep 1: Computing nuclei-focused entropy for {len(unlabel_data_idx)} unlabeled images...")
    print(f"Will select top {candidate_num} uncertain images (4x of final {num_data_to_label})")
    
    all_entropies = []
    epsilon = 1e-10
    
    with torch.no_grad():
        for img_name in tqdm(unlabel_data_idx, desc="Nuclei Entropy"):
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            OUTPUT = model(INPUT)  # [1, 1, H, W]
            prob = OUTPUT.squeeze()  # [H, W]
            
            prob_clamped = torch.clamp(prob, epsilon, 1 - epsilon)
            entropy = -(prob_clamped * torch.log(prob_clamped) + 
                       (1 - prob_clamped) * torch.log(1 - prob_clamped))
            
            # 找出預測為 nuclei 的 pixels (prob > 0.5)
            nuclei_mask = prob > 0.5
            
            if nuclei_mask.sum() > 0:
                nuclei_entropy_value = entropy[nuclei_mask].mean().item()
            else:
                nuclei_entropy_value = entropy.mean().item()
            
            all_entropies.append(nuclei_entropy_value)
    
    all_entropies = np.array(all_entropies)
    
    # 選擇 entropy 最高的 10x 樣本
    sorted_indices = np.argsort(all_entropies)[::-1]  # 降序
    candidate_indices = sorted_indices[:candidate_num]
    candidate_files = [unlabel_data_idx[i] for i in candidate_indices]
    
    print(f"✓ Selected {len(candidate_files)} candidates with highest nuclei entropy")
    print(f"  Entropy range: [{all_entropies[candidate_indices].min():.4f}, {all_entropies[candidate_indices].max():.4f}]")
    
    # ===== Step 2: 提取 U-Net embeddings (512維) =====
    print(f"\nStep 2: Extracting 512-dim embeddings from U-Net bottleneck...")
    
    embeddings = []
    bottleneck_features = []
    
    def hook_fn(module, input, output):
        feat = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
        feat = feat.view(feat.size(0), -1)  # [batch, 512]
        bottleneck_features.append(feat.cpu())
    
    # 註冊 hook
    try:
        hook_handle = model.Conv5.register_forward_hook(hook_fn)
    except AttributeError:
        print("Warning: Cannot find Conv5 in model, trying alternative...")
        try:
            hook_handle = model.Maxpool4.register_forward_hook(hook_fn)
        except:
            raise AttributeError("Cannot find bottleneck layer in U-Net model")
    
    with torch.no_grad():
        for img_name in tqdm(candidate_files, desc="Extracting Embeddings"):
            bottleneck_features.clear()
            
            img_sub = o_data(opath, [img_name], width, height)
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            
            _ = model(INPUT)
            
            if len(bottleneck_features) > 0:
                embeddings.append(bottleneck_features[0].numpy())
    
    hook_handle.remove()
    
    embeddings = np.vstack(embeddings)  # [candidate_num, 512]
    print(f"✓ Extracted embeddings shape: {embeddings.shape}")
    
    # ===== Step 3: KMeans++ Clustering =====
    print(f"\nStep 3: Performing KMeans++ clustering into {num_data_to_label} clusters...")
    
    kmeans = KMeans(
        n_clusters=num_data_to_label,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_centers = kmeans.cluster_centers_
    
    print(f"✓ Clustering completed")
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")
    
    # ===== Step 4: 從每個 cluster 選出最接近 centroid 的影像 =====
    print(f"\nStep 4: Selecting one sample from each cluster (closest to centroid)...")
    
    selected_indices = []
    
    for cluster_id in range(num_data_to_label):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_embeddings = embeddings[cluster_mask]
        cluster_files_idx = np.where(cluster_mask)[0]
        
        centroid = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        closest_idx = cluster_files_idx[np.argmin(distances)]
        selected_indices.append(closest_idx)
    
    to_label_data_idx = [candidate_files[i] for i in selected_indices]
    
    print(f"✓ Selected {len(to_label_data_idx)} diverse samples")
    
    # 創建資訊字典
    info_dict = {
        "strategy": "nuclei_entropy_clustering",
        "candidate_num": candidate_num,
        "final_num": num_data_to_label,
        "selected_files": to_label_data_idx,
        "entropy_scores": {candidate_files[i]: float(all_entropies[candidate_indices[i]]) 
                          for i in selected_indices}
    }
    
    return to_label_data_idx, info_dict