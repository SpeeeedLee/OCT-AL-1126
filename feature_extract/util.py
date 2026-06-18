import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from feature_extract.extractor import ResNetSimCLR 

def get_latent_features(data_dir, train_idx, extractor, device, split='train', class_label=False):
    '''
    maybe need to finetune: https://github.com/facebookresearch/dinov2/issues/92
    '''
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
    ])
    if split=='train':
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                            transform=data_transform)
    elif split=='val':
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                            transform=data_transform)
    else:
        raise ValueError()
    subset = Subset(full_dataset, train_idx)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=4) # no shuffle! very important!

    print(f'load extractor model: {extractor}')
    if extractor in ['resnet18_pretrained', 'resnet18_simclr']:
        model = ResNetSimCLR('resnet18', 32).to(device)
        if extractor == 'resnet18_simclr':
            simclr_path = './SSL/simclr/resnet18_simclr_lr0.0002_bs128_ep100.pkl'
            print(f"Loading pretrained weights from {simclr_path}")
            state_dict = torch.load(simclr_path, map_location=device)    
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}):")
                for key in missing_keys:
                    print(f"  - {key}")
            if unexpected_keys:
                print(f"Unexpected keys ({len(unexpected_keys)}):")
                for key in unexpected_keys:
                    print(f"  - {key}")
        backbone = model.backbone               # standard ResNet
        modules = list(backbone.children())[:-1] # all layers up to avgpool
        feature_extractor = nn.Sequential(*modules)
    else:
        raise NotImplementedError(f"Extractor '{extractor}' not implemented.")
        
    feature_extractor.to(device).eval()
    print(feature_extractor)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    
    total_params = sum(p.numel() for p in feature_extractor.parameters())
    print(f"Total parameters: {total_params:,}")

    # 4) Inference & collect features
    feature_dict = {}
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc='Extracting features')):
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)            # [B, 512, 1, 1]
            feats = feats.view(feats.size(0), -1)      # [B, 512]
            feats = feats.cpu().numpy()
            labs = labels.cpu().numpy()

            start = batch_idx * loader.batch_size
            idxs = subset.indices[start : start + feats.shape[0]]
            for orig_idx, feat, lab in zip(idxs, feats, labs):
                if class_label:
                    feature_dict[orig_idx] = (feat, int(lab))
                else:
                    feature_dict[orig_idx] = feat


    # 4) Sanity check
    example_idx = train_idx[0]
    val = feature_dict[example_idx]
    if class_label:
        feat, lab = val
        print(f"Image {example_idx} feature shape:", feat.shape, "label:", lab)
    else:
        print(f"Image {example_idx} feature shape:", val.shape)
    return feature_dict



from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms.functional import to_pil_image
def get_latent_features_vit(data_dir, train_idx, extractor, device, split='train', class_label=False):
    # 1) Dataset with your original ToTensor()+Normalize
    data_transform = transforms.Compose([
        transforms.ToTensor(),                          # => [0,1]
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),  # => [-1,1]
    ])
    if split == 'train':
        full_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=data_transform
        )
    elif split == 'val':
        full_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'val'),
            transform=data_transform
        )
    else:
        raise ValueError(f"Invalid split '{split}'; expected 'train' or 'val'.")

    subset = Subset(full_dataset, train_idx)
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=4)

    # 2) Load model based on extractor type
    if extractor.startswith('dinov2'):
        # DINOv2 models
        if extractor == 'dinov2_base':
            model_name = 'facebook/dinov2-base'
        elif extractor == 'dinov2_small':
            model_name = 'facebook/dinov2-small'
        elif extractor == 'dinov2_large':
            model_name = 'facebook/dinov2-large'
        elif extractor == 'dinov2_giant':
            model_name = 'facebook/dinov2-giant'
        else:
            raise NotImplementedError(f"DINOv2 extractor '{extractor}' not implemented.")
        
        print(f"Loading DINOv2 extractor: {extractor}")
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        model_type = 'dinov2'
        
        # DINOv2 只有視覺編碼器，所以計算全部參數
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
    elif extractor.startswith('clip'):
        # CLIP models
        if extractor == 'clip_base_32':
            model_name = 'openai/clip-vit-base-patch32'
        elif extractor == 'clip_base_16':
            model_name = 'openai/clip-vit-base-patch16'
        elif extractor == 'clip_large_14':
            model_name = 'openai/clip-vit-large-patch14'
        elif extractor == 'clip_large_14_336':
            model_name = 'openai/clip-vit-large-patch14-336'
        else:
            raise NotImplementedError(f"CLIP extractor '{extractor}' not implemented.")
            
        print(f"Loading CLIP extractor: {extractor}")
        processor = AutoProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        model_type = 'clip'
        
        # 計算各部分參數量
        total_params = sum(p.numel() for p in model.parameters())
        vision_params = sum(p.numel() for p in model.vision_model.parameters())
        text_params = sum(p.numel() for p in model.text_model.parameters())
        projection_params = total_params - vision_params - text_params
        
        print(f"Total CLIP parameters: {total_params:,}")
        print(f"  - Vision Encoder: {vision_params:,}")
        print(f"  - Text Encoder: {text_params:,}")
        print(f"  - Projection layers: {projection_params:,}")
        
    else:
        raise NotImplementedError(f"Extractor '{extractor}' not supported. Use 'dinov2_*' or 'clip_*'.")

    # 3) Extract features
    feature_dict = {}
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc='Extracting features')):
            # a) undo Normalize -> back to [0,1]
            unnorm = (imgs + 1.0) / 2.0

            # b) convert each back to PIL.Image
            pil_imgs = [to_pil_image(img.cpu()) for img in unnorm]

            # c) processor + model -> embeddings
            if model_type == 'dinov2':
                inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim] - CLS token
            elif model_type == 'clip':
                inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
                outputs = model.get_image_features(**inputs)  # [B, hidden_dim] - image embeddings
                embeddings = outputs

            # d) collect into numpy + map to original indices
            feats = embeddings.cpu().numpy()
            labs = labels.cpu().numpy()
            start = batch_idx * loader.batch_size
            idxs = subset.indices[start : start + feats.shape[0]]
            for orig_idx, feat, lab in zip(idxs, feats, labs):
                if class_label:
                    feature_dict[orig_idx] = (feat, int(lab))
                else:
                    feature_dict[orig_idx] = feat

    # 4) Sanity check
    example_idx = train_idx[0]
    val = feature_dict[example_idx]
    if class_label:
        feat, lab = val
        print(f"Image {example_idx} feature shape:", feat.shape, "label:", lab)
    else:
        print(f"Image {example_idx} feature shape:", val.shape)

    return feature_dict


# 需要額外的 import
from transformers import CLIPModel, AutoProcessor



from sklearn.cluster import KMeans
import numpy as np

def k_means_centroid(
    feature_dict,
    k,
    random_state,
    feat_norm=None  # None, 'l2_cols', or 'zscore_cols'
):
    keys = list(feature_dict.keys())
    feats = np.stack([feature_dict[key] for key in keys], axis=0)  # (N, D)

    # —— feature-wise normalization/standardization —— #
    if feat_norm == 'l2_cols':
        # 对每一列做 ℓ2 归一化
        norms = np.linalg.norm(feats, axis=0, keepdims=True)  # (1, D)
        feats = feats / (norms + 1e-12)
    elif feat_norm == 'zscore_cols':
        # 对每一列做 zero-mean, unit-variance
        means = feats.mean(axis=0, keepdims=True)             # (1, D)
        stds  = feats.std(axis=0,  keepdims=True)             # (1, D)
        feats = (feats - means) / (stds + 1e-12)
    # —— end preprocessing —— #

    print(f'performing kmeans with k = {k} (feat_norm={feat_norm})...')
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
    kmeans.fit(feats)
    centers = kmeans.cluster_centers_  # (k, D)
    labels = kmeans.labels_            # (N,)

    print('find images nearest to centroid')
    nearest_keys = []
    for ci in range(k):
        member_idxs = np.where(labels == ci)[0]
        dists = np.linalg.norm(feats[member_idxs] - centers[ci], axis=1)
        best_idx = member_idxs[np.argmin(dists)]
        nearest_keys.append(keys[best_idx])

    return nearest_keys

from sklearn.metrics import pairwise_distances

def k_means_dense_center(
    feature_dict,
    k,
    n_neighbors=100,
    random_state=42,
    feat_norm=None  # None, 'l2_cols', or 'zscore_cols'
):
    keys = list(feature_dict.keys())
    feats = np.stack([feature_dict[key] for key in keys], axis=0)  # (N, D)

    # —— feature-wise normalization —— #
    if feat_norm == 'l2_cols':
        norms = np.linalg.norm(feats, axis=0, keepdims=True)
        feats = feats / (norms + 1e-12)
    elif feat_norm == 'zscore_cols':
        means = feats.mean(axis=0, keepdims=True)
        stds  = feats.std(axis=0, keepdims=True)
        feats = (feats - means) / (stds + 1e-12)

    print(f'performing kmeans with k = {k} (feat_norm={feat_norm})...')
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    kmeans.fit(feats)
    labels = kmeans.labels_  # (N,)

    print('find densest points per cluster...')
    dense_keys = []

    for ci in range(k):
        member_idxs = np.where(labels == ci)[0]
        member_feats = feats[member_idxs]  # shape (M, D)

        # 距離矩陣 (M, M)
        dists = pairwise_distances(member_feats)

        # 對每一行取最近 n_neighbors 的距離（不包括自己）
        sorted_dists = np.sort(dists, axis=1)  # 每列排序，第一個是 0（自己）
        avg_knn_dists = sorted_dists[:, 1:n_neighbors+1].mean(axis=1)  # (M,)

        # 取平均距離的倒數作為密度，越大越密
        density_scores = 1 / (avg_knn_dists + 1e-12)

        # 找到密度最大的點
        best_idx = member_idxs[np.argmax(density_scores)]
        dense_keys.append(keys[best_idx])

    return dense_keys

import numpy as np
from itertools import combinations
from sklearn.cluster import KMeans
from collections import defaultdict
def k_means_cluster_margin(feature_dict, k, m, random_state=42):
    """
    Args:
        feature_dict: dict[key -> 1D array of dim D]
        k: int, KMeans 的群数
        m: int, 要选出的最混淆的 cluster‑pair 数量
        random_state: 为了重复性

    Returns:
        selected_keys: list of length m，每个最混淆边界挑出的一个样本 key
        boundary_info: list of tuples ((ci, cj), avg_margin) 前 m 个 cluster‑pair 及其平均 margin
    """
    # 1) 准备数据矩阵 X
    keys = list(feature_dict.keys())
    X = np.stack([feature_dict[k] for k in keys], axis=0)  # (N, D)

    # 2) 做 KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state).fit(X)
    centers = kmeans.cluster_centers_  # (k, D)

    # 3) 计算每个点到所有中心的距离矩阵 D (N x k)
    D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)

    # 4) 取每行的最小和第二小
    sorted_idx = np.argsort(D, axis=1)       # (N, k) 排序后的中心索引
    nearest    = sorted_idx[:, 0]            # 最近中心的索引
    second     = sorted_idx[:, 1]            # 第二近中心的索引
    d1 = D[np.arange(X.shape[0]), nearest]   # 到最近中心的距离
    d2 = D[np.arange(X.shape[0]), second]    # 到第二近中心的距离

    # 5) margin = d2 - d1
    margins = d2 - d1

    # 6) 按 (最近, 第二近) 的 pair 分组，累积 margins 和样本 idx
    pair_margins = defaultdict(list)
    pair_indices = defaultdict(list)
    for i, (c1, c2) in enumerate(zip(nearest, second)):
        pair = tuple(sorted((int(c1), int(c2))))
        pair_margins[pair].append(margins[i])
        pair_indices[pair].append(i)

    # 7) 计算每个 pair 的平均 margin
    pair_avg = {pair: np.mean(m_list) for pair, m_list in pair_margins.items()}

    # 8) 找出平均 margin 最小的前 m 个 pair
    selected_pairs = sorted(pair_avg.items(), key=lambda x: x[1])[:m]

    # 9) 对每个选中的 pair，从它那组 samples 中挑一个 margin 最小的
    selected_keys = []
    boundary_info = []
    for pair, avg_m in selected_pairs:
        idxs = pair_indices[pair]
        local_margins = np.array([margins[i] for i in idxs])
        best_local_idx = idxs[np.argmin(local_margins)]
        selected_keys.append(keys[best_local_idx])
        boundary_info.append((pair, avg_m))

    return selected_keys, boundary_info


# def k_means_centroid(feature_dict, k, random_state=42):
#     keys = list(feature_dict.keys())
#     feats = np.stack([feature_dict[key] for key in keys], axis=0)

#     print(f'performing kmeans with k = {k}...')
#     kmeans = KMeans(n_clusters=k, random_state=random_state)
#     kmeans.fit(feats)
#     centers = kmeans.cluster_centers_       # shape = (k, D)
#     labels = kmeans.labels_                 # shape = (N,)

#     print('find images nearest to centoid')
#     nearest_keys = []
#     for ci in range(k):
#         member_idxs = np.where(labels == ci)[0]
#         dists = np.linalg.norm(feats[member_idxs] - centers[ci], axis=1)
#         best_idx = member_idxs[np.argmin(dists)]
#         nearest_keys.append(keys[best_idx])

#     return nearest_keys

import numpy as np
from sklearn.cluster import KMeans

import numpy as np
from sklearn.cluster import KMeans

def hierarchical_k_means_centroid(feature_dict, k, B, random_state=42):
    """
    Args:
        feature_dict: dict[key -> 1D numpy array of dim D]
        k: int, 初始做几群
        B: int, 最终想选多少个样本
        random_state: int, for reproducibility

    Returns:
        selected_keys: list of length B，对应被选样本在 feature_dict 中的 key
    """
    keys = list(feature_dict.keys())
    X = np.stack([feature_dict[k] for k in keys], axis=0)  # (N, D)
    N = X.shape[0]
    if B > N:
        raise ValueError(f"B={B} 超过样本总数 N={N}")

    # —— Step 1: 初始 KMeans(k) —— #
    km = KMeans(n_clusters=k, random_state=random_state).fit(X)
    labels = km.labels_
    centers = km.cluster_centers_  # (k, D)

    # —— Step 2: 计算每个 cluster 应分配的 B_i —— #
    # 基本份额
    base = B // k
    # 剩余要分配的
    extra = B - base * k
    # 按 cluster 大小排序，把 extra 分给最大的那几群
    cluster_sizes = np.bincount(labels, minlength=k)
    order = np.argsort(cluster_sizes)[::-1]  # 从大到小的 cluster 索引
    B_i = np.full(k, base, dtype=int)
    B_i[order[:extra]] += 1  # 给最大的 extra 个 cluster 多分 1

    # —— Step 3: 对每个 cluster 做子 KMeans(B_i[i]) —— #
    selected_keys = []
    for ci in range(k):
        idxs = np.where(labels == ci)[0]
        sub_X = X[idxs]
        bi = B_i[ci]

        if bi == 1:
            # 直接选 initial centroid 最近的点
            center = centers[ci]
            local_idx = idxs[np.argmin(np.linalg.norm(sub_X - center, axis=1))]
            selected_keys.append(keys[local_idx])
        else:
            # 在这个 cluster 内再做 KMeans(bi)
            sub_km = KMeans(n_clusters=bi, random_state=random_state).fit(sub_X)
            sub_centers = sub_km.cluster_centers_
            sub_labels  = sub_km.labels_
            # 对每个子中心，选最近的点
            for sj in range(bi):
                sub_idxs = idxs[sub_labels == sj]
                sub_feats = sub_X[sub_labels == sj]
                c = sub_centers[sj]
                local_idx = sub_idxs[np.argmin(np.linalg.norm(sub_feats - c, axis=1))]
                selected_keys.append(keys[local_idx])

    # 最终长度应当是 sum(B_i) == B
    assert len(selected_keys) == B
    return selected_keys
