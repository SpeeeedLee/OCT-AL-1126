"""
10-fold cross-validation data splitting for the HF-vs-4x augmentation robustness
study (thesis Ch5).

設計（**全部在程式層面做，完全不動 ds/ 底下的資料夾結構**）：
  - 把 train/(2032) + val/(509) 兩個資料夾的所有影像 pool 成一個全域索引空間
    （共 2541 張），全域索引 0..2031 = train/、2032..2540 = val/；class→idx 由
    ImageFolder 自動對齊（兩資料夾類別相同）。
  - 比例 8:1:1 → 切成 **N_CHUNKS=10** 個 chunk（每個 ~254 張）。chunk 的定義刻意
    讓 **fold 1 = 原本的 split**（可與既有結果對齊）：
        chunk 0..7 = train/(2032) 依類別 stratify 切成 8 份
        chunk 8    = 原本的 val  （= utils/data.py 對 val/ 做 50/50 stratified split 的 val 半）
        chunk 9    = 原本的 test （= 同上的 test 半，random_state=42）
  - fold 是 **1-indexed（1..10）**；fold f 把 10-chunk 視窗循環右移 (f-1) 格：
        fold f: train = {c_{s}, ..., c_{s+7}}, val = c_{s+8}, test = c_{s+9}   (s=f-1, 全部 mod 10)
    → fold 1 (s=0): train={c0..c7}=train/ 全部, val=c8=原 val, test=c9=原 test ＝**原本 split**。
    跑前 5 folds = fold 1..5（往右 shift 0..4）。
  - labeled 子集（portion%）在該 fold 的 train pool 內、用 random seed 抽樣；
    HF 與 4x 在相同 (fold, seed, portion) 下會拿到**完全相同**的 labeled 子集（公平比較）。
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import Image

from utils.data import AugmentedDataset   # 重用既有的 flip-based 離線增強

N_CHUNKS = 10          # 8:1:1 → 10 個 chunk
N_TRAIN_CHUNKS = 8     # 每個 fold 用 8 個 chunk 當 train
N_TRAIN_FOLDER = 8     # 前 8 個 chunk 來自 train/ 資料夾（其餘 2 個 = 原 val/test）
CHUNK_SEED = 42        # 固定的 chunk 劃分種子（與 fold / sampling seed 無關）
VAL_TEST_SEED = 42     # 對齊 utils/data.py 的 val/test 50/50 split


class _PathListDataset(Dataset):
    """以 (path, label) 清單 + 索引 + transform 建立 dataset；PIL 載入後套 transform。"""
    def __init__(self, samples, indices, transform):
        self.samples = samples            # list[(path, class_idx)]
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.samples[self.indices[i]]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_pool(data_dir):
    """把 train/ + val/ 兩資料夾的 samples 串成單一全域 (path,label) 清單。"""
    tr = datasets.ImageFolder(os.path.join(data_dir, 'train'))
    va = datasets.ImageFolder(os.path.join(data_dir, 'val'))
    assert tr.class_to_idx == va.class_to_idx, \
        f"class mapping mismatch:\n train/={tr.class_to_idx}\n val/={va.class_to_idx}"
    samples = list(tr.samples) + list(va.samples)     # [(path, class_idx), ...]
    n_train_folder = len(tr.samples)                  # 全域索引 < n → train/；>= n → val/
    targets = np.array([s[1] for s in samples])
    return samples, targets, n_train_folder


def _chunk_assignment(targets, n_train_folder):
    """
    回傳 chunk_id[i] ∈ [0, N_CHUNKS)，確定性（固定種子）：
      chunk 0..7 = train/ 依類別 stratify 切 8 份；chunk 8/9 = 原 val/test（對齊 utils/data.py）。
    """
    chunk_id = np.full(len(targets), -1, dtype=int)

    # --- train/ → 8 個 chunk（全域索引就是 0..n_train_folder-1）---
    tr_targets = targets[:n_train_folder]
    skf = StratifiedKFold(n_splits=N_TRAIN_FOLDER, shuffle=True, random_state=CHUNK_SEED)
    for cid, (_, idx) in enumerate(skf.split(np.zeros(n_train_folder), tr_targets)):
        chunk_id[idx] = cid                          # idx ∈ [0, n_train_folder)

    # --- val/ → 原本的 val(=chunk8) / test(=chunk9)，完全對齊 utils/data.py ---
    va_targets = targets[n_train_folder:]
    local = np.arange(len(va_targets))
    val_local, test_local = train_test_split(
        local, test_size=0.5, stratify=va_targets, random_state=VAL_TEST_SEED
    )
    chunk_id[n_train_folder + val_local] = N_TRAIN_FOLDER        # 8
    chunk_id[n_train_folder + test_local] = N_TRAIN_FOLDER + 1   # 9

    assert (chunk_id >= 0).all(), "some samples were not assigned to a chunk"
    return chunk_id


def fold_split(targets, n_train_folder, fold):
    """回傳該 fold（1-indexed, 1..N_CHUNKS）的 (train_idx, val_idx, test_idx)（全域、已排序）。"""
    if not (1 <= fold <= N_CHUNKS):
        raise ValueError(f"fold must be in [1,{N_CHUNKS}]; got {fold}")
    chunk_id = _chunk_assignment(targets, n_train_folder)
    shift = fold - 1
    order = [(shift + k) % N_CHUNKS for k in range(N_CHUNKS)]    # 視窗循環右移 shift 格
    train_chunks = order[:N_TRAIN_CHUNKS]
    val_chunk = order[N_TRAIN_CHUNKS]
    test_chunk = order[N_TRAIN_CHUNKS + 1]
    train_idx = np.where(np.isin(chunk_id, train_chunks))[0]
    val_idx = np.where(chunk_id == val_chunk)[0]
    test_idx = np.where(chunk_id == test_chunk)[0]
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def get_data_cv(data_dir, fold, portion, seed, batch_size=16,
                data_aug=True, aug_factor=4, flip_type='horizontal'):
    """
    建出某個 fold（1-indexed）、某個 portion%、某個 sampling seed 的 train/val/test loaders。

    回傳: (data_loaders, dataset_sizes, train_pool_size, n_labeled)
    """
    samples, targets, n_train_folder = _build_pool(data_dir)
    train_pool_idx, val_idx, test_idx = fold_split(targets, n_train_folder, fold)

    # 在該 fold 的 train pool 內，用 seed 抽 portion% 當 labeled 子集
    target_num = round(len(train_pool_idx) * portion / 100)
    rng = random.Random(seed)                       # 不污染全域 random 狀態
    label_idx = rng.sample(list(train_pool_idx), target_num)

    # 與 utils/data.py 對齊：ToTensor + Normalize([0.5]*3)，無 online jitter
    norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    base_train = _PathListDataset(samples, label_idx, norm)
    if data_aug:
        train_dataset = AugmentedDataset(base_train, aug_factor=aug_factor, flip_type=flip_type)
    else:
        train_dataset = base_train
    val_dataset = _PathListDataset(samples, val_idx, norm)
    test_dataset = _PathListDataset(samples, test_idx, norm)

    data_loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4),
    }
    dataset_sizes = {p: len(data_loaders[p].dataset) for p in ['train', 'val', 'test']}
    print(f"[CV fold {fold}/{N_CHUNKS}] train_pool={len(train_pool_idx)} "
          f"val={len(val_idx)} test={len(test_idx)} | "
          f"labeled={len(label_idx)} ({portion}%) | sizes={dataset_sizes}")
    return data_loaders, dataset_sizes, len(train_pool_idx), len(label_idx)
