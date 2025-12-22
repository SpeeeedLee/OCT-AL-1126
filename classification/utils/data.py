import os
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import Counter

def get_num_train(data_dir):
    train_image_dataset = datasets.ImageFolder(data_dir + "/train")
    num_train = len(train_image_dataset)
    return num_train

def get_num_test(data_dir):
    test_image_dataset = datasets.ImageFolder(data_dir + "/val")
    num_test = len(test_image_dataset)
    return num_test


class AugmentedDataset(Dataset):
    """
    將數據集擴增 N 倍：
    - aug_factor=2: 原圖 + 翻轉（由 flip_type 指定）
    - aug_factor=3: 原圖 + 水平翻轉 + 垂直翻轉
    - aug_factor=4: 原圖 + 水平翻轉 + 垂直翻轉 + 水平+垂直翻轉
    """
    def __init__(self, base_dataset, aug_factor=4, flip_type='horizontal'):
        """
        Args:
            base_dataset: 基礎數據集
            aug_factor: 增強倍數 (2, 3, 或 4)
            flip_type: 當 aug_factor=2 時的翻轉類型
                      'horizontal' - 水平翻轉
                      'vertical' - 垂直翻轉
        """
        self.base_dataset = base_dataset
        self.base_size = len(base_dataset)
        self.aug_factor = aug_factor
        self.flip_type = flip_type
        
        if aug_factor not in [2, 3, 4]:
            raise ValueError("aug_factor must be 2, 3, or 4")
        
        if aug_factor == 2 and flip_type not in ['horizontal', 'vertical']:
            raise ValueError("flip_type must be 'horizontal' or 'vertical' when aug_factor=2")
        
        if aug_factor > 2 and flip_type != 'horizontal':
            print(f"Warning: flip_type='{flip_type}' is ignored when aug_factor={aug_factor}")
        
    def __len__(self):
        return self.base_size * self.aug_factor
    
    def __getitem__(self, idx):
        # 確定是哪個原始樣本和哪種增強方式
        base_idx = idx % self.base_size
        aug_type = idx // self.base_size
        
        # 獲取原始圖片和標籤
        img, label = self.base_dataset[base_idx]
        
        # 根據增強類型應用變換
        if aug_type == 0:
            # 原圖，不做變換
            pass
        elif aug_type == 1:
            if self.aug_factor == 2:
                # aug_factor=2 時，根據 flip_type 決定
                if self.flip_type == 'horizontal':
                    img = transforms.functional.hflip(img)
                elif self.flip_type == 'vertical':
                    img = transforms.functional.vflip(img)
            else:
                # aug_factor=3 或 4 時，第二種增強固定是水平翻轉
                img = transforms.functional.hflip(img)
        elif aug_type == 2 and self.aug_factor >= 3:
            # 垂直翻轉
            img = transforms.functional.vflip(img)
        elif aug_type == 3 and self.aug_factor == 4:
            # 水平+垂直翻轉
            img = transforms.functional.hflip(img)
            img = transforms.functional.vflip(img)
        
        return img, label


def get_data(data_dir, labeled_train_idx=None, batch_size=8, data_aug=True, aug_factor=4, flip_type='horizontal'):
    '''
    Split original val data to val and test!
    
    Args:
        data_dir: 數據目錄
        labeled_train_idx: 訓練集索引（用於半監督學習）
        batch_size: 訓練批次大小
        data_aug: 是否啟用數據增強 (default: True)
        aug_factor: 增強倍數 (default: 4)
            - 2: 原圖 + 翻轉（由 flip_type 指定）
            - 3: 原圖 + 水平翻轉 + 垂直翻轉
            - 4: 原圖 + 水平翻轉 + 垂直翻轉 + 水平+垂直翻轉
        flip_type: 當 aug_factor=2 時的翻轉類型 (default: 'horizontal')
            - 'horizontal': 水平翻轉
            - 'vertical': 垂直翻轉
    '''    
    data_transforms = {
        phase: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]) for phase in ['train', 'val']
    } ## This is important, to align simclr pretraining!

    # 讀取 train 和原始的 val dataset
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), 
        data_transforms['train']
    )
    original_val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), 
        data_transforms['val']
    )

    # 如果有指定 labeled_train_idx，使用 Subset
    if labeled_train_idx is not None:
        train_dataset = Subset(train_dataset, labeled_train_idx)

    # 如果啟用數據增強，應用指定倍數的增強
    if data_aug:
        if aug_factor == 2:
            print(f"Applying {aug_factor}x data augmentation to train set (原圖 + {flip_type} flip)...")
        elif aug_factor == 3:
            print(f"Applying {aug_factor}x data augmentation to train set (原圖 + 水平翻轉 + 垂直翻轉)...")
        elif aug_factor == 4:
            print(f"Applying {aug_factor}x data augmentation to train set (原圖 + 水平 + 垂直 + 水平+垂直)...")
        
        train_dataset = AugmentedDataset(train_dataset, aug_factor=aug_factor, flip_type=flip_type)

    # 將 val 按類別平分成 val 和 test
    targets = np.array(original_val_dataset.targets)
    indices = np.arange(len(targets))
    
    # stratify 確保每個類別都是 50-50 分割
    val_idx, test_idx = train_test_split(
        indices,
        test_size=0.5,
        stratify=targets,
        random_state=42  # 固定隨機種子確保可重現
    )
    
    # 創建 val 和 test 的 Subset
    val_dataset = Subset(original_val_dataset, val_idx)
    test_dataset = Subset(original_val_dataset, test_idx)
    
    # 組織成字典
    image_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }

    # 創建 DataLoader
    data_loaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        ),
        'val': torch.utils.data.DataLoader(
            val_dataset,
            batch_size=16, # 設回小一點才可以同時在同一個gpu跑多個
            shuffle=False,
            num_workers=4,
        ),
        'test': torch.utils.data.DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
        )
    }

    dataset_sizes = {p: len(image_datasets[p]) for p in ['train', 'val', 'test']}
    print("dataset sizes:", dataset_sizes)
    # # 獲取類別名稱
    # class_names = original_val_dataset.classes
    
    # # 打印 Train 集的類別分布
    # print("\nTrain 類別分布:")
    # if isinstance(train_dataset, AugmentedDataset):
    #     train_targets = [train_dataset.base_dataset[i][1] for i in range(train_dataset.base_size)]
    #     train_counter = Counter(train_targets)
    #     for cls in sorted(train_counter.keys()):
    #         print(f"  {class_names[cls]}: {train_counter[cls] * aug_factor} 張")
    # else:
    #     train_targets = [train_dataset[i][1] for i in range(len(train_dataset))]
    #     train_counter = Counter(train_targets)
    #     for cls in sorted(train_counter.keys()):
    #         print(f"  {class_names[cls]}: {train_counter[cls]} 張")
    
    # # 打印 Val 集的類別分布
    # print("\nValidation 類別分布:")
    # val_targets = [original_val_dataset.targets[i] for i in val_idx]
    # val_counter = Counter(val_targets)
    # for cls in sorted(val_counter.keys()):
    #     print(f"  {class_names[cls]}: {val_counter[cls]} 張")
    
    # # 打印 Test 集的類別分布
    # print("\nTest 類別分布:")
    # test_targets = [original_val_dataset.targets[i] for i in test_idx]
    # test_counter = Counter(test_targets)
    # for cls in sorted(test_counter.keys()):
    #     print(f"  {class_names[cls]}: {test_counter[cls]} 張")

    return data_loaders, dataset_sizes