import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_num_train(data_dir):
    train_image_dataset = datasets.ImageFolder(data_dir + "/train")
    num_train = len(train_image_dataset)
    return num_train

def get_num_test(data_dir):
    test_image_dataset = datasets.ImageFolder(data_dir + "/val")
    num_test = len(test_image_dataset)
    return num_test


# def get_data(data_dir, labeled_train_idx=None, batch_size=8, classes=None):
#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ]),
#         'val': transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ]),
#     }

#     print("class names of train:", os.listdir(data_dir + "/train"))
#     print("class names of val:", os.listdir(data_dir + "/val"))

#     image_datasets = {
#         x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#         for x in ['train', 'val']
#     }

#     if labeled_train_idx is not None:
#         image_datasets['train'] = Subset(image_datasets['train'], labeled_train_idx)

#     data_loaders = {
#         x: torch.utils.data.DataLoader(
#             image_datasets[x],
#             batch_size=(batch_size if x == 'train' else 16),
#             shuffle=(x == 'train'),
#             num_workers=4
#         )
#         for x in ['train', 'val']
#     }


#     dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#     print("dataset sizes: ", dataset_sizes)

#     return data_loaders, dataset_sizes


# def get_data(data_dir, labeled_train_idx=None, batch_size=8, classes=None):
#     """
#     如果 classes 只含兩類，函式會：
#       1. 先用 wanted classes 過濾樣本 (Subset)
#       2. 再把原始 targets 重新貼標為 0 / 1
#          └ subclass0 → classes[0]；subclass1 → classes[1]

#     其餘行為與舊版一致。
#     """
#     # ----------------------- transforms -----------------------
#     data_transforms = {
#         phase: transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5]*3, [0.5]*3),
#         ]) for phase in ['train', 'val']
#     }

#     # ----------------------- load datasets -----------------------
#     image_datasets = {
#         phase: datasets.ImageFolder(os.path.join(data_dir, phase),
#                                     data_transforms[phase])
#         for phase in ['train', 'val']
#     }

#     # ----------------------- class filtering -----------------------
#     if classes is not None:
#         name2idx = image_datasets['train'].class_to_idx
#         wanted_idx = [name2idx[c] if isinstance(c, str) else int(c)
#                       for c in classes]

#         def _filter(ds, base=None):
#             pool = base if base is not None else range(len(ds))
#             return [i for i in pool if ds.targets[i] in wanted_idx]

#         train_pool = labeled_train_idx if labeled_train_idx is not None else None
#         train_idx  = _filter(image_datasets['train'], train_pool)
#         val_idx    = _filter(image_datasets['val'])

#         image_datasets['train'] = Subset(image_datasets['train'], train_idx)
#         image_datasets['val']   = Subset(image_datasets['val'],   val_idx)

#         # ---------- 重新貼標（僅當為二元分類） ----------
#         if len(wanted_idx) == 2:
#             orig_to_new = {wanted_idx[0]: 0, wanted_idx[1]: 1}

#             class RelabeledSubset(torch.utils.data.Dataset):
#                 def __init__(self, subset, mapping):
#                     self.subset, self.map = subset, mapping
#                 def __len__(self):
#                     return len(self.subset)
#                 def __getitem__(self, idx):
#                     x, y = self.subset[idx]
#                     return x, self.map[int(y)]

#             for split in ['train', 'val']:
#                 image_datasets[split] = RelabeledSubset(
#                     image_datasets[split], orig_to_new
#                 )
#             print(f"Relabeled to subclass0/1 → {classes[0]}→0, {classes[1]}→1")
#         else:
#             print("Multi-class (≠2) — keep original labels.")

#     elif labeled_train_idx is not None:
#         image_datasets['train'] = Subset(
#             image_datasets['train'], labeled_train_idx
#         )

#     # ----------------------- dataloaders -----------------------
#     data_loaders = {
#         phase: torch.utils.data.DataLoader(
#             image_datasets[phase],
#             batch_size=batch_size if phase == 'train' else 16,
#             shuffle=(phase == 'train'),
#             num_workers=4,
#         ) for phase in ['train', 'val']
#     }

#     dataset_sizes = {p: len(image_datasets[p]) for p in ['train', 'val']}
#     print("dataset sizes:", dataset_sizes)
#     return data_loaders, dataset_sizes

# def get_data(data_dir, labeled_train_idx=None, batch_size=8, classes=None, one_vs_all_target=None):
#     from torchvision import datasets, transforms
#     from torch.utils.data import Subset
#     import torch
#     import os

#     data_transforms = {
#         phase: transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5]*3, [0.5]*3),
#         ]) for phase in ['train', 'val']
#     }

#     image_datasets = {
#         phase: datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
#         for phase in ['train', 'val']
#     }

#     # ------------------ One-vs-All 運作模式 ------------------
#     if one_vs_all_target is not None:
#         name2idx = image_datasets['train'].class_to_idx
#         pos_idx = name2idx[one_vs_all_target]
#         print(f"Running one-vs-all: positive = '{one_vs_all_target}' (index={pos_idx})")

#         def _filter_ova(ds, base=None):
#             pool = base if base is not None else range(len(ds))
#             return [i for i in pool if ds.targets[i] == pos_idx or ds.targets[i] in name2idx.values()]

#         train_pool = labeled_train_idx if labeled_train_idx is not None else None
#         train_idx = _filter_ova(image_datasets['train'], train_pool)
#         val_idx = _filter_ova(image_datasets['val'])

#         image_datasets['train'] = Subset(image_datasets['train'], train_idx)
#         image_datasets['val'] = Subset(image_datasets['val'], val_idx)

#         class RelabeledOVADataset(torch.utils.data.Dataset):
#             def __init__(self, subset, pos_idx):
#                 self.subset = subset
#                 self.pos_idx = pos_idx
#             def __len__(self):
#                 return len(self.subset)
#             def __getitem__(self, idx):
#                 x, y = self.subset[idx]
#                 return x, 1 if y == self.pos_idx else 0

#         for split in ['train', 'val']:
#             image_datasets[split] = RelabeledOVADataset(image_datasets[split], pos_idx)

#         print(f"Relabeled to 1 for '{one_vs_all_target}', 0 for rest.")
    
#     # ------------------ 原本的 two-class/multi-class 模式 ------------------
#     elif classes is not None:
#         name2idx = image_datasets['train'].class_to_idx
#         wanted_idx = [name2idx[c] if isinstance(c, str) else int(c)
#                       for c in classes]

#         def _filter(ds, base=None):
#             pool = base if base is not None else range(len(ds))
#             return [i for i in pool if ds.targets[i] in wanted_idx]

#         train_pool = labeled_train_idx if labeled_train_idx is not None else None
#         train_idx  = _filter(image_datasets['train'], train_pool)
#         val_idx    = _filter(image_datasets['val'])

#         image_datasets['train'] = Subset(image_datasets['train'], train_idx)
#         image_datasets['val']   = Subset(image_datasets['val'],   val_idx)

#         if len(wanted_idx) == 2:
#             orig_to_new = {wanted_idx[0]: 0, wanted_idx[1]: 1}
#             class RelabeledSubset(torch.utils.data.Dataset):
#                 def __init__(self, subset, mapping):
#                     self.subset, self.map = subset, mapping
#                 def __len__(self):
#                     return len(self.subset)
#                 def __getitem__(self, idx):
#                     x, y = self.subset[idx]
#                     return x, self.map[int(y)]
#             for split in ['train', 'val']:
#                 image_datasets[split] = RelabeledSubset(image_datasets[split], orig_to_new)
#             print(f"Relabeled to subclass0/1 → {classes[0]}→0, {classes[1]}→1")
#         else:
#             print("Multi-class (≠2) — keep original labels.")

#     elif labeled_train_idx is not None:
#         image_datasets['train'] = Subset(image_datasets['train'], labeled_train_idx)

#     data_loaders = {
#         phase: torch.utils.data.DataLoader(
#             image_datasets[phase],
#             batch_size=batch_size if phase == 'train' else 16,
#             shuffle=(phase == 'train'),
#             num_workers=4,
#         ) for phase in ['train', 'val']
#     }

#     dataset_sizes = {p: len(image_datasets[p]) for p in ['train', 'val']}
#     print("dataset sizes:", dataset_sizes)
#     return data_loaders, dataset_sizes


def get_data(data_dir, labeled_train_idx=None, batch_size=8, classes=None, one_vs_all_target=None, subset_mode=False):
    from torchvision import datasets, transforms
    from torch.utils.data import Subset
    import torch
    import os

    data_transforms = {
        phase: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ]) for phase in ['train', 'val']
    }

    image_datasets = {
        phase: datasets.ImageFolder(os.path.join(data_dir, phase), data_transforms[phase])
        for phase in ['train', 'val']
    }

    # ------------------ One-vs-All 運作模式 ------------------
    if one_vs_all_target is not None:
        name2idx = image_datasets['train'].class_to_idx
        pos_idx = name2idx[one_vs_all_target]
        print(f"Running one-vs-all: positive = '{one_vs_all_target}' (index={pos_idx})")

        def _filter_ova(ds, base=None):
            pool = base if base is not None else range(len(ds))
            return [i for i in pool if ds.targets[i] == pos_idx or ds.targets[i] in name2idx.values()]

        train_pool = labeled_train_idx if labeled_train_idx is not None else None
        train_idx = _filter_ova(image_datasets['train'], train_pool)
        val_idx = _filter_ova(image_datasets['val'])

        image_datasets['train'] = Subset(image_datasets['train'], train_idx)
        image_datasets['val'] = Subset(image_datasets['val'], val_idx)

        class RelabeledOVADataset(torch.utils.data.Dataset):
            def __init__(self, subset, pos_idx):
                self.subset = subset
                self.pos_idx = pos_idx
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                x, y = self.subset[idx]
                return x, 1 if y == self.pos_idx else 0

        for split in ['train', 'val']:
            image_datasets[split] = RelabeledOVADataset(image_datasets[split], pos_idx)

        print(f"Relabeled to 1 for '{one_vs_all_target}', 0 for rest.")
    
    # ------------------ Subset Mode (新增) ------------------
    elif subset_mode and classes is not None:
        name2idx = image_datasets['train'].class_to_idx
        wanted_idx = [name2idx[c] if isinstance(c, str) else int(c)
                      for c in classes]

        def _filter(ds, base=None):
            pool = base if base is not None else range(len(ds))
            return [i for i in pool if ds.targets[i] in wanted_idx]

        train_pool = labeled_train_idx if labeled_train_idx is not None else None
        train_idx  = _filter(image_datasets['train'], train_pool)
        val_idx    = _filter(image_datasets['val'])

        image_datasets['train'] = Subset(image_datasets['train'], train_idx)
        image_datasets['val']   = Subset(image_datasets['val'],   val_idx)

        # 總是重新標記標籤到 [0, 1, 2, ...] 不管有多少類別
        orig_to_new = {wanted_idx[i]: i for i in range(len(wanted_idx))}
        
        class RelabeledSubset(torch.utils.data.Dataset):
            def __init__(self, subset, mapping):
                self.subset = subset
                self.mapping = mapping
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                x, y = self.subset[idx]
                return x, self.mapping[int(y)]

        for split in ['train', 'val']:
            image_datasets[split] = RelabeledSubset(image_datasets[split], orig_to_new)
        
        print(f"Subset mode: Relabeled {len(classes)} classes to [0, {len(classes)-1}]")
        print(f"Original classes {classes} → New labels {list(range(len(classes)))}")
    
    # ------------------ 原本的 two-class/multi-class 模式 ------------------
    elif classes is not None:
        name2idx = image_datasets['train'].class_to_idx
        wanted_idx = [name2idx[c] if isinstance(c, str) else int(c)
                      for c in classes]

        def _filter(ds, base=None):
            pool = base if base is not None else range(len(ds))
            return [i for i in pool if ds.targets[i] in wanted_idx]

        train_pool = labeled_train_idx if labeled_train_idx is not None else None
        train_idx  = _filter(image_datasets['train'], train_pool)
        val_idx    = _filter(image_datasets['val'])

        image_datasets['train'] = Subset(image_datasets['train'], train_idx)
        image_datasets['val']   = Subset(image_datasets['val'],   val_idx)

        if len(wanted_idx) == 2:
            orig_to_new = {wanted_idx[0]: 0, wanted_idx[1]: 1}
            class RelabeledSubset(torch.utils.data.Dataset):
                def __init__(self, subset, mapping):
                    self.subset, self.map = subset, mapping
                def __len__(self):
                    return len(self.subset)
                def __getitem__(self, idx):
                    x, y = self.subset[idx]
                    return x, self.map[int(y)]
            for split in ['train', 'val']:
                image_datasets[split] = RelabeledSubset(image_datasets[split], orig_to_new)
            print(f"Relabeled to subclass0/1 → {classes[0]}→0, {classes[1]}→1")
        else:
            print("Multi-class (≠2) — keep original labels.")

    elif labeled_train_idx is not None:
        image_datasets['train'] = Subset(image_datasets['train'], labeled_train_idx)

    data_loaders = {
        phase: torch.utils.data.DataLoader(
            image_datasets[phase],
            batch_size=batch_size if phase == 'train' else 16,
            shuffle=(phase == 'train'),
            num_workers=4,
        ) for phase in ['train', 'val']
    }

    dataset_sizes = {p: len(image_datasets[p]) for p in ['train', 'val']}
    print("dataset sizes:", dataset_sizes)
    return data_loaders, dataset_sizes




import os
import shutil
from torchvision import datasets

def copy_original_images_by_index(data_dir, idx_list, save_dir='./temp'):
    """
    根據 ImageFolder 原始 index，將對應影像複製到指定資料夾，並在檔名前標記 label。
    不經過 transform，不做 normalization。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 使用 ImageFolder 載入原始圖像路徑，不套 transform
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'))  # 或 'val'
    
    for i, idx in enumerate(idx_list):
        image_path, label = dataset.samples[idx]  # 拿到檔案路徑和類別標籤
        filename = os.path.basename(image_path)
        # 加上 label 和 index prefix，例如：0012_class3_dog.jpg
        new_filename = f"{i:04d}_class{label}_{filename}"
        new_path = os.path.join(save_dir, new_filename)
        shutil.copy(image_path, new_path)
    
    print(f"Copied {len(idx_list)} images to '{save_dir}'")
