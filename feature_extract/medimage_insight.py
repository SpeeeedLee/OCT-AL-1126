import os
import base64
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn as nn

from feature_extract.MedImageInsights.medimageinsightmodel import MedImageInsight

def read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def get_latent_features_med(
    data_dir: str,
    train_idx: list[int],
    device: torch.device,
    split: str = 'train',
    model_dir: str = "./feature_extract/MedImageInsights/2024.09.27",
    vision_model_name: str = "medimageinsigt-v1.0.0.pt",
    language_model_name: str = "language_model.pth",
    batch_size: int = 32,
    num_workers: int = 4,
    class_label: bool = False,
) -> dict[int, torch.Tensor] | dict[int, tuple[torch.Tensor, int]]:
    """
    透過 MedImageInsight 拿取圖像 embeddings，可選擇同時回傳 class label。

    Args:
      data_dir:      資料夾路徑，下層要有 train/val 子資料夾
      train_idx:     要處理的 subset indices (對應 full_dataset.samples 的索引)
      device:        torch.device("cuda") or torch.device("cpu")
      split:         'train' 或 'test'
      model_dir:     MedImageInsight 權重資料夾
      vision_model_name: 視覺模型檔名
      language_model_name: 語言模型檔名
      batch_size:    DataLoader batch size
      num_workers:   DataLoader num_workers
      class_label:   若 True，回傳 (feature, label) tuple；否則只回傳 feature

    Returns:
      feature_dict: 
        如果 class_label=False，格式 {idx: tensor(feature)}  
        如果 class_label=True，  格式 {idx: (tensor(feature), label)}
    """
    # 1) 準備 Dataset + Subset + DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    if split == 'train':
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                            transform=transform)
    elif split == 'val':
        full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                            transform=transform)
    else:
        raise ValueError(f"Invalid split '{split}'; expected 'train' or 'val'.")

    subset = Subset(full_dataset, train_idx)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2) 初始化並載入 MedImageInsight
    classifier = MedImageInsight(
        model_dir=model_dir,
        vision_model_name=vision_model_name,
        language_model_name=language_model_name
    )
    classifier.load_model()

    # 3) 按 batch 提取 embeddings
    feature_dict: dict[int, torch.Tensor] | dict[int, tuple[torch.Tensor, int]] = {}
    with torch.no_grad():
        for batch_idx, (_imgs, _labels) in enumerate(tqdm(loader, desc="Extracting MedImage features")):
            # 计算本批的原始索引
            start = batch_idx * batch_size
            idxs = subset.indices[start : start + _imgs.size(0)]

            # 3.1) 讀圖 -> base64
            b64_list = []
            for idx in idxs:
                img_path, _ = full_dataset.samples[idx]
                raw = read_image_bytes(img_path)
                b64_list.append(base64.encodebytes(raw).decode("utf-8"))

            # 3.2) 呼叫 encode，取得 numpy embeddings
            out = classifier.encode(images=b64_list)
            embeds_np = out['image_embeddings']  # shape = (B, D), dtype=float32

            # 3.3) 存入 feature_dict
            for orig_idx, row in zip(idxs, embeds_np):
                # row already is a NumPy array
                if class_label:
                    _, label = full_dataset.samples[orig_idx]
                    feature_dict[orig_idx] = (row, int(label))
                else:
                    feature_dict[orig_idx] = row

    # 4) Sanity check
    example_idx = train_idx[0]
    val = feature_dict[example_idx]
    if class_label:
        feat, lab = val
        print(f"[Sanity] idx={example_idx}, emb shape={feat.shape}, label={lab}")
    else:
        print(f"[Sanity] idx={example_idx}, emb shape={val.shape}")

    return feature_dict