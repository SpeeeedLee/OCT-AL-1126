"""
Grayscale SimCLR data pipeline for the OCT segmentation images.

Mirrors the classification SimCLR augmentation
(`SSL/simclr/data_aug/contrastive_learning_dataset.py`):
RandomResizedCrop + HF/VF + RandomAutocontrast + RandomEqualize + GaussianBlur,
which is ALREADY color-jitter-free → works directly on grayscale. Only changes:
1-channel load + 1-channel Normalize([0.5],[0.5]) + 1-channel GaussianBlur, and a
flat-folder dataset (seg images aren't in class subfolders, so no ImageFolder).

Pretrains on the fold-0 TRAIN split (736 imgs) by default — no val/test leakage.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GaussianBlur1ch(object):
    """1-channel version of the classification SimCLR GaussianBlur (PIL in/out)."""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(1, 1, (kernel_size, 1), padding=0, bias=False, groups=1)
        self.blur_v = nn.Conv2d(1, 1, (1, kernel_size), padding=0, bias=False, groups=1)
        self.k, self.r = kernel_size, radias
        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.to_tensor(img).unsqueeze(0)          # [1,1,H,W]
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma)); x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).float()
        self.blur_h.weight.data.copy_(x.view(1, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(1, 1, 1, self.k))
        with torch.no_grad():
            img = self.blur(img).squeeze(0)
        return self.to_pil(img)


class TwoViews(object):
    def __init__(self, base): self.base = base
    def __call__(self, x): return [self.base(x), self.base(x)]


def simclr_transform(size, blur=True):
    """SimCLR augmentation (grayscale). `size` = crop size."""
    t = [transforms.RandomResizedCrop(size=size, scale=(0.3, 0.7)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomAutocontrast(),
         transforms.RandomEqualize()]
    if blur:
        t.append(GaussianBlur1ch(kernel_size=int(0.1 * size)))
    t += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(t)


class FlatGrayDataset(Dataset):
    """A flat folder of grayscale PNGs → two SimCLR views per image."""
    def __init__(self, image_dir, file_list, transform):
        self.dir = image_dir
        self.files = list(file_list)
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.dir, self.files[i])).convert("L")  # grayscale
        return self.transform(img), 0   # label unused (SSL)


def build_pretrain_dataset(dataroot, size, fold=0, train_only=True, blur=True):
    """Returns (dataset, file_list). train_only → fold-0 train split (no leakage)."""
    image_dir = os.path.join(dataroot, "image")
    if train_only:
        # reuse the segmentation split so SSL sees only training images
        from thesis.chapter_5.segmentation.utils.data import data_loader
        train_LD, _, _ = data_loader(image_dir + "/", fold)
        files = []
        for b in train_LD:
            files.extend(b)
    else:
        files = sorted(os.listdir(image_dir))
    ds = FlatGrayDataset(image_dir, files, TwoViews(simclr_transform(size, blur)))
    return ds, files
