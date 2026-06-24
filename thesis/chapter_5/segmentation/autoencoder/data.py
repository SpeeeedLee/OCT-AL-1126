"""
Autoencoder pretraining data: the SAME images the downstream U-Net sees.

Reuses `o_data` (cv2 read / 255, padded into a 384x512 canvas) so the encoder
is pretrained on the identical [0,1] input distribution as segmentation — no
Normalize([0.5],[0.5]) shift. Light random flip aug (the paper flips H/V to
expand training data). Each item is a single image tensor; the trainer uses it
as BOTH input and reconstruction target.

Pretrains on the fold-0 TRAIN split (736 imgs) by default — no val/test leakage.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from thesis.chapter_5.segmentation.utils.data import data_loader, o_data

WIDTH, HEIGHT = 384, 512   # same as the downstream U-Net


class AEReconDataset(Dataset):
    def __init__(self, opath, files, flip=True):
        self.opath = opath
        self.files = list(files)
        self.flip = flip

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = o_data(self.opath, [self.files[i]], WIDTH, HEIGHT)[0]   # [1,384,512] in [0,1]
        if self.flip:
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :]        # flip width
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1]        # flip height
        return torch.from_numpy(np.ascontiguousarray(img)).float()


def build_ae_dataset(dataroot, fold=0, train_only=True, flip=True):
    """Returns (dataset, file_list)."""
    opath = os.path.join(dataroot, "image") + "/"
    if train_only:
        train_LD, _, _ = data_loader(opath, fold)
        files = []
        for b in train_LD:
            files.extend(b)
    else:
        files = sorted(os.listdir(opath))
    return AEReconDataset(opath, files, flip=flip), files
