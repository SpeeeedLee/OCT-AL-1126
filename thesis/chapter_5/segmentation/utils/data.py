"""
Data utilities for Chapter 5 nuclei-segmentation experiments.

Copied from `segmentation/utils/data.py` and EXTENDED with offline flip
augmentation (HF / VF / HF+VF), applied SYNCHRONOUSLY to image and mask.

Key conventions (unchanged from the source codebase):
  - 5-fold split is deterministic: `random.Random(42).shuffle(filenames)`.
  - Each OCT image is read at 500px wide and placed into a 384x512 canvas at
    columns [6:506]; the 6px borders on each side are symmetric, so a horizontal
    flip of the whole canvas keeps content in [6:506] (no padding leak).
  - Binary nuclei masks are thresholded at 127.

Augmentation token format:
  A "training item" is either a bare filename (no augmentation) or the string
  "name@@flip" where flip in {h, v, hv}. `o_data` / `g_data_cell_binary` parse
  the token, load the file, then flip the canvas. Validation/test always pass
  bare filenames (flip = none), so they are never augmented.
"""
import os
import random
import numpy as np
import cv2
import torch.utils.data as Data


# ---------------------------------------------------------------------------
# Offline flip augmentation
# ---------------------------------------------------------------------------
# aug_factor -> which flips to materialise for every training image.
#   1 = original only          (1x)
#   2 = original + HF           (2x)
#   3 = original + HF + VF      (3x)
#   4 = original + HF + VF + HFV(4x)   <- the Ch4 "aug4" setting
FLIP_SETS = {
    1: ["none"],
    2: ["none", "h"],
    3: ["none", "h", "v"],
    4: ["none", "h", "v", "hv"],
}
# Named flip-sets (for variants whose composition differs from the integer codes,
# e.g. VF-only or VF+HV). h=horizontal, v=vertical, hv=both (diagonal).
FLIP_SETS_NAMED = {
    "none": ["none"],
    "hf":   ["none", "h"],            # = aug2
    "vf":   ["none", "v"],            # VF only (2x)
    "hfvf": ["none", "h", "v"],       # = aug3
    "vfhv": ["none", "v", "hv"],      # VF + HV (3x)
    "4x":   ["none", "h", "v", "hv"], # = aug4
}


def parse_item(token):
    """Split a training token into (filename, flip)."""
    token = str(token)
    if "@@" in token:
        name, flip = token.split("@@")
        return name, flip
    return token, "none"


def apply_flip(arr, flip):
    """Flip a 2D [H, W] canvas. h=left-right, v=up-down, hv=both."""
    if flip == "h":
        return arr[:, ::-1]
    if flip == "v":
        return arr[::-1, :]
    if flip == "hv":
        return arr[::-1, ::-1]
    return arr


def expand_with_aug(file_list, aug_factor=None, flips=None):
    """
    Expand a list of filenames into augmented training tokens.

    Either pass an integer `aug_factor` (uses FLIP_SETS) OR an explicit `flips`
    list (e.g. FLIP_SETS_NAMED['vf']). Returns a new list; originals stay as bare
    filenames, flipped variants become "name@@<flip>".
    """
    if flips is None:
        flips = FLIP_SETS[int(aug_factor)]
    items = []
    for name in file_list:
        for fl in flips:
            items.append(name if fl == "none" else f"{name}@@{fl}")
    return items


# ---------------------------------------------------------------------------
# Leakage-free split for ds/segmentation_correct  (2026-06-24 rewrite)
# ---------------------------------------------------------------------------
# The OLD split shuffled all 1224 images and folded them -> the dataset's built-in
# H-flips (prefix 1_<->3_, 2_<->4_) landed in different splits = test leaked from
# train. The corrected dataset `ds/segmentation_correct` keeps ONLY the 612 non-
# flipped images (1_/2_) and is pre-split on disk into image|cell|layer / {train,val}.
# Here: train = the train/ folder as-is; val/ (122) is halved into valid+test
# (deterministically) -> overall 8:1:1, mirroring classification. Returned filename
# tokens INCLUDE the split subfolder ("train/<name>" / "val/<name>") so that
# o_data/g_data (which do `path + name`) read from the right subfolder.
def data_loader(opath, fold=0):
    """opath = <dataroot>/image/  (must contain train/ and val/ subfolders).
    `fold` kept for backward-compat but ignored (no cross-validation)."""
    train_list = ["train/" + f for f in sorted(os.listdir(os.path.join(opath, "train")))]
    val_all = ["val/" + f for f in sorted(os.listdir(os.path.join(opath, "val")))]
    random.Random(42).shuffle(val_all)            # deterministic val/test split
    half = len(val_all) // 2
    valid_list, test_list = val_all[:half], val_all[half:]

    print('train', len(train_list), 'valid', len(valid_list), 'test', len(test_list))

    train_data_LD = Data.DataLoader(dataset=train_list, batch_size=8, shuffle=True, num_workers=4)
    valid_data_LD = Data.DataLoader(dataset=valid_list, batch_size=8, shuffle=False, num_workers=4)
    test_data_LD = Data.DataLoader(dataset=test_list, batch_size=8, shuffle=False, num_workers=4)

    return train_data_LD, valid_data_LD, test_data_LD


# ---------------------------------------------------------------------------
# Loading (aug-aware: accepts bare filenames OR "name@@flip" tokens)
# ---------------------------------------------------------------------------
def o_data(opath, batch_num, width, height):
    """Read OCT images. shape: [batch, 1, width(384), height(512)]."""
    img_sub = np.zeros((len(batch_num), 1, width, height))
    for i, token in enumerate(np.array(batch_num)):
        name, flip = parse_item(token)
        canvas = np.zeros((width, height))
        canvas[:, 6:506] = cv2.imread(opath + name, 0) / 255
        img_sub[i, 0] = apply_flip(canvas, flip)
    return img_sub


def g_data_cell_binary(gpath, batch_num, width, height):
    """Read binary nuclei masks. shape: [batch, 1, width, height], values {0,1}."""
    img_sub = np.zeros((len(batch_num), 1, width, height), dtype=np.float32)
    for i, token in enumerate(np.array(batch_num)):
        name, flip = parse_item(token)
        canvas = np.zeros((width, height), dtype=np.float32)
        img = cv2.imread(gpath + name, 0)
        canvas[:, 6:506] = (img > 127).astype(np.float32)  # threshold at 127
        img_sub[i, 0] = apply_flip(canvas, flip)
    return img_sub
