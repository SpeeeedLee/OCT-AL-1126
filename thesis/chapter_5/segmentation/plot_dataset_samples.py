#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3x2 figure: 3 random TRAIN OCT images (left = Original images, right = Nuclei labels).
Run (repo root): python3 thesis/chapter_5/segmentation/plot_dataset_samples.py
"""
import os
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
IMG_DIR = "ds/segmentation/image"
CELL_DIR = "ds/segmentation/cell"
SEED = 42                      # reproducible pick
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})


def train_files(fold=0):
    import sys; sys.path.insert(0, os.getcwd())
    from thesis.chapter_5.segmentation.utils.data import data_loader
    tr, _, _ = data_loader(IMG_DIR + "/", fold)
    files = []
    for b in tr:
        files.extend(b)
    return files


def main():
    files = train_files()
    random.Random(SEED).shuffle(files)
    picks = files[:3]
    print("picked train images:", picks)

    # image is 500x384 (W x H); keep aspect
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0))
    for r, name in enumerate(picks):
        img = np.array(Image.open(os.path.join(IMG_DIR, name)).convert("L"))
        msk = np.array(Image.open(os.path.join(CELL_DIR, name)).convert("L"))
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 1].imshow(msk, cmap="gray", vmin=0, vmax=255)
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    axes[0, 0].set_title("Original images", fontsize=16, fontweight="bold", pad=10)
    axes[0, 1].set_title("Nuclei labels", fontsize=16, fontweight="bold", pad=10)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, wspace=0.04, hspace=0.06)
    out = os.path.join(HERE, "figs", "dataset_samples.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"saved -> {out}  (each image {img.shape[1]}x{img.shape[0]} px W x H)")


if __name__ == "__main__":
    main()
