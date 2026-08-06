#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense SimCLR concept figure — simple & clean, using OUR actual
augmentation pipeline on real OCT images (original aspect ratio preserved).

Two source images -> each augmented into two views -> shared encoder f(.) ->
embeddings; contrastive objective pulls the two views of the SAME image
together (attract) and pushes different images apart (repel).

NOTE: the pipeline here deliberately OMITS RandomVerticalFlip (present in the
real SSL/simclr pipeline) and the Normalize, purely for display.

Run from repo root:
    python3 thesis/oral_defense/plot_simclr_demo.py
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch, random
from torchvision import transforms

from SSL.simclr.data_aug.gaussian_blur import GaussianBlur   # our real blur

OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
IMGS = [
    "ds/classification/seven_class/train/Nevus/20210108_131900B.png",
    "ds/classification/seven_class/train/Solar lentigo/20201223_094913B.png",
]
ORIG_W, ORIG_H = 1000, 600          # native OCT size (aspect 5:3)
ASPECT = ORIG_W / ORIG_H            # w/h
CROP = (300, 500)                   # (h, w) keeps 5:3 aspect for aug views

# figure geometry
FIGW, FIGH = 16.0, 8.0
C_A, C_B = "#2C7FB8", "#E6862A"
C_PULL, C_PUSH = "#238B45", "#CB2A2A"


def simclr_view_transform():
    """Our SimCLR pipeline, minus VerticalFlip and Normalize (display only)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=CROP, scale=(0.3, 0.7)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),   # intentionally omitted for slides
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        GaussianBlur(kernel_size=25),
    ])


def to_np(pil):
    return np.asarray(pil.convert("RGB"))


def make_views(seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tf = simclr_view_transform()
    origs, views = [], []
    for path in IMGS:
        img = Image.open(os.path.join(os.getcwd(), path)).convert("RGB")
        origs.append(to_np(img.resize((ORIG_W, ORIG_H))))
        views.append([to_np(tf(img)), to_np(tf(img))])
    return origs, views


def place_img(fig, cx, cy, wfrac, arr, edge):
    """Place image centered at (cx,cy) with width wfrac; height preserves aspect."""
    hfrac = wfrac * (FIGW / FIGH) / ASPECT
    left, bottom = cx - wfrac / 2, cy - hfrac / 2
    ax = fig.add_axes([left, bottom, wfrac, hfrac])
    ax.imshow(arr); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(3)
    return left, bottom, wfrac, hfrac


def arrow(ax, p0, p1, color="#888888", lw=2.2, style="-|>", ms=15, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=1, shrinkB=1, zorder=5))


def main():
    origs, views = make_views(seed=29)
    fig = plt.figure(figsize=(FIGW, FIGH))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")

    # column x-centres
    x_orig, x_view, x_enc, x_z = 0.10, 0.37, 0.63, 0.88
    w_orig, w_view = 0.155, 0.135
    # view y-centres (2 per image); originals centred between their pair
    yA = [0.80, 0.60]; yB = [0.36, 0.16]
    ycA = sum(yA) / 2; ycB = sum(yB) / 2

    # originals
    rA = place_img(fig, x_orig, ycA, w_orig, origs[0], C_A)
    rB = place_img(fig, x_orig, ycB, w_orig, origs[1], C_B)
    bg.text(x_orig, ycA + rA[3]/2 + 0.03, r"$\mathbf{x}^{(1)}$", ha="center",
            va="bottom", fontsize=27, color=C_A, fontweight="bold")
    bg.text(x_orig, ycB + rB[3]/2 + 0.03, r"$\mathbf{x}^{(2)}$", ha="center",
            va="bottom", fontsize=27, color=C_B, fontweight="bold")

    # augmented views + aug arrows from original
    view_right = {}
    for arr, y, col, yo in [(views[0][0], yA[0], C_A, ycA), (views[0][1], yA[1], C_A, ycA),
                            (views[1][0], yB[0], C_B, ycB), (views[1][1], yB[1], C_B, ycB)]:
        rv = place_img(fig, x_view, y, w_view, arr, col)
        view_right[y] = (x_view + rv[2]/2, y)
        rad = 0.12 if y > yo else -0.12
        arrow(bg, (x_orig + w_orig/2 + 0.004, yo), (x_view - rv[2]/2 - 0.004, y),
              color="#999999", rad=rad)
    for yo in (ycA, ycB):
        bg.text((x_orig + x_view)/2, yo + 0.075, r"$t,\,t' \sim \mathcal{T}$",
                ha="center", va="center", fontsize=19, color="#555555")

    # shared encoder box
    w_enc, y0, y1 = 0.115, 0.11, 0.85
    bg.add_patch(FancyBboxPatch((x_enc - w_enc/2, y0), w_enc, y1 - y0,
                 boxstyle="round,pad=0.006,rounding_size=0.02",
                 linewidth=2.2, edgecolor="#555555", facecolor="#F2F2F2", zorder=2))
    bg.text(x_enc, 0.55, r"$f(\cdot)$", ha="center", va="center", fontsize=34, fontweight="bold")
    bg.text(x_enc, 0.44, "ResNet-18\nencoder", ha="center", va="center",
            fontsize=17, color="#333333")

    # view -> encoder (horizontal, aligned to each view's y)
    for (rx, ry) in view_right.values():
        arrow(bg, (rx + 0.004, ry), (x_enc - w_enc/2 - 0.004, ry), color="#999999")

    # embeddings: two dots per image (close = same image)
    z = {"A_i": (x_z - 0.02, 0.74), "A_j": (x_z + 0.045, 0.66),
         "B_i": (x_z - 0.02, 0.30), "B_j": (x_z + 0.045, 0.22)}
    zc = {"A_i": C_A, "A_j": C_A, "B_i": C_B, "B_j": C_B}
    for key, ysrc in [("A_i", yA[0]), ("A_j", yA[1]), ("B_i", yB[0]), ("B_j", yB[1])]:
        arrow(bg, (x_enc + w_enc/2 + 0.004, ysrc),
              (z[key][0] - 0.018, z[key][1]), color="#999999",
              rad=0.10 if z[key][1] > ysrc else -0.10)
    for key, (zx, zy) in z.items():
        bg.scatter([zx], [zy], s=460, color=zc[key], edgecolor="white", linewidth=2, zorder=6)
    bg.text(x_z + 0.012, 0.83, r"$\mathbf{z}^{(1)}$", ha="center", fontsize=23, color=C_A)
    bg.text(x_z + 0.012, 0.13, r"$\mathbf{z}^{(2)}$", ha="center", fontsize=23, color=C_B)

    # attract (same image) / repel (different images)
    arrow(bg, z["A_i"], z["A_j"], color=C_PULL, lw=3, style="<|-|>", ms=15)
    arrow(bg, z["B_i"], z["B_j"], color=C_PULL, lw=3, style="<|-|>", ms=15)
    bg.text(x_z + 0.085, 0.70, "attract", ha="left", va="center", fontsize=18,
            color=C_PULL, fontweight="bold")
    bg.text(x_z + 0.085, 0.26, "attract", ha="left", va="center", fontsize=18,
            color=C_PULL, fontweight="bold")
    arrow(bg, (z["A_i"][0] + 0.03, z["A_i"][1] - 0.10), (z["B_i"][0] + 0.03, z["B_i"][1] + 0.10),
          color=C_PUSH, lw=3, style="<|-|>", ms=15, rad=-0.28)
    bg.text(x_z + 0.10, 0.48, "repel", ha="left", va="center", fontsize=18,
            color=C_PUSH, fontweight="bold")

    # column captions
    for xc, txt in [(x_orig, "Input images"), (x_view, "Augmented views"),
                    (x_enc, "Shared encoder"), (x_z, "Latent space")]:
        bg.text(xc, 0.035, txt, ha="center", va="center", fontsize=17,
                color="#222222", fontstyle="italic")

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "simclr_demo.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
