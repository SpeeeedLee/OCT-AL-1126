#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense flip-augmentation figures (progressive build), same style as
thesis/chapter_4/plot_flip_illustration.py (Arial, gray OCT, thin panel frame).

Produces, into thesis/oral_defense/figs/ :
  1. flip_original.png   — single original image only.
  2. flip_horizontal.png — (a) Original + (b) Horizontal flip, stacked VERTICALLY.
  3. flip_illustration.png — 2x2 grid; left column aligns with flip_horizontal:
        (a) Original          (c) Vertical flip (VF)
        (b) Horizontal flip   (d) HVF

Run from repo root:
    python3 thesis/oral_defense/plot_flip_variants.py
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_PATH = "ds/classification/seven_class/train/Normal/20210303_143725B.png"
OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
CAP = 22
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5})


def load():
    img = Image.open(os.path.join(os.getcwd(), IMG_PATH)).convert("L")
    HF = img.transpose(Image.FLIP_LEFT_RIGHT)
    VF = img.transpose(Image.FLIP_TOP_BOTTOM)
    HVF = img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    return img, HF, VF, HVF


def _frame(ax, im, cap=None):
    ax.imshow(np.asarray(im), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.2); s.set_color("0.4")
    if cap:
        ax.set_xlabel(cap, fontsize=CAP, labelpad=10)


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    img, HF, VF, HVF = load()
    W, H = img.size
    pw = 4.0 * (W / H)

    # 1. single original
    fig, ax = plt.subplots(figsize=(pw, 4.0))
    _frame(ax, img)
    fig.tight_layout(pad=0.3)
    save(fig, "flip_original.png")

    # 2. original + horizontal flip, stacked vertically
    fig, axes = plt.subplots(2, 1, figsize=(pw, 2 * 4.0 + 0.8), constrained_layout=True)
    _frame(axes[0], img, "(a) Original image")
    _frame(axes[1], HF, "(b) Horizontal flip (HF)")
    save(fig, "flip_horizontal.png")

    # 3. 2x2: left column = Original / HF (aligns with flip_horizontal),
    #         right column = VF / HVF. axes.flat is row-major (TL,TR,BL,BR).
    panels = [(img, "(a) Original image"), (VF, "(c) Vertical flip (VF)"),
              (HF, "(b) Horizontal flip (HF)"), (HVF, "(d) Horizontal & Vertical flip (HVF)")]
    fig, axes = plt.subplots(2, 2, figsize=(2 * pw, 2 * 4.0 + 0.8), constrained_layout=True)
    for ax, (im, cap) in zip(axes.flat, panels):
        _frame(ax, im, cap)
    save(fig, "flip_illustration.png")


if __name__ == "__main__":
    main()
