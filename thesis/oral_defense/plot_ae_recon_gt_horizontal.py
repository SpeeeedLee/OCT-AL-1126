#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizontal version of the AE recon-vs-GT figure: 2 rows
(Original / θ_AE @ Epoch 2000) x 5 image columns, each overlaid with the RED
ground-truth cell-nuclei contour.

Run from repo root IN oct-env (needs cv2 for the mask loader):
    conda run -n oct-env python3 thesis/oral_defense/plot_ae_recon_gt_horizontal.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.getcwd())
from thesis.chapter_5.segmentation.utils.data import g_data_cell_binary

RECON = os.path.join("thesis", "chapter_5", "segmentation", "autoencoder", "recon")
DATAROOT = "./ds/segmentation_correct"
OUT = os.path.join("thesis", "oral_defense", "figs", "ae_recon_gt_horizontal.png")
W, H = 384, 512
EPOCH = 2000
N_SHOW = 5
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})


def _load_row(folder):
    pngs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    return [np.asarray(Image.open(os.path.join(folder, p)).convert("L"), np.float32) / 255
            for p in pngs]


def _read_names():
    names = {}
    with open(os.path.join(RECON, "images.txt")) as f:
        for line in f:
            i, nm = line.rstrip("\n").split("\t")
            names[int(i)] = nm
    return [names[i] for i in sorted(names)]


def main():
    names = _read_names()[:N_SHOW]
    gpath = os.path.join(DATAROOT, "cell") + "/"
    masks = [g_data_cell_binary(gpath, [nm], W, H)[0, 0] for nm in names]
    orig = _load_row(os.path.join(RECON, "original"))[:N_SHOW]
    ep = _load_row(os.path.join(RECON, f"ep{EPOCH:04d}"))[:N_SHOW]
    rows = [("Original", orig),
            (rf"$\theta_{{\mathrm{{AE}}}}$ @ Epoch {EPOCH}", ep)]

    nrow, ncol = len(rows), N_SHOW
    AR = 512.0 / 384.0                     # per-image W/H
    pw = 2.9; ph = pw / AR
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * pw, nrow * ph + 0.3),
                             gridspec_kw=dict(wspace=0.03, hspace=0.03))
    axes = np.atleast_2d(axes)
    for r, (label, imgs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.contour(masks[c], levels=[0.5], colors="red", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=14, fontweight="bold", labelpad=10)

    fig.subplots_adjust(left=0.06, right=0.995, top=0.985, bottom=0.03)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT}  ({nrow}x{ncol}, + red GT)")


if __name__ == "__main__":
    main()
