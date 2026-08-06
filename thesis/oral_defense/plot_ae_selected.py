#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense AE figures restricted to epochs {1, 10, 50, 500, 1000, 2000}:
  1. Reconstruction grid (Input + those 6 epoch rows, 5 image columns).
  2. MSE training-loss curve with red dots ONLY at those epochs.

Same visual style as thesis/chapter_5/segmentation/autoencoder/plot_ae_progress.py.

Run from repo root:
    python3 thesis/oral_defense/plot_ae_selected.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RECON = os.path.join(os.path.dirname(__file__), "..", "chapter_5", "segmentation",
                     "autoencoder", "recon")
OUTDIR = os.path.join(os.path.dirname(__file__), "figs")


def _load_row(folder):
    pngs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    return [np.asarray(Image.open(os.path.join(folder, p)).convert("L"), np.float32) / 255
            for p in pngs]
SEL = [1, 10, 50, 500, 1000, 2000]
N_SHOW = 5


def recon_grid(out):
    orig = os.path.join(RECON, "original")
    inputs = _load_row(orig)[:N_SHOW]
    eps = [e for e in SEL if os.path.isdir(os.path.join(RECON, f"ep{e:04d}"))]
    rows = [("Input", inputs)] + [(str(e), _load_row(os.path.join(RECON, f"ep{e:04d}"))[:N_SHOW])
                                  for e in eps]
    ncol, nrow = len(inputs), len(rows)
    AR, HEADER, wide, wspace = 512.0 / 384.0, 0.5, 1.3, 0.10
    ch = min(5.6 / (ncol * AR), (9.0 - HEADER) / nrow)
    cw = AR * ch
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * cw * wide, nrow * ch + HEADER),
                             gridspec_kw=dict(wspace=wspace, hspace=0.03))
    axes = np.atleast_2d(axes)
    for r, (label, imgs) in enumerate(rows):
        is_in = (r == 0)
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D62728" if is_in else "#BBBBBB")
                sp.set_linewidth(2.2 if is_in else 0.6)
            if c == 0:
                ax.set_ylabel(label, fontsize=13, fontweight="bold" if is_in else "normal",
                              color="#D62728" if is_in else "#333333", rotation=0,
                              ha="right", va="center", labelpad=14)
    fig_h = nrow * ch + HEADER
    fig.subplots_adjust(left=0.16, right=0.995, top=1 - HEADER / fig_h, bottom=0.006)
    x_lab = axes[0, 0].get_position().x0
    x_img_c = (x_lab + axes[0, -1].get_position().x1) / 2
    fig.suptitle(r"$\theta_{\mathrm{AE}}$ Reconstruction Results at Different Epochs",
                 fontsize=15, x=x_img_c, y=1 - 0.46 * HEADER / fig_h)
    y_top = axes[1, 0].get_position().y1
    y_bot = axes[-1, 0].get_position().y0
    fig.text(x_lab - 0.135, (y_top + y_bot) / 2, "Epoch", fontsize=14, fontweight="bold",
             color="#333333", ha="center", va="center", rotation=90)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}  (Input + rows {eps})")


def loss_curve(out):
    pts = {}
    with open(os.path.join(RECON, "loss_log.csv"), errors="ignore") as f:
        for line in f:
            p = line.replace("\x00", "").strip().split(",")
            if len(p) >= 2 and p[0].isdigit():
                try:
                    pts[int(p[0])] = float(p[1])
                except ValueError:
                    continue
    ep = sorted(pts); mse = [pts[e] for e in ep]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ep, mse, color="#1F77B4", lw=2.5)
    sx = [e for e in SEL if e in pts]; sy = [pts[e] for e in sx]
    ax.scatter(sx, sy, color="#D62728", zorder=5, s=90, label="Visualization checkpoints")
    ax.set_yscale("log")
    ax.set_xlabel("Pretraining Epoch", fontsize=26, labelpad=10)
    ax.set_ylabel("MSE Loss", fontsize=26, labelpad=10)
    ax.set_title(r"Training Loss Curve of $\theta_{\mathrm{AE}}$", fontsize=26, pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=20, width=1.5, length=6)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}  (red dots at {sx})")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    recon_grid(os.path.join(OUTDIR, "ae_reconstruction_selected.png"))
    loss_curve(os.path.join(OUTDIR, "ae_loss_curve_selected.png"))


if __name__ == "__main__":
    main()
