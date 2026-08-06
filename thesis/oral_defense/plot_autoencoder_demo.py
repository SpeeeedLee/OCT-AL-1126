#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense Autoencoder schematic: a U-Net-shaped encoder-decoder in the same
visual style as the segmentation U-Net figure, but WITHOUT skip connections
(that absence is what makes it an autoencoder).

Encoder (Conv + MaxPool, green down) -> bottleneck -> Decoder (Upsampling,
orange up) -> 1x1 conv (yellow) -> reconstructed grayscale image.

Run from repo root:
    python3 thesis/oral_defense/plot_autoencoder_demo.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "figs", "autoencoder_demo.png")

BLUE, YELLOW, GREEN, ORANGE = "#3F6FB4", "#F2B417", "#7DAB3C", "#ED7D31"
EDGE = "#2A2A2A"
BW = 0.34            # block width
GAP = 0.16           # gap between blocks in a stack

# level geometry (levels 0..4 encoder/decoder, then bottleneck)
ENC_X = [1.2, 2.7, 4.2, 5.7, 7.2]
LVL_Y = [8.8, 7.15, 5.65, 4.3, 3.1]
H     = [3.0, 2.35, 1.8, 1.35, 1.0]
BOTT  = (9.0, 1.75); BOTT_H = 0.7
DEC_X = [16.8, 15.3, 13.8, 12.3, 10.8]   # level 0..4 (mirror of encoder)
CH    = [32, 64, 128, 256, 512]


def block(ax, cx, cy, h, color=BLUE):
    ax.add_patch(Rectangle((cx - BW / 2, cy - h / 2), BW, h, facecolor=color,
                           edgecolor=EDGE, linewidth=1.2, zorder=3))


def stack(ax, cx, cy, h, n, color=BLUE):
    """n blocks side by side centered at cx; small forward arrows between them."""
    total = n * BW + (n - 1) * GAP
    x0 = cx - total / 2 + BW / 2
    xs = [x0 + i * (BW + GAP) for i in range(n)]
    for x in xs:
        block(ax, x, cy, h, color)
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + BW / 2, cy), (xs[i + 1] - BW / 2, cy),
                     arrowstyle="-|>", mutation_scale=11, color=EDGE, lw=1.3, zorder=4))
    return xs[0] - BW / 2, xs[-1] + BW / 2   # left edge, right edge


def arrow(ax, p0, p1, color, lw=2.6, ms=18):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                 color=color, lw=lw, zorder=2, shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.2, 19.6); ax.set_ylim(-2.6, 10.4); ax.axis("off")

    # ── input image block ────────────────────────────────────────────────
    block(ax, 0.45, LVL_Y[0], H[0])
    ax.text(0.45, LVL_Y[0] + H[0] / 2 + 0.15, "1", ha="center", va="bottom",
            fontsize=15, fontweight="bold")

    # ── encoder ──────────────────────────────────────────────────────────
    enc_edges = []
    prev_r = 0.45 + BW / 2
    for l in range(5):
        le, re = stack(ax, ENC_X[l], LVL_Y[l], H[l], 2)
        enc_edges.append((le, re))
        arrow(ax, (prev_r + 0.02, LVL_Y[max(0, l - 1)] if l == 0 else LVL_Y[l]),
              (le - 0.02, LVL_Y[l]), EDGE, lw=1.6, ms=12) if l == 0 else None
        ax.text(ENC_X[l], LVL_Y[l] + H[l] / 2 + 0.15, str(CH[l]), ha="center",
                va="bottom", fontsize=15, fontweight="bold")
        prev_r = re
    # green max-pool arrows (down between encoder levels)
    for l in range(4):
        arrow(ax, (ENC_X[l], LVL_Y[l] - H[l] / 2 - 0.03),
              (ENC_X[l + 1], LVL_Y[l + 1] + H[l + 1] / 2 + 0.03), GREEN)
    # encoder level4 -> bottleneck
    arrow(ax, (ENC_X[4], LVL_Y[4] - H[4] / 2 - 0.03),
          (BOTT[0], BOTT[1] + BOTT_H / 2 + 0.03), GREEN)

    # ── bottleneck ───────────────────────────────────────────────────────
    stack(ax, BOTT[0], BOTT[1], BOTT_H, 3)
    ax.text(BOTT[0], BOTT[1] - BOTT_H / 2 - 0.32, "1024", ha="center", va="top",
            fontsize=15, fontweight="bold")

    # ── decoder ──────────────────────────────────────────────────────────
    # bottleneck -> decoder level4 (orange up)
    arrow(ax, (BOTT[0], BOTT[1] + BOTT_H / 2 + 0.03),
          (DEC_X[4], LVL_Y[4] - H[4] / 2 - 0.03), ORANGE)
    dec_edges = []
    for l in range(4, -1, -1):
        le, re = stack(ax, DEC_X[l], LVL_Y[l], H[l], 2)
        dec_edges.append((l, le, re))
        ax.text(DEC_X[l], LVL_Y[l] + H[l] / 2 + 0.15, str(CH[l]), ha="center",
                va="bottom", fontsize=15, fontweight="bold")
    # orange upsampling arrows (up between decoder levels)
    for l in range(4, 0, -1):
        arrow(ax, (DEC_X[l], LVL_Y[l] + H[l] / 2 + 0.03),
              (DEC_X[l - 1], LVL_Y[l - 1] - H[l - 1] / 2 - 0.03), ORANGE)

    # ── output: 1x1 conv (yellow) -> reconstructed image ─────────────────
    dec0_r = DEC_X[0] + (2 * BW + GAP) / 2
    out_x = dec0_r + 1.0
    arrow(ax, (dec0_r + 0.02, LVL_Y[0]), (out_x - BW / 2 - 0.02, LVL_Y[0]),
          YELLOW, lw=2.6, ms=16)
    block(ax, out_x, LVL_Y[0], H[0])
    ax.text(out_x, LVL_Y[0] + H[0] / 2 + 0.15, "1", ha="center", va="bottom",
            fontsize=15, fontweight="bold")
    ax.text(0.45, LVL_Y[0] - H[0] / 2 - 0.35, "input", ha="center", va="top", fontsize=13)
    ax.text(out_x, LVL_Y[0] - H[0] / 2 - 0.35, "reconstruction", ha="center",
            va="top", fontsize=13)

    # ── "no skip connections" annotation across the gap ──────────────────
    ax.text(9.0, 8.9, "no skip connections", ha="center", va="center",
            fontsize=16, style="italic", color="#9A9A9A")

    # ── legend (bottom band, clear of the U) ─────────────────────────────
    lx, ly, dy = 3.6, -0.2, 0.62
    items = [
        (BLUE,   "Conv 3×3, k=3, s=1, p=1, InstanceNorm, LeakyReLU"),
        (YELLOW, "Conv 1×1, k=1, s=1, p=0"),
        (GREEN,  "Max Pooling"),
        (ORANGE, "Upsampling (scaling factor=2), Conv 1×1, InstanceNorm, LeakyReLU"),
    ]
    for i, (c, txt) in enumerate(items):
        y = ly - i * dy
        ax.add_patch(FancyArrowPatch((lx, y), (lx + 0.7, y), arrowstyle="-|>",
                     mutation_scale=16, color=c, lw=3.0))
        ax.text(lx + 0.95, y, txt, ha="left", va="center", fontsize=13)

    ax.set_title("Autoencoder — U-Net backbone without skip connections",
                 fontsize=22, pad=6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
