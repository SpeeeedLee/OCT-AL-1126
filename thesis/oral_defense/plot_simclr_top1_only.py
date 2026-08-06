#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense version of the (b) Instance-Discrimination Accuracy panel from
thesis/chapter_4/plot_simclr_pretrain_curve.py (compare mode), but with ONLY
the Top-1 curves (Top-5 removed) and only the Random/ImageNet legend (no
Top-1/Top-5 linestyle legend). Same fonts/figure proportions.

Run from repo root:
    python3 thesis/oral_defense/plot_simclr_top1_only.py
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from plot_simclr_pretrain_curve import (load_history, style_ax, C_T1, C_T2,
                                        FONT_LABEL, FONT_TICK, FONT_TITLE, FONT_LEGEND)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JSON_DIR = os.path.join(ROOT, "SSL", "simclr", "json")
OUT = os.path.join(os.path.dirname(__file__), "figs", "simclr_top1_only.png")
BS, EP, LR = "256", "500", "0.0002"


def main():
    d1 = load_history("theta1", LR, BS, EP, JSON_DIR)   # random init (orange)
    d2 = load_history("theta2", LR, BS, EP, JSON_DIR)   # imagenet init (blue)

    fig, ax = plt.subplots(figsize=(9, 7))              # = one panel of the (18,7) compare fig
    ax.plot(d1["ep"], d1["top1"], color=C_T1, linewidth=3, label="Random Init")
    ax.plot(d2["ep"], d2["top1"], color=C_T2, linewidth=3, label="ImageNet Init")
    ax.set_ylabel("Contrastive Accuracy (%)", fontsize=FONT_LABEL, labelpad=8)
    ax.set_xlabel("Pretraining Epoch", fontsize=FONT_LABEL, labelpad=8)
    ax.set_title("(b) Instance-Discrimination Accuracy", fontsize=FONT_TITLE, pad=10)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="lower right")
    style_ax(ax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
