#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 輔助圖（垂直版）：Core-set vs TypiClust 選樣對比（2×1 子圖，(a) 上 (b) 下）
ρ=10%，扣掉初始隨機 2.5%（只框 AL 自選）。

從 repo root：
    python3 thesis/chapter_5/plot_5_3_umap_pair_vertical_typiclust.py
輸出：thesis/chapter_5/figs/umap/_pair/5_3_umap_pair_coreset_typiclust_p10_vertical.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import (get_embedding, load_selected, CLASS_COLORS, CLASS_MARKERS,  # noqa: E402
                           NAME_ABBR, REPO)
from plot_5_3_umap_pair import draw_panel, draw_grid  # noqa: E402

FROZEN = os.path.join(REPO, "SSL", "simclr", "ckpt", "resnet18_simclr_lr0.0002_bs256_ep500.pkl")
OUT = os.path.join(HERE, "figs", "umap", "_pair",
                   "5_3_umap_pair_coreset_typiclust_p10_vertical.png")
PORTION, SEED = 10.0, 42

PANELS = [("a", "coreset", "Core-set"), ("b", "typiclust", "TypiClust")]

# 標號 1–3 沿用 Core-set；此圖從 7 開始（延續前圖 1–6）
ARROWS = [
    # coreset: 3 與 coreset-entropy 圖同位置；7,8,9 為此圖新增
    dict(panel="coreset", num=7, fx=0.075, fy=0.70, tx=0.04, ty=0.88),
    dict(panel="coreset", num=3, fx=0.72, fy=0.51, tx=0.76, ty=0.68),   # 同 coreset-entropy ③
    dict(panel="coreset", num=8, fx=0.55, fy=0.28, tx=0.40, ty=0.20),
    dict(panel="coreset", num=9, fx=0.76, fy=0.15, tx=0.82, ty=0.035),
]
# TypiClust 紅框 10–13
BOXES = [
    dict(panel="typiclust", num=13, fx=0.28, fy=0.70, bw=0.20, bh=0.20),
    dict(panel="typiclust", num=10, fx=0.315, fy=0.32, bw=0.165, bh=0.205),
    dict(panel="typiclust", num=11, fx=0.515, fy=0.495, bw=0.22, bh=0.18),
    dict(panel="typiclust", num=12, fx=0.91, fy=0.21, bw=0.1, bh=0.14),
]


def overlay_annotations(axd):
    numcirc = dict(boxstyle="circle,pad=0.30", fc="white", ec="black", lw=1.8)
    numcirc_red = dict(boxstyle="circle,pad=0.30", fc="white", ec="red", lw=1.8)
    for a in ARROWS:
        if a["panel"] not in axd:
            continue
        ax = axd[a["panel"]]
        ax.annotate(str(a["num"]), xy=(a["fx"], a["fy"]), xytext=(a["tx"], a["ty"]),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=19, fontweight="bold", ha="center", va="center",
                    bbox=numcirc, zorder=10,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, shrinkA=12, shrinkB=2))
    for b in BOXES:
        if b["panel"] not in axd:
            continue
        ax = axd[b["panel"]]
        ax.add_patch(Rectangle((b["fx"] - b["bw"] / 2, b["fy"] - b["bh"] / 2), b["bw"], b["bh"],
                               transform=ax.transAxes, fill=False, edgecolor="red",
                               linewidth=2.6, zorder=9))
        ax.text(b["fx"] - b["bw"] / 2, b["fy"] + b["bh"] / 2, str(b["num"]),
                transform=ax.transAxes, fontsize=19, fontweight="bold", color="red",
                ha="center", va="center", bbox=numcirc_red, zorder=10)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    emb, labels, classes = get_embedding(FROZEN, "cuda:9", method="umap", split="train")
    fig, axes = plt.subplots(2, 1, figsize=(12, 19))
    axd = {}
    shared_handles = None
    for i, (ax, (letter, strat, title)) in enumerate(zip(axes, PANELS)):
        handles = draw_panel(ax, strat, title, letter, emb, labels, classes, show_ylabel=True)
        axd[strat] = ax
        if shared_handles is None:
            shared_handles = handles
    overlay_annotations(axd)
    if args.grid:
        for ax in axes:
            draw_grid(ax)

    fig.legend(handles=shared_handles, fontsize=18, framealpha=0.95,
               loc="lower center", ncol=len(shared_handles),
               bbox_to_anchor=(0.5, 0.01), handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.07, hspace=0.18)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
