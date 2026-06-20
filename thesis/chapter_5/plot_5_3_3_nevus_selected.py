#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 Nevus 代表性影像 — 標註圖（第一步：標出手選 idx）
========================================================
在 frozen-SimCLR base UMAP（(10,9) 尺寸/風格）上，把手選的 Nevus idx 用黑色空心圈
標出（無 idx 數字）。之後再依指示加紅圈分兩處（與 Healthy A/B/C 流程一致）。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_3_nevus_selected.py
"""
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import get_embedding, CLASS_COLORS, CLASS_MARKERS, NAME_ABBR, REPO

FROZEN = os.path.join(REPO, "SSL", "simclr", "ckpt",
                      "resnet18_simclr_lr0.0002_bs256_ep500.pkl")

SELECTED = [244, 254, 266, 267, 279, 308, 330, 383, 399, 402, 403, 404,
            406, 407, 408, 409, 410, 411, 437, 489]
SEL_NAME = "Nevus"

# 兩處區塊（承接 Healthy A/B/C）：D = 左上獨立群；E = 與 Healthy 較接近群（各 10 點）
REGIONS = {
    "D": [399, 402, 403, 404, 406, 407, 408, 409, 410, 411],
    "E": [244, 254, 266, 267, 279, 308, 330, 383, 437, 489],
}
REGION_RED = "#D62728"


def plot(emb, labels, classes, sel, out_path):
    disp = [NAME_ABBR.get(c, c) for c in classes]
    n = len(classes)
    counts = np.array([int((labels == ci).sum()) for ci in range(n)])
    order = list(np.argsort(counts)[::-1])

    fig, ax = plt.subplots(figsize=(10, 9))
    for ci in range(n):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=90, c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
                   alpha=0.55, linewidths=0, zorder=2)

    pts = emb[np.array(sel)]
    ax.scatter(pts[:, 0], pts[:, 1], s=230, facecolors="none", edgecolors="#111111",
               linewidths=2.2, marker="o", zorder=4)

    # 兩處紅色大圈 + 紅色 C/D（每區自適應半徑，把整群圈進去）
    for tag, ids in REGIONS.items():
        p = emb[ids]
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        r = max(np.hypot(p[:, 0] - cx, p[:, 1] - cy).max() + 0.45, 0.75)
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=REGION_RED,
                            linewidth=3.0, zorder=5, clip_on=False))
        ax.annotate(tag, (cx, cy + r), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=30, fontweight="bold",
                    color=REGION_RED, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=11, label=disp[ci]) for ci in order]
    sel_h = Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="#111111", markeredgewidth=2.2, markersize=13,
                   label=f"Selected {SEL_NAME} (n={len(sel)})")
    ax.legend(handles=class_h + [sel_h], fontsize=13, framealpha=0.95, loc="upper right")

    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.07)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}  (n={len(sel)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:6")
    args = ap.parse_args()
    emb, labels, classes = get_embedding(FROZEN, args.device, method="umap", split="train")
    # 核對所選都是 Nevus
    cls_of = {NAME_ABBR.get(c, c): i for i, c in enumerate(classes)}
    bad = [i for i in SELECTED if labels[i] != cls_of["Nevus"]]
    if bad:
        print(f"[warn] 非 Nevus 的 idx: {[(i, classes[labels[i]]) for i in bad]}")
    out = os.path.join(HERE, "figs", "umap", "representative", "5_3_3_nevus_selected.png")
    plot(emb, labels, classes, SELECTED, out)


if __name__ == "__main__":
    main()
