#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 Healthy 代表性影像 — 標註圖（第一版：標出手選的 idx）
=============================================================
在 frozen-SimCLR base UMAP（同 plot_base 的 (10,9) 尺寸/風格）上，把手動挑選的
Healthy idx 用「醒目空心圈 + idx 數字」標出（不用黑框），方便核對與分組。

idx 空間 = ImageFolder(seven_class/train).samples，與 UMAP cache 對齊。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_3_healthy_selected.py
    python3 thesis/chapter_5/plot_5_3_3_healthy_selected.py --groups   # 三區分色（給 --g1/--g2/--g3）
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

# 使用者 hover 後手選的 30 張 Healthy
SELECTED = [661, 671, 673, 674, 677, 678, 679, 680, 695, 714, 884, 887, 894,
            906, 940, 1009, 1010, 1011, 1092, 1097, 1130, 1133, 1139, 1182,
            1205, 1209, 1248, 1260, 1286, 1291]

# 三大區塊（A/B/C）：用紅色大圈標出大致位置
REGIONS = {
    "A": [661, 671, 673, 674, 677, 678, 679, 680, 695, 714],
    "B": [884, 887, 894, 906, 940, 1009, 1010, 1011, 1092, 1097],
    "C": [1130, 1133, 1139, 1182, 1205, 1209, 1248, 1260, 1286, 1291],
}
REGION_RADIUS = 0.95     # data units（視覺上的大紅圈半徑）
REGION_RED = "#D62728"


def plot(emb, labels, classes, sel, out_path):
    disp = [NAME_ABBR.get(c, c) for c in classes]
    n = len(classes)
    counts = np.array([int((labels == ci).sum()) for ci in range(n)])
    order = list(np.argsort(counts)[::-1])

    fig, ax = plt.subplots(figsize=(10, 9))
    # 背景：全部點，類別著色（同 plot_base）
    for ci in range(n):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=90, c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
                   alpha=0.55, linewidths=0, zorder=2)

    # 選取點：黑色空心圈（不標 idx）
    pts = emb[np.array(sel)]
    ax.scatter(pts[:, 0], pts[:, 1], s=230, facecolors="none", edgecolors="#111111",
               linewidths=2.2, marker="o", zorder=4)

    # 三大區塊：紅色大圈 + 紅色 A/B/C
    for tag, ids in REGIONS.items():
        p = emb[ids]
        cx, cy = p[:, 0].mean(), p[:, 1].mean()
        ax.add_patch(Circle((cx, cy), REGION_RADIUS, fill=False, edgecolor=REGION_RED,
                            linewidth=3.0, zorder=5, clip_on=False))
        ax.annotate(tag, (cx, cy + REGION_RADIUS), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=30, fontweight="bold", color=REGION_RED, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # 單一 legend → 右上角：7 類 + 最後一列 Selected Healthy
    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=11, label=disp[ci]) for ci in order]
    sel_h = Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                   markeredgecolor="#111111", markeredgewidth=2.2, markersize=13,
                   label=f"Selected Healthy (n={len(sel)})")
    ax.legend(handles=class_h + [sel_h], fontsize=13, framealpha=0.95, loc="upper right")

    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.97, bottom=0.09)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}  (n={len(sel)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:6")
    args = ap.parse_args()

    emb, labels, classes = get_embedding(FROZEN, args.device, method="umap", split="train")
    out_dir = os.path.join(HERE, "figs", "umap", "representative")
    out = os.path.join(out_dir, "5_3_3_healthy_selected.png")
    plot(emb, labels, classes, SELECTED, out)


if __name__ == "__main__":
    main()
