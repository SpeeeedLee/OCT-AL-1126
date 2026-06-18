#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 輔助圖：Margin vs Core-set 選樣對比（1×2 子圖 (a)/(b)）
=============================================================
左 (a) = Margin、右 (b) = Core-set，皆 ρ=10%、box 標註、**扣掉初始隨機 2.5%**（只框 AL 自選 n=152）。
畫布 = frozen SimCLR 特徵（與 Fig. 5-26 同一個 UMAP，故位置一致）。

另以 **紅色框（red box）** 與 **黑色箭頭（arrow）** 標註兩法選樣差異較大的區塊，
編號 1–6 **兩子圖共用、不重複**，方便正文逐點討論。座標用 axes-fraction（0–1），易微調。

從 repo root：
    python3 thesis/chapter_5/plot_5_3_umap_pair.py
輸出：thesis/chapter_5/figs/umap/_pair/5_3_umap_pair_margin_coreset_p10.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import (get_embedding, load_selected, CLASS_COLORS, CLASS_MARKERS,  # noqa: E402
                           NAME_ABBR, REPO)

FROZEN = os.path.join(REPO, "SSL", "simclr", "ckpt", "resnet18_simclr_lr0.0002_bs256_ep500.pkl")
OUT = os.path.join(HERE, "figs", "umap", "_pair", "5_3_umap_pair_coreset_entropy_p10.png")
PORTION, SEED = 10.0, 42

# 子圖：(letter, strategy, title)
PANELS = [("a", "coreset", "Core-set"), ("b", "entropy", "Entropy")]

# ---- 標註（座標 = axes fraction 0~1；x:左0右1, y:下0上1）----
# type: "arrow"(黑箭頭) / "box"(紅框)
# panel: "margin" / "coreset"
# arrow: (fx,fy)=箭頭指向點, (tx,ty)=數字(箭尾)位置
# box:   (fx,fy)=框中心, (bw,bh)=框寬高
ARROWS = [
    # —— Core-set：選「遠/稀疏」之點（依 user 紫色箭頭定位）——
    dict(panel="coreset", num=1, fx=0.24, fy=0.95, tx=0.10, ty=0.90),   # 頂端孤立藍色(Nevus)群
    dict(panel="coreset", num=2, fx=0.61, fy=0.14, tx=0.63, ty=0.04),   # 中下部紫色(Psoriasis)選很多
    dict(panel="coreset", num=3, fx=0.72, fy=0.51, tx=0.76, ty=0.68),   # 中右狹長線完整被選（label右、箭頭向左）
    # —— Entropy：選「類別交界 / decision boundary」之點 ——
    dict(panel="entropy", num=4, fx=0.91, fy=0.15, tx=0.97, ty=0.05),   # 右下 Healthy 團邊緣（水平向左）
    dict(panel="entropy", num=5, fx=0.472, fy=0.6425, tx=0.48, ty=0.87),   # 左上label→右下邊界混合區
]
BOXES = [
    # —— Entropy：中部綠色密集處「沒選到」一大塊（依 user 重框）——
    dict(panel="entropy", num=6, fx=0.487, fy=0.46, bw=0.123, bh=0.11),
]


def draw_panel(ax, strategy, title, letter, emb, labels, classes, show_ylabel=True):
    n = len(classes)
    disp = [NAME_ABBR.get(c, c) for c in classes]
    sel_idx = load_selected(strategy, PORTION, SEED, "cumulative", exclude_init=True)
    sel = np.zeros(len(labels), dtype=bool)
    if sel_idx:
        sel[np.array(sel_idx)] = True
    counts = np.array([int((labels == ci).sum()) for ci in range(n)])
    order = list(np.argsort(counts)[::-1])

    PT = 70
    for ci in range(n):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=PT, c=CLASS_COLORS[ci], marker=CLASS_MARKERS[ci],
                   alpha=0.65, linewidths=0, zorder=2)
    if sel.any():
        ax.scatter(emb[sel, 0], emb[sel, 1], s=190, facecolors="none", edgecolors="black",
                   linewidths=1.4, marker="s", zorder=4)
    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=18, label=disp[ci]) for ci in order]
    sel_h = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
                    markeredgecolor="black", markersize=14, label=f"Selected (n={int(sel.sum())})")]

    # "Selected" 留在各子圖右上角
    ax.legend(handles=sel_h, fontsize=15, framealpha=0.95, loc="upper right")

    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    if show_ylabel:
        ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_title(f"({letter}) {title}   (ρ={PORTION:g}%)", fontsize=26, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    return class_h  # 類別 handles 交給 fig.legend()


def overlay_annotations(axd):
    numcirc = dict(boxstyle="circle,pad=0.30", fc="white", ec="black", lw=1.8)
    numcirc_red = dict(boxstyle="circle,pad=0.30", fc="white", ec="red", lw=1.8)
    for a in ARROWS:
        ax = axd[a["panel"]]
        ax.annotate(str(a["num"]), xy=(a["fx"], a["fy"]), xytext=(a["tx"], a["ty"]),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=19, fontweight="bold", ha="center", va="center",
                    bbox=numcirc, zorder=10,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2, shrinkA=12, shrinkB=2))
    for b in BOXES:
        ax = axd[b["panel"]]
        ax.add_patch(Rectangle((b["fx"] - b["bw"] / 2, b["fy"] - b["bh"] / 2), b["bw"], b["bh"],
                               transform=ax.transAxes, fill=False, edgecolor="red",
                               linewidth=2.6, zorder=9))
        ax.text(b["fx"] - b["bw"] / 2, b["fy"] + b["bh"] / 2, str(b["num"]),
                transform=ax.transAxes, fontsize=19, fontweight="bold", color="red",
                ha="center", va="center", bbox=numcirc_red, zorder=10)


def draw_grid(ax):
    """除錯用：畫 axes-fraction 0~1 每 0.1 一格的格線與座標標籤，方便回報精準座標。"""
    for f in np.arange(0.0, 1.001, 0.1):
        ax.plot([f, f], [0, 1], color="0.6", lw=0.6, ls=":", transform=ax.transAxes, zorder=8)
        ax.plot([0, 1], [f, f], color="0.6", lw=0.6, ls=":", transform=ax.transAxes, zorder=8)
        ax.text(f, 1.005, f"{f:.1f}", transform=ax.transAxes, fontsize=8, color="blue",
                ha="center", va="bottom", zorder=11)
        ax.text(-0.005, f, f"{f:.1f}", transform=ax.transAxes, fontsize=8, color="blue",
                ha="right", va="center", zorder=11)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="疊 axes-fraction 座標格線（除錯/定位用）")
    args = ap.parse_args()

    emb, labels, classes = get_embedding(FROZEN, "cuda:9", method="umap", split="train")
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axd = {}
    shared_handles = None
    for i, (ax, (letter, strat, title)) in enumerate(zip(axes, PANELS)):
        handles = draw_panel(ax, strat, title, letter, emb, labels, classes, show_ylabel=(i == 0))
        axd[strat] = ax
        if shared_handles is None:
            shared_handles = handles  # 兩圖類別相同，取第一張即可
    overlay_annotations(axd)
    if args.grid:
        for ax in axes:
            draw_grid(ax)
    # 共享類別 legend 置底部中央，一排排開
    fig.legend(handles=shared_handles, fontsize=22, framealpha=0.95,
               loc="lower center", ncol=len(shared_handles),
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.0)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.11, wspace=0.04)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
