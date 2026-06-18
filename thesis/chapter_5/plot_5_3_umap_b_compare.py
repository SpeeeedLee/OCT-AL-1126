#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3 輔助圖：Margin ρ=22.5% seed=24，比較不同 batch size 的選取模式（2×1 垂直）
(a) b=2.5%（細粒度，8 輪迭代）
(b) b=20%（粗粒度，1 輪大批）
初始 b₀=2.5%（51 張隨機）不標出，只框顯 AL 自選部分。

從 repo root：
    python3 thesis/chapter_5/plot_5_3_umap_b_compare.py
輸出：thesis/chapter_5/figs/umap/_pair/5_3_umap_b_compare_margin_p22.5_seed24.png
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import get_embedding, CLASS_COLORS, CLASS_MARKERS, NAME_ABBR, REPO  # noqa: E402
from plot_5_3_umap_pair import draw_grid  # noqa: E402

FROZEN = os.path.join(REPO, "SSL", "simclr", "ckpt", "resnet18_simclr_lr0.0002_bs256_ep500.pkl")
OUT = os.path.join(HERE, "figs", "umap", "_pair",
                   "5_3_umap_b_compare_margin_p22.5_seed24.png")

PORTION, SEED = 22.5, 24

# 兩條軌跡的 labeled_ids 路徑；letter 也用作 axd key
TRAJ = [
    dict(letter="a", title="Margin (ρ=22.5%, b=2.5%)",
         path=os.path.join(REPO, "classification", "exp_results",
                           "classification_hard", "AL_simclr", "labeled_ids",
                           "margin_seed24_bs16.json")),
    dict(letter="b", title="Margin (ρ=22.5%, b=20%)",
         path=os.path.join(REPO, "classification", "exp_results",
                           "ch5_b_ablation", "b_20", "classification_hard",
                           "AL_simclr", "labeled_ids", "margin_seed24_bs16.json")),
]

# ── annotations ─────────────────────────────────────────────────────────────
# panel: "a"=上圖(b=2.5%), "b"=下圖(b=20%)
# arrow: (fx,fy)=箭頭指向點(axes fraction), (tx,ty)=標號文字位置
# box:   (fx,fy)=框中心, (bw,bh)=框寬高
ARROWS = [
    dict(panel="a", num=16, fx=0.22, fy=0.81, tx=0.15, ty=0.92),   # 暫放左下角，user 自行調整
]
BOXES = [
    dict(panel="a", num=14,  fx=0.10, fy=0.41, bw=0.14, bh=0.20),  # 上圖左下角 box①，user 自行調整
    dict(panel="b", num=14,  fx=0.10, fy=0.41, bw=0.14, bh=0.20),  # 上圖左下角 box①，user 自行調整

    dict(panel="a", num=15,  fx=0.155, fy=0.235, bw=0.11, bh=0.122),  # 上圖左下角 box②，user 自行調
    dict(panel="b", num=15,  fx=0.155, fy=0.235, bw=0.11, bh=0.122),  # 上圖左下角 box②，user 自行調

    dict(panel="a", num=12, fx=0.91,  fy=0.21,  bw=0.10,  bh=0.14),   # 同 typiclust 圖 ⑫
    dict(panel="b", num=12, fx=0.91,  fy=0.21,  bw=0.10,  bh=0.14),   # 同 typiclust 圖 ⑫
 ]


def load_ids(path, portion):
    """讀 labeled_ids 並扣掉初始 b₀ 隨機池，只留 AL 自選影像。"""
    d = json.load(open(path))
    pk = str(float(portion))
    cum = list(d[pk]["cumulative"])
    init_key = min(d.keys(), key=float)
    init_set = set(d[init_key].get("cumulative", []))
    sel = [i for i in cum if i not in init_set]
    print(f"  {os.path.basename(os.path.dirname(os.path.dirname(path)))} "
          f"ρ={portion}%: cumulative={len(cum)}, exclude_init={len(init_set)}, AL_only={len(sel)}")
    return sel


def draw_panel(ax, traj, emb, labels, classes):
    sel_idx = load_ids(traj["path"], PORTION)
    sel = np.zeros(len(labels), dtype=bool)
    sel[np.array(sel_idx)] = True

    counts = np.array([(labels == ci).sum() for ci in range(len(classes))])
    order = list(np.argsort(counts)[::-1])
    disp = [NAME_ABBR.get(c, c) for c in classes]

    for ci in range(len(classes)):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=70, c=CLASS_COLORS[ci],
                   marker=CLASS_MARKERS[ci], alpha=0.65, linewidths=0, zorder=2)
    if sel.any():
        ax.scatter(emb[sel, 0], emb[sel, 1], s=190, facecolors="none",
                   edgecolors="black", linewidths=1.4, marker="s", zorder=4)

    sel_h = [Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
                    markeredgecolor="black", markersize=14,
                    label=f"Selected (n={int(sel.sum())})")]
    ax.legend(handles=sel_h, fontsize=15, framealpha=0.95, loc="upper right")

    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_title(f"({traj['letter']}) {traj['title']}", fontsize=24, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)

    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=18, label=disp[ci]) for ci in order]
    return class_h


def overlay_annotations(axd):
    numcirc     = dict(boxstyle="circle,pad=0.30", fc="white", ec="black", lw=1.8)
    numcirc_red = dict(boxstyle="circle,pad=0.30", fc="white", ec="red",   lw=1.8)
    for a in ARROWS:
        if a["panel"] not in axd:
            continue
        ax = axd[a["panel"]]
        ax.annotate(str(a["num"]), xy=(a["fx"], a["fy"]), xytext=(a["tx"], a["ty"]),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=19, fontweight="bold", ha="center", va="center",
                    bbox=numcirc, zorder=10,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2,
                                   shrinkA=12, shrinkB=2))
    for b in BOXES:
        if b["panel"] not in axd:
            continue
        ax = axd[b["panel"]]
        ax.add_patch(Rectangle(
            (b["fx"] - b["bw"] / 2, b["fy"] - b["bh"] / 2), b["bw"], b["bh"],
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
    for ax, traj in zip(axes, TRAJ):
        handles = draw_panel(ax, traj, emb, labels, classes)
        axd[traj["letter"]] = ax
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
