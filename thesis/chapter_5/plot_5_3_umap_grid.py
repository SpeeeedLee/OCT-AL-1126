#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3　特徵空間隨 finetune portion 演變：2×3 UMAP/t-SNE 拼圖
=========================================================

六格 = 在 ρ∈{0,15,30,50,70,100}% finetune 的模型，各自對 train（或 test）set 的特徵空間。
  - ρ=0%   = 完全沒 finetune = frozen SimCLR θ² backbone（.pkl）
  - ρ=100% = simclr_p100_4x.pth
  - ρ=15/30/50/70% = simclr_p{ρ}_4x.pth（由 finetune_full_model.py 產生；缺的格子留空）
子圖標 (a)–(f)，(a) ρ=0% ... (f) ρ=100%。每格只畫類別著色（無 AL 選取標註）。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_umap_grid.py --method umap --split train --device cuda:6
    python3 thesis/chapter_5/plot_5_3_umap_grid.py --method umap --split test  --device cuda:6
缺 checkpoint 的格子會顯示「(x) ρ=..%  (pending)」，補完 checkpoint 後重跑即補上。
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import (get_embedding, CLASS_COLORS, CLASS_MARKERS, NAME_ABBR,  # noqa: E402
                           CKPT_DIR, OUT_DIR, REPO)

FROZEN_SIMCLR = os.path.join(REPO, "SSL", "simclr", "ckpt",
                             "resnet18_simclr_lr0.0002_bs256_ep500.pkl")

# (rho, ckpt)；順序 = (a)..(f)
PANELS = [
    (0,   FROZEN_SIMCLR),
    (15,  os.path.join(CKPT_DIR, "simclr_p15_4x.pth")),
    (30,  os.path.join(CKPT_DIR, "simclr_p30_4x.pth")),
    (50,  os.path.join(CKPT_DIR, "simclr_p50_4x.pth")),
    (70,  os.path.join(CKPT_DIR, "simclr_p70_4x.pth")),
    (100, os.path.join(CKPT_DIR, "simclr_p100_4x.pth")),
]
LETTERS = "abcdef"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["umap", "tsne"], default="umap")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    axname = "t-SNE" if args.method == "tsne" else "UMAP"
    pt_size = 12 if args.split == "train" else 42   # 約原本 1.5 倍
    n_cls = len(CLASS_COLORS)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    classes_ref, order_ref = None, None

    for i, (rho, ck) in enumerate(PANELS):
        ax = axes.flat[i]
        letter = LETTERS[i]
        if not os.path.isfile(ck):
            ax.text(0.5, 0.5, f"({letter}) ρ={rho}%\n(checkpoint pending)",
                    ha="center", va="center", fontsize=24, color="0.4",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(1.5)
            continue
        emb, labels, classes = get_embedding(ck, args.device, method=args.method,
                                             split=args.split, recompute=args.recompute)
        if classes_ref is None:
            classes_ref = classes
            counts = np.array([int((labels == ci).sum()) for ci in range(n_cls)])
            order_ref = list(np.argsort(counts)[::-1])
        for ci in range(n_cls):
            m = labels == ci
            ax.scatter(emb[m, 0], emb[m, 1], s=pt_size, c=CLASS_COLORS[ci],
                       marker=CLASS_MARKERS[ci], alpha=0.7, linewidths=0)
        ax.set_title(f"({letter}) ρ={rho}%", fontsize=28, pad=10)
        # 每個子圖各自的軸標：六張是各自獨立的 UMAP，兩維物理意義不同、不可共用
        ax.set_xlabel(f"{axname}-1", fontsize=17)
        ax.set_ylabel(f"{axname}-2", fontsize=17)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(1.5)

    # 每格自己的軸標（上面已畫）；底部留共用 legend
    fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.13, wspace=0.14, hspace=0.24)
    if classes_ref is not None:
        disp = [NAME_ABBR.get(c, c) for c in classes_ref]
        handles = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                          markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                          markersize=18, label=disp[ci]) for ci in order_ref]
        fig.legend(handles=handles, loc="lower center", ncol=n_cls, fontsize=22,
                   framealpha=0.95, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2,
                   handletextpad=0.4)
    out = os.path.join(args.out_dir, args.method, "_grid",
                       f"5_3_{args.method}_grid_{args.split}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
