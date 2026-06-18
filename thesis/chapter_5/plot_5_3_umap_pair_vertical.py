#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 輔助圖（垂直版）：Core-set vs Entropy 選樣對比（2×1 子圖，(a) 上 (b) 下）
同一份標註 ARROWS / BOXES，座標 axes-fraction 故不受版型影響。

從 repo root：
    python3 thesis/chapter_5/plot_5_3_umap_pair_vertical.py
輸出：thesis/chapter_5/figs/umap/_pair/5_3_umap_pair_coreset_entropy_p10_vertical.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import (get_embedding, load_selected, CLASS_COLORS, CLASS_MARKERS,  # noqa: E402
                           NAME_ABBR, REPO)
from plot_5_3_umap_pair import (PANELS, ARROWS, BOXES, FROZEN, PORTION, SEED,  # noqa: E402
                                draw_panel, overlay_annotations, draw_grid)

OUT = os.path.join(HERE, "figs", "umap", "_pair",
                   "5_3_umap_pair_coreset_entropy_p10_vertical.png")


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

    # 共享類別 legend 置底部中央
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
