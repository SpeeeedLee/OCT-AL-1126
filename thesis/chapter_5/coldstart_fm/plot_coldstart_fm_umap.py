#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
§5.2 Cold-Start FM Selection — UMAP 視覺化
==========================================
用 θ²-SimCLR 的 UMAP embedding，把 FM 主動選取的初始標注集
（simclr__resnet18 labeled_ids，2.5% = 51 張）以黑色方框標出。

style 與 §5.3 plot_5_3_umap.py 完全一致（figsize, legend, 字體等）。

從 repo root 執行：
    python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm_umap.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

# 複用 §5.3 的 plot_umap / plot_base
sys.path.insert(0, os.path.join(REPO, "thesis", "chapter_5"))
from plot_5_3_umap import plot_umap, NAME_ABBR   # noqa: E402

UMAP_CACHE  = os.path.join(REPO, "thesis", "chapter_5", "umap_cache",
                            "resnet18_simclr_lr0.0002_bs256_ep500_umap.npz")
LABELED_IDS = os.path.join(HERE, "labeled_ids", "simclr__resnet18.json")
OUT_DIR     = os.path.join(REPO, "thesis", "chapter_5", "figs")


def main():
    # 載入 θ² UMAP 2D embedding
    d = np.load(UMAP_CACHE, allow_pickle=True)
    emb     = d["emb"]       # (2032, 2)
    labels  = d["labels"]    # (2032,)
    classes = list(d["classes"])
    print(f"UMAP loaded: {emb.shape}, classes={classes}")

    # 載入 FM 選取的 2.5% (51 張) 索引
    ids = json.load(open(LABELED_IDS))
    selected = list(ids["2.5"]["cumulative"])
    print(f"FM selected: {len(selected)} images @ ρ=2.5%")

    # 畫圖（box highlight = 黑色空心方框）
    out = os.path.join(OUT_DIR, "5_2_coldstart_fm_umap_selection.png")
    plot_umap(
        emb, labels, classes,
        selected_idx=selected,
        strategy="",
        portion=2.5,
        seed=None,
        out_path=out,
        highlight="box",
        method="umap",
        title=r"$\theta^2_{\mathrm{SimCLR}}$ Guided Initial $\rho$=2.5% Data Selection",
    )


if __name__ == "__main__":
    main()
