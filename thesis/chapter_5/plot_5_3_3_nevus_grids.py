#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 Nevus 兩處區塊（C/D）代表性 OCT 排版圖
=============================================
每區 6 張 → 3 列 × 2 欄，標 (a)–(f)，沿用 plot_representative_figures.py 的 style。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_3_nevus_grids.py
"""
import os, sys
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import REPO
from plot_representative_figures import make_figure
from plot_5_3_3_nevus_selected import REGIONS

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class", "train")
OUT_DIR = os.path.join(HERE, "figs", "umap", "representative")


def main():
    ds = datasets.ImageFolder(DATA_DIR)
    paths = [p for p, _ in ds.samples]
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, ids in REGIONS.items():
        img_paths = [paths[i] for i in ids]
        out = os.path.join(OUT_DIR, f"5_3_3_nevus_region_{tag}.png")
        make_figure(img_paths, out)
        print(f"[saved] Region {tag} ({len(img_paths)} imgs) → {out}")


if __name__ == "__main__":
    main()
