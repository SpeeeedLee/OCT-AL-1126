#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3.3 Healthy 三大區塊（A/B/C）代表性 OCT 排版圖
==================================================
每區 10 張 → 5 列 × 2 欄，標 (a)–(j)，沿用 plot_representative_figures.py 的
論文用 style（FIG_W=6.3、A4 文字寬、黑底白字標籤）。

idx → OCT 檔案 = ImageFolder(seven_class/train).samples[idx]，與 UMAP 標註圖同一空間。

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_3_healthy_grids.py
"""
import os, sys
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import REPO
from plot_representative_figures import make_figure
from plot_5_3_3_healthy_selected import REGIONS

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class", "train")
OUT_DIR = os.path.join(HERE, "figs", "umap", "representative")


def main():
    ds = datasets.ImageFolder(DATA_DIR)
    paths = [p for p, _ in ds.samples]
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, ids in REGIONS.items():
        img_paths = [paths[i] for i in ids]
        out = os.path.join(OUT_DIR, f"5_3_3_healthy_region_{tag}.png")
        make_figure(img_paths, out)
        print(f"[saved] Region {tag} ({len(img_paths)} imgs) → {out}")


if __name__ == "__main__":
    main()
