#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense plain UMAP: same embedding/size as thesis/chapter_5 plot_5_3_umap.py
`--base`, but every point is a single black-grey circle and there is NO
lesion-type legend.

Reuses the cached 2D embedding (thesis/chapter_5/umap_cache), so no GPU/model
needed. Matches the original base figure: figsize (10,9), s=90, dpi 300.

Run from repo root:
    python3 thesis/oral_defense/plot_umap_grey.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# θ²-SimCLR feature extractor = the frozen SimCLR backbone (same space as the
# §5.2 cold-start cluster figure), NOT the finetuned p100 classifier.
CACHE = os.path.join(REPO, "thesis", "chapter_5", "umap_cache",
                     "resnet18_simclr_lr0.0002_bs256_ep500_umap.npz")
OUT = os.path.join(os.path.dirname(__file__), "figs", "umap_grey.png")


def main():
    d = np.load(CACHE, allow_pickle=True)
    emb = d["emb"]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb[:, 0], emb[:, 1], s=90, c="#4D4D4D", marker="o",
               alpha=0.7, linewidths=0, zorder=2)
    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}  ({len(emb)} points)")


if __name__ == "__main__":
    main()
