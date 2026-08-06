#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense variant of §5.2 cold-start FM cluster UMAP (Version B), but with
ONLY the 3 cluster colors + legend. Every point is a circle; no per-lesion-type
marker shapes and no lesion-type legend. Selected initial ρ=2.5% kept as black
squares.

Reuses the exact clustering from
thesis/chapter_5/coldstart_fm/plot_coldstart_fm_umap_clusters.py
(t-SNE 2D + KMeans k=3, seed=0). Same figure size as the original Version B.

Run from repo root:
    python3 thesis/oral_defense/plot_coldstart_clusters_only.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFM = os.path.join(REPO, "thesis", "chapter_5", "coldstart_fm")
sys.path.insert(0, CFM)
import plot_coldstart_fm_umap_clusters as C   # reuse load_data / clustering / colors

OUT = os.path.join(os.path.dirname(__file__), "figs", "coldstart_clusters_only.png")


def main():
    emb, labels, classes, hi_emb, indices, selected = C.load_data()
    all_cluster_labels, cluster_labels_51, sel_rows = C.get_cluster_assignments(
        hi_emb, indices, selected, k=3)

    fig, ax = plt.subplots(figsize=(10, 9))

    # all points: circle, colored ONLY by cluster
    for k in range(3):
        m = all_cluster_labels == k
        ax.scatter(emb[m, 0], emb[m, 1], s=55, c=C.CLUSTER_COLORS[k], marker="o",
                   alpha=0.65, linewidths=0, zorder=2)

    # selected initial ρ=2.5% : black open squares
    sel_emb = emb[selected]
    ax.scatter(sel_emb[:, 0], sel_emb[:, 1], s=230, facecolors="none",
               edgecolors="black", linewidths=1.8, marker="s", zorder=5)

    handles = [Line2D([0], [0], marker="o", linestyle="none",
                      markerfacecolor=C.CLUSTER_COLORS[k], markeredgecolor="none",
                      markersize=12, label=f"Cluster {k+1}") for k in range(3)]
    handles.append(Line2D([0], [0], marker="s", linestyle="none",
                          markerfacecolor="none", markeredgecolor="black",
                          markeredgewidth=1.8, markersize=11,
                          label=r"Selected ($\rho$=2.5%)"))
    ax.legend(handles=handles, fontsize=15, framealpha=0.95, loc="upper right")

    ax.set_title(r"$\theta^2_{\mathrm{SimCLR}}$ Guided Initial $\rho$=2.5% Data Selection",
                 fontsize=22, pad=12)
    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
