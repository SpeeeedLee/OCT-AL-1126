#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
§5.2 Cold-Start FM Selection — UMAP cluster 視覺化（兩種）
=========================================================
Version A: 圓圈版 — 用紅色圓形圈出 k̂=3 個 cluster 位置
Version B: 著色版 — 51 張 selected points 按 cluster 用不同顏色，class marker shape 保留

cluster 分配：用高維 512-dim SimCLR embedding 做 KMeans(k=3)
（與原始選取時一致，原始是對 t-SNE 2D 做，此處用高維版近似，
UMAP 視覺化上的 cluster 分布應與 t-SNE 版高度吻合）

從 repo root 執行：
    python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm_umap_clusters.py
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "thesis", "chapter_5"))
from plot_5_3_umap import NAME_ABBR, CLASS_COLORS, CLASS_MARKERS  # noqa

UMAP_CACHE  = os.path.join(REPO, "thesis", "chapter_5", "umap_cache",
                            "resnet18_simclr_lr0.0002_bs256_ep500_umap.npz")
EMB_CACHE   = os.path.join(HERE, "embeddings", "simclr__resnet18.pt")
LABELED_IDS = os.path.join(HERE, "labeled_ids", "simclr__resnet18.json")
OUT_DIR     = os.path.join(REPO, "thesis", "chapter_5", "figs")

# cluster 顏色（3 個，柔和易區分，與 class color 不衝突）
CLUSTER_COLORS = ["#D62728", "#9467BD", "#17BECF"]   # crimson / purple / teal
CLUSTER_LABELS = ["Cluster 1", "Cluster 2", "Cluster 3"]


def load_data():
    # UMAP 2D
    d = np.load(UMAP_CACHE, allow_pickle=True)
    emb     = d["emb"]          # (2032, 2)
    labels  = d["labels"]       # (2032,)
    classes = list(d["classes"])

    # 高維 512-dim embedding
    pt = torch.load(EMB_CACHE, map_location="cpu", weights_only=False)
    hi_emb  = pt["embeddings"].numpy()   # (2032, 512)
    indices = pt["indices"].numpy()      # (2032,) — 對應 dataset 的絕對 index

    # FM selected @ 2.5%
    ids = json.load(open(LABELED_IDS))
    selected = np.array(ids["2.5"]["cumulative"])   # 51 個絕對 index

    return emb, labels, classes, hi_emb, indices, selected


def get_cluster_assignments(hi_emb, indices, selected, k=3):
    """復現 Levy et al. 原始 pipeline：對全部 embedding 做 t-SNE 2D，再 KMeans(k=3)。
    seed=0 與 select_coldstart.py SEED=0 一致，保證 cluster 分配完全相同。
    回傳全部 2032 點的 cluster label，以及 selected 51 張對應的 label。"""
    from sklearn.manifold import TSNE

    SEED = 0
    idx_map = {int(idx): i for i, idx in enumerate(indices)}

    print("  Running t-SNE 2D on all 2032 embeddings (same as original run)...")
    feats_2d = TSNE(n_components=2, random_state=SEED,
                    init="pca").fit_transform(hi_emb.astype(np.float32))
    print("  t-SNE done.")

    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    all_labels = km.fit_predict(feats_2d)   # (2032,)

    sel_rows = np.array([idx_map[s] for s in selected])
    cluster_labels_51 = all_labels[sel_rows]
    return all_labels, cluster_labels_51, sel_rows


# ── Version A: 紅色圓圈版 ──────────────────────────────────────────────── #

def plot_circle_version(emb, labels, classes, selected, cluster_labels_51,
                        sel_rows, out_path):
    disp = [NAME_ABBR.get(c, c) for c in classes]
    fig, ax = plt.subplots(figsize=(10, 9))

    # 底層：所有點按 class 著色
    for ci, cls in enumerate(classes):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=55, c=CLASS_COLORS[ci],
                   marker=CLASS_MARKERS[ci], alpha=0.55, linewidths=0)

    # 51 張 selected：黑色方框 overlay（與原圖一致）
    sel_emb = emb[selected]
    ax.scatter(sel_emb[:, 0], sel_emb[:, 1], s=220, facecolors="none",
               edgecolors="black", linewidths=1.8, marker="s", zorder=5,
               label="Selected (ρ=2.5%)")

    # 畫 3 個 cluster 圓圈
    for k in range(3):
        mask = cluster_labels_51 == k
        pts  = emb[selected[mask]]           # cluster k 的 selected 點 UMAP 座標
        cent = pts.mean(axis=0)              # cluster 重心（UMAP 2D）

        # 半徑 = 最遠點距重心距離 × 1.15（留白）
        dists = np.linalg.norm(pts - cent, axis=1)
        r = dists.max() * 1.15 if len(dists) > 0 else 0.5

        circle = mpatches.Circle(cent, r, fill=False,
                                 edgecolor=CLUSTER_COLORS[k],
                                 linewidth=2.8, linestyle="-",
                                 zorder=6, alpha=0.9)
        ax.add_patch(circle)

        # cluster 標籤放在圓圈正上方
        ax.text(cent[0], cent[1] + r + 0.15, f"C{k+1}",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=CLUSTER_COLORS[k], zorder=7)

    # Legend — class
    class_h = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                      markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
                      markersize=8, label=disp[ci])
               for ci in range(len(classes))]
    sel_h   = [Line2D([0], [0], marker="s", linestyle="none", markersize=9,
                      markerfacecolor="none", markeredgecolor="black",
                      markeredgewidth=1.8, label=r"Selected ($\rho$=2.5%)")]
    circ_h  = [Line2D([0], [0], color=CLUSTER_COLORS[k], linewidth=2.5,
                      label=f"Cluster {k+1}", linestyle="-")
               for k in range(3)]
    ax.legend(handles=class_h + sel_h + circ_h,
              fontsize=11, framealpha=0.85, loc="lower left",
              ncol=2, handletextpad=0.4, columnspacing=0.8)

    ax.set_title(r"$\theta^2_{\mathrm{SimCLR}}$ Guided Initial $\rho$=2.5% Data Selection"
                 "\n(k̂=3 Clusters, Circle Overlay)",
                 fontsize=13, pad=10)
    ax.set_xlabel("UMAP-1", fontsize=12)
    ax.set_ylabel("UMAP-2", fontsize=12)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── Version B: cluster 著色版 ─────────────────────────────────────────── #

def plot_cluster_color_version(emb, labels, classes, selected,
                               all_cluster_labels, out_path):
    """全部 2032 點按 cluster 著色 + class marker；51 selected 疊加黑色空心方框。"""
    disp = [NAME_ABBR.get(c, c) for c in classes]
    fig, ax = plt.subplots(figsize=(10, 9))

    # 全部點：cluster 顏色 + class marker shape
    for ci in range(len(classes)):
        for k in range(3):
            m = (labels == ci) & (all_cluster_labels == k)
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=55,
                       c=CLUSTER_COLORS[k],
                       marker=CLASS_MARKERS[ci],
                       alpha=0.65, linewidths=0, zorder=2)

    # 51 張 selected：疊加黑色空心方框
    sel_emb = emb[selected]
    ax.scatter(sel_emb[:, 0], sel_emb[:, 1], s=230,
               facecolors="none", edgecolors="black",
               linewidths=1.8, marker="s", zorder=5)

    # Legend col1: Cluster 1/2/3 + Selected (4 items), pad to 7 with invisibles
    col1 = [Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=CLUSTER_COLORS[k], markeredgecolor="none",
                   markersize=10, label=f"Cluster {k+1}")
            for k in range(3)]
    col1 += [Line2D([0], [0], marker="s", linestyle="none",
                    markerfacecolor="none", markeredgecolor="black",
                    markeredgewidth=1.8, markersize=10,
                    label=r"Selected ($\rho$=2.5%)")]
    while len(col1) < len(classes):   # pad to 7 rows
        col1.append(Line2D([0], [0], linestyle="none", marker="none",
                           color="none", label=" "))

    # Legend col2: 7 class marker shapes
    col2 = [Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
                   markerfacecolor="#777777", markeredgecolor="none",
                   markersize=9, label=disp[ci])
            for ci in range(len(classes))]

    ax.legend(handles=col1 + col2,
              fontsize=13, framealpha=0.95, loc="upper right",
              ncol=2, handletextpad=0.4, columnspacing=0.8)

    ax.set_title(r"$\theta^2_{\mathrm{SimCLR}}$ Guided Initial $\rho$=2.5% Data Selection",
                 fontsize=22, pad=12)
    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out_path}")


def main():
    print("Loading data...")
    emb, labels, classes, hi_emb, indices, selected = load_data()
    print(f"  UMAP: {emb.shape}, selected: {len(selected)}")

    print("Reproducing Levy et al. clustering (t-SNE 2D + KMeans k=3, seed=0)...")
    all_cluster_labels, cluster_labels_51, sel_rows = get_cluster_assignments(
        hi_emb, indices, selected, k=3)
    for k in range(3):
        n = (cluster_labels_51 == k).sum()
        print(f"  Cluster {k+1}: {n} selected images  (total in cluster: {(all_cluster_labels==k).sum()})")

    # Version A
    plot_circle_version(
        emb, labels, classes, selected,
        cluster_labels_51, sel_rows,
        os.path.join(OUT_DIR, "5_2_coldstart_fm_umap_clusters_circle.png"))

    # Version B
    plot_cluster_color_version(
        emb, labels, classes, selected,
        all_cluster_labels,
        os.path.join(OUT_DIR, "5_2_coldstart_fm_umap_clusters_color.png"))


if __name__ == "__main__":
    main()
