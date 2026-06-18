"""
5.3.3 代表性影像 UMAP 圖
========================
各策略一張圖，同時標示：
  ■ Top-10% D_train^L  (n=152)：黑色空心方框（同 5.3.2 風格）
  ● Bottom-10% D_train^U (n=203)：橘紅色 'X' 標記

所用 seed = 42（為三種策略中唯一有 ρ=90% labeled_ids 的 seed）。

Usage (from repo root):
    python3 thesis/chapter_5/plot_5_3_3_representative_umap.py --strategy margin
    python3 thesis/chapter_5/plot_5_3_3_representative_umap.py --strategy coreset
    python3 thesis/chapter_5/plot_5_3_3_representative_umap.py --strategy cluster_margin
    # --grid 疊座標格線方便調整標註
"""
import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_5_3_umap import get_embedding, CLASS_COLORS, CLASS_MARKERS, NAME_ABBR, REPO
from plot_5_3_umap_pair import draw_grid

FROZEN = os.path.join(REPO, "SSL", "simclr", "ckpt",
                      "resnet18_simclr_lr0.0002_bs256_ep500.pkl")
SEED   = 42
DATA_TOTAL = 2032

LABELED_IDS_DIRS = [
    os.path.join(REPO, "classification/exp_results/ch5_5_3_3_extend/"
                 "classification_hard/AL_simclr/labeled_ids"),
    os.path.join(REPO, "classification/exp_results/"
                 "classification_hard/AL_simclr/labeled_ids"),
]

STRATEGY_TITLE = {
    "margin":         "Margin",
    "coreset":        "Core-set",
    "cluster_margin": "Cluster-Margin",
}


def load_merged(strategy, seed):
    """所有 labeled_ids 檔合併成一個 dict（portion → entry）。"""
    fname = f"{strategy}_seed{seed}_bs16.json"
    merged = {}
    for d in LABELED_IDS_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            merged.update(json.load(open(p)))
    if not merged:
        raise FileNotFoundError(f"No labeled_ids found for {strategy} seed{seed}")
    return merged


def load_top10(strategy, seed):
    """Top-10% D_train^L: cumulative@10% minus init@2.5%"""
    d = load_merged(strategy, seed)
    cum10 = set(d["10.0"]["cumulative"])
    init  = set(d["2.5"]["cumulative"])
    top   = sorted(cum10 - init)
    print(f"  Top-10%  ({strategy} seed{seed}): "
          f"cum@10={len(cum10)}, init={len(init)}, AL-selected={len(top)}")
    return top


def load_bottom10(strategy, seed):
    """Bottom-10% D_train^U: all indices NOT in cumulative@90%"""
    d = load_merged(strategy, seed)
    if "90.0" not in d:
        avail = sorted(d.keys(), key=float)
        raise ValueError(
            f"ρ=90% not found for {strategy} seed{seed}. "
            f"Available: {avail}. Run AL extension first.")
    cum90  = set(d["90.0"]["cumulative"])
    bottom = sorted(set(range(DATA_TOTAL)) - cum90)
    print(f"  Bottom-10% ({strategy} seed{seed}): "
          f"cum@90={len(cum90)}, bottom={len(bottom)}")
    return bottom


def plot_representative_umap(strategy, device, out_dir, grid=False):
    title = STRATEGY_TITLE[strategy]
    emb, labels, classes = get_embedding(FROZEN, device, method="umap", split="train")

    top_idx    = load_top10(strategy, SEED)
    bottom_idx = load_bottom10(strategy, SEED)

    top_mask    = np.zeros(len(labels), bool)
    bottom_mask = np.zeros(len(labels), bool)
    top_mask[np.array(top_idx)]    = True
    bottom_mask[np.array(bottom_idx)] = True

    counts = np.array([(labels == ci).sum() for ci in range(len(classes))])
    order  = list(np.argsort(counts)[::-1])
    disp   = [NAME_ABBR.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(12, 9))

    # ── background: all images (class-colored) ──────────────────────────────
    for ci in range(len(classes)):
        m = labels == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=60, c=CLASS_COLORS[ci],
                   marker=CLASS_MARKERS[ci], alpha=0.55, linewidths=0, zorder=2)

    # ── Bottom-10% D_train^U: deep rose hollow triangles ────────────────────
    ax.scatter(emb[bottom_mask, 0], emb[bottom_mask, 1],
               s=200, facecolors="none", edgecolors="#C2185B",
               linewidths=1.8, marker="^", zorder=3,
               label=f"Bottom-10% $D_{{\\rm train}}^U$ (n={bottom_mask.sum()})")

    # ── Top-10% D_train^L: black hollow squares ──────────────────────────────
    ax.scatter(emb[top_mask, 0], emb[top_mask, 1],
               s=190, facecolors="none", edgecolors="black",
               linewidths=1.4, marker="s", zorder=4,
               label=f"Top-10% $D_{{\\rm train}}^L$ (n={top_mask.sum()})")

    if grid:
        draw_grid(ax)

    # ── legends ─────────────────────────────────────────────────────────────
    # highlight legend (upper right)
    hl_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
               markeredgecolor="black", markersize=13,
               label=f"Top-10% $D_{{\\rm train}}^L$ (n={top_mask.sum()})"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="none",
               markeredgecolor="#C2185B", markeredgewidth=1.8, markersize=13,
               label=f"Bottom-10% $D_{{\\rm train}}^U$ (n={bottom_mask.sum()})"),
    ]
    ax.legend(handles=hl_handles, fontsize=15, framealpha=0.95, loc="upper right")

    # class legend (bottom)
    class_handles = [
        Line2D([0], [0], marker=CLASS_MARKERS[ci], linestyle="none",
               markerfacecolor=CLASS_COLORS[ci], markeredgecolor="none",
               markersize=16, label=disp[ci])
        for ci in order
    ]
    fig.legend(handles=class_handles, fontsize=16, framealpha=0.95,
               loc="lower center", ncol=len(class_handles),
               bbox_to_anchor=(0.5, 0.00), handletextpad=0.4, columnspacing=1.0)

    ax.set_xlabel("UMAP-1", fontsize=20, labelpad=8)
    ax.set_ylabel("UMAP-2", fontsize=20, labelpad=8)
    ax.set_title(title, fontsize=24, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.5)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.10)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"5_3_3_umap_representative_{strategy}_seed{SEED}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["margin", "coreset", "cluster_margin"])
    ap.add_argument("--device", default="cuda:9")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "figs", "umap", "representative")
    plot_representative_umap(args.strategy, args.device, out_dir, grid=args.grid)


if __name__ == "__main__":
    main()
