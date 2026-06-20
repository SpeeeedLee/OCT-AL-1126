"""
製作 representative_images_final 的論文用排版圖。
每個 (class, subfolder) → 一張 PNG，5 列 × 2 欄，10 張影像，標 (a)–(j)。
圖寬符合 A4 碩論文字寬，底部留 20% 給 Caption。

Usage (from repo root):
    python3 thesis/chapter_5/plot_representative_figures.py
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SRC_BASE = "thesis/chapter_5/representative_images_final"
OUT_BASE = "thesis/chapter_5/figs/representative"

SUBFOLDER_ORDER = ["Random", "Margin_Top_10", "Margin_Bot_10",
                   "Coreset_Top_10", "Coreset_Bot_10", "TypiClust_Top_10"]

LABELS = list("abcdefghij")

# A4 text width ~16cm with 2.5cm margins → ~6.3 inch
# Total page height A4 with margins ~24.7cm → ~9.7 inch
# Images occupy top 80% → ~7.75 inch, bottom 20% for caption
FIG_W = 6.3          # inches
FIG_H_PER_ROW = 1.94 # inches per row (5 rows = 9.7 inches total)
BOTTOM_PAD = 0.01
NCOLS = 2

def load_images(folder, n=10):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, ext))
    paths = sorted(paths)[:n]
    return paths

def make_figure(img_paths, out_path):
    n = len(img_paths)
    nrows = max(1, -(-n // NCOLS))   # ceil(n / NCOLS)
    fig_h = FIG_H_PER_ROW * nrows
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(FIG_W, fig_h))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(img_paths):
            img = mpimg.imread(img_paths[i])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                      aspect="auto")
            # label in top-left corner
            ax.text(0.03, 0.97, f"({LABELS[i]})",
                    transform=ax.transAxes,
                    fontsize=11, fontweight="bold", color="white",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black",
                              ec="none", alpha=0.55))

    fig.subplots_adjust(left=0.02, right=0.98,
                        top=1.0 - 0.01,
                        bottom=BOTTOM_PAD,
                        hspace=0.04, wspace=0.04)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)

def main():
    classes = sorted(os.listdir(SRC_BASE))
    for cls in classes:
        cls_path = os.path.join(SRC_BASE, cls)
        if not os.path.isdir(cls_path):
            continue
        for sub in SUBFOLDER_ORDER:
            sub_path = os.path.join(cls_path, sub)
            if not os.path.isdir(sub_path):
                continue
            imgs = load_images(sub_path)
            if not imgs:
                print(f"  [skip] {cls}/{sub}: 0 images")
                continue
            out_path = os.path.join(OUT_BASE, cls, f"{sub}.png")
            make_figure(imgs, out_path)
            print(f"  [saved] {cls}/{sub}  ({len(imgs)} imgs) → {out_path}")

if __name__ == "__main__":
    main()
