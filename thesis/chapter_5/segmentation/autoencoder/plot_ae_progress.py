"""
Plot the AE training progress from the folders written by train_ae.py:
  1) recon/loss_log.csv          -> figs/ae_loss_curve.png   (MSE vs epoch, log-y)
  2) recon/original + recon/ep*  -> figs/ae_reconstruction.png (Input + each epoch row)

Reads whatever is on disk so far, so it works mid-training (run it any time to see
current progress). Run (repo root):
  python3 thesis/chapter_5/segmentation/autoencoder/plot_ae_progress.py
"""
import os
import sys
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.getcwd())
from thesis.chapter_5.segmentation.autoencoder.data import WIDTH, HEIGHT

HERE = os.path.dirname(__file__)
RECON = os.path.join(HERE, "recon")      # 10-image display set
RECON3 = os.path.join(HERE, "recon3")    # 3-image set (same images as dataset_samples.png)
RECON_C5 = os.path.join(HERE, "recon_custom5")   # 5 hand-picked images (see images.txt)
FIGS = os.path.normpath(os.path.join(HERE, "..", "figs"))
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"]})


def _load_row(folder):
    pngs = sorted(f for f in os.listdir(folder) if f.endswith(".png"))
    return [np.asarray(Image.open(os.path.join(folder, p)).convert("L"), np.float32) / 255
            for p in pngs]


def loss_curve(out):
    pts = {}
    # live per-epoch log first (tolerant of null-corrupted lines) ...
    path = os.path.join(RECON, "loss_log.csv")
    if os.path.isfile(path):
        with open(path, errors="ignore") as f:
            for line in f:
                p = line.replace("\x00", "").strip().split(",")
                if len(p) == 3 and p[0].isdigit():
                    try:
                        pts[int(p[0])] = float(p[1])
                    except ValueError:
                        continue
    # ... then recovered snapshot losses (recomputed from ckpts) OVERRIDE, so the
    # trusted values win at the snapshot epochs even if the live log was corrupted.
    snap = os.path.join(RECON, "loss_snapshots.csv")
    if os.path.isfile(snap):
        with open(snap) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2 and p[0].isdigit():
                    pts[int(p[0])] = float(p[1])
    # fully-recovered per-epoch log (e.g. parsed from tmux scrollback) — authoritative
    full = os.path.join(RECON, "loss_full.csv")
    if os.path.isfile(full):
        with open(full) as f:
            for line in f:
                p = line.strip().split(",")
                if len(p) >= 2 and p[0].isdigit():
                    try:
                        pts[int(p[0])] = float(p[1])
                    except ValueError:
                        continue
    if not pts:
        print("  [skip] no loss data yet"); return
    ep = sorted(pts); mse = [pts[e] for e in ep]
    fig, ax = plt.subplots(figsize=(12, 8))           # match the other thesis curve figs
    ax.plot(ep, mse, color="#1F77B4", lw=2.5)
    SNAP = {1, 3, 10, 30, 50, 100, 200, 500, 1000, 1500, 2000}
    sx = [e for e in ep if e in SNAP]; sy = [mse[ep.index(e)] for e in sx]
    ax.scatter(sx, sy, color="#D62728", zorder=5, s=70, label="Visualization checkpoints")
    ax.set_yscale("log")
    ax.set_xlabel("Pretraining Epoch", fontsize=26, labelpad=10)
    ax.set_ylabel("MSE Loss", fontsize=26, labelpad=10)
    ax.set_title(r"Training Loss Curve of $\theta_{\mathrm{AE}}$", fontsize=26, pad=12)
    ax.grid(True, which="both", ls="--", alpha=0.4, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=20, width=1.5, length=6)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}  (epochs 1..{ep[-1]})")


def recon_grid(out, recon_dir=RECON, n_show=None, wide=1.0, wspace=0.03):
    orig = os.path.join(recon_dir, "original")
    if not os.path.isdir(orig):
        print(f"  [skip] no {os.path.basename(recon_dir)}/original yet"); return
    sl = slice(None) if n_show is None else slice(0, n_show)   # first n_show images
    inputs = _load_row(orig)[sl]
    eps = sorted(int(d[2:]) for d in os.listdir(recon_dir)
                 if d.startswith("ep") and d[2:].isdigit())
    # row labels: "Input" then the bare epoch NUMBER (a single "Epoch" header is drawn
    # once on the left, so we don't repeat the word "Epoch" on every row).
    rows = [("Input", inputs)] + [(str(e), _load_row(os.path.join(recon_dir, f"ep{e:04d}"))[sl])
                                  for e in eps]
    ncol = len(inputs); nrow = len(rows)
    # size to fit one thesis page (content <= 5.6 x 9.0 in; row labels + tight bbox
    # add a little) while keeping each image's true aspect (512x384 -> W/H=1.333).
    # `wide` widens the whole figure (images + gaps); `wspace` is the inter-image gap.
    AR, HEADER = 512.0 / 384.0, 0.5
    ch = min(5.6 / (ncol * AR), (9.0 - HEADER) / nrow)
    cw = AR * ch
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * cw * wide, nrow * ch + HEADER),
                             gridspec_kw=dict(wspace=wspace, hspace=0.03))
    axes = np.atleast_2d(axes)
    for r, (label, imgs) in enumerate(rows):
        is_in = (r == 0)
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#D62728" if is_in else "#BBBBBB")
                sp.set_linewidth(2.2 if is_in else 0.6)
            if c == 0:
                ax.set_ylabel(label, fontsize=13,
                              fontweight="bold" if is_in else "normal",
                              color="#D62728" if is_in else "#333333",
                              rotation=0, ha="right", va="center", labelpad=14)
    fig_h = nrow * ch + HEADER
    fig.subplots_adjust(left=0.16, right=0.995, top=1 - HEADER / fig_h, bottom=0.006)
    # center the title over the IMAGE columns (not the whole figure), so the left "Epoch"
    # margin doesn't pull it left.
    x_lab = axes[0, 0].get_position().x0
    x_img_c = (x_lab + axes[0, -1].get_position().x1) / 2
    fig.suptitle(r"$\theta_{\mathrm{AE}}$ Reconstruction Results at Different Epochs",
                 fontsize=15, x=x_img_c, y=1 - 0.46 * HEADER / fig_h)
    # a SINGLE vertical "Epoch" label down the left side, spanning the epoch-number rows
    # (rows 1..end) — the word "Epoch" appears once instead of on every row.
    y_top = axes[1, 0].get_position().y1        # top of first epoch row
    y_bot = axes[-1, 0].get_position().y0       # bottom of last row
    fig.text(x_lab - 0.135, (y_top + y_bot) / 2, "Epoch", fontsize=14, fontweight="bold",
             color="#333333", ha="center", va="center", rotation=90)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}  ({ncol} imgs, Input + {len(eps)} epoch rows: {eps})")


def _latest_ep(recon_dir):
    """Largest epoch among the ep<NNNN>/ folders, or None."""
    eps = [int(d[2:]) for d in os.listdir(recon_dir)
           if d.startswith("ep") and d[2:].isdigit()] if os.path.isdir(recon_dir) else []
    return max(eps) if eps else None


def _read_names(recon_dir=RECON):
    names = {}
    with open(os.path.join(recon_dir, "images.txt")) as f:
        for line in f:
            i, nm = line.rstrip("\n").split("\t")
            names[int(i)] = nm
    return [names[i] for i in sorted(names)]


def recon_vs_gt(out, dataroot, recon_dir=RECON, eps=(1000,), n_show=5, figsize=(6.3, 9.7)):
    """n_show rows x (1+len(eps)) cols: Original | Epoch{eps...}, each overlaid with
    the RED ground-truth cell-nuclei contour."""
    need = ["original"] + [f"ep{e:04d}" for e in eps]
    miss = [d for d in need if not os.path.isdir(os.path.join(recon_dir, d))]
    if miss or not os.path.isfile(os.path.join(recon_dir, "images.txt")):
        print(f"  [skip] not ready (missing: {miss or 'images.txt'})"); return
    from thesis.chapter_5.segmentation.utils.data import g_data_cell_binary
    names = _read_names(recon_dir)[:n_show]
    gpath = os.path.join(dataroot, "cell") + "/"
    masks = [g_data_cell_binary(gpath, [nm], WIDTH, HEIGHT)[0, 0] for nm in names]
    cols = [("Original", _load_row(os.path.join(recon_dir, "original"))[:n_show])]
    cols += [(rf"$\theta_{{\mathrm{{AE}}}}$ @ Epoch {e}",
              _load_row(os.path.join(recon_dir, f"ep{e:04d}"))[:n_show]) for e in eps]
    nrow, ncol = len(masks), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize,
                             gridspec_kw=dict(wspace=0.02, hspace=0.03))
    axes = np.atleast_2d(axes)
    for r in range(nrow):
        for c, (title, imgs) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(imgs[r], cmap="gray", vmin=0, vmax=1)
            ax.contour(masks[r], levels=[0.5], colors="red", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=16, fontweight="bold", pad=6)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}  ({n_show} imgs x [Original, {','.join('ep'+str(e) for e in eps)}] + red GT)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="./ds/segmentation_correct")
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    print("— loss curve —");      loss_curve(os.path.join(FIGS, "ae_loss_curve.png"))
    print("— recon grid (10) —"); recon_grid(os.path.join(FIGS, "ae_reconstruction.png"), RECON)
    print("— recon grid (5) —");  recon_grid(os.path.join(FIGS, "ae_reconstruction_5.png"), RECON,
                                             n_show=5, wide=1.3, wspace=0.10)
    print("— recon vs GT (10) —");recon_vs_gt(os.path.join(FIGS, "ae_recon_gt.png"), a.dataroot,
                                              RECON, n_show=5, figsize=(6.3, 9.7))
    # same but using the LATEST trained epoch (not fixed 1000): Original | Epoch{latest} + GT
    le = _latest_ep(RECON)
    if le:
        print(f"— recon vs GT latest=ep{le} (5) —")
        recon_vs_gt(os.path.join(FIGS, "ae_recon_gt_5.png"), a.dataroot,
                    RECON, eps=(le,), n_show=5, figsize=(6.3, 9.7))
    # 3-image versions (same 3 images as dataset_samples.png)
    print("— recon grid (3) —");  recon_grid(os.path.join(FIGS, "ae_reconstruction_3.png"), RECON3)
    print("— recon vs GT (3) —"); recon_vs_gt(os.path.join(FIGS, "ae_recon_gt_3.png"), a.dataroot,
                                              RECON3, n_show=3, figsize=(7.2, 7.0))
    # custom 5 hand-picked images
    print("— recon grid (custom5) —");  recon_grid(os.path.join(FIGS, "ae_reconstruction_custom5.png"), RECON_C5)
    print("— recon vs GT (custom5) —"); recon_vs_gt(os.path.join(FIGS, "ae_recon_gt_custom5.png"), a.dataroot,
                                                    RECON_C5, n_show=5, figsize=(6.3, 9.7))


if __name__ == "__main__":
    main()
