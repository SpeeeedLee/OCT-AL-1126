#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2x2 qualitative figure for the nuclei prediction of the three random-init models
(seed 42, portion 10%, best-lr) on ONE OCT image (the first AE-reconstruction image,
a train image -> inferred on the fly from each ckpt):

  (a) Original          : grayscale OCT + RED ground-truth nuclei contour
  (b) w/o Aug           : TP/FP/TN/FN overlay  (+ this image's Dice in the title)
  (c) HF                : "
  (d) HF+VF+HVF (4x)    : "

Per-pixel colour key (per user):
  YELLOW = TP (hit nucleus) | PURPLE/MAGENTA = FP | BLUE = TN | BLACK = FN

Run (repo root, needs the ckpts from run_3aug_ckpt_infer.sh):
  python3 thesis/chapter_5/segmentation/plot_4panel_pred.py --device cuda:0
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.getcwd())
from thesis.chapter_5.segmentation.utils.data import o_data, g_data_cell_binary, data_loader
from thesis.chapter_5.segmentation.utils.model import Optim_U_Net
from thesis.chapter_5.segmentation.utils.tool import compute_dice_binary

HERE = os.path.dirname(__file__)
CKDIR = os.path.join(HERE, "ckpts")
W, H = 384, 512                       # o_data canvas (width, height)
# a TEST-split image (qualitative demo must be held-out); override with --image
IMG_DEFAULT = "val/1_20200206_100602_11_cheek.png"
# per-pixel colours (0-1 RGB)
COL = {"TP": (1.00, 0.85, 0.00),      # yellow
       "FP": (0.85, 0.10, 0.85),      # purple / magenta
       "TN": (0.12, 0.18, 0.55),      # blue
       "FN": (0.00, 0.00, 0.00)}      # black
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"]})


def load_model(path, device):
    m = Optim_U_Net(img_ch=1, output_ch=1, USE_DS=False, USE_DFS=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def predict(model, img_np, device):
    INPUT = torch.from_numpy(img_np.astype(np.float32)).to(device=device, dtype=torch.float)
    out = model(INPUT)
    return (out > 0.5).float().cpu().numpy()[0, 0]    # (W,H) binary


def overlay(pred2d, gt2d):
    p, g = pred2d.astype(bool), gt2d.astype(bool)
    ov = np.zeros((*pred2d.shape, 3), np.float32)
    ov[p & g] = COL["TP"]; ov[p & ~g] = COL["FP"]
    ov[~p & ~g] = COL["TN"]; ov[~p & g] = COL["FN"]
    return ov


def find_ckpt(tag, seed, portion):
    hits = glob.glob(os.path.join(CKDIR, f"unet_nuclei_seed{seed}_p{portion:g}_{tag}_lr*.pkl"))
    if not hits:
        raise FileNotFoundError(f"no ckpt for tag={tag} (run run_3aug_ckpt_infer.sh first)")
    return sorted(hits)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="./ds/segmentation_correct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--portion", type=float, default=10)
    ap.add_argument("--image", default=IMG_DEFAULT, help="a TEST-split image name")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "pred_4panel.png"))
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    # sanity: the demo image must be in the held-out TEST split
    _, _, te = data_loader(a.dataroot + "/image/", 0)
    test_files = [x for b in te for x in b]
    if a.image not in test_files:
        print(f"[warn] {a.image} is NOT in the test split (qualitative demo should be held-out)")

    img = o_data(a.dataroot + "/image/", [a.image], W, H)             # (1,1,W,H)
    gt = g_data_cell_binary(a.dataroot + "/cell/", [a.image], W, H)
    img2d, gt2d = img[0, 0], gt[0, 0]

    models = [("noaug", "w/o Aug"), ("hf", "HF (2x)"), ("4x", "HF+VF+HVF (4x)")]
    panels = []
    for tag, name in models:
        ck = find_ckpt(tag, a.seed, a.portion)
        pred = predict(load_model(ck, device), img, device)
        dice = compute_dice_binary(pred, gt2d)
        panels.append((name, overlay(pred, gt2d), dice))
        print(f"  {name:16}: Dice {dice:.3f}  ({os.path.basename(ck)})")

    # ---- 2x2, thesis style (true aspect, no ticks, 300 dpi) ----
    AR = H / W                                   # 512/384 = 1.333 (W:H displayed)
    pw = 3.25; ph = pw / AR                       # per-panel inches
    fig, axes = plt.subplots(2, 2, figsize=(2 * pw, 2 * ph + 0.9),
                             gridspec_kw=dict(wspace=0.04, hspace=0.30))
    TITLE_FS = 14

    ax = axes[0, 0]
    ax.imshow(img2d, cmap="gray", vmin=0, vmax=1)
    ax.contour(gt2d, levels=[0.5], colors="red", linewidths=0.8)
    ax.set_title("(a) Original\n ", fontsize=TITLE_FS, pad=5)   # blank 2nd line keeps row aligned

    tags = ["(b)", "(c)", "(d)"]
    cells = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for tg, ax, (name, ov, dice) in zip(tags, cells, panels):
        ax.imshow(ov)
        ax.set_title(f"{tg} {name}\nDice = {dice:.3f}", fontsize=TITLE_FS, pad=5)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # compact colour key under the panels
    handles = [Patch(facecolor=COL["TP"], label="TP"),
               Patch(facecolor=COL["FP"], label="FP"),
               Patch(facecolor=COL["TN"], label="TN"),
               Patch(facecolor=COL["FN"], edgecolor="#888", label="FN")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=2.0,
               handlelength=1.3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
