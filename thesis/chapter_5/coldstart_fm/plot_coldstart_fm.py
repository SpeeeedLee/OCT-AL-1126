#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
§5.2　Cold-start FM 初始選樣 → one-shot fine-tune 結果（長條圖）
================================================================================
每個 portion 一張長條圖（title = "ρ = XX%"），比較「用不同 foundation model 當 cold-start
特徵抽取器所選的初始標註集」在 one-shot fine-tune 後的 test accuracy。

聚合慣例與全論文一致（§4.4 / 5.3）：每個 seed 先挑自己的 best-lr（該 seed 各 lr 的 run 平均取最大），
得每 seed 一個 acc → 再對 5 seeds 取 **mean ± std (ddof=1)**。黑色 error bar = std over seeds。

同一 model family 用相近色系（不同 size 深淺不同）；Random（θ² cold-start 隨機選）為灰色參考。
x 軸 = family 名；多 size 的 family 在每根柱子上方標 size。

從 repo root 執行（一次畫 2.5/10/20 三張）：
    python3 thesis/chapter_5/coldstart_fm/plot_coldstart_fm.py
圖存到 thesis/chapter_5/figs/5_2_coldstart_fm_rho{p}.png，並在 terminal 印 structured 結果。
"""
import os
import sys
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "thesis", "chapter_4"))
from plot_al_curve import (_per_seed_best_curve, _per_seed_best, _fmt_lr, _acc_list,  # noqa: E402
                           random_baseline, style_ax,
                           FONT_LABEL, FONT_TICK, FONT_LEGEND)

COLD_FM_DIR = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                           "coldstart_fm_simclr")
OUT_DIR = os.path.join(REPO, "thesis", "chapter_5", "figs")
DEFAULT_PORTIONS = [2.5, 10.0, 20.0]
AUG = "aug4"

# --------------------------------------------------------------------------- #
# Layout: ordered families.  Each member = (model_id, size_label, color).
#   - same family = same hue, light→dark by size
#   - size_label shown above the bar only when a family has >1 size
#   - Random (grey) + SimCLR-ours (gold) are the reference points, placed first
# --------------------------------------------------------------------------- #
FAMILIES = [
    ("Random",                  [("random",               "",    "#7F7F7F")]),
    ("ResNet\n(ImageNet)", [("resnet_imagenet:resnet18",  "18",  "#9ECAE1"),
                            ("resnet_imagenet:resnet50",   "50",  "#6BAED6"),
                            ("resnet_imagenet:resnet101",  "101", "#2171B5"),
                            ("resnet_imagenet:resnet152",  "152", "#08306B")]),
    ("ResNet\n(RadImageNet)", [("radimagenet:resnet50",    "50",  "#E6550D")]),
    ("ResNet\n(SimCLR; Ours)",  [("simclr:resnet18",      "18",  "#E6A817")]),
    ("DINOv2",             [("dinov2:small",               "S",   "#A1D99B"),
                            ("dinov2:base",                "B",   "#41AB5D"),
                            ("dinov2:large",               "L",   "#006D2C")]),
    ("CLIP",               [("clip:base",                  "B",   "#BCBDDC"),
                            ("clip:large",                 "L",   "#6A51A3")]),
    ("BiomedCLIP",         [("biomedclip:base",            "",    "#1FA191")]),
    ("RETFound\n(OCT)",    [("retfound:oct",               "",    "#CB181D")]),
    ("MedImageInsight",    [("medimageinsight:base",       "",    "#8C6D31")]),
]


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def _model_table(model_id, aug=AUG):
    """coldstart_fm_simclr/{model}[_seed*]_bs16.json → {portion(float): {file: {lr:[runs]}}}.
    Matches BOTH the seedless file ({model}_bs16.json) and any legacy {model}_seed*_bs16.json;
    both are pooled as extra reps (the selected set is identical, seeds only add noise)."""
    fname = model_id.replace(":", "__")
    by_portion = {}
    if not os.path.isdir(COLD_FM_DIR):
        return by_portion
    for f in os.listdir(COLD_FM_DIR):
        if not f.endswith("_bs16.json"):
            continue
        core = re.sub(r"_seed\d+$", "", f[:-len("_bs16.json")])   # strip optional _seed{n}
        if core != fname:
            continue
        d = json.load(open(os.path.join(COLD_FM_DIR, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            by_portion.setdefault(float(p), {})[f] = lrd          # key by filename (distinct)
    return by_portion


def _pooled_best(by_portion):
    """The cold-start selected set is DETERMINISTIC (fixed per model×portion), so the
    seed only perturbs training noise — there is NO subset variance to average over.
    Therefore: pool all runs across seeds per lr, pick the lr with the best mean, and
    report mean±std (ddof=1) over THAT lr's runs (the thesis "fixed-subset" convention,
    just with extra reps from the already-run seeds).
    Returns {portion: {'mean','std','lr','n','bylr': {lr:(mean%, n)}}}."""
    out = {}
    for p, seeds in by_portion.items():
        bylr = {}
        for lrd in seeds.values():
            for lr, v in lrd.items():
                bylr.setdefault(lr, []).extend(_acc_list(v))
        lr_mean = {lr: (float(np.mean(r)) * 100.0, len(r)) for lr, r in bylr.items() if r}
        if not lr_mean:
            continue
        best_lr = max(lr_mean, key=lambda k: lr_mean[k][0])
        runs = bylr[best_lr]
        std = float(np.std(runs, ddof=1)) * 100.0 if len(runs) > 1 else 0.0
        out[p] = {"mean": float(np.mean(runs)) * 100.0, "std": std,
                  "lr": best_lr, "n": len(runs), "bylr": lr_mean}
    return out


def load_all(aug=AUG):
    """Return {model_id: {portion: {'mean','std','lr','n','bylr'}}} + Random baseline."""
    out = {}
    for _, members in FAMILIES:
        for mid, _, _ in members:
            if mid == "random":
                continue
            out[mid] = _pooled_best(_model_table(mid, aug))
    rb = random_baseline(aug)
    return out, rb


def value_for(model_id, portion, data, rb):
    """(mean, std) or None."""
    if model_id == "random":
        return rb.get(portion)
    d = data.get(model_id, {}).get(portion)
    return (d["mean"], d["std"]) if d else None


# --------------------------------------------------------------------------- #
# terminal structured print
# --------------------------------------------------------------------------- #
def print_structured(portion, data, rb):
    print("\n" + "=" * 88)
    print(f"  Cold-start FM one-shot — ρ = {portion:g}%   (aug={AUG})")
    print(f"  FIXED selected set → tune lr → best-lr → mean±std (ddof=1) over its runs.")
    print(f"  (seeds = extra training-noise reps on the SAME images; Random = θ² cold-start")
    print(f"   over 5 random subsets, so its std reflects subset variance, not just noise.)")
    print("=" * 88)
    head = f"  {'model':<26} | {'best-lr':>8} {'n':>3} ‖ {'mean':>6} {'std':>5}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for disp, members in FAMILIES:
        flat = disp.replace("\n", " ")
        for mid, size, _ in members:
            name = flat + (f" [{size}]" if size else "")
            if mid == "random":
                v = rb.get(portion)
                agg = f" ‖ {v[0]:6.2f} {v[1]:5.2f}" if v else f" ‖ {'—':>6} {'—':>5}"
                print(f"  {name:<26} | {'(5 subsets)':>8} {'—':>3}{agg}")
                continue
            d = data.get(mid, {}).get(portion)
            if not d:
                print(f"  {name:<26} | {'—':>8} {'—':>3} ‖ {'—':>6} {'—':>5}")
                continue
            print(f"  {name:<26} | {_fmt_lr(d['lr']):>8} {d['n']:>3} ‖ {d['mean']:6.2f} {d['std']:5.2f}")
            parts = [f"{_fmt_lr(lr)}:{m:.2f}({n}){'*' if lr == d['lr'] else ''}"
                     for lr, (m, n) in sorted(d["bylr"].items(), key=lambda kv: float(kv[0]))]
            print(f"  {'':<26} |   lr means: " + "  ".join(parts))
    print("=" * 88)


# --------------------------------------------------------------------------- #
# bar chart per portion
# --------------------------------------------------------------------------- #
def _text_color(hex_color):
    """Black on light bars, white on dark bars (for the in-bar size token)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if lum > 0.6 else "white"


_HATCH_CNN, _HATCH_VIT = "//", "oo"


def _hatch_for(disp):
    """Two textures by backbone family (color = model identity; hatch = architecture):
    CNN-based (all ResNet variants) vs ViT-based (everything else — DINOv2/CLIP/BiomedCLIP/
    RETFound/MedImageInsight's DaViT). Random = the baseline → solid."""
    d = disp.replace("\n", " ")
    if d == "Random":
        return ""                       # baseline, solid
    if d.startswith("ResNet"):
        return _HATCH_CNN               # CNN-based
    return _HATCH_VIT                   # ViT-based (all others)



def draw(portion, data, rb, out):
    # assign x positions: intra-family gap 1.0, inter-family extra gap
    INTRA, INTER = 1.0, 0.9
    bars = []          # (xpos, mean, std, color, size_label, multi, hatch)
    fam_ticks = []     # (display, center_x)
    x = 0.0
    missing = []
    for disp, members in FAMILIES:
        xs = []
        multi = len(members) > 1
        hatch = _hatch_for(disp)
        for mid, size, color in members:
            v = value_for(mid, portion, data, rb)
            if v is None:
                missing.append(mid)
                xs.append(x); x += INTRA
                continue
            bars.append((x, v[0], v[1], color, size, multi, hatch))
            xs.append(x); x += INTRA
        fam_ticks.append((disp, sum(xs) / len(xs)))
        x += INTER

    if not bars:
        print(f"  [warn] ρ={portion:g}%: no data at all — skip figure")
        return

    fig, ax = plt.subplots(figsize=(16, 8))
    means = [b[1] for b in bars]
    stds = [b[2] for b in bars]
    from matplotlib.colors import to_rgba
    for xpos, mean, std, color, size, multi, hatch in bars:
        ax.bar(xpos, mean, width=0.8, color=to_rgba(color, 0.78), edgecolor="black",
               linewidth=1.2, hatch=hatch or None, yerr=std, capsize=5,
               error_kw=dict(ecolor="black", elinewidth=1.8, capthick=1.8), zorder=3)

    # y-limits focused on the data range (bars compared, not absolute-from-0)
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    y0, y1 = max(0, np.floor((lo - 5) / 5) * 5), min(100, hi + 4)
    ax.set_ylim(y0, y1)

    # annotations (after ylim is fixed): accuracy ON TOP, size token in the MIDDLE of the bar
    size_tokens = []                              # (Text artist, color) → underline later
    for xpos, mean, std, color, size, multi, hatch in bars:
        ax.text(xpos, mean + std + (y1 - y0) * 0.012, f"{mean:.1f}", ha="center",
                va="bottom", fontsize=FONT_TICK - 6, color="#222222")
        if size:                                  # italic, inside the bar (underlined below)
            t = ax.text(xpos, (y0 + mean) / 2.0, size, ha="center", va="center",
                        fontsize=FONT_TICK, fontstyle="italic", fontweight="bold",
                        color=_text_color(color), zorder=4)
            size_tokens.append((t, color))

    # span ALL family ticks (incl. trailing families with no bar yet, e.g. MedImageInsight),
    # else matplotlib auto-clips x to the drawn bars and drops their tick labels.
    ax.set_xlim(fam_ticks[0][1] - 1.0, fam_ticks[-1][1] + 1.0)
    ax.set_xticks([c for _, c in fam_ticks])
    # ha="right" + va="top" + rotation_mode="anchor": anchors the TOP of each label block at
    # the tick → every label's first line ("ResNet"/"RETFound"/"Random"/…) top-aligns,
    # and 2nd lines hang below (no overflow past the axis edges).
    ax.set_xticklabels([d for d, _ in fam_ticks], fontsize=FONT_LEGEND + 2,
                       rotation=45, ha="right", va="top", rotation_mode="anchor")
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(rf"$\rho$ = {portion:g}%", fontsize=FONT_LABEL, pad=14)
    style_ax(ax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    # Random reference line across the whole axis
    rbv = rb.get(portion)
    if rbv:
        ax.axhline(rbv[0], color="#7F7F7F", linestyle=(0, (6, 4)), linewidth=2, alpha=0.7, zorder=1)

    # texture legend: CNN-based (//) vs ViT-based (oo)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch=_HATCH_CNN, label="CNN-based"),
                       Patch(facecolor="white", edgecolor="black", hatch=_HATCH_VIT, label="ViT-based")],
              loc="upper right", fontsize=FONT_LEGEND, framealpha=0.9, ncol=2,
              columnspacing=1.0, handlelength=1.6)

    # underline each italic size token (mathtext has no \underline here): draw a line
    # under the rendered text's bbox, in data coords.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    rend = fig.canvas.get_renderer()
    for t, color in size_tokens:
        bb = t.get_window_extent(renderer=rend)
        (xa, ya), (xb, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y0)])
        pad = (xb - xa) * 0.08
        ax.plot([xa - pad, xb + pad], [ya, ya], color=t.get_color(),
                linewidth=1.8, zorder=5, solid_capstyle="butt")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if missing:
        print(f"  [warn] ρ={portion:g}%: no data for {missing} (skipped those bars)")
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portions", type=float, nargs="+", default=DEFAULT_PORTIONS)
    ap.add_argument("--aug", default=AUG)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    data, rb = load_all(args.aug)
    for p in args.portions:
        print_structured(p, data, rb)
        out = os.path.join(args.out_dir, f"5_2_coldstart_fm_rho{p:g}.png")
        draw(p, data, rb, out)


if __name__ == "__main__":
    main()
