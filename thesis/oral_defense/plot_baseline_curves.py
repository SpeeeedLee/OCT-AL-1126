#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense baseline curves: performance vs. labeled-data ratio for the
"no strategy" setting (random weight init + random subset selection).

Two standalone figures, IDENTICAL style, a single dark line each:
  1. Classification accuracy (7-class, task_type=hard)  -> cls_baseline_curve.png
  2. Nuclei segmentation Dice                            -> seg_baseline_curve.png

Both use random-init, 4x augmentation, 5 seeds, ρ=2.5–90%.
Aggregation (thesis convention): per-seed best learning rate, then mean ± std
across seeds (shaded band).

Run from repo root:
    python3 thesis/oral_defense/plot_baseline_curves.py
"""
import os, glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(os.path.dirname(__file__), "figs")

# ── shared style ──────────────────────────────────────────────────────────
FONT_LABEL, FONT_TICK = 26, 20
LINE_COLOR = "#333333"
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
XTICKS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
KEEP = {2.5, 5.0} | {float(x) for x in range(10, 101, 10)}  # 2.5,5,10,20,…,100


def _agg_over_seeds(by_portion):
    """by_portion: {p: {seed: best_lr_value}} -> {p: (mean, std)} over seeds."""
    out = {}
    for p, seeds in by_portion.items():
        vals = list(seeds.values())
        mu = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[p] = (mu, std)
    return out


def load_classification():
    """cold_start_random, aug4: per-seed best-lr accuracy (%)."""
    d_dir = os.path.join(ROOT, "classification/exp_results/classification_hard/cold_start_random")
    by_portion = {}
    for f in glob.glob(os.path.join(d_dir, "random*_bs16_ep20.json")):
        m = re.search(r"random(\d+)", os.path.basename(f))
        seed = m.group(1) if m else f
        d = json.load(open(f)).get("aug4", {})
        for p, lrd in d.items():
            p = float(p)
            if p not in KEEP:
                continue
            cand = []
            for reps in lrd.values():           # reps = list of accuracies
                reps = [x for x in reps if x is not None]
                if reps:
                    cand.append(np.mean(reps))
            if cand:
                by_portion.setdefault(p, {})[seed] = max(cand) * 100.0
    return _agg_over_seeds(by_portion)


def load_segmentation():
    """aug_curve/aug4 cold_start_random: per-seed best-lr Dice."""
    d_dir = os.path.join(ROOT, "thesis/chapter_5/segmentation/exp_results/aug_curve/aug4/nuclei/cold_start_random")
    by_portion = {}
    for f in glob.glob(os.path.join(d_dir, "random_*_bs8.json")):
        seed = os.path.basename(f).split("_")[1]
        d = json.load(open(f))
        for p, lrd in d.items():
            p = float(p)
            if p not in KEEP:
                continue
            cand = [float(np.mean([r["test_dice"] for r in runs])) for runs in lrd.values()]
            if cand:
                by_portion.setdefault(p, {})[seed] = max(cand)
    return _agg_over_seeds(by_portion)


FONT_TITLE = 30


def plot_curve(curve, ylabel, out_name, title):
    ps = sorted(curve)
    mean = np.array([curve[p][0] for p in ps])
    std = np.array([curve[p][1] for p in ps])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ps, mean, marker="o", color=LINE_COLOR, linewidth=3, markersize=10)
    ax.fill_between(ps, mean - std, mean + std, color=LINE_COLOR, alpha=0.15, linewidth=0)
    ax.set_xlabel(r"Labeling Training Images Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(title, fontsize=FONT_TITLE, pad=14)
    ax.set_xticks(XTICKS)
    ax.set_xlim(0, 103)
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, out_name)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}  ({len(ps)} portions {ps[0]:g}->{ps[-1]:g})")
    for p in ps:
        print(f"   rho={p:>5g}%   {curve[p][0]:.3f} ± {curve[p][1]:.3f}")


def main():
    print("== Classification (accuracy %) ==")
    plot_curve(load_classification(), "Accuracy (%)", "cls_baseline_curve.png",
               "Lesion Classification")
    print("\n== Nuclei segmentation (Dice) ==")
    plot_curve(load_segmentation(), "Dice", "seg_baseline_curve.png",
               "Cell Nuclei Segmentation")


if __name__ == "__main__":
    main()
