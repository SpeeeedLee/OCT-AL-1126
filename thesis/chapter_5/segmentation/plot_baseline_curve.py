#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Dice-vs-rho curve for the plainest setup (no strategy at all):
random weight init + no augmentation + random subset selection.
One line, NO legend. Thesis style (same fonts/size as the other curve figures).

  --aug aug1 (default, no-aug) | aug4 (4x) | aug2 (HF) | ...
Run: python3 thesis/chapter_5/segmentation/plot_baseline_curve.py
"""
import os, glob, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "exp_results")
FONT_LABEL, FONT_TICK = 26, 20
plt.rcParams.update({"font.size": 16, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5})
CANON = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def _curve(pattern):
    byp = {}
    for f in glob.glob(pattern):
        seed = os.path.basename(f).split("_")[1]
        d = json.load(open(f))
        for p, lrd in d.items():
            best = max(float(np.mean([r["test_dice"] for r in runs])) for runs in lrd.values())
            byp.setdefault(float(p), {}).setdefault(seed, []).append(best)
    out = {}
    for p, seeds in byp.items():
        vals = [np.mean(v) for v in seeds.values()]
        if len(vals) == 1:
            reps = list(seeds.values())[0]
            out[p] = (float(np.mean(reps)), float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0)
        else:
            a = np.array(vals, float); out[p] = (float(a.mean()), float(np.std(a, ddof=1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug1")
    ap.add_argument("--out", default=os.path.join(HERE, "figs", "baseline_curve.png"))
    a = ap.parse_args()
    c = _curve(f"{EXP}/aug_curve/{a.aug}/nuclei/cold_start_random/random_*_bs8.json")
    ps = sorted(p for p in c if p in CANON)
    if not ps:
        print(f"  [skip] no data in aug_curve/{a.aug}"); return
    mean = np.array([c[p][0] for p in ps]); std = np.array([c[p][1] for p in ps])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ps, mean, marker="o", color="#333333", linewidth=3, markersize=10)
    ax.fill_between(ps, mean - std, mean + std, color="#333333", alpha=0.15)
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(0, 103)
    # ax.set_ylim(0.46, 0.72)        # (disabled) common y-range across all Dice-vs-ρ figures
    # _yt = [0.46, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72]
    # ax.set_yticks(_yt); ax.set_yticklabels([f"{v:.2f}" for v in _yt])
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {a.out}  ({a.aug}, {len(ps)} portions {ps[0]:g}->{ps[-1]:g})")


if __name__ == "__main__":
    main()
