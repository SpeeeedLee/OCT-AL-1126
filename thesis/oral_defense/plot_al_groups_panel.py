#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense combined AL panel: three subplots side by side
(left=Uncertainty, center=Diversity, right=Hybrid), each drawing its own
strategies + Random baseline + Target line. Shared Accuracy (y) and
labeled-training-data-portion (x) axis titles.

Same data/aggregation/colors as thesis/chapter_4/plot_al_curve.py.

Run from repo root:
    python3 thesis/oral_defense/plot_al_groups_panel.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from plot_al_curve import GROUPS, pool_strategy, random_baseline   # reuse loaders/colors

OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
FONT_AXIS, FONT_TITLE, FONT_TICK, FONT_LEG = 30, 28, 20, 17
AUG = "aug4"
TARGET = 88.2
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
PANELS = ["Uncertainty", "Diversity", "Hybrid"]


def main():
    groups = dict(GROUPS)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)

    rb = random_baseline(AUG)
    for ax, gname in zip(axes, PANELS):
        for key, label, color, marker in groups[gname]:
            data = pool_strategy(key, AUG)
            if not data:
                continue
            ps = sorted(data)
            mean = np.array([data[p][0] for p in ps])
            std = np.array([data[p][1] for p in ps])
            ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=9, label=label)
            ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
        # Random baseline (grey dashed)
        if rb:
            ps = sorted(rb)
            mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
            ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=9,
                    linestyle="--", label="Random")
            ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.12)
        # three reference lines (ρ=100% full-finetune ceilings)
        for yv in (96.2, 92.3, 88.2):
            ax.axhline(y=yv, color="black", linestyle=(0, (8, 4)), linewidth=2.0,
                       alpha=0.85, label=f"{yv:g}%")

        ax.set_title(gname, fontsize=FONT_TITLE, pad=12)
        ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
        ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(1.5)
        ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc="lower right")

    fig.supxlabel(r"Labeled Training Data Portion $\rho$ (%)", fontsize=FONT_AXIS)
    fig.supylabel("Accuracy (%)", fontsize=FONT_AXIS)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "al_groups_panel.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
