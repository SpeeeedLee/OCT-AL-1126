#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense version of the Ch4.3 weight-init portion curve, but with ONLY two
lines: random init vs SimCLR (θ²). Same data/aggregation/style as
thesis/chapter_4/plot_portion_curve.py (per-seed best-lr -> mean±std over seeds).

Legend: θ_random, θ_simclr.

Run from repo root:
    python3 thesis/oral_defense/plot_init_random_vs_simclr.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from aggregate_results import EXP, pool_seed_files   # identical loader/numbers

OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
CANON = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AUG = "aug4"
CFG = "simclr_lr0.0002_simclr_bs256_simclr_ep500"


def main():
    series = [
        (pool_seed_files(os.path.join(EXP, "cold_start_random"),
                         lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), AUG),
         r"$\theta_{\mathrm{random}}$", "#7F7F7F", "s"),
        (pool_seed_files(os.path.join(EXP, "cold_start_simclr"),
                         lambda f: CFG in f, AUG),
         r"$\theta_{\mathrm{simclr}}$", "#8E44AD", "o"),
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    for data, label, color, marker in series:
        ps = sorted(p for p in data if p in CANON)
        mean = np.array([data[p][0] for p in ps])
        std = np.array([data[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=10, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.15)
        print(f"  {label}: {len(ps)} portions ({ps[0]:g}->{ps[-1]:g})")

    ax.set_xlabel(r"Labeling Training Images Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="lower right")
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "init_random_vs_simclr.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
