#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense figure: accuracy gap θ²_SimCLR − θ_ImageNet vs. labeling ratio ρ.

Same data/aggregation as thesis/chapter_4/plot_portion_curve.py
(per-seed best-lr -> mean±std over seeds, aug4). One line = difference of the
two per-portion means; light band = combined std sqrt(s1^2 + s2^2).

Run from repo root:
    python3 thesis/oral_defense/plot_simclr_minus_imagenet.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_4"))
from aggregate_results import EXP, pool_seed_files

OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
FONT_LABEL, FONT_TICK = 26, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
CANON = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
AUG = "aug4"
CFG = "simclr_lr0.0002_simclr_bs256_simclr_ep500"


def main():
    random = pool_seed_files(os.path.join(EXP, "cold_start_random"),
                             lambda f: f.startswith("random") and f.endswith("_bs16_ep20.json"), AUG)
    simclr = pool_seed_files(os.path.join(EXP, "cold_start_simclr"),
                             lambda f: CFG in f, AUG)

    ps = [p for p in CANON if p >= 10 and p in random and p in simclr]
    diff = np.array([simclr[p][0] - random[p][0] for p in ps])
    comb = np.array([np.hypot(simclr[p][1], random[p][1]) for p in ps])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axhline(y=0, color="#888888", linestyle=(0, (8, 4)), linewidth=2.0, alpha=0.9)
    ax.fill_between(ps, diff - comb, diff + comb, color="#8E44AD", alpha=0.15, linewidth=0)
    ax.plot(ps, diff, marker="o", color="#8E44AD", linewidth=3, markersize=10)

    ax.set_xlabel(r"Labeling Training Images Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel(r"$\theta^{2}_{\mathrm{SimCLR}} - \theta_{\mathrm{random}}$  (%)",
                  fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "simclr_minus_random.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"\n{'rho':>6} | {'SimCLR':>8} | {'random':>8} | {'diff':>7}")
    for p in ps:
        print(f"{p:>6g} | {simclr[p][0]:>8.2f} | {random[p][0]:>8.2f} | {simclr[p][0]-random[p][0]:>+7.2f}")


if __name__ == "__main__":
    main()
