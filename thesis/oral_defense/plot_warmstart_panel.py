#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense combined panel: Warm-Start vs. Cold-Start AL across three
strategies (Margin / Core-set / Cluster-Margin), side by side, shared axes.

Big suptitle "Warm-Start vs. Cold-Start AL"; each subplot titled by strategy.
Same data/aggregation as thesis/chapter_5/plot_5_x_warmstart_al.py.

Run from repo root:
    python3 thesis/oral_defense/plot_warmstart_panel.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "chapter_5"))
import plot_5_x_warmstart_al as W   # pool_strategy, pool_warmstart, random_baseline, configs

OUTDIR = os.path.join(os.path.dirname(__file__), "figs")
FONT_AXIS, FONT_SUP, FONT_TITLE, FONT_TICK, FONT_LEG = 30, 34, 28, 20, 16
AUG = "aug4"
TARGET = 88.2
STRATS = ["margin", "coreset", "cluster_margin"]
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})


ZOOM_LO, ZOOM_HI = 2.5, 15.0


def add_circular_zoom(ax, cold, warm, cold_color, warm_color, cold_marker, warm_marker):
    """Pulled-out magnifier (lower area) of the ρ=2.5–15% region; cold vs warm
    only, with its own x/y axis labels, plus a region box + connector lines."""
    reg = [p for p in sorted(cold) if ZOOM_LO <= p <= ZOOM_HI]
    if not reg:
        return

    axins = ax.inset_axes([0.46, 0.13, 0.40, 0.40])
    axins.set_facecolor("white")

    cm = [cold[p][0] for p in reg]
    axins.plot(reg, cm, marker=cold_marker, color=cold_color, linewidth=2.5,
               markersize=7, linestyle="-", zorder=3)
    wreg = [p for p in reg if p in warm]
    if wreg:
        wm = [warm[p][0] for p in wreg]
        axins.plot(wreg, wm, marker=warm_marker, color=warm_color, linewidth=2.5,
                   markersize=8, linestyle="--", markeredgecolor="black",
                   markeredgewidth=1.4, zorder=4)

    vals = cm + ([warm[p][0] for p in wreg] if wreg else [])
    lo, hi = min(vals), max(vals)
    ypad = (hi - lo) * 0.12 + 0.8
    axins.set_xlim(ZOOM_LO - 0.6, ZOOM_HI + 0.6)
    axins.set_ylim(lo - ypad, hi + ypad)
    axins.set_xticks([5, 10, 15])
    axins.locator_params(axis="y", nbins=4)
    axins.tick_params(labelsize=12, width=1.3, length=4)
    axins.set_xlabel(r"$\rho$ (%)", fontsize=14, labelpad=2)
    axins.set_ylabel("Accuracy (%)", fontsize=14, labelpad=2)
    axins.grid(True, linestyle="--", alpha=0.4, linewidth=0.9)
    axins.set_axisbelow(True)
    for s in axins.spines.values():
        s.set_linewidth(1.3)

    # region box on the main axis around the ρ=2.5–15% data
    # (hug the top so it clears the legend, more room below)
    bx0, bx1 = ZOOM_LO - 0.5, ZOOM_HI + 0.6
    by0, by1 = lo - ((hi - lo) * 0.18 + 1.5), hi + 1.2
    ax.add_patch(mpatches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                                    edgecolor="0.3", linewidth=1.8, zorder=8))
    # connector lines from box corners to the inset (pulled-out lens)
    for (rx, ry), (cx, cy) in [((bx1, by1), (0.0, 1.0)), ((bx1, by0), (0.0, 0.0))]:
        con = mpatches.ConnectionPatch(xyA=(rx, ry), coordsA=ax.transData,
                                       xyB=(cx, cy), coordsB=axins.transAxes,
                                       color="0.5", linewidth=1.2, zorder=7)
        ax.add_artist(con)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5), sharey=True)
    rb = W.random_baseline(AUG)

    for ax, strat in zip(axes, STRATS):
        label, cold_color = W.METHODS[strat]
        warm_color = W.WARM_COLOR[strat]

        # cold-start (solid)
        cold = W.pool_strategy(strat, AUG)
        if cold:
            ps = sorted(cold)
            mean = np.array([cold[p][0] for p in ps]); std = np.array([cold[p][1] for p in ps])
            ax.plot(ps, mean, marker=W.COLD_MARKER[strat], color=cold_color,
                    linewidth=3, markersize=9, linestyle="-", label="Cold-start")
            ax.fill_between(ps, mean - std, mean + std, color=cold_color, alpha=0.12)

        # warm-start (dashed, θ²-FM initial pool)
        warm = W.pool_warmstart(strat, AUG)
        if warm:
            ps = sorted(warm)
            mean = np.array([warm[p][0] for p in ps]); std = np.array([warm[p][1] for p in ps])
            ax.errorbar(ps, mean, yerr=std, marker=W.WARM_MARKER[strat], color=warm_color,
                        linewidth=3, markersize=10, linestyle="--",
                        markeredgecolor="black", markeredgewidth=2.0,
                        capsize=5, elinewidth=1.6, capthick=1.6, ecolor=warm_color,
                        label="Warm-start")

        # Random passive baseline
        if rb:
            ps = sorted(rb)
            mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
            ax.plot(ps, mean, marker="X", color="#404040", linewidth=2.5, markersize=8,
                    linestyle=(0, (6, 3)), alpha=0.85, label="Random")
            ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.10)

        # Target
        ax.axhline(y=TARGET, color="black", linestyle=(0, (8, 4)), linewidth=2.2,
                   alpha=0.85, label="Target")

        # circular magnifier of the ρ=2.5–10% region (cold vs warm)
        if cold and warm:
            add_circular_zoom(ax, cold, warm, cold_color, warm_color,
                              W.COLD_MARKER[strat], W.WARM_MARKER[strat])

        ax.set_title(label, fontsize=FONT_TITLE, pad=12)
        ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
        ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_linewidth(1.5)
        # legend order: Warm-start first
        h, l = ax.get_legend_handles_labels()
        order = ["Warm-start", "Cold-start"]   # keep Random/Target lines, drop from legend
        pairs = [(l.index(o), o) for o in order if o in l]
        ax.legend([h[i] for i, _ in pairs], [o for _, o in pairs],
                  fontsize=FONT_LEG, framealpha=0.9, loc="upper left")

    fig.supxlabel(r"Labeled Training Data Portion $\rho$ (%)", fontsize=FONT_AXIS)
    fig.supylabel("Accuracy (%)", fontsize=FONT_AXIS)
    fig.suptitle("Warm-Start vs. Cold-Start Active Learning", fontsize=FONT_SUP, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "warmstart_panel.png")
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
