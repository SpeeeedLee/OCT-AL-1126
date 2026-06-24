#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aug_curve_focus.png — the augmentation curve with a "magnifier" inset that zooms
the low-ρ region (2.5–20%) to emphasise the 4x vs HF gap. Paper style.

Main panel: all aug series (none/HF/4x...). Inset (lower-right, with connector box):
ONLY the series flagged zoom=True (4x and HF; w/o-Aug excluded per request).
SERIES is easy to extend (e.g. add VF(2x) later) — set zoom=True to include it.

Run (repo root): python3 thesis/chapter_5/segmentation/plot_aug_curve_focus.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

sys.path.insert(0, os.path.dirname(__file__))
from plot_aug_curve import curve  # reuse the per-aug best-lr -> mean±std loader

FONT_LABEL, FONT_TICK = 26, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})

# (tree, label, color, marker, linestyle, zoom?, bound?) — colors match Ch4.
# zoom=True  → drawn in the magnifier inset.
# bound=True → counts toward the inset's y-limits. (w/o Aug is drawn but does NOT
#   set the y-limits, so the box stays fixed to the augmented-lines range; w/o Aug
#   just shows the part that falls inside and is clipped below.)
SERIES = [
    ("aug4",  "HF+VF+HVF (4x)", "#1F77B4", "o", "-", True, True),
    ("vfhv3", "VF+HV (3x)",     "#D62728", "s", "-", True, True),
    ("aug2",  "HF (2x)",        "#8C564B", "D", "-", True, True),
    ("vf2",   "VF (2x)",        "#17BECF", "^", "-", True, True),
    ("aug1",  "w/o Aug (1x)",   "#7F7F7F", "v", "-", True, False),
]
ZOOM_EDGE = "black"            # color of the magnifier box + connector lines
ZOOM_LO, ZOOM_HI = 2.5, 20.0   # magnified ρ range


def _plot_series(ax, A, color, marker, ls, label, pmax=None, ms=9, lw=3, band=True):
    c = curve(A)
    if not c:
        return False
    ps = [p for p in sorted(c) if (pmax is None or p <= pmax)]
    if not ps:
        return False
    mean = np.array([c[p][0] for p in ps]); std = np.array([c[p][1] for p in ps])
    ax.plot(ps, mean, marker=marker, color=color, linewidth=lw, markersize=ms,
            linestyle=ls, label=label)
    if band:
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
    return True


def main():
    fig, ax = plt.subplots(figsize=(12, 8))   # match aug_curve.png

    # --- main panel: all series, full range ---
    for A, label, color, marker, ls, _, _ in SERIES:
        _plot_series(ax, A, color, marker, ls, label)
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(0, 103)
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.legend(fontsize=16, framealpha=0.95, loc="upper left", ncol=2,
              columnspacing=1.2, handlelength=1.8)  # 2 cols → shorter, avoids the lines

    # --- magnifier inset: only zoom=True series, ρ in [ZOOM_LO, ZOOM_HI] ---
    axins = ax.inset_axes([0.40, 0.085, 0.56, 0.44])   # [x,y,w,h] axes-fraction (lower-right)
    ylo, yhi = [], []
    for A, label, color, marker, ls, zoom, bound in SERIES:
        if not zoom:
            continue
        _plot_series(axins, A, color, marker, ls, label, pmax=ZOOM_HI, ms=8, lw=2.6)
        if bound:   # only 4x/HF set the y-limits; w/o Aug is drawn but doesn't expand them
            c = curve(A)
            for p in c:
                if p <= ZOOM_HI:
                    ylo.append(c[p][0] - c[p][1]); yhi.append(c[p][0] + c[p][1])
    if ylo:  # fixed box: hug 4x/HF (+ bands); w/o Aug below this is clipped
        axins.set_ylim(min(ylo) - 0.004, max(yhi) + 0.004)
    axins.set_xlim(ZOOM_LO - 1.2, ZOOM_HI + 1.2)
    axins.set_xticks([5, 10, 15, 20])                       # integer ticks (no decimals)
    axins.yaxis.set_major_locator(MultipleLocator(0.02))    # 2-decimal-friendly steps
    axins.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # 2 decimal places
    axins.tick_params(axis="both", labelsize=14, width=1.2, length=4)
    axins.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)
    for s in axins.spines.values():
        s.set_linewidth(1.8); s.set_edgecolor(ZOOM_EDGE)
    axins.set_title(r"zoom: $\rho=2.5$–$20\%$", fontsize=14, pad=4)

    # connector box (solid) + extending connector lines (DASHED) — black
    ind = ax.indicate_inset_zoom(axins, edgecolor=ZOOM_EDGE, linewidth=1.8, alpha=0.9)
    conns = getattr(ind, "connectors", None)
    if conns is None and isinstance(ind, (tuple, list)) and len(ind) > 1:
        conns = ind[1]
    for c in (conns or []):
        c.set_linestyle((0, (5, 3)))   # dashed connector lines

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figs", "aug_curve_focus.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
