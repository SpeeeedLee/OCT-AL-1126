#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmentation ablation CURVE for nuclei segmentation (Task 0): none / HF(2x) / 4x
across ρ = 2.5,5,10,20,...,90,100, each at per-seed best-lr -> mean±std over seeds.
Same visual style as the Ch4 classification curves, y-axis = Dice.

Aggregation convention (matches thesis): per seed pick that seed's best-lr (mean of
its runs at each lr -> max); one value per seed; then mean±std (ddof=1) over seeds.
ρ=100 is seed-independent -> single seed, std taken over its best-lr reps.

Run (repo root): python3 thesis/chapter_5/segmentation/plot_aug_curve.py
"""
import os, json, glob
import numpy as np
import matplotlib.pyplot as plt

EXP = os.path.join(os.path.dirname(__file__), "exp_results", "aug_curve")
FONT_LABEL, FONT_TICK = 26, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})

# (aug_factor, label, color, marker, linestyle) -- colors/markers MATCH the Ch4
# classification aug plot (classification/exp/data_aug/plot_all.py):
#   w/o Aug(1x)=gray v, HF(2x)=brown D, HF+VF+HVF(4x)=blue o.
SERIES = [
    (4, "HF+VF+HVF (4x)", "#1F77B4", "o", "-"),
    (2, "HF (2x)",        "#8C564B", "D", "-"),
    (1, "w/o Aug (1x)",   "#7F7F7F", "v", "-"),
]


EXP_ROOT = os.path.join(os.path.dirname(__file__), "exp_results")
# old single-seed full passive run, used as a fallback for the 4x(=Random) line so
# it matches the Random line in plot_al_groups (both: new aug_curve + old base_v2).
OLD_FALLBACK = {4: f"{EXP_ROOT}/base_v2/nuclei/cold_start_random/random_*_bs8.json"}


def _collect(files):
    by_p = {}
    for f in glob.glob(files):
        seed = os.path.basename(f).split("_")[1]
        d = json.load(open(f))
        for p, lrd in d.items():
            best = max(float(np.mean([r["test_dice"] for r in runs])) for runs in lrd.values())
            by_p.setdefault(float(p), {}).setdefault(seed, []).append(best)
    return by_p


def curve(aug):
    """{portion: (mean, std, n_seeds)} via per-seed best-lr. Reads ONLY the new
    5-seed lr-swept aug_curve data (no old-data fallback → matches report_table)."""
    out = {}
    by_p = _collect(f"{EXP}/aug{aug}/nuclei/cold_start_random/random_*_bs8.json")
    for p, seeds in by_p.items():
        # one value per seed (its best-lr). If a single seed (ρ=100), use its reps for std.
        vals = [np.mean(v) for v in seeds.values()]
        if len(vals) == 1:
            # std over that seed's best-lr reps if available
            only = list(seeds.values())[0]
            reps = only
            m = float(np.mean(reps))
            s = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
            out[p] = (m, s, 1)
        else:
            a = np.array(vals, float)
            out[p] = (float(a.mean()), float(np.std(a, ddof=1)), len(a))
    return out


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    any_data = False
    for A, label, color, marker, ls in SERIES:
        c = curve(A)
        if not c:
            print(f"  [skip] aug{A}: no data"); continue
        ps = sorted(c)
        mean = np.array([c[p][0] for p in ps])
        std = np.array([c[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=9,
                linestyle=ls, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
        any_data = True
        print(f"  [ok] aug{A}: " + "  ".join(f"{p:g}:{c[p][0]:.4f}±{c[p][1]:.4f}(n{c[p][2]})" for p in ps))
    if not any_data:
        print("no aug_curve data yet"); return
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(0, 103)   # left padding so ρ=2.5 isn't flush against the y-axis
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "figs", "aug_curve.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
