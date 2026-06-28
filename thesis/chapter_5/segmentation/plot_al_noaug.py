#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AL curves under W/O AUG (aug1) — to test whether AL helps when there is NO data
augmentation (hypothesis: 4x aug saturates the gain, AL may only help w/o Aug).

Reads the ISOLATED no-aug AL tree `exp_results/al_noaug/...` (run via
  AUG=1 EXP=.../al_noaug bash scripts/run_al.sh) and the no-aug Random baseline
(aug_curve/aug1 cold-start). Does NOT touch the 4x `al_sweep` figures.

Produces (all titled "w/o Aug"):
  figs/al_noaug_strategies.png   : Margin + Core-set + Cluster-Margin + Random + Target
  figs/al_noaug_margin.png       : Margin         + Random + Target
  figs/al_noaug_coreset.png      : Core-set        + Random + Target
  figs/al_noaug_cluster_margin.png: Cluster-Margin + Random + Target

Run (repo root): python3 thesis/chapter_5/segmentation/plot_al_noaug.py
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt

import plot_al_groups as G   # reuse _collect / _per_seed_best / style_ax / fonts / colors

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "exp_results")
# margin/coreset/cluster_margin (G.COMBINED) + TypiClust (low-budget AL, the point of
# the w/o-Aug test). TypiClust uses diversity-group light-green "v" like in al_groups.
STRATS = G.COMBINED + [("typiclust", "TypiClust", "#74C476", "v")]


def al_curve_noaug(strat):
    files = glob.glob(f"{EXP}/al_noaug/nuclei/AL_random/{strat}_seed*_bs8.json")
    return G._per_seed_best(G._collect(files, strat))


def random_noaug():
    """Random baseline + Target both from the NO-AUG cold-start (aug_curve/aug1)."""
    full = G._per_seed_best(G._collect(glob.glob(
        f"{EXP}/aug_curve/aug1/nuclei/cold_start_random/random_*_bs8.json")))
    rnd = {p: v for p, v in full.items() if p in G.RANDOM_PORTIONS}
    target = full.get(100.0, (None,))[0]      # w/o-Aug @ 100% = no-aug ceiling
    return rnd, target


def draw(out, items, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    drew = False
    for key, label, color, marker in items:
        c = al_curve_noaug(key)
        if not c:
            print(f"  [skip] {key}: no data"); continue
        ps = sorted(c)
        mean = np.array([c[p][0] for p in ps]); std = np.array([c[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=9, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
        drew = True
        print(f"  [ok]  {key}: {len(ps)} pts")
    rnd, target = random_noaug()
    if rnd:
        ps = sorted(rnd); mean = np.array([rnd[p][0] for p in ps]); std = np.array([rnd[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=9,
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.12)
    # Target line removed per request (uncomment to restore):
    # if target is not None:
    #     ax.axhline(y=target, color="black", linestyle=(0, (8, 4)), linewidth=2.2,
    #                alpha=0.85, label="Target")
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=G.FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=G.FONT_LABEL, labelpad=10)
    ax.set_title(title, fontsize=G.FONT_LABEL, pad=12)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.set_xlim(0, 62)
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    G.style_ax(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    figs = os.path.join(HERE, "figs")
    print("— al_noaug (combined) —")
    draw(os.path.join(figs, "al_noaug_strategies.png"), STRATS, "w/o Aug")
    for key, label, color, marker in STRATS:
        print(f"— al_noaug ({key}) —")
        draw(os.path.join(figs, f"al_noaug_{key}.png"),
             [(key, label, color, marker)], f"w/o Aug — {label}")


if __name__ == "__main__":
    main()
