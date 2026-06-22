#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grouped AL curves for nuclei segmentation — SAME style/colors/markers as the Ch4
classification figure (thesis/chapter_4/plot_al_curve.py), y-axis = Dice.

Produces THREE figures (one per strategy family), each = that group's strategies
(solid, group colors) + Random baseline (gray dashed X) + Target (ρ=100% full-data,
black dashed hline):
    figs/uncertainty.png : Confidence, Margin, Entropy
    figs/diversity.png   : Core-set, TypiClust
    figs/hybrid.png      : BADGE, Cluster-Margin

Aggregation (matches thesis): per seed pick that seed's best-lr (mean of its runs
at each lr -> max) -> one value/seed -> mean±std (ddof=1) over seeds. Strategies
with no data yet are silently skipped, so this can be run anytime.

Run (repo root): python3 thesis/chapter_5/segmentation/plot_al_groups.py
"""
import os, glob, json
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "exp_results")
FONT_LABEL, FONT_TICK = 26, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})

# Same colors/markers as classification GROUPS.
GROUPS = {
    # Confidence omitted: for BINARY masks least-confidence == margin (identical
    # selection), so we only run Margin + Entropy.
    "uncertainty": [
        ("margin",  "Margin",  "#3182BD", "s"),
        ("entropy", "Entropy", "#6BAED6", "^"),
    ],
    "diversity": [
        ("coreset",   "Core-set",  "#238B45", "D"),
        ("typiclust", "TypiClust", "#74C476", "v"),
    ],
    "hybrid": [
        ("badge",          "BADGE",          "#A50F15", "P"),
        ("cluster_margin", "Cluster-Margin", "#FB6A4A", "*"),
    ],
}
# Random baseline drawn all the way to full data (2.5, 5, 10, 20, ..., 90, 100),
# matching plot_margin_random.py. AL strategies themselves only span 2.5->60.
RANDOM_PORTIONS = {2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}


def _per_seed_best(by_p):
    """by_p: {portion: {seed: [bestlr_dice per rep]}} -> {portion: (mean,std,n)}."""
    out = {}
    for p, seeds in by_p.items():
        vals = [float(np.mean(v)) for v in seeds.values()]
        if len(vals) == 1:
            reps = list(seeds.values())[0]
            out[p] = (float(np.mean(reps)),
                      float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0, 1)
        else:
            a = np.array(vals, float)
            out[p] = (float(a.mean()), float(np.std(a, ddof=1)), len(a))
    return out


def _collect(files, strat=None):
    """-> {portion: {seed: [best-lr dice per rep]}} from a set of result files."""
    by_p = {}
    for f in files:
        base = os.path.basename(f)
        if strat is not None and base.split("_seed")[0] != strat:
            continue                              # avoid cluster_margin vs margin clash
        if strat is not None:
            seed = base.split("_seed")[1].split("_")[0]
        else:
            seed = base.split("_")[1]             # random_<seed>_bs8.json
        d = json.load(open(f))
        for p, lrd in d.items():
            best = max(float(np.mean([r["test_dice"] for r in runs])) for runs in lrd.values())
            by_p.setdefault(float(p), {}).setdefault(seed, []).append(best)
    return by_p


def _prefer(new, old):
    """Per portion use NEW (5-seed lr-swept) if present, else fall back to OLD (full)."""
    return {p: (new[p] if p in new else old[p]) for p in set(new) | set(old)}


def al_curve(strat):
    """Per-seed best-lr -> mean±std. Prefer the new *_sweep run, fall back to the
    old single-seed main_aug4 run so the curve is complete while the 5-seed run fills."""
    new = _collect(glob.glob(f"{EXP}/*_sweep/nuclei/AL_random/{strat}_seed*_bs8.json"), strat)
    old = _collect(glob.glob(f"{EXP}/main_aug4/nuclei/AL_random/{strat}_seed*_bs8.json"), strat)
    return _per_seed_best(_prefer(new, old))


def random_curve():
    """Random (passive) baseline: prefer new 5-seed aug_curve/aug4, fall back to base_v2."""
    new = _collect(glob.glob(f"{EXP}/aug_curve/aug4/nuclei/cold_start_random/random_*_bs8.json"))
    old = _collect(glob.glob(f"{EXP}/base_v2/nuclei/cold_start_random/random_*_bs8.json"))
    full = _per_seed_best(_prefer(new, old))
    target = full.get(100.0, (None,))[0]
    rnd = {p: v for p, v in full.items() if p in RANDOM_PORTIONS}
    return rnd, target


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def draw_group(gname, items, rnd, target):
    fig, ax = plt.subplots(figsize=(12, 8))
    drew = False
    for key, label, color, marker in items:
        c = al_curve(key)
        if not c:
            print(f"  [skip] {gname}/{key}: no data"); continue
        ps = sorted(c)
        mean = np.array([c[p][0] for p in ps]); std = np.array([c[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=9, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
        drew = True
        print(f"  [ok]  {gname}/{key}: {len(ps)} pts")
    if rnd:
        ps = sorted(rnd); mean = np.array([rnd[p][0] for p in ps]); std = np.array([rnd[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040", linewidth=3, markersize=9,
                linestyle="--", label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.12)
        drew = True
    if not drew:
        plt.close(fig); print(f"  [skip group] {gname}: nothing to draw"); return
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(0, 103)   # left padding so ρ=2.5 isn't flush; Random runs to 100
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(HERE, "figs", f"{gname}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


def main():
    rnd, target = random_curve()
    for gname, items in GROUPS.items():
        print(f"— {gname} —")
        draw_group(gname, items, rnd, target)


if __name__ == "__main__":
    main()
