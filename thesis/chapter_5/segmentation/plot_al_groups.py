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
# Combined overlay (one figure, all AL strategies together) — same colors/markers
# as the per-group figures above. cluster_margin has no data yet → auto-skipped.
COMBINED = [
    ("margin",         "Margin",         "#3182BD", "s"),
    ("coreset",        "Core-set",       "#238B45", "D"),
    ("cluster_margin", "Cluster-Margin", "#FB6A4A", "*"),
]

# Random baseline drawn over the AL range (2.5->60), like the classification
# figure; the full-data (rho=100) point is shown as the Target horizontal line.
RANDOM_PORTIONS = {2.5, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0}


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
    """Per-seed best-lr -> mean±std. Reads ONLY the new *_sweep 5-seed runs (no old
    single-seed main_aug4 fallback → consistent with report_table)."""
    new = _collect(glob.glob(f"{EXP}/*_sweep/nuclei/AL_random/{strat}_seed*_bs8.json"), strat)
    return _per_seed_best(new)


def random_curve():
    """Random (passive) baseline = new 5-seed aug_curve/aug4 ONLY (no base_v2 fallback).
    Target (ceiling) line = w/o-Aug full-data result = aug1 @ rho=100%."""
    new = _collect(glob.glob(f"{EXP}/aug_curve/aug4/nuclei/cold_start_random/random_*_bs8.json"))
    full = _per_seed_best(new)
    rnd = {p: v for p, v in full.items() if p in RANDOM_PORTIONS}
    aug1 = _per_seed_best(_collect(glob.glob(
        f"{EXP}/aug_curve/aug1/nuclei/cold_start_random/random_*_bs8.json")))
    target = aug1.get(100.0, (None,))[0]   # w/o-Aug @ 100%
    return rnd, target


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def draw_group(gname, items, rnd, target, title=None):
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
    # Target line removed per request (uncomment to restore):
    # if target is not None:   # ceiling = w/o-Aug full-data (rho=100%)
    #     ax.axhline(y=target, color="black", linestyle=(0, (8, 4)), linewidth=2.2,
    #                alpha=0.85, label="Target")
    if not drew:
        plt.close(fig); print(f"  [skip group] {gname}: nothing to draw"); return
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    if title:
        ax.set_title(title, fontsize=FONT_LABEL, pad=12)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.set_xlim(0, 62)    # left padding so ρ=2.5 isn't flush; AL range 2.5->60
    # ax.set_ylim(0.46, 0.72)  # (disabled) common y-range across all Dice-vs-ρ figures
    # _yt = [0.46, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72]
    # ax.set_yticks(_yt); ax.set_yticklabels([f"{v:.2f}" for v in _yt])
    ax.legend(fontsize=18, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    out = os.path.join(HERE, "figs", f"{gname}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


def main():
    rnd, target = random_curve()
    TITLE_4X = "HF+VF+HVF (4x)"             # mark the 4x-aug AL figures (vs the al_noaug ones)
    TITLED = {"diversity", "hybrid"}        # + al_strategies below (uncertainty left untitled per request)
    for gname, items in GROUPS.items():
        print(f"— {gname} —")
        draw_group(gname, items, rnd, target, title=TITLE_4X if gname in TITLED else None)
    # combined overlay: Margin + Core-set + Cluster-Margin on one figure
    print("— al_strategies (combined) —")
    draw_group("al_strategies", COMBINED, rnd, target, title=TITLE_4X)


if __name__ == "__main__":
    main()
