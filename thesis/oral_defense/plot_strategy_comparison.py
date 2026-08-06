#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oral-defense teaser: Lesion-Classification accuracy vs. labeling budget,
TWO lines only —
  (a) baseline: no strategy at all (random init, aug4, random subset selection).
  (b) with our strategies applied: mean of the three active-learning query
      strategies (Margin / Core-set / Cluster-Margin) on the full pipeline
      (aug4 + SimCLR θ² init + active learning).

Same visual language as plot_baseline_curves.py (dark theme, big fonts).
Run from repo root:
    python3 thesis/oral_defense/plot_strategy_comparison.py
"""
import os, glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTDIR = os.path.join(os.path.dirname(__file__), "figs")

FONT_LABEL, FONT_TICK, FONT_TITLE, FONT_LEG = 26, 20, 30, 20
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})

CLS = os.path.join(ROOT, "classification/exp_results/classification_hard")
AL_DIR = os.path.join(CLS, "AL_simclr")
CS_RAND = os.path.join(CLS, "cold_start_imagenet")   # baseline = ImageNet pretrain, passive
STRATS = ["margin", "coreset", "cluster_margin"]


def _acc(leaf):
    v = leaf["acc"] if isinstance(leaf, dict) else leaf
    v = [float(x) for x in v if x is not None]
    return v


def _seed_best_mean(by_portion):
    """by_portion: {p:{seed:{lr:leaf}}} -> {p: mean-over-seeds of per-seed best-lr acc%}."""
    out = {}
    for p, seeds in by_portion.items():
        seed_best = []
        for lrd in seeds.values():
            cand = [np.mean(_acc(v)) for v in lrd.values() if _acc(v)]
            if cand:
                seed_best.append(max(cand) * 100.0)
        if seed_best:
            out[p] = float(np.mean(seed_best))
    return out


def load_baseline():
    """cold_start_random aug4 -> per-seed best-lr, mean±std over seeds (%)."""
    KEEP = {2.5, 5.0} | {float(x) for x in range(10, 61, 10)}
    by_portion = {}
    for f in glob.glob(os.path.join(CS_RAND, "random*_bs16_ep20.json")):
        seed = re.search(r"random(\d+)", os.path.basename(f)).group(1)
        for p, lrd in json.load(open(f)).get("aug4", {}).items():
            p = float(p)
            if p in KEEP:
                by_portion.setdefault(p, {})[seed] = lrd
    mean = _seed_best_mean(by_portion)
    std = {}
    for p, seeds in by_portion.items():
        vals = []
        for lrd in seeds.values():
            cand = [np.mean(_acc(v)) for v in lrd.values() if _acc(v)]
            if cand:
                vals.append(max(cand) * 100.0)
        std[p] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def load_strategy(strat):
    """AL_simclr/{strat}_seed*_bs16.json aug4 -> mean-over-seeds curve (%)."""
    by_portion = {}
    for f in glob.glob(os.path.join(AL_DIR, f"{strat}_seed*_bs16.json")):
        if os.path.basename(f).split("_seed")[0] != strat:
            continue
        seed = re.search(r"seed(\d+)", os.path.basename(f)).group(1)
        for p, lrd in json.load(open(f)).get("aug4", {}).items():
            by_portion.setdefault(float(p), {})[seed] = lrd
    return _seed_best_mean(by_portion)


def main():
    base_mean, base_std = load_baseline()
    bp = sorted(base_mean)
    bmu = np.array([base_mean[p] for p in bp])
    bsd = np.array([base_std[p] for p in bp])

    curves = {s: load_strategy(s) for s in STRATS}
    common = sorted(set.intersection(*[set(c) for c in curves.values()]))
    stacked = np.array([[curves[s][p] for s in STRATS] for p in common])  # [P,3]
    al_mean = stacked.mean(axis=1)
    al_lo, al_hi = stacked.min(axis=1), stacked.max(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    # (b) with strategies — mean of 3 AL strategies, envelope band
    ax.fill_between(common, al_lo, al_hi, color="#3182BD", alpha=0.15, linewidth=0)
    ax.plot(common, al_mean, color="#3182BD", linewidth=3, marker="o", markersize=10,
            label="With 3 strategies")

    # (a) baseline — no strategy
    ax.fill_between(bp, bmu - bsd, bmu + bsd, color="#333333", alpha=0.12, linewidth=0)
    ax.plot(bp, bmu, color="#333333", linewidth=3, marker="X", markersize=11,
            linestyle="--", label="No strategy")

    ax.set_xlabel(r"Labeling Training Images Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title("Lesion Classification", fontsize=FONT_TITLE, pad=14)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    ax.set_xlim(0, 63)
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(1.5)
    ax.legend(fontsize=FONT_LEG, framealpha=0.9, loc="lower right")

    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "cls_strategy_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"\n{'rho':>6} | {'baseline':>10} | {'AL-mean':>8} | " + " ".join(f"{s:>13}" for s in STRATS))
    for p in sorted(set(bp) | set(common)):
        b = f"{base_mean[p]:.1f}" if p in base_mean else "—"
        if p in common:
            a = f"{al_mean[common.index(p)]:.1f}"
            each = " ".join(f"{curves[s][p]:>13.1f}" for s in STRATS)
        else:
            a, each = "—", ""
        print(f"{p:>6g} | {b:>10} | {a:>8} | {each}")


if __name__ == "__main__":
    main()
