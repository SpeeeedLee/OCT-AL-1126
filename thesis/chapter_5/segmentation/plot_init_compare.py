#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight-init comparison for Ch5 nuclei segmentation — SAME style as the Ch4 portion
curve (thesis/chapter_4/plot_portion_curve.py), y-axis = Dice.

Produces TWO figures, each comparing theta_random vs theta_AE (+ any warm-up
variants), holding the augmentation fixed:
    figs/init_compare.png        : 4x aug  (aug_curve/aug4  vs  ae_init/<ep>/aug4*)
    figs/init_compare_noaug.png  : no aug  (aug_curve/aug1  vs  ae_init/<ep>/aug1*)

theta_random -> cold_start_random ; theta_AE -> cold_start_ae (encoder=AE, decoder
random). aug<N>_wu<K> trees show up as a separate "theta_AE WU-K" curve. The AE
epoch is auto-picked as the largest ae_init/ep* that has data for that aug.

Aggregation = per-seed best-lr -> mean±std (ddof=1) over seeds. Curves with no data
are skipped (run anytime).

Run (repo root): python3 thesis/chapter_5/segmentation/plot_init_compare.py
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "exp_results")
FONT_LABEL, FONT_TICK, FONT_LEGEND = 26, 20, 18
plt.rcParams.update({
    "font.size": 16, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.linewidth": 1.5,
})
CANON = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# color/marker cycle for the AE variants (one per (epoch, warm-up) combo)
AE_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
             "#E377C2", "#17BECF", "#BCBD22", "#393B79", "#637939", "#8C6D31"]
AE_MARKERS = ["o", "^", "D", "P", "*", "v", "X", "<", ">", "p", "h", "d"]


def _curve(pattern):
    """per-seed best-lr -> {portion: (mean, std, n)}."""
    byp = {}
    for f in glob.glob(pattern):
        seed = os.path.basename(f).split("_")[1]
        d = json.load(open(f))
        for p, lrd in d.items():
            best = max(float(np.mean([r["test_dice"] for r in runs])) for runs in lrd.values())
            byp.setdefault(float(p), {}).setdefault(seed, []).append(best)
    out = {}
    for p, seeds in byp.items():
        vals = [np.mean(v) for v in seeds.values()]
        if len(vals) == 1:
            reps = list(seeds.values())[0]
            out[p] = (float(np.mean(reps)),
                      float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0, 1)
        else:
            a = np.array(vals, float)
            out[p] = (float(a.mean()), float(np.std(a, ddof=1)), len(a))
    return out


def _ae_variants(aug):
    """ALL (epoch, warmup, dir) with <aug>* data, sorted by (epoch, warmup)."""
    out = []
    for d in glob.glob(f"{EXP}/ae_init/ep*/{aug}*/nuclei/cold_start_ae"):
        if not glob.glob(os.path.join(d, "random_*_bs8.json")):
            continue
        parts = d.split(os.sep)
        ep = int(next(p for p in parts if p.startswith("ep") and p[2:].isdigit())[2:])
        at = next(p for p in parts if p.startswith(aug))
        wu = int(at.split("_wu")[1]) if "_wu" in at else 0
        out.append((ep, wu, d))
    return sorted(out)


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    for s in ax.spines.values():
        s.set_linewidth(1.5)


def draw(aug, out, title=None):
    series = [(_curve(f"{EXP}/aug_curve/{aug}/nuclei/cold_start_random/random_*_bs8.json"),
               r"$\theta_{\mathrm{random}}$", "#7F7F7F", "s")]
    for ep, wu, d in _ae_variants(aug):        # keep ONLY the epoch-2000, no-warm-up AE
        if ep != 2000 or wu != 0:
            continue
        series.append((_curve(os.path.join(d, "random_*_bs8.json")),
                       r"$\theta_{\mathrm{AE}}$", "#1F77B4", "o"))

    fig, ax = plt.subplots(figsize=(12, 8))
    drew = 0
    for data, label, color, marker in series:
        ps = sorted(p for p in data if p in CANON)
        if not ps:
            print(f"    [skip] {label}: no data"); continue
        mean = np.array([data[p][0] for p in ps]); std = np.array([data[p][1] for p in ps])
        ax.plot(ps, mean, marker=marker, color=color, linewidth=3, markersize=10, label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.15)
        drew += 1
        print(f"    [ok] {label}: {len(ps)} portions ({ps[0]:g}->{ps[-1]:g})")
    if not drew:
        plt.close(fig); print(f"  [skip] {os.path.basename(out)}: no data yet"); return
    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Dice", fontsize=FONT_LABEL, labelpad=10)
    if title:
        ax.set_title(title, fontsize=FONT_LABEL, pad=12)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax.set_ylim(0.46, 0.72)        # (disabled) common y-range across all Dice-vs-ρ figures
    # _yt = [0.46, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72]
    # ax.set_yticks(_yt); ax.set_yticklabels([f"{v:.2f}" for v in _yt])
    # legend adapts to how many curves there are (avoid covering the lines)
    leg_fs = FONT_LEGEND if drew <= 5 else (15 if drew <= 8 else 13)
    ncol = 1 if drew <= 5 else 2
    ax.legend(fontsize=leg_fs, framealpha=0.9, loc="lower right", ncol=ncol)
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {out}")


def main():
    print("- init_compare (4x aug) -")
    draw("aug4", os.path.join(HERE, "figs", "init_compare.png"), title="HF+VF+HVF (4x)")
    print("- init_compare_noaug (no aug) -")
    draw("aug1", os.path.join(HERE, "figs", "init_compare_noaug.png"), title="w/o Aug")


if __name__ == "__main__":
    main()
