#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
§5.x  FM Warm-start AL vs Cold-start AL 比較圖
===============================================
三張圖（Margin / Core-set / Cluster-Margin），各比較：
  - Cold-start AL（實線，ch4 AL_simclr/）      — 隨機初始池
  - Warm-start AL（虛線，ch5_fm_init_al/）      — θ²-FM 主動選初始池
  - Random passive baseline（灰虛線）           — 參考下界
  - Target（黑水平線 93.2%）                   — ρ=100% 上界

聚合慣例：per-seed best-lr → over seeds mean±std (ddof=1)。
  - Cold-start: 5 seeds (10,24,38,42,57)
  - Warm-start: 3 seeds (10,24,42)，初始池固定，seed 只影響訓練雜訊

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_x_warmstart_al.py
"""
import os, sys, re, json, argparse
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "chapter_4"))
from plot_al_curve import (pool_strategy, random_baseline,
                           _per_seed_best_curve, style_ax,
                           AL_DIR, FONT_LABEL, FONT_TICK, FONT_LEGEND)

REPO = os.path.dirname(os.path.dirname(HERE))
WARMSTART_DIR = os.path.join(REPO, "classification", "exp_results",
                              "ch5_fm_init_al", "classification_hard", "AL_simclr")
OUT_DIR = os.path.join(HERE, "figs")
TARGET_ACC = 88.2   # consistent with ch4 al_curve
AUG = "aug4"

# ── visual config ──────────────────────────────────────────────────────────── #
METHODS = {
    "margin":         ("Margin",         "#3182BD"),
    "coreset":        ("Core-set",       "#238B45"),
    "cluster_margin": ("Cluster-Margin", "#FB6A4A"),
}
# cold-start markers (same as ch4 / redistribution)
COLD_MARKER = {"margin": "s", "coreset": "D", "cluster_margin": "*"}
# warm-start: lighter color + very different marker shape
WARM_COLOR  = {"margin": "#9ECAE1", "coreset": "#74C476", "cluster_margin": "#FC9272"}
WARM_MARKER = {"margin": "p",       "coreset": "^",        "cluster_margin": "h"}


def pool_warmstart(strat, aug=AUG):
    """讀 WARMSTART_DIR/{strat}_seed*.json → per-seed best-lr → mean±std。"""
    if not os.path.isdir(WARMSTART_DIR):
        return {}
    by_portion = {}
    for f in os.listdir(WARMSTART_DIR):
        if not f.endswith(".json") or f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        seed = m.group(1) if m else f
        d = json.load(open(os.path.join(WARMSTART_DIR, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            by_portion.setdefault(float(p), {})[seed] = lrd
    return _per_seed_best_curve(by_portion)


def _best_lr_acc(lrd):
    best = -1
    for lr, run in lrd.items():
        acc_list = run.get("test_acc", run.get("acc", []))
        acc = max(acc_list) if isinstance(acc_list, list) else acc_list
        if acc > best:
            best = acc
    return best


def _print_table(strat, aug, cold, warm):
    label = METHODS[strat][0]
    # raw warm-start per-seed data
    ws_raw = {}
    if os.path.isdir(WARMSTART_DIR):
        for f in sorted(os.listdir(WARMSTART_DIR)):
            if not f.endswith(".json") or f.split("_seed")[0] != strat:
                continue
            m = re.search(r"seed(\d+)", f)
            seed = m.group(1) if m else f
            d = json.load(open(os.path.join(WARMSTART_DIR, f)))
            if aug not in d:
                continue
            for p, lrd in d[aug].items():
                ws_raw.setdefault(float(p), {})[seed] = _best_lr_acc(lrd)

    all_p = sorted(set(list(cold.keys()) + list(ws_raw.keys())))
    ws_seeds = sorted({s for pv in ws_raw.values() for s in pv})

    W = 54 + len(ws_seeds) * 10
    print(f"\n{'='*W}")
    print(f"  {label}")
    seed_hdr = "  ".join(f"s{s:>3}" for s in ws_seeds)
    print(f"  {'ρ':>6}  {'cold mean±std':>14}  | warm: {seed_hdr}  mean±std")
    print(f"  {'-'*W}")
    for p in all_p:
        cold_str = "---"
        if p in cold:
            cv = cold[p]
            m, s = cv[0], cv[1]
            cold_str = f"{m:6.2f}±{s:.2f}"
        warm_vals = ws_raw.get(p, {})
        seed_cols = "  ".join(f"{warm_vals[s]*100:>6.2f}" if s in warm_vals else f"{'---':>6}"
                              for s in ws_seeds)
        warm_agg = ""
        if warm_vals:
            vs = list(warm_vals.values())
            wm = np.mean(vs) * 100
            ws = np.std(vs, ddof=1) * 100 if len(vs) > 1 else 0.0
            warm_agg = f"  {wm:6.2f}±{ws:.2f}(n={len(vs)})"
        print(f"  {p:>5.1f}%  {cold_str:>14}  | {seed_cols}{warm_agg}")
    print()


def draw(strat, aug, out):
    label, cold_color = METHODS[strat]
    warm_color  = WARM_COLOR[strat]
    cold_marker = COLD_MARKER[strat]
    warm_marker = WARM_MARKER[strat]

    fig, ax = plt.subplots(figsize=(12, 8))

    # ── Cold-start AL（實線）─────────────────────────────────────────────── #
    cold = pool_strategy(strat, aug)
    warm = pool_warmstart(strat, aug)
    _print_table(strat, aug, cold, warm)
    if cold:
        ps   = sorted(cold)
        mean = np.array([cold[p][0] for p in ps])
        std  = np.array([cold[p][1] for p in ps])
        ax.plot(ps, mean, marker=cold_marker, color=cold_color,
                linewidth=3, markersize=9, linestyle="-",
                label=f"{label} (cold-start)")
        ax.fill_between(ps, mean - std, mean + std, color=cold_color, alpha=0.12)
    else:
        print(f"  [warn] no cold-start data for {strat}")

    # ── Warm-start AL（虛線 + 不同 marker + 較淺色）─────────────────────── #
    if warm:
        ps   = sorted(warm)
        mean = np.array([warm[p][0] for p in ps])
        std  = np.array([warm[p][1] for p in ps])
        ax.errorbar(ps, mean, yerr=std, marker=warm_marker, color="black",
                    linewidth=3, markersize=10, linestyle="--",
                    markeredgecolor="black", markeredgewidth=2.2,
                    markerfacecolor=warm_color,
                    capsize=5, elinewidth=1.8, capthick=1.8, ecolor="black",
                    label=f"{label} (warm-start by $\\theta^2_{{\\mathrm{{SimCLR}}}}$)")
    else:
        print(f"  [warn] no warm-start data for {strat} — 先跑指令以產生資料")

    # ── Random passive baseline（灰虛線）────────────────────────────────── #
    rb = random_baseline(aug)
    if rb:
        ps   = sorted(rb)
        mean = np.array([rb[p][0] for p in ps])
        std  = np.array([rb[p][1] for p in ps])
        ax.plot(ps, mean, marker="X", color="#404040",
                linewidth=2.5, markersize=8, linestyle=(0, (6, 3)),
                alpha=0.85, label="Random")
        ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.10)

    # ── Target 水平線（ρ=100%）──────────────────────────────────────────── #
    ax.axhline(y=TARGET_ACC, color="black", linestyle=(0, (8, 4)),
               linewidth=2.2, alpha=0.85, label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)",
                  fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(f"Warm-Start vs. Cold-Start AL ({label})",
                 fontsize=FONT_LABEL, pad=14)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])

    # legend: 固定順序 cold → warm → random → target
    order = [f"{label} (cold-start)", f"{label} (warm-start by $\\theta^2_{{\\mathrm{{SimCLR}}}}$)",
             "Random", "Target"]
    h, l = ax.get_legend_handles_labels()
    pairs = [(l.index(o), o) for o in order if o in l]
    ax.legend([h[i] for i, _ in pairs], [o for _, o in pairs],
              fontsize=FONT_LEGEND + 1, framealpha=0.9, loc="lower right")

    style_ax(ax)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default=AUG)
    ap.add_argument("--strategies", nargs="+",
                    default=["margin", "coreset", "cluster_margin"])
    args = ap.parse_args()

    for strat in args.strategies:
        if strat not in METHODS:
            print(f"[skip] unknown strategy: {strat}"); continue
        out = os.path.join(OUT_DIR, f"5_x_warmstart_al_{strat}.png")
        draw(strat, args.aug, out)


if __name__ == "__main__":
    main()
