#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3　類別重分布消融圖
=====================
三張圖（Margin / Core-set / Cluster-Margin），各比較：
  - 原始 AL 曲線（實線，AL_simclr/）—— 逐影像挑選 + 類別重分布
  - 「Class Redistribution Only」（虛線，redistribution_simclr/）—— 只沿用各類張數、影像隨機抽
  - Random baseline（灰虛線，θ² cold-start）—— 參考下界（被動隨機）
title = "Ablation on <Method>"。

聚合慣例與 4.4 完全一致：per-seed best-lr → over 5 seeds mean±std (ddof=1)。
（redistribution 只有 ρ=10/20/30 三點 → 以散點+虛線連接。）

從 repo root 執行：
    python3 thesis/chapter_5/plot_5_3_redistribution.py
    python3 thesis/chapter_5/plot_5_3_redistribution.py --no_random   # 不畫 Random 參考線
圖存到 thesis/chapter_5/figs/。
"""
import os
import sys
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "chapter_4"))
from plot_al_curve import (pool_strategy, random_baseline,            # noqa: E402
                           _per_seed_best_curve, _per_seed_best, _fmt_lr,
                           AL_DIR, style_ax, FONT_LABEL)

REPO = os.path.dirname(os.path.dirname(HERE))
REDIST_DIR = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                          "redistribution_simclr")
OUT_DIR = os.path.join(HERE, "figs")

# 顯示名 + 顏色（對齊 Ch4 / 5.3 其他圖）
METHODS = {
    "margin":         ("Margin",         "#3182BD"),
    "coreset":        ("Core-set",       "#238B45"),
    "cluster_margin": ("Cluster-Margin", "#FB6A4A"),
}
MARKER = {"margin": "s", "coreset": "D", "cluster_margin": "*"}


def pool_redistribution(strat, aug="aug4"):
    """讀 redistribution_simclr/{strat}_seed*.json → per-seed best-lr → mean±std。"""
    if not os.path.isdir(REDIST_DIR):
        return {}
    by_portion = {}
    for f in os.listdir(REDIST_DIR):
        if not f.endswith(".json") or f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        seed = m.group(1) if m else f
        d = json.load(open(os.path.join(REDIST_DIR, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            by_portion.setdefault(float(p), {})[seed] = lrd
    return _per_seed_best_curve(by_portion)


def _per_seed_best_table(folder, strat, aug):
    """讀 {folder}/{strat}_seed*.json → {seed(int): {portion(float): (acc%, best_lr)}}。"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for f in os.listdir(folder):
        if not f.endswith(".json") or f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        if not m:
            continue
        seed = int(m.group(1))
        d = json.load(open(os.path.join(folder, f)))
        if aug not in d:
            continue
        for p, lrd in d[aug].items():
            b = _per_seed_best(lrd)
            if b:
                out.setdefault(seed, {})[float(p)] = b
    return out


def print_structured(strat, aug):
    """terminal 印 per-seed best-lr 結果（AL 與 Redistribution 並排）+ mean±std。"""
    label = METHODS[strat][0]
    al = _per_seed_best_table(AL_DIR, strat, aug)
    rd = _per_seed_best_table(REDIST_DIR, strat, aug)
    portions = sorted({p for tbl in (al, rd) for sd in tbl.values() for p in sd})
    # 只關注有 redistribution 的 portion（預設 10/20/30）；若還沒跑就全列
    rd_portions = sorted({p for sd in rd.values() for p in sd})
    focus = rd_portions or portions
    seeds = sorted({s for tbl in (al, rd) for s in tbl})

    print("\n" + "=" * 84)
    print(f"  Class-Redistribution Ablation — {label}  (aug={aug})")
    print(f"  格子 = test_acc(%) (best-lr)   |   per-seed best-lr → over-seeds mean±std (ddof=1)")
    print("=" * 84)

    def block(title, tbl):
        print(f"\n  ▸ {title}")
        head = f"    {'ρ(%)':>6} | " + " ".join(f"seed{s:<9}" for s in seeds) + f" ‖ {'mean':>6} {'std':>5}"
        print(head)
        print("    " + "-" * (len(head) - 4))
        for p in focus:
            cells, accs = [], []
            for s in seeds:
                hit = tbl.get(s, {}).get(p)
                if hit:
                    cells.append(f"{hit[0]:5.2f}({_fmt_lr(hit[1])})".ljust(13))
                    accs.append(hit[0])
                else:
                    cells.append("—".center(13))
            if accs:
                mean = float(np.mean(accs))
                std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
                agg = f" ‖ {mean:6.2f} {std:5.2f}"
            else:
                agg = f" ‖ {'—':>6} {'—':>5}"
            print(f"    {p:>6.1f} | " + " ".join(cells) + agg)

    if al:
        block(f"{label}  (original AL: per-image + redistribution)", al)
    if rd:
        block(f"{label} (Class Redistribution Only: random within class)", rd)
    else:
        print("\n  ▸ (Class Redistribution Only: 尚無資料 — 先跑 run_5_3_redistribution.sh)")

    # Δ = redistribution − AL（同 portion；正=重分布即可達到/超越 AL）
    if al and rd:
        print(f"\n  ▸ Δ (Redistribution − AL) at matched ρ   [越接近 0 → 類別比例即足以解釋]")
        for p in focus:
            a = [al[s][p][0] for s in seeds if al.get(s, {}).get(p)]
            r = [rd[s][p][0] for s in seeds if rd.get(s, {}).get(p)]
            if a and r:
                print(f"    ρ={p:>4g}% :  AL {np.mean(a):6.2f}   Redist {np.mean(r):6.2f}   "
                      f"Δ {np.mean(r) - np.mean(a):+5.2f} pp")
    print("=" * 84)


def draw(strat, aug, with_random, out):
    label, color = METHODS[strat]
    fig, ax = plt.subplots(figsize=(12, 8))

    # 原始 AL（實線）
    al = pool_strategy(strat, aug)
    if al:
        ps = sorted(al)
        mean = np.array([al[p][0] for p in ps]); std = np.array([al[p][1] for p in ps])
        ax.plot(ps, mean, marker=MARKER[strat], color=color, linewidth=3, markersize=9,
                label=label)
        ax.fill_between(ps, mean - std, mean + std, color=color, alpha=0.12)
    else:
        print(f"  [warn] no AL data for {strat}")

    # Class Redistribution Only（虛線）
    rd = pool_redistribution(strat, aug)
    if rd:
        ps = sorted(rd)
        mean = np.array([rd[p][0] for p in ps]); std = np.array([rd[p][1] for p in ps])
        ax.errorbar(ps, mean, yerr=std, marker=MARKER[strat], color=color, linewidth=3,
                    markersize=11, linestyle="--", markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=2.2, capsize=5, elinewidth=2,
                    label=f"{label} (Class Redistribution Only)")
        for p in ps:
            print(f"    redist {strat} ρ={p:g}%: {rd[p][0]:.2f} ± {rd[p][1]:.2f}")
    else:
        print(f"  [warn] no redistribution data for {strat} (先跑 run_5_3_redistribution.sh)")

    # Random baseline（灰虛線，參考）
    if with_random:
        rb = random_baseline(aug)
        if rb:
            ps = sorted(rb)
            mean = np.array([rb[p][0] for p in ps]); std = np.array([rb[p][1] for p in ps])
            ax.plot(ps, mean, marker="X", color="#404040", linewidth=2.5, markersize=8,
                    linestyle=(0, (6, 3)), alpha=0.85, label="Random")
            ax.fill_between(ps, mean - std, mean + std, color="#404040", alpha=0.10)

    # Target performance 水平參考線（88.2%，與 Ch4 al_curve 一致）
    ax.axhline(y=88.2, color="black", linestyle=(0, (8, 4)), linewidth=2.2, alpha=0.85,
               label="Target")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_title(f"Ablation on Class Re-distribution ({label})", fontsize=FONT_LABEL, pad=14)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60])
    # legend 固定順序：AL → Class Redistribution Only → Random → Target
    order = [label, f"{label} (Class Redistribution Only)", "Random", "Target"]
    h, l = ax.get_legend_handles_labels()
    pairs = [(l.index(o), o) for o in order if o in l]
    ax.legend([h[i] for i, _ in pairs], [o for _, o in pairs],
              fontsize=17, framealpha=0.9, loc="lower right")
    style_ax(ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[saved] {out}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", default="aug4")
    ap.add_argument("--no_random", action="store_true", help="不畫 Random 參考線")
    ap.add_argument("--strategy", nargs="+", default=list(METHODS.keys()))
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    for strat in args.strategy:
        print_structured(strat, args.aug)
        out = os.path.join(args.out_dir, f"5_3_redistribution_{strat}.png")
        draw(strat, args.aug, not args.no_random, out)


if __name__ == "__main__":
    main()
