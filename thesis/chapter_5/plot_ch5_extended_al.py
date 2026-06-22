"""
Ch5: Margin + Coreset AL curves extended from ρ=60% to ρ=90%.

- ρ=2.5–60% (2.5% interval): 5-seed mean ± std from main AL_simclr dir.
- ρ=65–90% (5% interval):  seed 42 only, no shading, from ch5_5_3_3_extend.
- Random baseline: cold-start θ² per-seed best-lr, all available portions.
- Target: horizontal line at 88.2%.

Usage (from repo root):
    python3 thesis/chapter_5/plot_ch5_extended_al.py
"""
import os, json, re
import numpy as np
import matplotlib.pyplot as plt

MAIN_AL_DIR = "classification/exp_results/classification_hard/AL_simclr"
EXT_DIR     = "classification/exp_results/ch5_5_3_3_extend/classification_hard/AL_simclr"
CS_DIR      = "classification/exp_results/classification_hard/cold_start_simclr"
OUT         = "thesis/chapter_5/figs/ch5_extended_al.png"
AUG         = "aug4"
TARGET      = 88.2
CS_CFG      = "simclr_lr0.0002_simclr_bs256_simclr_ep500"

FONT_LABEL, FONT_TICK, FONT_LEGEND = 24, 18, 16

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.linewidth": 1.5,
})

STRATEGIES = [
    ("margin",  "Margin",   "#3182BD", "s"),
    ("coreset", "Core-set", "#238B45", "D"),
]


def _acc_list(v):
    """Handles both cold-start (list) and AL ({'acc': list, ...}) formats."""
    if isinstance(v, dict):
        v = v.get("acc", [])
    if isinstance(v, list):
        return [float(x) for x in v if x is not None]
    return []


def _per_seed_best_curve(by_portion):
    out = {}
    for p, seeds in by_portion.items():
        accs = []
        for lrd in seeds.values():
            cand = [np.mean(_acc_list(v)) for v in lrd.values() if _acc_list(v)]
            if cand:
                accs.append(max(cand) * 100.0)
        if accs:
            std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            out[p] = (float(np.mean(accs)), std)
    return out


def load_main_al(strat):
    """5-seed AL data, ρ=2.5–60%."""
    by_portion = {}
    for f in os.listdir(MAIN_AL_DIR):
        if not f.endswith(".json"):
            continue
        if f.split("_seed")[0] != strat:
            continue
        m = re.search(r"seed(\d+)", f)
        seed = m.group(1) if m else f
        d = json.load(open(os.path.join(MAIN_AL_DIR, f)))
        if AUG not in d:
            continue
        for p, lrd in d[AUG].items():
            by_portion.setdefault(float(p), {})[seed] = lrd
    return _per_seed_best_curve(by_portion)


def load_ext_al(strat):
    """Seed-42-only data for ρ=65–90% (skip ρ=60 — already in main)."""
    f = os.path.join(EXT_DIR, f"{strat}_seed42_bs16.json")
    d = json.load(open(f))
    out = {}
    for p, lrd in d.get(AUG, {}).items():
        p = float(p)
        if p == 60.0:
            continue
        cand = [np.mean(_acc_list(v)) for v in lrd.values() if _acc_list(v)]
        if cand:
            out[p] = max(cand) * 100.0
    return out


def load_random():
    """Cold-start θ² per-seed best-lr, all available portions ≤ 90%."""
    by_portion = {}
    # aggregate file (seed42 only; covers 5,10,…,65,70%)
    f42 = os.path.join(CS_DIR, "random42_bs16.json")
    if os.path.isfile(f42):
        d = json.load(open(f42))
        if AUG in d:
            for p, lrd in d[AUG].items():
                by_portion.setdefault(float(p), {})["42"] = lrd
    # per-seed cfg files (cover 2.5,5,10,20,…,90,100%)
    for fn in os.listdir(CS_DIR):
        if CS_CFG not in fn or not fn.endswith(".json"):
            continue
        m = re.search(r"random(\d+)", fn)
        seed = m.group(1) if m else fn
        d = json.load(open(os.path.join(CS_DIR, fn)))
        if AUG not in d:
            continue
        for p, lrd in d[AUG].items():
            by_portion.setdefault(float(p), {})[seed] = lrd
    # only show: 2.5, 5, then 10% intervals (10, 20, ..., 90)
    KEEP = {2.5, 5.0} | set(range(10, 91, 10))
    return _per_seed_best_curve({p: v for p, v in by_portion.items() if p in KEEP})


def style_ax(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=1.5, length=6)
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=1.0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)


def main():
    fig, ax = plt.subplots(figsize=(12, 7))

    for strat, label, color, marker in STRATEGIES:
        main_data = load_main_al(strat)
        ext_data  = load_ext_al(strat)

        # ── segment 1: 2.5–60%, 5 seeds, mean ± std ──────────────────────
        ps_m  = sorted([p for p in main_data if p <= 60.0])
        mu_m  = np.array([main_data[p][0] for p in ps_m])
        std_m = np.array([main_data[p][1] for p in ps_m])
        ax.plot(ps_m, mu_m, color=color, linewidth=2.5,
                marker=marker, markersize=8, label=label)
        ax.fill_between(ps_m, mu_m - std_m, mu_m + std_m,
                        color=color, alpha=0.15, linewidth=0)

        # ── segment 2: 65–90%, seed 42 only, no shading ──────────────────
        ps_e  = sorted(ext_data.keys())
        acc_e = [ext_data[p] for p in ps_e]
        # anchor from the 5-seed mean at ρ=60
        xs60 = [60.0] + ps_e
        ys60 = [main_data[60.0][0]] + acc_e
        ax.plot(xs60, ys60, color=color, linewidth=2.5,
                marker=marker, markersize=8, linestyle="-")

    # ── Random baseline ───────────────────────────────────────────────────
    rd = load_random()
    ps_r  = sorted(rd)
    mu_r  = np.array([rd[p][0] for p in ps_r])
    std_r = np.array([rd[p][1] for p in ps_r])
    ax.plot(ps_r, mu_r, color="#404040", linewidth=2.5,
            marker="X", markersize=8, linestyle="--", label="Random")
    ax.fill_between(ps_r, mu_r - std_r, mu_r + std_r,
                    color="#404040", alpha=0.12, linewidth=0)

    # ── Target line ───────────────────────────────────────────────────────
    ax.axhline(y=TARGET, color="black", linestyle=(0, (8, 4)),
               linewidth=2.0, alpha=0.85, label="Target")

    # ── vertical separator at ρ=60 (single-seed region starts at 65) ─────
    ax.axvline(x=60.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.6)
    ax.text(61.5, ax.get_ylim()[0] + 1.0 if ax.get_ylim()[1] > 0 else 70,
            "seed 42 only", fontsize=11, color="gray", va="bottom")

    ax.set_xlabel(r"Labeled Training Data Ratio $\rho$ (%)",
                  fontsize=FONT_LABEL, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL, labelpad=10)
    ax.set_xticks([5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    style_ax(ax)

    ax.legend(fontsize=FONT_LEGEND, framealpha=0.9, loc="lower right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved → {OUT}")

    # Print summary table
    print(f"\n{'ρ':>6} | {'Margin':>12} | {'Coreset':>12} | {'Random':>12}")
    print("-" * 50)
    all_ps = sorted(set(
        list(load_main_al("margin")) + list(load_ext_al("margin").keys()) +
        list(load_random())
    ))
    main_m = load_main_al("margin"); ext_m = load_ext_al("margin")
    main_c = load_main_al("coreset"); ext_c = load_ext_al("coreset")
    rd = load_random()
    for p in all_ps:
        if p > 90: continue
        def fmt(d_main, d_ext, pp):
            if pp in d_main:
                mu, std = d_main[pp]
                return f"{mu:.1f}±{std:.1f}"
            elif pp in d_ext:
                return f"{d_ext[pp]:.1f}  "
            return "—"
        r_str = f"{rd[p][0]:.1f}±{rd[p][1]:.1f}" if p in rd else "—"
        print(f"{p:>6.1f} | {fmt(main_m, ext_m, p):>12} | {fmt(main_c, ext_c, p):>12} | {r_str:>12}")


if __name__ == "__main__":
    main()
