#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One command to (1) regenerate the 4 figures (aug_curve + uncertainty/diversity/
hybrid) and (2) print structured DICE tables for every method: per portion, a
seed x lr grid of mean test-Dice, with each seed's BEST lr in bold, plus the
per-seed-best -> mean±std aggregate (the thesis convention).

Run (repo root):
    python3 thesis/chapter_5/segmentation/report_table.py
"""
import os, glob, json
import numpy as np

HERE = os.path.dirname(__file__)
EXP = os.path.join(HERE, "exp_results")
B, E = "\033[1m", "\033[0m"   # ANSI bold

# method label -> (kind, glob pattern, strat-or-None)
METHODS = [
    ("Aug w/o (1x)",   f"{EXP}/aug_curve/aug1/nuclei/cold_start_random/random_*_bs8.json", None),
    ("Aug HF (2x)",    f"{EXP}/aug_curve/aug2/nuclei/cold_start_random/random_*_bs8.json", None),
    ("Aug 4x (=Random)", f"{EXP}/aug_curve/aug4/nuclei/cold_start_random/random_*_bs8.json", None),
    ("Margin",         f"{EXP}/*_sweep/nuclei/AL_random/margin_seed*_bs8.json",         "margin"),
    ("Entropy",        f"{EXP}/*_sweep/nuclei/AL_random/entropy_seed*_bs8.json",        "entropy"),
    ("Core-set",       f"{EXP}/*_sweep/nuclei/AL_random/coreset_seed*_bs8.json",        "coreset"),
    ("Cluster-Margin", f"{EXP}/*_sweep/nuclei/AL_random/cluster_margin_seed*_bs8.json", "cluster_margin"),
]


def load(pattern, strat):
    """-> {portion(float): {seed(str): {lr(str): mean_dice}}}."""
    out = {}
    for f in glob.glob(pattern):
        base = os.path.basename(f)
        if strat is None:
            seed = base.split("_")[1]                 # random_<seed>_bs8
        else:
            if base.split("_seed")[0] != strat:        # cluster_margin vs margin
                continue
            seed = base.split("_seed")[1].split("_")[0]
        d = json.load(open(f))
        for p, lrd in d.items():
            for lr, runs in lrd.items():
                m = float(np.mean([r["test_dice"] for r in runs]))
                out.setdefault(float(p), {}).setdefault(seed, {})[lr] = m
    return out


def print_method(label, data):
    print("\n" + "=" * 78)
    print(f"  METHOD: {label}")
    print("=" * 78)
    if not data:
        print("  (no data yet)")
        return
    for p in sorted(data):
        seeds = data[p]
        lrs = sorted({lr for sd in seeds.values() for lr in sd}, key=float)
        header = "  ρ={:>5}% | ".format(p) + " ".join(f"{('lr'+lr):>10}" for lr in lrs)
        print("\n" + header)
        print("  " + "-" * (len(header) - 2))
        per_seed_best = []
        for s in sorted(seeds, key=lambda x: int(x) if x.isdigit() else x):
            row = seeds[s]
            best_lr = max(row, key=row.get) if row else None
            if best_lr is not None:
                per_seed_best.append(row[best_lr])
            cells = []
            for lr in lrs:
                if lr in row:
                    txt = f"{row[lr]:.4f}"
                    cells.append(f"{B}{txt:>10}{E}" if lr == best_lr else f"{txt:>10}")
                else:
                    cells.append(f"{'-':>10}")
            print(f"  seed{s:>5} | " + " ".join(cells))
        if per_seed_best:
            a = np.array(per_seed_best)
            std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
            print(f"  {'agg':>9} | per-seed-best -> mean±std = "
                  f"{B}{a.mean():.4f}{E} ± {std:.4f}  (n={len(a)} seeds)")


def main():
    for label, pattern, strat in METHODS:
        print_method(label, load(pattern, strat))

    # regenerate the 4 figures
    print("\n" + "#" * 78)
    print("# regenerating figures")
    print("#" * 78)
    import importlib, sys
    sys.path.insert(0, HERE)
    for mod in ("plot_aug_curve", "plot_al_groups"):
        m = importlib.import_module(mod)
        m.main()


if __name__ == "__main__":
    main()
