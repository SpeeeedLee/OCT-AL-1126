"""
Aggregate + plot Ch5 nuclei-segmentation results.

Convention (mirrors the thesis): per seed, pick that seed's best-lr run (mean of
its runs at each lr), giving one representative test Dice per seed; then report
mean +/- std (ddof=1) over seeds. With the current fixed-lr protocol there is
only one lr, so this reduces to mean+/-std over seeds.

Usage (from repo root):
  python3 thesis/chapter_5/segmentation/aggregate.py            # print tables
  python3 thesis/chapter_5/segmentation/aggregate.py --plot     # + save AL curve
"""
import os
import sys
import json
import glob
import argparse
import numpy as np

ROOT = os.path.join("thesis", "chapter_5", "segmentation", "exp_results", "nuclei")


def _per_seed_best(runs_by_lr):
    """runs_by_lr: {lr: [test_dice,...]} -> representative test Dice (best-lr mean)."""
    best = None
    for lr, vals in runs_by_lr.items():
        if not vals:
            continue
        m = float(np.mean(vals))
        if best is None or m > best:
            best = m
    return best


def _coldstart_test_list(run_list):
    """cold-start run_list = [run_dict,...] -> list of test_dice."""
    return [r["test_dice"] for r in run_list]


def aggregate_coldstart(init="random"):
    """Return {portion: (mean, std, n_seeds)}."""
    files = glob.glob(os.path.join(ROOT, f"cold_start_{init}", "random_*_bs*.json"))
    # portion -> seed -> {lr: [dice]}
    table = {}
    for f in files:
        seed = os.path.basename(f).split("_")[1]
        data = json.load(open(f))
        for portion, lrs in data.items():
            for lr, run_list in lrs.items():
                table.setdefault(portion, {}).setdefault(seed, {}).setdefault(lr, [])
                table[portion][seed][lr].extend(_coldstart_test_list(run_list))
    out = {}
    for portion, seeds in table.items():
        reps = [_per_seed_best(lrs) for lrs in seeds.values()]
        reps = [r for r in reps if r is not None]
        if reps:
            out[portion] = (float(np.mean(reps)),
                            float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0,
                            len(reps))
    return out


def aggregate_al(strategy, init="random"):
    """Return {portion: (mean, std, n_seeds)} for an AL strategy."""
    files = glob.glob(os.path.join(ROOT, f"AL_{init}", f"{strategy}_seed*_bs*.json"))
    table = {}
    for f in files:
        seed = os.path.basename(f).split("_seed")[1].split("_")[0]
        data = json.load(open(f))
        for portion, lrs in data.items():
            for lr, run_list in lrs.items():
                dices = [r["test_dice"] for r in run_list]
                table.setdefault(portion, {}).setdefault(seed, {}).setdefault(lr, []).extend(dices)
    out = {}
    for portion, seeds in table.items():
        reps = [_per_seed_best(lrs) for lrs in seeds.values()]
        reps = [r for r in reps if r is not None]
        if reps:
            out[portion] = (float(np.mean(reps)),
                            float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0,
                            len(reps))
    return out


def _print_curve(name, curve):
    if not curve:
        print(f"  [{name}] no data")
        return
    print(f"  [{name}]")
    for p in sorted(curve, key=float):
        m, s, n = curve[p]
        print(f"    rho={float(p):6.1f}%  Dice={m:.4f} +/- {s:.4f}  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="random")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--strategies", default="random margin coreset cluster_margin")
    args = ap.parse_args()

    print("=" * 70)
    print("COLD-START (passive random baseline)")
    print("=" * 70)
    cs = aggregate_coldstart(args.init)
    _print_curve("cold_start", cs)

    print("\n" + "=" * 70)
    print("ACTIVE LEARNING")
    print("=" * 70)
    curves = {}
    for strat in args.strategies.split():
        curves[strat] = aggregate_al(strat, args.init)
        _print_curve(strat, curves[strat])

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        styles = {"random": ("gray", "--"), "margin": ("tab:blue", "-"),
                  "coreset": ("tab:green", "-"), "cluster_margin": ("tab:red", "-")}
        for strat, curve in curves.items():
            if not curve:
                continue
            xs = sorted(curve, key=float)
            x = [float(p) for p in xs]
            y = [curve[p][0] for p in xs]
            e = [curve[p][1] for p in xs]
            c, ls = styles.get(strat, ("black", "-"))
            ax.errorbar(x, y, yerr=e, label=strat, color=c, linestyle=ls,
                        marker="o", markersize=3, capsize=2)
        ax.set_xlabel("Labeled portion (%)")
        ax.set_ylabel("Test Dice")
        ax.set_title("Active Learning for Nuclei Segmentation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_dir = os.path.join("thesis", "chapter_5", "segmentation", "figs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "al_curve.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\n✓ Saved AL curve -> {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
