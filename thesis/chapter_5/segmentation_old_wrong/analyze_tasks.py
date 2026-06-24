"""
Print answers to the 5 Ch5-segmentation questions from whatever results exist.
Graceful with partial data. Run from repo root anytime:
    python3 thesis/chapter_5/segmentation/analyze_tasks.py
"""
import os, sys, json, glob
import numpy as np

BASE = "thesis/chapter_5/segmentation/exp_results"


def load_seed_files(tree):
    """tree/nuclei/cold_start_random/random_<seed>_bs8.json -> {seed: data}."""
    out = {}
    for f in glob.glob(os.path.join(BASE, tree, "nuclei", "cold_start_random", "random_*_bs8.json")):
        seed = os.path.basename(f).split("_")[1]
        out[seed] = json.load(open(f))
    return out


def dices_at(data, portion, lr=None):
    """list of test_dice at a portion (optionally a specific lr) within one seed file."""
    pk = str(float(portion))
    if pk not in data:
        return []
    vals = []
    for lrk, runs in data[pk].items():
        if lr is not None and lrk != str(lr):
            continue
        vals += [r["test_dice"] for r in runs]
    return vals


def ms(v):
    if not v:
        return "n/a"
    a = np.array(v, float)
    s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return f"{a.mean():.4f} +/- {s:.4f} (n={len(a)})"


print("=" * 72)
print("TASK 1 & 2 : Augmentation ablation (lr=1e-3 fixed)")
print("=" * 72)
for portion in (100, 10):
    print(f"  portion {portion}% :")
    for A in (1, 2, 4):
        seeds = load_seed_files(f"aug_ablation/aug{A}")
        pooled = []
        for sd in seeds.values():
            pooled += dices_at(sd, portion)
        tag = {1: "no-aug ", 2: "2x(HF) ", 4: "4x(HFV)"}[A]
        print(f"    aug{A} [{tag}] : {ms(pooled)}")

print("\n" + "=" * 72)
print("TASK 3 : LR sensitivity (aug4, seed42)")
print("=" * 72)
lrsw = load_seed_files("lr_sweep")
for portion in (5, 50):
    print(f"  portion {portion}% :")
    data42 = lrsw.get("42", {})
    pk = str(float(portion))
    if pk in data42:
        for lrk in sorted(data42[pk], key=float):
            v = [r["test_dice"] for r in data42[pk][lrk]]
            print(f"    lr={float(lrk):<8.5f} : {ms(v)}")
    else:
        print("    (no data yet)")

print("\n" + "=" * 72)
print("TASK 4 : 10% random, 5 seeds, per-seed best-lr -> mean")
print("=" * 72)
reps = []
for seed in ("10", "24", "38", "42", "57"):
    data = lrsw.get(seed, {})
    pk = str(float(10))
    if pk not in data:
        print(f"  seed {seed}: (no data)")
        continue
    best_lr, best_m = None, -1
    for lrk, runs in data[pk].items():
        m = float(np.mean([r["test_dice"] for r in runs]))
        if m > best_m:
            best_m, best_lr = m, lrk
    reps.append(best_m)
    print(f"  seed {seed}: best lr={best_lr} -> Dice {best_m:.4f}")
if reps:
    a = np.array(reps)
    s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    print(f"  >>> MEAN over {len(reps)} seeds (each best-lr): {a.mean():.4f} +/- {s:.4f}")

print("\n" + "=" * 72)
print("TASK 5 : Margin AL vs random baseline (aug4, seed42)")
print("=" * 72)
base = load_seed_files("main_aug4").get("42", {})
print("  random baseline (passive):")
brow = {}
for pk in sorted(base, key=float):
    v = [r["test_dice"] for r in base[pk].values().__iter__().__next__()] if base[pk] else []
    # simpler: first lr's runs
    lr0 = sorted(base[pk], key=float)[0]
    d = base[pk][lr0][-1]["test_dice"]
    brow[float(pk)] = d
    print(f"    rho={float(pk):5.1f}% : {d}")
alf = os.path.join(BASE, "main_aug4", "nuclei", "AL_random", "margin_seed42_bs8.json")
if os.path.isfile(alf):
    al = json.load(open(alf))
    print("  margin AL  (rho : dice  | random@same | delta):")
    for pk in sorted(al, key=float):
        lr0 = sorted(al[pk], key=float)[0]
        d = al[pk][lr0][-1]["test_dice"]
        # nearest baseline portion
        rp = brow.get(float(pk))
        delta = f"{d-rp:+.4f}" if rp is not None else " n/a"
        rstr = f"{rp}" if rp is not None else "n/a"
        print(f"    {float(pk):5.1f}% : {d}  | {rstr} | {delta}")
else:
    print("  (margin AL json not present yet)")
