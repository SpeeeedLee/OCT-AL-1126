"""
Analyzer for the RACE-SAFE re-run trees:
  aug_lr/   : fair aug ablation, each (portion,aug) swept over lr -> best-lr
  lr_v2/    : Task3 (p5,p50 lr sensitivity) + Task4 (p10 x 5 seeds)
  base_v2/  : Task5 passive random baseline
  main_aug4/AL_random/margin_seed42 : Task5 margin AL (single-writer, trusted)
Run from repo root: python3 thesis/chapter_5/segmentation/analyze_v2.py
"""
import os, json, glob
import numpy as np

B = "thesis/chapter_5/segmentation/exp_results"


def read_one(path):
    """A cold_start file with a single portion/lr -> (portion, lr, [dice])."""
    d = json.load(open(path))
    p = list(d)[0]; lr = list(d[p])[0]
    return float(p), float(lr), [r["test_dice"] for r in d[p][lr]]


def ms(v):
    if not v: return "n/a"
    a = np.array(v, float)
    s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return f"{a.mean():.4f}+/-{s:.4f}(n={len(a)})"


# ---------- Task 1&2 : fair aug ablation (best-lr per aug) ----------
print("=" * 74)
print("TASK 1&2 : FAIR aug ablation  (each aug at its OWN best-lr)  [aug_lr/, seed42]")
print("=" * 74)
for P in (100, 10):
    print(f"  portion {P}% :")
    for A in (1, 2, 4):
        files = glob.glob(f"{B}/aug_lr/p{P}_a{A}_lr*/nuclei/cold_start_random/random_42_bs8.json")
        curve = {}
        for f in files:
            _, lr, dl = read_one(f)
            if dl: curve[lr] = float(np.mean(dl))
        if not curve:
            print(f"    aug{A}: (no data yet)"); continue
        best_lr = max(curve, key=curve.get)
        grid = "  ".join(f"{lr:g}:{curve[lr]:.4f}" for lr in sorted(curve))
        tag = {1: "none", 2: "2xHF", 4: "4xHFV"}[A]
        print(f"    aug{A}[{tag:5}] best lr={best_lr:g} -> {curve[best_lr]:.4f}   | grid: {grid}")


# ---------- Task 3 : lr sensitivity ----------
print("\n" + "=" * 74)
print("TASK 3 : LR sensitivity (aug4, seed42)  [lr_v2/]")
print("=" * 74)
f42 = f"{B}/lr_v2/nuclei/cold_start_random/random_42_bs8.json"
if os.path.isfile(f42):
    d = json.load(open(f42))
    for P in (5, 50):
        pk = str(float(P))
        if pk in d:
            print(f"  portion {P}% :")
            for lr in sorted(d[pk], key=float):
                print(f"    lr={float(lr):<8g}: {ms([r['test_dice'] for r in d[pk][lr]])}")
        else:
            print(f"  portion {P}%: (no data yet)")
else:
    print("  (no lr_v2 seed42 file yet)")


# ---------- Task 4 : 10% random, 5 seeds, per-seed best-lr ----------
print("\n" + "=" * 74)
print("TASK 4 : 10% random, 5 seeds, per-seed best-lr -> mean  [lr_v2/]")
print("=" * 74)
reps = []
for S in ("10", "24", "38", "42", "57"):
    f = f"{B}/lr_v2/nuclei/cold_start_random/random_{S}_bs8.json"
    if not os.path.isfile(f):
        print(f"  seed {S}: (no data)"); continue
    d = json.load(open(f)); pk = str(float(10))
    if pk not in d:
        print(f"  seed {S}: (no p10 data)"); continue
    cur = {lr: float(np.mean([r["test_dice"] for r in d[pk][lr]])) for lr in d[pk]}
    blr = max(cur, key=cur.get); reps.append(cur[blr])
    print(f"  seed {S}: best lr={float(blr):g} -> {cur[blr]:.4f}   (grid {len(cur)} lrs)")
if reps:
    a = np.array(reps)
    s = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    print(f"  >>> MEAN over {len(reps)} seeds: {a.mean():.4f} +/- {s:.4f}")


# ---------- Task 5 : margin AL vs passive baseline ----------
print("\n" + "=" * 74)
print("TASK 5 : margin AL vs passive random baseline (aug4, seed42)")
print("=" * 74)
rand = {}
bf = f"{B}/base_v2/nuclei/cold_start_random/random_42_bs8.json"
if os.path.isfile(bf):
    d = json.load(open(bf))
    for pk in d:
        lr = sorted(d[pk], key=float)[0]
        rand[float(pk)] = d[pk][lr][-1]["test_dice"]
# dense low-rho random from preliminary AL-random run (aug4,lr1e-3,seed42)
old = f"{B}/nuclei/AL_random/random_seed42_bs8.json"
if os.path.isfile(old):
    d = json.load(open(old))
    for pk in d:
        lr = sorted(d[pk], key=float)[0]
        rand.setdefault(float(pk), d[pk][lr][-1]["test_dice"])
mf = f"{B}/main_aug4/nuclei/AL_random/margin_seed42_bs8.json"
if os.path.isfile(mf):
    d = json.load(open(mf))
    print(f"  {'rho%':>6} {'margin':>8} {'random':>8} {'delta':>8}")
    for pk in sorted(d, key=float):
        lr = sorted(d[pk], key=float)[0]
        m = d[pk][lr][-1]["test_dice"]
        p = float(pk); r = rand.get(p)
        dd = f"{m-r:+.4f}" if r is not None else "  --"
        rs = f"{r:.4f}" if r is not None else "  --"
        print(f"  {p:6.1f} {m:8.4f} {rs:>8} {dd:>8}")
else:
    print("  (margin AL json not present)")
