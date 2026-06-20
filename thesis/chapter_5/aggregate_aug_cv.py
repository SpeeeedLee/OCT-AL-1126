#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ch5 — HF vs 4x augmentation 的 5-fold CV 結果彙整 / 印表。

讀 classification/exp_results/chapter5_aug_cv/classification_hard/aug_cv_simclr/
     fold{F}_seed{S}_bs16_ep20.json   （結構：aug_key → portion → lr → [runs]）

彙整邏輯（與論文慣例一致，all per-seed best-lr → 再往上聚合）：
  1. 每 (aug, portion, fold, seed)：挑「該 seed runs 平均最高」的 lr → 該 (fold,seed) 代表值。
  2. 每 (aug, portion, fold)：對 seeds 取平均 → 該 fold 的值
       （portion 100 只有 seed42，故 fold 值 = seed42 代表值）。
  3. 每 (aug, portion)：對 5 個 fold 取 mean ± std（ddof=1）= 報表主數字。
     （= 標準 5-fold CV 報法；fold 是主軸，seed 只是 fold 內選 labeled 子集的變異。）

用法（repo 根）：
  python3 thesis/chapter_5/aggregate_aug_cv.py            # 主表
  python3 thesis/chapter_5/aggregate_aug_cv.py --detail   # 加每個 fold 的細節
"""
import os
import json
import argparse
import numpy as np

AUG_LABEL = {
    "no_aug": "w/o (1x)",
    "aug2_horizontal": "HF (2x)",
    "aug2_vertical": "VF (2x)",
    "aug3": "HF+VF (3x)",
    "aug4": "HF+VF+HVF (4x)",
}
AUG_ORDER = ["no_aug", "aug2_horizontal", "aug2_vertical", "aug3", "aug4"]
PORTIONS = ["5.0", "50.0", "100.0"]


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def _sstd(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def load_records(base_dir):
    """回傳 nested[aug][portion][fold][seed] = {lr: [runs]}。"""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"results dir not found: {base_dir}")
    rec = {}
    for fn in sorted(os.listdir(base_dir)):
        if not (fn.startswith("fold") and fn.endswith(".json")):
            continue
        # fold{F}_seed{S}_bs..._ep...
        try:
            fold = int(fn.split("fold")[1].split("_")[0])
            seed = int(fn.split("seed")[1].split("_")[0])
        except (IndexError, ValueError):
            print(f"  [skip] 無法解析檔名: {fn}")
            continue
        with open(os.path.join(base_dir, fn), "r", encoding="utf-8") as f:
            data = json.load(f)
        for aug_k, by_portion in data.items():
            for portion_k, by_lr in by_portion.items():
                (rec.setdefault(aug_k, {}).setdefault(portion_k, {})
                    .setdefault(fold, {})[seed]) = by_lr
    return rec


def per_seed_best(by_lr):
    """挑該 seed 平均最高的 lr → (rep_acc, best_lr, n_runs, runs_of_best)。"""
    best = None
    for lr_k, runs in by_lr.items():
        if not runs:
            continue
        m = _mean(runs)
        if best is None or m > best[0]:
            best = (m, lr_k, len(runs), list(runs))
    return best  # 可能 None


def aggregate(rec, aug_k, portion_k, pool_runs=False, only_fold=None):
    """
    回傳 (reps, folds, detail)：
      reps   = 主數字的樣本集（mean±std over reps）。
               - 一般 portion：per-seed best-lr 代表值（單 fold→over seeds；多 fold→seed+fold 變異）。
               - pool_runs=True（ρ=100，單 seed）：改收 best-lr 的**個別 runs**
                 → std 來自 best-lr runs（對齊 §4.2 SPECIAL_RHO_100，ddof=1）。
      only_fold（非 None）：只看該 fold（per-fold 表用）。
      folds  = 有資料的 fold 集合
      detail = [(fold, fold_mean, seed_reps), ...]；seed_reps = [(seed, rep_mean, best_lr, n_runs), ...]
    """
    reps, folds, detail = [], set(), []
    by_fold = rec.get(aug_k, {}).get(portion_k, {})
    for fold in sorted(by_fold.keys()):
        if only_fold is not None and fold != only_fold:
            continue
        seed_reps = []
        for seed in sorted(by_fold[fold].keys()):
            b = per_seed_best(by_fold[fold][seed])
            if b is None:
                continue
            seed_reps.append((seed, b[0], b[1], b[2]))   # seed, rep(mean), best_lr, n_runs
            reps.extend(b[3] if pool_runs else [b[0]])   # ρ100: 收 runs；其餘: 收 mean
        if not seed_reps:
            continue
        folds.add(fold)
        detail.append((fold, _mean([r[1] for r in seed_reps]), seed_reps))
    return reps, folds, detail


def fmt(m, s):
    return f"{m*100:5.2f}±{s*100:4.2f}" if not np.isnan(m) else "   NA   "


def print_table(rec, title, only_fold=None):
    """印一張表：rows = portion，columns = 五種 aug，每格 mean±std。
    only_fold=None → 所有 fold pooled；指定 fold → 該 fold（ρ5/50 over seeds、ρ100 over best-lr runs）。"""
    cov_label = "#folds" if only_fold is None else "#seeds"
    cols = "".join(f"{AUG_LABEL[a]:^16}|" for a in AUG_ORDER)
    header = f"{'ρ(%)':>5} |{cols} {cov_label}"
    print("\n" + title)
    print(header)
    print("-" * len(header))
    any_row = False
    for portion_k in PORTIONS:
        pool_runs = (portion_k == "100.0")
        cells, cov = "", []
        for aug_k in AUG_ORDER:
            reps, folds, detail = aggregate(rec, aug_k, portion_k,
                                            pool_runs=pool_runs, only_fold=only_fold)
            cells += f"{fmt(_mean(reps), _sstd(reps)):^16}|"
            if only_fold is None:
                cov.append(len(folds))
            else:
                cov.append(len(detail[0][2]) if detail else 0)   # #seeds in this fold
        if any(c for c in cov):
            any_row = True
        rho = portion_k.rstrip("0").rstrip(".")
        print(f"{rho:>5} |{cells} {'/'.join(str(c) for c in cov)}")
    if not any_row:
        print("   （此 fold 尚無資料）")


def print_lr_detail(rec, seeds=(10, 24, 38, 42, 57)):
    """每個 fold×portion×aug 印一張 seed×lr 明細表（每格 = mean±std over runs，標該 seed best-lr）。"""
    present_folds = set()
    for a in rec:
        for p in rec[a]:
            present_folds.update(rec[a][p].keys())
    print("\n" + "#" * 78)
    print("# LR 明細：每格 = mean±std over runs（標 ◀best = 該 seed 挑中的 lr）")
    print("#" * 78)
    for fold in sorted(present_folds):
        for portion_k in PORTIONS:
            for aug_k in AUG_ORDER:
                # 收集此 (fold,portion,aug) 出現過的所有 lr
                lrset = set()
                for seed in seeds:
                    lrset |= set(rec.get(aug_k, {}).get(portion_k, {}).get(fold, {}).get(seed, {}).keys())
                if not lrset:
                    continue
                lrs = sorted(lrset, key=float)
                rho = portion_k.rstrip("0").rstrip(".")
                print(f"\n=== fold {fold} | ρ={rho}% | {AUG_LABEL[aug_k]} ===")
                print("seed  | " + " | ".join(f"{lr:^11}" for lr in lrs) + " | best")
                for seed in seeds:
                    by_lr = rec.get(aug_k, {}).get(portion_k, {}).get(fold, {}).get(seed)
                    if not by_lr:
                        print(f"{seed:<5} | (no data)")
                        continue
                    b = per_seed_best(by_lr)
                    best_lr = b[1] if b else None
                    cells = []
                    for lr in lrs:
                        r = by_lr.get(lr)
                        if not r:
                            cells.append(f"{'--':^11}")
                        else:
                            s = np.std(r, ddof=1) * 100 if len(r) > 1 else 0.0
                            cells.append(f"{np.mean(r)*100:5.2f}±{s:4.2f}")
                    print(f"{seed:<5} | " + " | ".join(cells) + f" | {best_lr}")


def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--base_dir", default=os.path.join(
        repo_root, "classification/exp_results/chapter5_aug_cv/classification_hard/aug_cv_imagenet"))
    ap.add_argument("--detail", action="store_true", help="印每個 fold 的細節（seed/best-lr/n）")
    ap.add_argument("--lr_detail", action="store_true",
                    help="印每個 fold×portion×aug 的 seed×lr 明細表（每格 mean±std over runs，標 best）")
    args = ap.parse_args()

    rec = load_records(args.base_dir)

    print("\n" + "=" * 92)
    print("Ch5  augmentation — 10-fold CV (init=ImageNet, one-shot, no AL)")
    print("每 fold 一張表：mean±std over seeds（ρ100=單 seed→best-lr runs，ddof=1）；fold 1 = §4.2 原本 split")
    print(f"來源: {args.base_dir}")
    print("=" * 92)

    # 找出目前有資料的 folds
    present_folds = set()
    for aug_k in rec:
        for portion_k in rec[aug_k]:
            present_folds.update(rec[aug_k][portion_k].keys())
    present_folds = sorted(present_folds)

    # ---- 每個 fold 各一張表 ----
    if not present_folds:
        print("\n（尚無任何 fold 資料）")
    for f in present_folds:
        print_table(rec, f"【fold {f}】" + ("（= §4.2 原本 split）" if f == 1 else ""), only_fold=f)

    # ---- 另附：所有 fold pooled（總覽）----
    print_table(rec, "【ALL folds pooled】mean±std over 所有 (fold×seed) reps", only_fold=None)

    # ---- 細節 ----
    if args.detail:
        print("\n" + "=" * 72)
        print("Per-fold 細節（fold: 值  [seed=rep@best-lr×n_runs]）")
        print("=" * 72)
        for portion_k in PORTIONS:
            rho = portion_k.rstrip("0").rstrip(".")
            for aug_k in AUG_ORDER:
                reps, folds, detail = aggregate(rec, aug_k, portion_k, pool_runs=(portion_k == "100.0"))
                print(f"\nρ={rho}%  {AUG_LABEL[aug_k]}  "
                      f"→ mean±std over {len(reps)} reps = {fmt(_mean(reps), _sstd(reps)).strip()}")
                for fold, fval, seed_reps in detail:
                    parts = ", ".join(
                        f"s{seed}={rep*100:.2f}@{lr}×{n}" for seed, rep, lr, n in seed_reps)
                    print(f"   fold {fold}: {fval*100:5.2f}   [{parts}]")

    # ---- lr 明細表 ----
    if args.lr_detail:
        print_lr_detail(rec)

    print("\n注：per-fold 表的 #seeds = 該 fold 各 aug 覆蓋的 seed 數（ρ100 恆為 1）；")
    print("    pooled 表的 #folds = 各 aug 覆蓋的 fold 數。fold 1 來自 §4.2 匯入。")
    print()


if __name__ == "__main__":
    main()
