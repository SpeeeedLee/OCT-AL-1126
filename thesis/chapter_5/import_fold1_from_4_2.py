#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 §4.2 data-aug 實驗（ImageNet init、原本 8:1:1 split）的結果匯入成本 CV 的 **fold 1**。

理由：fold 1（10-chunk 視窗 shift 0）= 原本的 train/val/test split，且 §4.2 就是在這個 split 上
用 ImageNet init 跑 w/o·HF·VF·HF+VF·HF+VF+HVF 五種 aug → fold 1 的資料**已經有了**，不必重跑。

來源： classification/exp_results/classification_hard/cold_start_imagenet/random{seed}_bs16_ep20.json
目標： classification/exp_results/chapter5_aug_cv/classification_hard/aug_cv_imagenet/
              fold1_seed{seed}_bs16_ep20.json
只搬： aug ∈ {no_aug, aug2_horizontal, aug2_vertical, aug3, aug4}（跳過 *_jitter_* 變體）
       portion ∈ {5, 50, 100}；每個 lr 的 runs 原樣複製（彙整時 per-seed best-lr）。

用法（repo 根）：
  python3 thesis/chapter_5/import_fold1_from_4_2.py            # 實際寫入
  python3 thesis/chapter_5/import_fold1_from_4_2.py --dry_run  # 只印會搬什麼
"""
import os
import json
import argparse

SEEDS = [10, 24, 38, 42, 57]
AUGS = ["no_aug", "aug2_horizontal", "aug2_vertical", "aug3", "aug4"]
PORTIONS = ["5.0", "50.0", "100.0"]


def main():
    ap = argparse.ArgumentParser()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--src_dir", default=os.path.join(
        repo_root, "classification/exp_results/classification_hard/cold_start_imagenet"))
    ap.add_argument("--dst_dir", default=os.path.join(
        repo_root, "classification/exp_results/chapter5_aug_cv/classification_hard/aug_cv_imagenet"))
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    total_cells = 0
    for seed in SEEDS:
        src = os.path.join(args.src_dir, f"random{seed}_bs16_ep20.json")
        if not os.path.isfile(src):
            print(f"[skip] 來源不存在: {src}")
            continue
        with open(src, "r", encoding="utf-8") as f:
            data = json.load(f)

        out, n_cells = {}, 0
        for aug in AUGS:
            if aug not in data:
                continue
            for p in PORTIONS:
                if p not in data[aug]:
                    continue
                lrs = {lr: list(runs) for lr, runs in data[aug][p].items()}
                out.setdefault(aug, {})[p] = lrs
                n_cells += 1
        if not out:
            print(f"[skip] seed {seed}: 無可搬資料")
            continue

        dst = os.path.join(args.dst_dir, f"fold1_seed{seed}_bs16_ep20.json")
        cov = ", ".join(f"{aug}:{sorted(out[aug].keys())}" for aug in out)
        print(f"seed {seed}: {n_cells} cells → {os.path.basename(dst)}\n    {cov}")
        total_cells += n_cells
        if not args.dry_run:
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)

    print(f"\n{'[dry-run] ' if args.dry_run else ''}fold 1 匯入完成：共 {total_cells} 個 (aug×portion) cell。")
    if not args.dry_run:
        print("→ 之後 run_5_x_aug_cv.sh 只需跑 folds 2..5；aggregate_aug_cv.py 會把 fold 1 一起算。")


if __name__ == "__main__":
    main()
