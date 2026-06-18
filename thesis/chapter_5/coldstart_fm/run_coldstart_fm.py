#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot training on a foundation-model cold-start selected set (Ch.5 §5.2).
================================================================================
Trains the standard θ²-SimCLR one-shot classifier on the image IDs that
`select_coldstart.py` chose for a given (foundation model, portion), so we can
compare a smart cold-start initial set against the random baseline at the SAME
budget — using the EXACT same protocol as run_5_3_redistribution.py
(ResNetSimCLR θ² backbone -> 7-class fc, aug4, AdamW + LinearLR(1->0) + CE,
batch16, epoch20; sweep lr, RUNS_PER_LR runs each).

This is a *standalone* runner — it does NOT modify any core training file. The
AL-initial-pool path instead reuses the existing
`run_AL.py --resume_labeled_ids <json> --resume_from <portion>` (no core change).

Results are written to an isolated tree mirroring redistribution_simclr:
    classification/exp_results/classification_hard/coldstart_fm_simclr/
        {model_id}_seed{seed}_bs16.json   {"aug4": {"<portion>": {"<lr>": [acc,...]}}}

Run from repo root:
    python3 thesis/chapter_5/coldstart_fm/run_coldstart_fm.py \
        --model dinov2:base --portion 10 --seed 42 --device cuda:4
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from classification.utils.data import get_data                       # noqa: E402
from classification.utils.train_eval import train_model              # noqa: E402
# reuse the redistribution one-shot helpers verbatim (same protocol)
from thesis.chapter_5.run_5_3_redistribution import (                # noqa: E402
    build_simclr_classifier, lr_grid_for, load_json, save_json,
    AUG_KEY, RUNS_PER_LR,
)

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
SEL_DIR = os.path.join(REPO, "thesis", "chapter_5", "coldstart_fm", "labeled_ids")
OUT_ROOT = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                        "coldstart_fm_simclr")


def load_selected_idx(model_id, portion):
    path = os.path.join(SEL_DIR, model_id.replace(":", "__") + ".json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no selection json {path}; run select_coldstart.py first")
    d = json.load(open(path))
    pk = str(float(portion))
    if pk not in d:
        raise KeyError(f"portion {pk} not in {path} (have {list(d)})")
    return list(d[pk]["selected"])


def _train_once(lr, label_idx, args, run_tag):
    """One independent training run: fresh DataLoader + fresh model. Returns test_acc.
    Used both sequentially and (with --parallel_runs) concurrently on one GPU, so it
    must NOT share loaders/model across calls."""
    # vary RNG per run so concurrent reps differ (mirrors the sequential reps' variance)
    base = (args.seed if args.seed is not None else 0) * 1000 + run_tag
    torch.manual_seed(base)
    np.random.seed(base)
    data_loaders, dataset_sizes = get_data(DATA_DIR, label_idx, batch_size=16,
                                           data_aug=True, aug_factor=args.aug_factor)
    model = build_simclr_classifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                      end_factor=0.0, total_iters=args.epoch)
    _, test_acc, _ = train_model(model, args.device, data_loaders, dataset_sizes,
                                 criterion, optimizer, scheduler, num_epochs=args.epoch)
    return round(float(test_acc), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id, e.g. dinov2:base")
    ap.add_argument("--portion", type=float, required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="選樣是 deterministic（固定 ID）→ 預設不需 seed，寫 {model}_bs16.json、"
                         "跑 --runs 次取 best-lr。給 seed 則寫 {model}_seed{seed}_bs16.json（legacy）。")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--aug_factor", type=int, default=4)
    ap.add_argument("--lr_grid", nargs="+", default=None)
    ap.add_argument("--runs", type=int, default=RUNS_PER_LR)
    ap.add_argument("--parallel_runs", type=int, default=1,
                    help="同一個 lr 的多個 run 同時丟到同一張 GPU（ThreadPoolExecutor）。"
                         "1=循序（預設）；設 3 = 一個 lr 的 3 個 run 並行。")
    args = ap.parse_args()

    lr_grid = args.lr_grid or lr_grid_for(args.portion)
    pk = str(float(args.portion))
    tag = f"_seed{args.seed}" if args.seed is not None else ""
    out_path = os.path.join(OUT_ROOT, f"{args.model.replace(':', '__')}{tag}_bs16.json")
    label_idx = load_selected_idx(args.model, args.portion)

    print("=" * 70)
    print(f" Cold-start FM one-shot: {args.model}  ρ={args.portion}%  "
          f"seed={'(none, fixed set)' if args.seed is None else args.seed}")
    print(f" |labeled|={len(label_idx)}  lr grid={lr_grid}  runs/lr={args.runs}  dev={args.device}")
    print(f" out = {out_path}")
    print("=" * 70)

    # the labeled set is fixed by selection; runs only vary training stochasticity.
    for lr_str in lr_grid:
        lr = float(lr_str)
        lk = str(lr)
        data = load_json(out_path)
        done = len(data.get(AUG_KEY, {}).get(pk, {}).get(lk, []))
        if done >= args.runs:
            print(f"[skip] lr={lk}: already {done}/{args.runs} runs")
            continue
        run_tags = list(range(done, args.runs))          # which reps still need doing

        if args.parallel_runs > 1 and len(run_tags) > 1:
            from concurrent.futures import ThreadPoolExecutor
            n_par = min(args.parallel_runs, len(run_tags))
            print(f"\n----- lr={lk}  running reps {run_tags} CONCURRENTLY "
                  f"(x{n_par} on {args.device}) -----")
            with ThreadPoolExecutor(max_workers=n_par) as ex:
                accs = list(ex.map(lambda r: _train_once(lr, label_idx, args, r), run_tags))
            data = load_json(out_path)
            data.setdefault(AUG_KEY, {}).setdefault(pk, {}).setdefault(lk, []).extend(accs)
            save_json(data, out_path)
            print(f"  test_accs = {accs}   [saved] {out_path}")
        else:
            for r in run_tags:
                print(f"\n----- lr={lk}  run {r + 1}/{args.runs} -----")
                test_acc = _train_once(lr, label_idx, args, r)
                print(f"  test_acc = {test_acc}")
                data = load_json(out_path)
                data.setdefault(AUG_KEY, {}).setdefault(pk, {}).setdefault(lk, []).append(test_acc)
                save_json(data, out_path)
                print(f"  [saved] {out_path}")

    print("\n[done]", out_path)


if __name__ == "__main__":
    main()
