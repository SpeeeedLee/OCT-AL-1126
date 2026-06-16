#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.3　類別重分布消融 (Class-Redistribution Ablation)
====================================================
問題：AL 之所以贏，是因為它把「各類別的標註比例」重分布到特定比重？還是「逐影像挑選」
本身也關鍵？本腳本切開這兩者：

  - **沿用** 某 AL 策略 (margin / coreset / cluster_margin) 在某 portion、跨 5 seeds 的
    「各類別累積選取張數的平均」當作 target 各類張數 (class redistribution)。
  - **但實際影像於對應類別中『隨機』抽取**（每個 seed 各自一組隨機抽樣、可 reproduce）。

訓練協定與 cold-start (θ²-SimCLR / aug4 one-shot) 完全一致：
  ResNetSimCLR(θ²)→換 7-class fc、AdamW + LinearLR(1→0) + CE、batch16、epoch20。
  每個 seed 在多個 lr 各 run 3 次，取「該 seed best-mean lr」；最終曲線 = over 5 seeds 的 mean±std。
  （非 iterative：每個 portion 獨立 one-shot，預設只做 ρ=10/20/30。）

結果存到隔離樹（勿與主實驗混）：
  classification/exp_results/classification_hard/redistribution_simclr/
      {strategy}_seed{seed}_bs16.json     結構 {"aug4": {"<portion>": {"<lr>": [acc,...]}}}
  （與 AL_simclr/ 同結構 → plot_5_3_redistribution.py 用同一套 per-seed-best 聚合。）

從 repo root 執行（單 seed 單 portion，便於分卡）：
    python3 thesis/chapter_5/run_5_3_redistribution.py \
        --strategy margin --portion 10 --seed 42 --device cuda:6
批次請用 thesis/chapter_5/run_5_3_redistribution.sh。
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from classification.utils.data import get_data                       # noqa: E402
from classification.utils.train_eval import train_model              # noqa: E402
from classification.model.simclr.resnet_simclr import ResNetSimCLR   # noqa: E402

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
LABELED_IDS_DIR = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                               "AL_simclr", "labeled_ids")
OUT_ROOT = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                        "redistribution_simclr")
SIMCLR_CKPT = os.path.join(REPO, "SSL", "simclr", "ckpt",
                           "resnet18_simclr_lr0.0002_bs256_ep500.pkl")

# target 各類張數的來源 seeds（= 4.4 AL 的 5 seeds）；與訓練用 seed 概念不同。
SRC_SEEDS = [10, 24, 38, 42, 57]
NUM_CLASSES = 7
AUG_KEY = "aug4"
RUNS_PER_LR = 3


def lr_grid_for(portion):
    """沿用 4.3 simclr finetune 精簡 lr 網格（run_4_3_simclr_finetune.sh）。"""
    p = float(portion)
    if p <= 10:
        return ["3e-5", "5e-5", "1e-4", "3e-4"]
    return ["5e-5", "1e-4", "5e-4"]     # 20/30/40


# --------------------------------------------------------------------------- #
# target 各類張數：AL 跨 src_seeds 的 cumulative 各類張數平均 → largest-remainder 取整
# --------------------------------------------------------------------------- #
def _class_of_each_train_index():
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
    return np.array(ds.targets), list(ds.classes)


def al_mean_class_counts(strategy, portion, src_seeds=SRC_SEEDS):
    """讀 labeled_ids/{strategy}_seed{s}_bs16.json，取各 seed 在 portion 的 cumulative
    各類張數，回傳跨 seed 平均 (float, len=7)。"""
    targets, _ = _class_of_each_train_index()
    pk = str(float(portion))
    rows = []
    for s in src_seeds:
        path = os.path.join(LABELED_IDS_DIR, f"{strategy}_seed{s}_bs16.json")
        if not os.path.isfile(path):
            print(f"  [warn] missing {path}")
            continue
        d = json.load(open(path))
        if pk not in d or "cumulative" not in d[pk]:
            print(f"  [warn] portion {pk}/cumulative not in {os.path.basename(path)}")
            continue
        cnt = np.zeros(NUM_CLASSES, dtype=int)
        for idx in d[pk]["cumulative"]:
            cnt[targets[idx]] += 1
        rows.append(cnt)
    if not rows:
        raise RuntimeError(f"no labeled_ids found for {strategy} @ {portion}%")
    return np.array(rows).mean(axis=0), len(rows)


def largest_remainder(float_counts, total):
    """把浮點各類張數四捨五入成整數、且總和剛好 = total（最大餘數法）。"""
    floor = np.floor(float_counts).astype(int)
    rem = total - int(floor.sum())
    if rem > 0:                       # 把剩餘名額給小數部分最大的類別
        order = np.argsort(-(float_counts - floor))
        for i in range(rem):
            floor[order[i % len(floor)]] += 1
    elif rem < 0:                     # 多了則從小數部分最小的扣
        order = np.argsort(float_counts - floor)
        for i in range(-rem):
            floor[order[i % len(floor)]] -= 1
    return floor


def build_redistributed_idx(strategy, portion, seed):
    """回傳 (label_idx, target_counts)：各類張數 = AL 平均；影像於該類隨機抽 (seeded)。"""
    targets, class_names = _class_of_each_train_index()
    mean_counts, n_src = al_mean_class_counts(strategy, portion)
    total = int(round(mean_counts.sum()))
    target = largest_remainder(mean_counts, total)

    class_pools = {c: np.where(targets == c)[0].tolist() for c in range(NUM_CLASSES)}
    rng = random.Random(seed)
    label_idx = []
    for c in range(NUM_CLASSES):
        pool, n = class_pools[c], int(target[c])
        if n > len(pool):
            print(f"  [warn] class {class_names[c]}: need {n} > pool {len(pool)} → cap")
            n = len(pool)
        label_idx += rng.sample(pool, n)

    print(f"  redistribution target ({strategy} @ {portion}%, from {n_src} AL seeds, "
          f"total={target.sum()}):")
    for c in range(NUM_CLASSES):
        print(f"    {class_names[c]:24s}: {int(target[c]):4d}  "
              f"(AL mean {mean_counts[c]:6.1f})")
    return label_idx, target


# --------------------------------------------------------------------------- #
# 模型 + JSON
# --------------------------------------------------------------------------- #
def build_simclr_classifier():
    model = ResNetSimCLR("resnet18", 32)
    sd = torch.load(SIMCLR_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    in_feat = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_feat, NUM_CLASSES, bias=True)
    return model


def load_json(path):
    if os.path.isfile(path):
        try:
            return json.load(open(path))
        except json.JSONDecodeError:
            print(f"  [warn] corrupt JSON {path}, recreating")
    return {}


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # sort: aug → portion(float) → lr(float)
    out = {}
    for ak in sorted(data):
        out[ak] = {}
        for pk in sorted(data[ak], key=float):
            out[ak][pk] = {lk: data[ak][pk][lk] for lk in sorted(data[ak][pk], key=float)}
    json.dump(out, open(path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["margin", "coreset", "cluster_margin"])
    ap.add_argument("--portion", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--aug_factor", type=int, default=4)
    ap.add_argument("--lr_grid", nargs="+", default=None,
                    help="不指定時用 lr_grid_for(portion)")
    ap.add_argument("--runs", type=int, default=RUNS_PER_LR)
    args = ap.parse_args()

    if not os.path.isfile(SIMCLR_CKPT):
        raise FileNotFoundError(SIMCLR_CKPT)
    lr_grid = args.lr_grid or lr_grid_for(args.portion)
    pk = str(float(args.portion))
    out_path = os.path.join(OUT_ROOT, f"{args.strategy}_seed{args.seed}_bs16.json")

    print("=" * 70)
    print(f" Class-redistribution ablation: {args.strategy}  ρ={args.portion}%  seed={args.seed}")
    print(f" lr grid = {lr_grid}   runs/lr = {args.runs}   device = {args.device}")
    print(f" out = {out_path}")
    print("=" * 70)

    # 重分布後的 labeled set（此 seed 的隨機抽樣，target 張數來自 AL 平均）
    label_idx, _ = build_redistributed_idx(args.strategy, args.portion, args.seed)
    data_loaders, dataset_sizes = get_data(DATA_DIR, label_idx, batch_size=16,
                                           data_aug=True, aug_factor=args.aug_factor)
    print("dataset_sizes:", dataset_sizes)

    for lr_str in lr_grid:
        lr = float(lr_str)
        lk = str(lr)                      # 與 cold-start 一致："5e-05"/"0.0001"/...
        data = load_json(out_path)
        done = len(data.get(AUG_KEY, {}).get(pk, {}).get(lk, []))
        if done >= args.runs:
            print(f"[skip] lr={lk}: already {done}/{args.runs} runs")
            continue
        for r in range(done, args.runs):
            print(f"\n----- lr={lk}  run {r + 1}/{args.runs} -----")
            model = build_simclr_classifier()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=lr)
            scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0,
                                              end_factor=0.0, total_iters=args.epoch)
            _, test_acc, _ = train_model(model, args.device, data_loaders, dataset_sizes,
                                         criterion, optimizer, scheduler, num_epochs=args.epoch)
            test_acc = round(float(test_acc), 4)
            print(f"  test_acc = {test_acc}")
            data = load_json(out_path)         # re-read (其他並行 run 可能已寫)
            data.setdefault(AUG_KEY, {}).setdefault(pk, {}).setdefault(lk, []).append(test_acc)
            save_json(data, out_path)
            print(f"  [saved] {out_path}")

    print("\n[done]", out_path)


if __name__ == "__main__":
    main()
