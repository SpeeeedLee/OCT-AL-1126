#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5.2　混淆矩陣資料產生器 (Confusion-matrix dumper)
==================================================
為 Random + 七種 AL 策略在某 portion（預設 ρ=30%）重訓一次、在 D_test 上 inference，
存下畫混淆矩陣所需的資訊（7×7 counts + 整體 acc + per-class recall）。

協定（與主線一致）：
  - 標註集：
      random      → 該 seed 的 cold-start 隨機子集（random.seed→sample，與 cold_start 完全相同）。
      AL 策略     → labeled_ids/{strategy}_seed{seed}_bs16.json 的該 portion "cumulative"。
  - lr：查既有結果挑「該 (portion,seed) mean-acc 最高的 lr」（random 查 cold_start_simclr，
        AL 查 AL_simclr），**只用該 best-lr 訓練一次**（confusion matrix 無法 mean，故單 run/單 seed）。
  - 模型：θ²-SimCLR(lr2e-4/bs256/ep500) backbone→7-class fc、aug4、AdamW+LinearLR、bs16、ep20。

輸出：thesis/chapter_5/confusion/{strategy}_p{portion:g}_seed{seed}.json
  {strategy, portion, seed, lr, n_labeled, acc, classes[7], matrix[7][7] (rows=true, cols=pred)}

從 repo root 執行（預設一次跑 random + 7 AL）：
    python3 thesis/chapter_5/run_5_2_confusion.py --device cuda:6
單一策略：--strategy margin ；已存在會跳過（--force 覆寫）。
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from classification.utils.data import get_data, get_num_train          # noqa: E402
from classification.utils.train_eval import train_model               # noqa: E402
from classification.model.simclr.resnet_simclr import ResNetSimCLR    # noqa: E402

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
EXP = os.path.join(REPO, "classification", "exp_results", "classification_hard")
AL_DIR = os.path.join(EXP, "AL_simclr")
LABELED_IDS_DIR = os.path.join(AL_DIR, "labeled_ids")
COLDSTART = os.path.join(EXP, "cold_start_simclr", "random{seed}_bs16.json")
SIMCLR_CKPT = os.path.join(REPO, "SSL", "simclr", "ckpt",
                           "resnet18_simclr_lr0.0002_bs256_ep500.pkl")
CONF_DIR = os.path.join(REPO, "thesis", "chapter_5", "confusion")

NUM_CLASSES = 7
AUG_KEY = "aug4"
ALL_STRATEGIES = ["random", "conf", "margin", "entropy", "coreset",
                  "typiclust", "badge", "cluster_margin"]


def _acc_list(v):
    """支援兩種格式：plain list（cold-start）或 {"acc":[...]}（AL_simclr）。"""
    if isinstance(v, dict):
        v = v.get("acc", [])
    return [x for x in v if isinstance(x, (int, float))]


def best_lr(lr_dict):
    """{lr: vals} → mean-acc 最高的 lr (float)。"""
    best = None
    for lr, v in lr_dict.items():
        a = _acc_list(v)
        if not a:
            continue
        m = sum(a) / len(a)
        if best is None or m > best[0]:
            best = (m, float(lr))
    if best is None:
        raise RuntimeError("no acc found for lr lookup")
    return best[1]


def lookup_lr_and_idx(strategy, portion, seed):
    """回傳 (lr, label_idx)。"""
    pk = str(float(portion))
    if strategy == "random":
        path = COLDSTART.format(seed=seed)
        d = json.load(open(path))
        lr = best_lr(d[AUG_KEY][pk])
        tot = get_num_train(DATA_DIR)
        target = round(tot * portion / 100)
        random.seed(seed)
        label_idx = random.sample(list(range(tot)), target)
        return lr, label_idx
    # AL 策略
    al_path = os.path.join(AL_DIR, f"{strategy}_seed{seed}_bs16.json")
    lr = best_lr(json.load(open(al_path))[AUG_KEY][pk])
    ids_path = os.path.join(LABELED_IDS_DIR, f"{strategy}_seed{seed}_bs16.json")
    label_idx = json.load(open(ids_path))[pk]["cumulative"]
    return lr, label_idx


def build_model():
    model = ResNetSimCLR("resnet18", 32)
    sd = torch.load(SIMCLR_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    in_feat = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_feat, NUM_CLASSES, bias=True)
    return model


@torch.no_grad()
def confusion_on_test(model, device, test_loader):
    """回傳 7×7 counts（rows=true, cols=pred）。"""
    model.eval().to(device)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for inputs, labels in test_loader:
        preds = model(inputs.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            cm[t, p] += 1
    return cm


def run_one(strategy, portion, seed, device, epoch=20, aug_factor=4):
    lr, label_idx = lookup_lr_and_idx(strategy, portion, seed)
    print(f"\n{'=' * 60}\n {strategy}  ρ={portion}%  seed={seed}  "
          f"best-lr={lr}  n_labeled={len(label_idx)}\n{'=' * 60}")

    data_loaders, dataset_sizes = get_data(DATA_DIR, label_idx, batch_size=16,
                                           data_aug=True, aug_factor=aug_factor)
    from torchvision import datasets
    classes = list(datasets.ImageFolder(os.path.join(DATA_DIR, "train")).classes)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                      total_iters=epoch)
    model, test_acc, _ = train_model(model, device, data_loaders, dataset_sizes,
                                     criterion, optimizer, scheduler, num_epochs=epoch)

    cm = confusion_on_test(model, device, data_loaders["test"])
    acc_from_cm = float(np.trace(cm) / cm.sum())
    print(f" test_acc(train_model)={test_acc:.4f}  acc(from CM)={acc_from_cm:.4f}  "
          f"(CM total={cm.sum()})")

    os.makedirs(CONF_DIR, exist_ok=True)
    out = os.path.join(CONF_DIR, f"{strategy}_p{portion:g}_seed{seed}.json")
    json.dump({
        "strategy": strategy, "portion": float(portion), "seed": seed,
        "lr": lr, "n_labeled": len(label_idx),
        "acc": acc_from_cm, "classes": classes,
        "matrix": cm.tolist(),
    }, open(out, "w"), indent=2)
    print(f" [saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=ALL_STRATEGIES,
                    help=f"預設全部：{ALL_STRATEGIES}")
    ap.add_argument("--portion", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="已存在也重跑")
    args = ap.parse_args()

    if not os.path.isfile(SIMCLR_CKPT):
        raise FileNotFoundError(SIMCLR_CKPT)
    for strat in args.strategy:
        out = os.path.join(CONF_DIR, f"{strat}_p{args.portion:g}_seed{args.seed}.json")
        if os.path.isfile(out) and not args.force:
            print(f"[skip] {strat}: {out} 已存在（--force 覆寫）")
            continue
        run_one(strat, args.portion, args.seed, args.device, epoch=args.epoch)
    print("\n[done]")


if __name__ == "__main__":
    main()
