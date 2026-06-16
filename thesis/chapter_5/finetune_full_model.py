#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finetune 一個 full-data 模型，供 5.3 的 UMAP 特徵空間視覺化使用，
並把 .pth 存到 thesis/gradcam/ckpt/（與既有 imagenet_p100_*.pth 同處）。

default = θ²-SimCLR (lr0.0002/bs256/ep500) backbone → 換 7-class fc，aug4、ρ=100% finetune。
訓練協定與 classification/run_first_iter_simclr.py 一致：AdamW + LinearLR(1→0) + CrossEntropy、
batch16、epoch20。存 **純 state_dict**（與 gradcam ckpt 格式相容，可被 gradcam_view / UMAP 直接載）。

從 repo root 執行（挑空 GPU，例：實體5 = cuda:6）：
    python3 thesis/chapter_5/finetune_full_model.py --device cuda:6
產出：thesis/gradcam/ckpt/simclr_p100_4x.pth
"""
import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)  # 讓 `classification.*` 可被 import（從 repo root 跑）

from classification.utils.data import get_data, get_num_train          # noqa: E402
from classification.utils.train_eval import train_model                # noqa: E402
from classification.model.simclr.resnet_simclr import ResNetSimCLR     # noqa: E402

CKPT_DIR = os.path.join(REPO, "thesis", "gradcam", "ckpt")
DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class")
# θ² cold-start 彙整檔（4.3 曲線來源）：用來查各 portion 的 best-lr（per-seed）
COLDSTART_AGG = os.path.join(REPO, "classification", "exp_results", "classification_hard",
                             "cold_start_simclr", "random{seed}_bs16.json")


def lookup_best_lr(portion, seed, aug_key="aug4"):
    """從 cold_start_simclr/random{seed}_bs16.json 撈該 (aug, portion) 下 mean-acc 最高的 lr。
    查不到回傳 None。與論文主線（per-seed best-lr）一致。"""
    import json
    path = COLDSTART_AGG.format(seed=seed)
    if not os.path.isfile(path):
        return None
    data = json.load(open(path))
    pk = str(float(portion))
    if aug_key not in data or pk not in data[aug_key]:
        return None
    def mean_acc(v):
        a = v["acc"] if isinstance(v, dict) else v
        return sum(a) / len(a) if a else -1
    lrd = data[aug_key][pk]
    best_lr = max(lrd.items(), key=lambda kv: mean_acc(kv[1]))[0]
    return float(best_lr)


def build_simclr_classifier(num_classes, simclr_ckpt):
    model = ResNetSimCLR("resnet18", 32)
    sd = torch.load(simclr_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    in_feat = model.backbone.fc[0].in_features          # 512
    model.backbone.fc = nn.Linear(in_feat, num_classes, bias=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:6")
    ap.add_argument("--portion", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=None,
                    help="不指定時自動查 cold_start_simclr 該 (portion,seed) 的 best-lr；查不到退回 5e-4")
    ap.add_argument("--aug_factor", type=int, default=4)
    ap.add_argument("--simclr_lr", default="0.0002")
    ap.add_argument("--simclr_bs", default="256")
    ap.add_argument("--simclr_ep", default="500")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    num_classes = 7
    simclr_ckpt = os.path.join(REPO, "SSL", "simclr", "ckpt",
                               f"resnet18_simclr_lr{args.simclr_lr}_bs{args.simclr_bs}_ep{args.simclr_ep}.pkl")
    if not os.path.isfile(simclr_ckpt):
        raise FileNotFoundError(simclr_ckpt)

    # lr：未指定 → 查該 (portion, seed) 的 cold-start best-lr（per-seed，跟論文主線一致）
    if args.lr is None:
        best = lookup_best_lr(args.portion, args.seed)
        args.lr = best if best is not None else 5e-4
        print(f"lr = {args.lr} ({'best-lr from cold_start_simclr' if best is not None else 'fallback 5e-4 (查無)'})")
    else:
        print(f"lr = {args.lr} (user-specified)")

    tot = get_num_train(DATA_DIR)
    target = round(tot * args.portion / 100)
    random.seed(args.seed)
    label_idx = random.sample(list(range(tot)), target)
    print(f"train pool={tot}, labeled={len(label_idx)} (portion={args.portion}%)")

    data_loaders, dataset_sizes = get_data(DATA_DIR, label_idx, batch_size=16,
                                           data_aug=True, aug_factor=args.aug_factor)
    print("dataset_sizes:", dataset_sizes)

    model = build_simclr_classifier(num_classes, simclr_ckpt)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.epoch)

    model, test_acc, best_val = train_model(model, args.device, data_loaders, dataset_sizes,
                                            criterion, optimizer, scheduler, num_epochs=args.epoch)
    print(f"done. test_acc={test_acc:.4f} best_val_loss={best_val:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    out = args.out or os.path.join(CKPT_DIR, f"simclr_p{args.portion:g}_{args.aug_factor}x.pth")
    # 存純 state_dict（與 imagenet_p100_*.pth 相同格式，gradcam_view / UMAP 可 strict 載入）
    torch.save(model.state_dict(), out)
    print(f"[saved] {out}  (test_acc={test_acc:.4f})")


if __name__ == "__main__":
    main()
