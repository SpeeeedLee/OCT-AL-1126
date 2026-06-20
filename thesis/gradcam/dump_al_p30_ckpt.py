#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匯出 AL 策略在 ρ=30% 的 fine-tuned ckpt（供 Grad-CAM 使用）。
================================================================
協定與 thesis/chapter_5/run_5_2_confusion.py 位元級一致：
  - 標註集：labeled_ids/{strategy}_seed{seed}_bs16.json 的該 portion "cumulative"
            （seed42、b0=2.5→b=2.5 AL 軌跡，ρ=30% → 610 張）。
  - lr    ：查 AL_simclr 結果挑該 (portion,seed) mean-acc 最高的 best-lr，只訓練一次。
  - 模型  ：θ²-SimCLR(lr2e-4/bs256/ep500) backbone → 7-class fc、aug4、AdamW+LinearLR、bs16、ep20。
唯一差別：train 完把 model.state_dict() 存成 .pth（confusion 那支沒存）。

輸出：thesis/gradcam/ckpt/{strategy}_p{portion:g}_seed{seed}_{aug}x.pth

從 repo root 執行（預設 sequential 跑 margin/coreset/cluster_margin/typiclust）：
    python3 thesis/gradcam/dump_al_p30_ckpt.py --device cuda:6
單一策略：--strategy margin ；已存在會跳過（--force 覆寫）。
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

# 直接複用 confusion 那支的 lookup / build，確保協定完全一致
from thesis.chapter_5.run_5_2_confusion import (   # noqa: E402
    build_model, lookup_lr_and_idx, DATA_DIR, SIMCLR_CKPT,
)
from classification.utils.data import get_data           # noqa: E402
from classification.utils.train_eval import train_model  # noqa: E402

CKPT_DIR = os.path.join(REPO, "thesis", "gradcam", "ckpt")
DEFAULT_STRATEGIES = ["margin", "coreset", "cluster_margin", "typiclust"]


def dump_one(strategy, portion, seed, device, epoch=20, aug_factor=4, force=False):
    out = os.path.join(CKPT_DIR, f"{strategy}_p{portion:g}_seed{seed}_{aug_factor}x.pth")
    if os.path.isfile(out) and not force:
        print(f"[skip] 已存在 {out}（--force 可覆寫）")
        return

    lr, label_idx = lookup_lr_and_idx(strategy, portion, seed)
    print(f"\n{'=' * 60}\n {strategy}  rho={portion}%  seed={seed}  "
          f"best-lr={lr}  n_labeled={len(label_idx)}\n{'=' * 60}")

    data_loaders, dataset_sizes = get_data(DATA_DIR, label_idx, batch_size=16,
                                           data_aug=True, aug_factor=aug_factor)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                      total_iters=epoch)
    model, test_acc, _ = train_model(model, device, data_loaders, dataset_sizes,
                                     criterion, optimizer, scheduler, num_epochs=epoch)

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f" test_acc={test_acc:.4f}\n [saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", nargs="+", default=DEFAULT_STRATEGIES,
                    help="預設 margin coreset cluster_margin typiclust（sequential）")
    ap.add_argument("--portion", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--aug_factor", type=int, default=4)
    ap.add_argument("--device", required=True, help="cuda:N（這台手動指定 GPU）")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(SIMCLR_CKPT):
        raise FileNotFoundError(SIMCLR_CKPT)
    for strat in args.strategy:   # sequential
        dump_one(strat, args.portion, args.seed, args.device,
                 epoch=args.epoch, aug_factor=args.aug_factor, force=args.force)


if __name__ == "__main__":
    main()
