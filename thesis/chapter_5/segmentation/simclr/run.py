"""
U-Net encoder SimCLR pretraining — entry point. Mirrors the classification SimCLR
config (bs 256, ep 500, Adam, cosine, τ=0.07, out_dim 32) on grayscale OCT.

Run from repo root, e.g.:
  python3 thesis/chapter_5/segmentation/simclr/run.py \
      --batch_size 256 --epochs 500 --lr 2e-4 --crop_size 128 --micro_bs 32 --device cuda:0

Output: thesis/chapter_5/segmentation/simclr/ckpt/unet_simclr_lr{lr}_bs{bs}_ep{ep}.pkl
        (+ ..._encoder.pkl = Conv1..Conv6 only). Use downstream via
        run_first_iter.py / run_AL.py  --init simclr --simclr_path <encoder.pkl>
        (loads into Optim_U_Net strict=False; decoder stays random).

NOTE: from-scratch SimCLR (no ImageNet for this custom encoder) → SWEEP lr, don't
fix 2e-4. With 736 train imgs, bs256 = few batches/epoch; --all_images uses 1224.
"""
import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from thesis.chapter_5.segmentation.simclr.data import build_pretrain_dataset
from thesis.chapter_5.segmentation.simclr.model import UNetEncoderSimCLR
from thesis.chapter_5.segmentation.simclr.trainer import SimCLRTrainer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default="./ds/segmentation_correct")
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--lr', type=float, default=2e-4, help='SWEEP this (from-scratch != 2e-4)')
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--out_dim', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--n_views', type=int, default=2)
    p.add_argument('--crop_size', type=int, default=128, help='smaller than cls (256) for VRAM')
    p.add_argument('--gradcache', action='store_true',
                   help='use GradCache micro-batching; default OFF = plain full-batch '
                        '(preferred when VRAM allows — unambiguously true bs InfoNCE)')
    p.add_argument('--micro_bs', type=int, default=32, help='GradCache micro-batch (only if --gradcache)')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--all_images', action='store_true', help='use all 1224 (default: 736 train only)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    a.device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    ds, files = build_pretrain_dataset(a.dataroot, a.crop_size, a.fold,
                                       train_only=not a.all_images)
    a.n_train = len(files)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=False, pin_memory=True)

    a.full_batch = not a.gradcache   # default: full-batch (no gradcache)

    model = UNetEncoderSimCLR(img_ch=1, out_dim=a.out_dim)
    opt = torch.optim.Adam(model.parameters(), a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=0)

    SimCLRTrainer(model, opt, sched, a).train(loader, min(a.micro_bs, a.batch_size))


if __name__ == "__main__":
    main()
