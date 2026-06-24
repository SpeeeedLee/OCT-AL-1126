"""
U-Net-encoder AUTOENCODER pretraining — entry point (Fu et al., TMI 2024 recipe).

Plain (skip-free) autoencoder reconstructs the OCT images (MSE); only the encoder
(Conv1..Conv6) is saved for downstream transfer. Paper config: AdamW lr 1e-5,
weight_decay 0.01, batch size 8, 100 epochs, 5-epoch lr warm-up.

Run from repo root, e.g.:
  python3 thesis/chapter_5/segmentation/autoencoder/run.py \
      --batch_size 8 --epochs 100 --lr 1e-5 --device cuda:5

Output: thesis/chapter_5/segmentation/autoencoder/ckpt/
        unet_ae_lr{lr}_bs{bs}_ep{ep}.pkl           (full AE)
        unet_ae_lr{lr}_bs{bs}_ep{ep}_encoder.pkl   (Conv1..Conv6 only)
Use downstream:  run_first_iter.py --init ae --simclr_path <...>_encoder.pkl
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

from thesis.chapter_5.segmentation.autoencoder.data import build_ae_dataset
from thesis.chapter_5.segmentation.autoencoder.model import UNetAutoencoder
from thesis.chapter_5.segmentation.autoencoder.trainer import AETrainer

HERE = os.path.dirname(__file__)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', default="./ds/segmentation_correct")
    p.add_argument('--batch_size', type=int, default=8)      # paper
    p.add_argument('--epochs', type=int, default=100)        # paper
    p.add_argument('--lr', type=float, default=1e-5)         # paper (AdamW)
    p.add_argument('--wd', type=float, default=0.01)         # paper
    p.add_argument('--warmup', type=int, default=5)          # paper: 5-epoch warm-up
    p.add_argument('--save_every', type=int, default=25, help='dump encoder ckpt every N epochs')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--all_images', action='store_true', help='use all imgs (default: 736 train only)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--ckpt_dir', default=os.path.join(HERE, 'ckpt'))
    return p.parse_args()


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    a.device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    ds, files = build_ae_dataset(a.dataroot, a.fold, train_only=not a.all_images)
    a.n_train = len(files)
    print(f"AE pretrain on {len(files)} images (bs={a.batch_size}, {a.epochs} ep, "
          f"AdamW lr={a.lr} wd={a.wd}, warmup={a.warmup})")
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=False, pin_memory=True)

    model = UNetAutoencoder(img_ch=1, output_ch=1)
    AETrainer(model, a).train(loader)


if __name__ == "__main__":
    main()
