"""
Train the skip-free U-Net AUTOENCODER (Fu et al. TMI'24 recipe) and save, at the
snapshot epochs, BOTH the FULL U-Net weights (encoder+decoder) and the 10 fixed
images' reconstructions — into a structured folder. Also logs per-epoch MSE.

LR: 5-ep linear warm-up (2e-6->1e-5) then COSINE decay 1e-5->0 over the full run.
Prints EVERY epoch.  Use plot_ae_progress.py (separately) to draw the loss curve
and the reconstruction grid from the folders below.

Outputs (HERE = autoencoder/):
  ckpt/unet_ae_lr{lr}_bs{bs}_run{EP}_ep{E:04d}_full.pkl   full enc+dec, at each snapshot
  recon/original/{00..09}.png                              the 10 input images
  recon/ep{E:04d}/{00..09}.png                             reconstructions at epoch E
  recon/loss_log.csv                                       epoch,recon_mse,lr  (every epoch)
  recon/images.txt                                         which dataset file each index is

Run (repo root):
  python3 thesis/chapter_5/segmentation/autoencoder/train_ae.py --epochs 2000 --device cuda:5
"""
import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from thesis.chapter_5.segmentation.autoencoder.data import WIDTH, HEIGHT
from thesis.chapter_5.segmentation.autoencoder.model import UNetAutoencoder
from thesis.chapter_5.segmentation.utils.data import data_loader, o_data

HERE = os.path.dirname(__file__)
CKPT = os.path.join(HERE, "ckpt")
RECON = os.path.join(HERE, "recon")
# epochs at which we save full ckpt + reconstructions
SNAP = [1, 3, 10, 30, 50, 100, 200, 500, 1000, 1500, 2000]
N_IMGS = 10


class CachedFlipDataset(Dataset):
    """736 base images preloaded once ([0,1]); random H/V flip per item (online 4x)."""
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img = self.base[i]
        if np.random.rand() < 0.5: img = img[:, ::-1, :]
        if np.random.rand() < 0.5: img = img[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img)).float()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", default="./ds/segmentation_correct")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick_seed", type=int, default=42, help="seed for the 10 display images")
    p.add_argument("--device", default="cuda:5")
    return p.parse_args()


def lr_at(ep, base, warmup, epochs):
    if warmup and ep <= warmup:
        return 2e-6 + (ep / warmup) * (base - 2e-6)
    prog = (ep - warmup) / max(epochs - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * min(prog, 1.0)))


def save_row(arrs, folder):
    os.makedirs(folder, exist_ok=True)
    for k, a in enumerate(arrs):
        img = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(os.path.join(folder, f"{k:02d}.png"))


@torch.no_grad()
def reconstruct(model, fixed, device):
    model.eval()
    return model(fixed.to(device)).cpu().numpy()[:, 0]


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT, exist_ok=True); os.makedirs(RECON, exist_ok=True)

    opath = os.path.join(a.dataroot, "image") + "/"
    train_LD, _, _ = data_loader(opath, 0)
    files = [f for b in train_LD for f in b]
    base = np.stack([o_data(opath, [f], WIDTH, HEIGHT)[0] for f in files]).astype(np.float32)
    loader = DataLoader(CachedFlipDataset(base), batch_size=a.batch_size, shuffle=True,
                        num_workers=8, drop_last=False, pin_memory=True)

    # 10 fixed display images (canonical orientation) -> recon/original/
    # NOTE: data_loader's file ORDER is not deterministic (the train/val/test SET is,
    # but the within-set order varies), so sample from sorted(files) to make the pick
    # reproducible across runs / regenerations.
    pick = random.Random(a.pick_seed).sample(sorted(files), N_IMGS)
    fixed_np = np.stack([o_data(opath, [f], WIDTH, HEIGHT)[0] for f in pick]).astype(np.float32)
    fixed = torch.from_numpy(np.ascontiguousarray(fixed_np)).float()
    save_row(fixed_np[:, 0], os.path.join(RECON, "original"))
    with open(os.path.join(RECON, "images.txt"), "w") as f:
        for k, name in enumerate(pick):
            f.write(f"{k:02d}\t{name}\n")
    print(f"saved 10 inputs -> recon/original/  ({pick})", flush=True)

    model = UNetAutoencoder(img_ch=1, output_ch=1).to(device)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    tag = f"unet_ae_lr{a.lr}_bs{a.batch_size}_run{a.epochs}"

    log = open(os.path.join(RECON, "loss_log.csv"), "w")
    log.write("epoch,recon_mse,lr\n"); log.flush()

    for ep in range(1, a.epochs + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(ep, a.lr, a.warmup, a.epochs)
        model.train()
        tot = 0.0
        for x in loader:
            x = x.to(device, non_blocking=True)
            loss = mse(model(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        avg = tot / len(loader)
        cur_lr = opt.param_groups[0]["lr"]
        log.write(f"{ep},{avg:.6f},{cur_lr:.3e}\n"); log.flush()

        mark = "  [SNAPSHOT: full ckpt + recon saved]" if ep in SNAP else ""
        print(f"Epoch {ep:4d}/{a.epochs} | recon_mse = {avg:.6f} | lr = {cur_lr:.2e}{mark}", flush=True)

        if ep in SNAP:
            save_row(reconstruct(model, fixed, device), os.path.join(RECON, f"ep{ep:04d}"))
            torch.save(model.state_dict(), os.path.join(CKPT, f"{tag}_ep{ep:04d}_full.pkl"))

    log.close()
    print("\n✓ training done. full ckpts in ckpt/, reconstructions in recon/, loss in recon/loss_log.csv")


if __name__ == "__main__":
    main()
