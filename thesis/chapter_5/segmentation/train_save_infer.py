#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ONE nuclei U-Net (random init) at a given (aug, portion, seed, lr), SAVE the
best-val-loss checkpoint under a human-readable name, then run inference over the
WHOLE test split and dump the predicted binary masks + per-image Dice.

This re-uses the exact training protocol of run_first_iter.py (train_unet), so the
saved model matches the cold-start_random numbers.

Run from repo root, e.g.:
  python3 thesis/chapter_5/segmentation/train_save_infer.py \
      --aug 4 --portion 10 --seed 42 --lr 0.001 --epoch 25 \
      --tag 4x --device cuda:0

Outputs:
  ckpts/unet_nuclei_seed<seed>_p<portion>_<tag>_lr<lr>.pkl     (state_dict)
  preds/<tag>/test_preds.pkl   { filename: pred_mask(uint8 HxW) }
  preds/<tag>/test_dice.json   { per_image: {name: dice}, mean_dice, meta... }
"""
import os
import sys
import json
import pickle
import argparse
import random
import numpy as np
import torch
import torch.utils.data as Data

sys.path.insert(0, os.getcwd())
from thesis.chapter_5.segmentation.utils.data import (
    data_loader, o_data, g_data_cell_binary,
)
from thesis.chapter_5.segmentation.utils.train import train_unet, WIDTH, HEIGHT
from thesis.chapter_5.segmentation.utils.tool import compute_dice_binary

HERE = os.path.dirname(__file__)
AUG_TAG = {1: "noaug", 2: "hf", 3: "vfhv", 4: "4x"}


def pick_label_idx(opath, fold, portion, seed):
    """Same seeded portion subset as run_first_iter.py."""
    train_LD, _, _ = data_loader(opath, fold)
    all_train = [x for b in train_LD for x in b]
    target = round(len(all_train) * portion / 100)
    random.seed(seed)
    return random.sample(all_train, target)


@torch.no_grad()
def infer_test(model, opath, gpath_cell, device):
    """Predicted binary mask + Dice for every test image."""
    model.eval()
    _, _, test_LD = data_loader(opath, 0)
    test_files = [x for b in test_LD for x in b]
    preds, dices = {}, {}
    loader = Data.DataLoader(dataset=list(test_files), batch_size=8,
                             shuffle=False, num_workers=4)
    for batch in loader:
        img = o_data(opath, batch, WIDTH, HEIGHT)
        gt = g_data_cell_binary(gpath_cell, batch, WIDTH, HEIGHT)
        INPUT = torch.from_numpy(img.astype(np.float32)).to(device=device, dtype=torch.float)
        out = model(INPUT)
        pred = (out > 0.5).float().cpu().numpy()        # [B,1,H,W]
        for k, name in enumerate(batch):
            preds[name] = pred[k, 0].astype(np.uint8)
            dices[name] = float(compute_dice_binary(pred[k:k + 1], gt[k:k + 1]))
    return preds, dices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="./ds/segmentation_correct")
    ap.add_argument("--aug", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--portion", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--input_nc", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None, help="short name for files (default from --aug)")
    args = ap.parse_args()
    tag = args.tag or AUG_TAG[args.aug]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(HERE, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)
    pred_dir = os.path.join(HERE, "preds", tag); os.makedirs(pred_dir, exist_ok=True)
    p_str = f"{args.portion:g}"
    ckpt_path = os.path.join(
        ckpt_dir, f"unet_nuclei_seed{args.seed}_p{p_str}_{tag}_lr{args.lr:g}.pkl")

    # ---- train (reuse the exact protocol) ----
    opath = args.dataroot + "/image/"
    label_idx = pick_label_idx(opath, args.fold, args.portion, args.seed)
    print(f"[{tag}] aug={args.aug} portion={p_str}% seed={args.seed} lr={args.lr:g} "
          f"-> {len(label_idx)} labeled imgs")

    class Opt:  # train_unet reads attributes off this
        pass
    o = Opt()
    o.dataroot, o.fold, o.input_nc = args.dataroot, args.fold, args.input_nc
    o.lr, o.step, o.epoch, o.batch_size = args.lr, 10, args.epoch, args.batch_size
    o.aug_factor, o.flip_set, o.simclr_path, o.warmup = args.aug, None, None, 0

    out = train_unet(label_idx, o, device)
    model = out["model"]
    torch.save(model.state_dict(), ckpt_path)
    print(f"✓ ckpt saved -> {ckpt_path}  (test Dice {out['test_dice']})")

    # ---- inference over the whole test split ----
    preds, dices = infer_test(model, opath, args.dataroot + "/cell/", device)
    with open(os.path.join(pred_dir, "test_preds.pkl"), "wb") as f:
        pickle.dump(preds, f)
    meta = {"tag": tag, "aug": args.aug, "portion": args.portion, "seed": args.seed,
            "lr": args.lr, "ckpt": os.path.basename(ckpt_path),
            "mean_dice": float(np.mean(list(dices.values()))),
            "n_test": len(dices), "per_image": dices}
    with open(os.path.join(pred_dir, "test_dice.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ test preds + dice saved -> {pred_dir}  "
          f"(mean test Dice over {len(dices)} imgs = {meta['mean_dice']:.4f})")


if __name__ == "__main__":
    main()
