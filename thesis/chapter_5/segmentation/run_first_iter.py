"""
Cold-start baseline for Ch5 nuclei segmentation.

Trains a pure U-Net on a random `portion`% subset of the training images
(seeded), with offline flip augmentation, and reports the per-image mean Dice on
the TEST split (model picked at the lowest-validation-loss epoch). Per-epoch
train/val loss & Dice are recorded.

Run from repo root, e.g.:
  python3 thesis/chapter_5/segmentation/run_first_iter.py \
      --dataroot ./ds/segmentation --portion 100 --seed 42 \
      --aug_factor 4 --lr 0.001 --epoch 25 --device cuda:0

Results JSON:
  thesis/chapter_5/segmentation/exp_results/nuclei/cold_start_<init>/random_<seed>_bs<bs>.json
  schema: { portion: { lr: [ run_dict, ... ] } }
"""
import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import torch

from thesis.chapter_5.segmentation.utils.data import data_loader
from thesis.chapter_5.segmentation.utils.train import train_unet
from thesis.chapter_5.segmentation.utils.jsonio import save_compact_json


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', required=True)
    p.add_argument('--fold', type=int, default=0, help='single fold (no cross-validation)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--input_nc', type=int, default=1)
    p.add_argument('--output_nc', type=int, default=1)  # kept for CLI compat; binary
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--step', type=int, default=10)
    p.add_argument('--epoch', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--portion', type=float, required=True, help='1-100')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--aug_factor', type=int, default=4, choices=[1, 2, 3, 4],
                   help='offline flip factor: 1=none 2=+HF 3=+HF+VF 4=+HF+VF+HFV')
    p.add_argument('--init', type=str, default='random', choices=['random', 'simclr'],
                   help='encoder init (simclr requires --simclr_path)')
    p.add_argument('--simclr_path', type=str, default=None)
    p.add_argument('--max_runs', type=int, default=5)
    p.add_argument('--exp_path', type=str,
                   default='./thesis/chapter_5/segmentation/exp_results')
    return p.parse_args()


def main():
    args = parse_arguments()
    if not (0 < args.portion <= 100):
        raise ValueError(f"portion must be in (0,100], got {args.portion}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(args.exp_path, "nuclei", f"cold_start_{args.init}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"random_{args.seed}_bs{args.batch_size}.json")

    portion_key = str(float(args.portion))
    lr_key = str(args.lr)

    # Load-or-create + max_runs guard
    data = {}
    if os.path.isfile(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    existing = data.get(portion_key, {}).get(lr_key, [])
    if len(existing) >= args.max_runs:
        raise RuntimeError(f"Already {len(existing)}/{args.max_runs} runs for "
                           f"portion={portion_key} lr={lr_key} in {file_path}")

    # ----- pick the portion subset (seeded), mirroring the source convention -----
    opath = args.dataroot + "/image/"
    train_LD, _, _ = data_loader(opath, args.fold)
    all_train = []
    for b in train_LD:
        all_train.extend(b)
    target_num = round(len(all_train) * args.portion / 100)
    random.seed(args.seed)
    label_idx = random.sample(all_train, target_num)
    print(f"Cold-start: {args.portion}% -> {target_num}/{len(all_train)} imgs (seed {args.seed})")

    # ----- train + evaluate -----
    out = train_unet(label_idx, args, device)

    run_dict = {
        "test_dice": out["test_dice"],
        "best_val_epoch": out["best_val_epoch"],
        "best_val_loss": out["best_val_loss"],
        "best_val_dice": out["best_val_dice"],
        "aug_factor": out["aug_factor"],
        "seed": args.seed,
        "n_labeled": out["n_labeled"],
        "per_epoch": out["per_epoch"],
    }

    # Race-safe append: hold an exclusive lock, RE-READ the file fresh (it may
    # have changed during the minutes of training), append, then save. The
    # critical section is short (no training inside it), so concurrent jobs that
    # share this file serialize their writes instead of clobbering each other.
    import fcntl
    lock_path = file_path + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        fresh = {}
        if os.path.isfile(file_path):
            try:
                with open(file_path, encoding="utf-8") as f:
                    fresh = json.load(f)
            except json.JSONDecodeError:
                fresh = {}
        n_now = len(fresh.get(portion_key, {}).get(lr_key, []))
        fresh.setdefault(portion_key, {}).setdefault(lr_key, []).append(run_dict)
        sorted_data = {pk: {lk: fresh[pk][lk] for lk in sorted(fresh[pk], key=float)}
                       for pk in sorted(fresh, key=float)}
        save_compact_json(sorted_data, file_path)
        fcntl.flock(lf, fcntl.LOCK_UN)
    print(f"\n✓ Saved -> {file_path}  (test Dice {out['test_dice']}, "
          f"run {n_now + 1}/{args.max_runs})")


if __name__ == "__main__":
    main()
