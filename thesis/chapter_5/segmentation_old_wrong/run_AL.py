"""
Active-learning loop for Ch5 nuclei segmentation.

Strategies (image-level selection, no patches):
  - margin          : highest mean-over-pixels margin uncertainty
  - coreset         : k-Center-Greedy on encoder embeddings (conditioned on labeled)
  - cluster_margin  : margin candidate pool -> HAC -> round-robin
  - random          : passive baseline (random pick each step)

Protocol: FIXED learning rate across portions (the .py accepts any --lr/--seed;
the run scripts keep lr fixed and vary seed over {10,24,38,42,57}). Fresh U-Net
each portion, trained on the grown labeled set with offline flip augmentation;
the PREVIOUS portion's best-val-loss model is the selector for the next batch.
Headline metric per portion = per-image mean Dice on the TEST split.

Run from repo root, e.g.:
  python3 thesis/chapter_5/segmentation/run_AL.py --AL_strategy margin \
      --dataroot ./ds/segmentation --portion_start 5 --portion_end 60 \
      --portion_interval 2.5 --seed 42 --aug_factor 4 --lr 0.001 --device cuda:0

Results JSON:
  thesis/chapter_5/segmentation/exp_results/nuclei/AL_<init>/<strategy>_seed<seed>_bs<bs>.json
  schema: { portion: { lr: [ run_dict_with_selected_and_cumulative_indices ] } }
"""
import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np
import torch

from thesis.chapter_5.segmentation.utils.data import data_loader
from thesis.chapter_5.segmentation.utils.train import train_unet
from thesis.chapter_5.segmentation.utils.jsonio import save_compact_json
from thesis.chapter_5.segmentation.AL_strategy.uncertainty import margin, confidence, entropy
from thesis.chapter_5.segmentation.AL_strategy.diversity import coreset
from thesis.chapter_5.segmentation.AL_strategy.hybrid import cluster_margin


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--AL_strategy', required=True,
                   choices=['random', 'margin', 'confidence', 'entropy',
                            'coreset', 'cluster_margin'])
    p.add_argument('--portion_start', type=float, required=True)
    p.add_argument('--portion_end', type=float, required=True)
    p.add_argument('--portion_interval', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--dataroot', required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--input_nc', type=int, default=1)
    p.add_argument('--output_nc', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.001,
                   help='used only when --lr_schedule fixed')
    p.add_argument('--lr_schedule', type=str, default='sweep', choices=['sweep', 'fixed'],
                   help="sweep = per-portion lr sweep + best-val-loss selector (option A, "
                        "matches classification); fixed = single --lr across all portions")
    p.add_argument('--lr_grid', type=str, default=None,
                   help='comma-sep lr list overriding the per-portion default grid')
    p.add_argument('--step', type=int, default=10)
    p.add_argument('--epoch', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--aug_factor', type=int, default=4, choices=[1, 2, 3, 4])
    p.add_argument('--init', type=str, default='random', choices=['random', 'simclr'])
    p.add_argument('--simclr_path', type=str, default=None)
    p.add_argument('--exp_path', type=str,
                   default='./thesis/chapter_5/segmentation/exp_results')
    return p.parse_args()


def lr_grid_for(portion, override=None):
    """Per-portion candidate lr grid (informed by the Task-3 lr-sensitivity sweep:
    low ρ peaks ~1e-3, high ρ peaks ~3e-4). Override with --lr_grid."""
    if override:
        return [float(x) for x in override.split(',')]
    if portion < 15:
        return [5e-4, 1e-3, 3e-3]
    if portion < 50:
        return [3e-4, 1e-3, 3e-3]
    return [3e-4, 5e-4, 1e-3]


class _A:
    """Lightweight per-lr args view for train_unet (shares everything but lr)."""
    def __init__(self, base, lr):
        self.__dict__.update(vars(base))
        self.lr = lr


def select_batch(strategy, model, opath, gpath_cell, unlabeled, labeled, k, device, seed):
    """Dispatch to the AL strategy. Returns (to_label_files, info)."""
    if strategy == 'random':
        picks = random.sample(unlabeled, k)
        return picks, {"strategy": "random", "selected_files": picks}
    if strategy == 'margin':
        return margin(model, opath, gpath_cell, unlabeled, k, device)
    if strategy == 'confidence':
        return confidence(model, opath, gpath_cell, unlabeled, k, device)
    if strategy == 'entropy':
        return entropy(model, opath, gpath_cell, unlabeled, k, device)
    if strategy == 'coreset':
        return coreset(model, opath, unlabeled, labeled, k, device)
    if strategy == 'cluster_margin':
        return cluster_margin(model, opath, gpath_cell, unlabeled, k, device)
    raise NotImplementedError(strategy)


def main():
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    opath = args.dataroot + "/image/"
    gpath_cell = args.dataroot + "/cell/"

    train_LD, _, _ = data_loader(opath, args.fold)
    all_train = []
    for b in train_LD:
        all_train.extend(b)
    tot = len(all_train)

    save_dir = os.path.join(args.exp_path, "nuclei", f"AL_{args.init}")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(
        save_dir, f"{args.AL_strategy}_seed{args.seed}_bs{args.batch_size}.json")

    data = {}
    if os.path.isfile(file_path):
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    print('=' * 80)
    lr_desc = f'fixed lr={args.lr}' if args.lr_schedule == 'fixed' else 'lr=per-portion sweep'
    print(f'AL SEGMENTATION | strategy={args.AL_strategy} seed={args.seed} '
          f'{lr_desc} aug{args.aug_factor} | train pool={tot}')
    print(f'portions {args.portion_start}->{args.portion_end} step {args.portion_interval}')
    print('=' * 80)

    label_idx = []
    unlabeled = all_train.copy()
    last_model = None

    portions = np.arange(args.portion_start,
                         args.portion_end + args.portion_interval / 2,
                         args.portion_interval)
    for portion in portions:
        portion_key = str(float(portion))
        target_num = round(tot * portion / 100)
        k = target_num - len(label_idx)
        if k <= 0:
            print(f"[skip] portion {portion}: nothing new to label")
            continue

        print('\n' + '=' * 80)
        print(f'PORTION {portion}%  -> label {k} more (cumulative target {target_num})')
        print('=' * 80)

        if portion == args.portion_start or last_model is None:
            to_label = random.sample(unlabeled, k)
            info = {"strategy": "random_init", "selected_files": to_label}
            print(f"Initial pool: random {k}")
        else:
            to_label, info = select_batch(args.AL_strategy, last_model, opath, gpath_cell,
                                          unlabeled, label_idx, k, device, args.seed)

        label_idx.extend(to_label)
        unlabeled = list(set(unlabeled) - set(to_label))
        assert len(label_idx) == len(set(label_idx)), "duplicate labels!"

        # ----- learning-rate handling -----
        if args.lr_schedule == 'fixed':
            grid = [args.lr]
        else:
            grid = lr_grid_for(portion, args.lr_grid)

        # Train each candidate lr fresh; the SELECTOR for the next AL batch is the
        # model with the lowest validation loss (val-based, no test leakage).
        best_val = float('inf')
        for lr in grid:
            out = train_unet(label_idx, _A(args, lr), device)
            if out["best_val_loss"] < best_val:
                best_val = out["best_val_loss"]
                last_model = out["model"]
            run_dict = {
                "test_dice": out["test_dice"],
                "best_val_epoch": out["best_val_epoch"],
                "best_val_loss": out["best_val_loss"],
                "best_val_dice": out["best_val_dice"],
                "aug_factor": out["aug_factor"],
                "n_labeled": len(label_idx),
                "selected_idx": list(to_label),       # picked THIS round
                "labeled_idx": list(label_idx),       # cumulative labeled set
                "per_epoch": out["per_epoch"],
            }
            data.setdefault(portion_key, {}).setdefault(str(lr), []).append(run_dict)

        sorted_data = {pk: {lk: data[pk][lk] for lk in sorted(data[pk], key=float)}
                       for pk in sorted(data, key=float)}
        save_compact_json(sorted_data, file_path)
        best_dice = max(data[portion_key][lk][-1]["test_dice"] for lk in data[portion_key])
        print(f"✓ portion {portion}: best-lr test Dice {best_dice:.4f} "
              f"(swept {len(grid)} lrs) -> saved {file_path}")

    print('\n' + '=' * 80)
    print('ACTIVE LEARNING COMPLETED')
    print('=' * 80)


if __name__ == "__main__":
    main()
