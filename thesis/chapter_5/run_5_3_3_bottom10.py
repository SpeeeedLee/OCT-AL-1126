"""
Bottom-10% D_train^U one-shot training script.

取 ρ=90% labeled_ids 的 cumulative，
Bottom-10% = 全部 train indices - cumulative@90%，
用這 ~203 張影像做一次性訓練（tune LR）。

Usage (from repo root):
    python3 thesis/chapter_5/run_5_3_3_bottom10.py \
        --strategy margin --seed 42 --lr 1e-4 --device cuda:0

LR sweep example:
    for lr in 3e-5 5e-5 1e-4 3e-4 5e-4; do
        python3 thesis/chapter_5/run_5_3_3_bottom10.py \
            --strategy margin --seed 42 --lr $lr --device cuda:0
    done

Results saved to:
    classification/exp_results/ch5_5_3_3_bottom10/
        classification_hard/AL_simclr/{strategy}_seed{seed}_bs16.json
    JSON structure: {"aug4": {"bottom10": {lr_str: [acc]}}}
"""
import os, sys, json, glob
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR
from classification.utils.train_eval import train_model

SIMCLR_CKPT = "SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"
DATA_DIR     = "ds/classification/seven_class"
BATCH_SIZE   = 16
EPOCH        = 20
AUG_KEY      = "aug4"
PORTION_KEY  = "bottom10"

# labeled_ids 搜尋路徑（優先 ch5 extension，再 base AL_simclr）
LABELED_IDS_DIRS = [
    "classification/exp_results/ch5_5_3_3_extend/classification_hard/AL_simclr/labeled_ids",
    "classification/exp_results/classification_hard/AL_simclr/labeled_ids",
]


def find_labeled_ids(strategy, seed):
    fname = f"{strategy}_seed{seed}_bs16.json"
    for d in LABELED_IDS_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Cannot find labeled_ids for {strategy} seed{seed}. "
        f"Need ρ=90% data — run the AL extension first.")


def get_bottom10_idx(strategy, seed):
    path = find_labeled_ids(strategy, seed)
    d = json.load(open(path))
    if "90.0" not in d:
        avail = sorted(d.keys(), key=float)
        raise ValueError(
            f"ρ=90% not found in {path}. "
            f"Available portions: {avail}. "
            f"Please extend the AL run to ρ=90% first.")
    cumulative_90 = set(d["90.0"]["cumulative"])
    tot = get_num_train(DATA_DIR)
    bottom = [i for i in range(tot) if i not in cumulative_90]
    print(f"  [bottom-10%] strategy={strategy} seed={seed}: "
          f"total={tot}, labeled@90%={len(cumulative_90)}, bottom={len(bottom)}")
    return bottom


def save_result(out_path, lr_key, acc):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = json.load(open(out_path)) if os.path.exists(out_path) else {}
    data.setdefault(AUG_KEY, {}).setdefault(PORTION_KEY, {}).setdefault(lr_key, [])
    data[AUG_KEY][PORTION_KEY][lr_key].append(acc)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [saved] {out_path}  lr={lr_key}  acc={acc:.4f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True,
                    choices=["margin", "coreset", "cluster_margin"])
    ap.add_argument("--seed",   type=int, required=True)
    ap.add_argument("--lr",     type=float, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out_path = os.path.join(
        "classification/exp_results/ch5_5_3_3_bottom10",
        "classification_hard/AL_simclr",
        f"{args.strategy}_seed{args.seed}_bs16.json")

    lr_key = str(args.lr)

    # skip if already done
    if os.path.exists(out_path):
        d = json.load(open(out_path))
        existing = d.get(AUG_KEY, {}).get(PORTION_KEY, {}).get(lr_key, [])
        if existing:
            print(f"[skip] {args.strategy} seed{args.seed} lr={lr_key} already done: {existing}")
            return

    bottom_idx = get_bottom10_idx(args.strategy, args.seed)

    data_loaders, dataset_sizes = get_data(
        DATA_DIR, bottom_idx, BATCH_SIZE,
        data_aug=True, aug_factor=4, flip_type="horizontal")
    print(dataset_sizes)

    # SimCLR init
    model = ResNetSimCLR("resnet18", 32)
    state_dict = torch.load(SIMCLR_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, 7, bias=True)

    device = torch.device(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                       total_iters=EPOCH)
    criterion = nn.CrossEntropyLoss()

    print(f"\n===== Bottom-10% training: {args.strategy} seed{args.seed} "
          f"lr={args.lr} device={args.device} =====")
    _, acc, _ = train_model(
        model, device, data_loaders, dataset_sizes,
        criterion, optimizer, scheduler,
        num_epochs=EPOCH)

    save_result(out_path, lr_key, float(acc))


if __name__ == "__main__":
    main()
