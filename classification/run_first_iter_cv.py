"""
One-shot training (NO active learning) under 10-fold cross-validation, for the
augmentation robustness study (thesis Ch5；w/o · HF · VF · HF+VF · HF+VF+HVF）。

跟 run_first_iter.py 一樣是「在固定 portion 下訓練一次、存 test_acc」，差別只在：
  - 用 utils/data_cv.get_data_cv 做 in-code 的 10-fold train/val/test 切分（見該檔說明）。
  - 多一個 --fold 維度（1-indexed；fold 1 = 原本 split）；結果存成每 (fold, seed) 一個 JSON。
  - init 與 §4.2 data-aug 實驗一致 = **ImageNet**（fold 1 直接沿用 §4.2 既有資料）。

結果結構（沿用 4.2 慣例，可被 aggregate_aug_cv.py 直接讀）：
  {exp_path}/classification_{task}/aug_cv_{init}/fold{F}_seed{S}_bs{BS}_ep{EP}.json
      └─ aug_key（no_aug / aug2_horizontal / aug2_vertical / aug3 / aug4）→ portion → lr → [runs]

從 classification/ 目錄執行（與 run_first_iter.py 相同）：
  python3 run_first_iter_cv.py --task_type hard --fold 2 --portion 50 --seed 42 \
      --pretrained_weights imagenet --aug_factor 4 --lr 1e-4 --device cuda:0
"""
import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.data_cv import get_data_cv
from utils.train_eval import train_model
# 重用 run_first_iter.py 的 model 初始化 / 存檔 / 既有結果檢查
from run_first_iter import (
    initialize_model,
    initialize_simclr_model,
    save_compact_json,
    check_existing_results,
)


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--fold', type=int, required=True,
                        help='CV fold (1-indexed, 1..10)；fold 1 = 原本 8:1:1 split')
    parser.add_argument('--portion', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_path', type=str, default='./exp_results/chapter5_aug_cv')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--pretrained_weights', type=str,
                        choices=['random', 'imagenet', 'simclr'], default='imagenet', required=True)
    parser.add_argument('--simclr_path', type=str, default=None)
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false', default=True)
    parser.add_argument('--aug_factor', type=int, default=4)
    parser.add_argument('--flip_type', type=str, default='horizontal')
    return parser.parse_args()


def main():
    args = parse_arguments()

    task_config = {
        'easy': (2, '../ds/classification/two_class'),
        'medium': (4, '../ds/classification/four_class'),
        'hard': (7, '../ds/classification/seven_class'),
    }
    num_classes, data_dir = task_config[args.task_type]
    if not os.path.isdir(data_dir):
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(_repo_root, data_dir.replace('../', '', 1))

    args.batch_size = 16

    # aug_key（與 run_first_iter.py 一致）
    if not args.data_aug:
        aug_key = "no_aug"
    elif args.aug_factor == 2:
        aug_key = f"aug{args.aug_factor}_{args.flip_type}"
    else:
        aug_key = f"aug{args.aug_factor}"

    portion_key = str(float(args.portion))
    lr_key = str(args.lr)

    file_name = f"fold{args.fold}_seed{args.seed}_bs{args.batch_size}_ep{args.epoch}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
    print(f'Exp name: {file_name}')

    save_path = os.path.join(args.exp_path, f"classification_{args.task_type}",
                             f"aug_cv_{args.pretrained_weights}")
    file_path = os.path.join(save_path, file_name)

    # ===== 既有結果檢查（每 aug×portion×lr 最多 3 runs；滿了就 raise，被腳本 || true 跳過）=====
    print("\nChecking existing results...")
    check_existing_results(file_path, aug_key, portion_key, lr_key, max_runs=3)
    print("Check passed. Proceeding with training...\n")

    # ===== 載入該 fold / portion / seed 的資料 =====
    print(f'===== Load fold {args.fold} | {args.portion}% data | seed {args.seed} =====')
    if args.data_aug and args.aug_factor is None:
        raise ValueError("aug_factor is required when data_aug is True")
    data_loaders, dataset_sizes, train_pool_size, n_labeled = get_data_cv(
        data_dir, fold=args.fold, portion=args.portion, seed=args.seed,
        batch_size=args.batch_size,
        data_aug=args.data_aug, aug_factor=args.aug_factor, flip_type=args.flip_type,
    )

    # ===== 初始化模型 =====
    if args.pretrained_weights == 'random':
        print('Initialize ResNet18 without pretrained weights')
        model = initialize_model(num_classes, False)
    elif args.pretrained_weights == 'simclr':
        print(f'Initialize ResNet18 using SimCLR weights: {args.simclr_path}')
        if not args.simclr_path:
            raise ValueError("--simclr_path is required when pretrained_weights=simclr")
        model = initialize_simclr_model(num_classes, args.simclr_path)
    elif args.pretrained_weights == 'imagenet':
        print('Initialize ResNet18 using ImageNet pretrained weights')
        model = initialize_model(num_classes, True)
    else:
        raise NotImplementedError()

    criterion = nn.CrossEntropyLoss()
    if args.weight_decay is not None:
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler_ = lr_scheduler.LinearLR(
        optimizer_, start_factor=1.0, end_factor=0.0, total_iters=args.epoch
    )

    print(f'===== Train (fold {args.fold}, {args.portion}%, seed {args.seed}, '
          f'{aug_key}, lr {args.lr}) =====')
    print('-' * 50)
    print(f'{"Fold":<20}: {args.fold}')
    print(f'{"Aug key":<20}: {aug_key}')
    print(f'{"Batch Size":<20}: {args.batch_size}')
    print(f'{"Learning Rate":<20}: {args.lr}')
    print(f'{"Train pool":<20}: {train_pool_size}')
    print(f'{"Labeled samples":<20}: {n_labeled}')
    print('-' * 50)

    _, final_acc, _ = train_model(
        model, args.device, data_loaders, dataset_sizes,
        criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch,
    )
    final_acc = round(final_acc, 4)
    print(f"Final Acc: {final_acc}")

    # ===== 存結果（用檔案鎖保護 load→append→save，讓同一 fold/seed 檔可被多個
    #        平行 process（多個 lr × 多個 run 同卡）安全 append，不會 race 掉資料）=====
    import fcntl
    os.makedirs(save_path, exist_ok=True)
    lock_path = file_path + ".lock"
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)            # 阻塞直到取得獨佔鎖（critical section 很短）
        if os.path.isfile(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error in {file_path}: {e}\nCreating new data structure...")
                data = {}
        else:
            data = {}

        data.setdefault(aug_key, {}).setdefault(portion_key, {}).setdefault(lr_key, [])
        data[aug_key][portion_key][lr_key].append(final_acc)

        # 排序 aug / portion / lr
        sorted_data = {}
        for aug_k in sorted(data.keys()):
            sorted_portions = {}
            for portion_k in sorted(data[aug_k].keys(), key=float):
                sorted_lrs = {lr_k: data[aug_k][portion_k][lr_k]
                              for lr_k in sorted(data[aug_k][portion_k].keys(), key=float)}
                sorted_portions[portion_k] = sorted_lrs
            sorted_data[aug_k] = sorted_portions

        save_compact_json(sorted_data, file_path)
        fcntl.flock(lockf, fcntl.LOCK_UN)
    print(f'Result saved to {file_path}!')


if __name__ == "__main__":
    main()
