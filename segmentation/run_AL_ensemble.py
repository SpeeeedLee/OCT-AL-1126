import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from segmentation.utils.data import data_loader, o_data, g_data_cell_binary
from segmentation.utils.model import Optim_U_Net
import segmentation.utils.loss as L
from segmentation.utils.tool import compute_dice_binary
from segmentation.AL_strategy.bald_ensemble import bald_ensemble_mean, bald_ensemble_nuclei, variance_ensemble


"""
=================================================================================
ENSEMBLE BALD ACTIVE LEARNING
=================================================================================
此版本使用 **多個獨立訓練的模型** 進行 ensemble BALD。

與 MC Dropout 的差異：
- MC Dropout: 同一個模型 + Dropout，推論 5 次
- Ensemble: 5 個獨立模型，各推論 1 次

Ensemble 優點：
1. 不需要在模型中加入 Dropout
2. 每個模型都是完整訓練的，預測更穩定
3. 學界公認最好的 BALD 實現方式
4. 正好配合你的實驗設計（每個 portion 跑 5 次）

工作流程：
1. 每個 portion 訓練 5 個模型（不同 seed）
2. 用這 5 個模型做 ensemble BALD
3. 選出下一批資料
4. 繼續下一個 portion...
=================================================================================
"""


def train_single_model(opt, device, seed):
    """
    訓練單一模型
    
    Args:
        opt: 訓練參數
        device: 訓練設備
        seed: 隨機種子
    
    Returns:
        model: 訓練完成的模型
        final_dice: 最終驗證 Dice score
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    fold = opt.fold
    print(f'\n{"="*80}')
    print(f'Training Model #{seed} - Fold: {fold}')
    print(f'{"="*80}')
    
    # Creating model
    model = Optim_U_Net(img_ch=opt.input_nc, output_ch=1, USE_DS=False, USE_DFS=False)
    model = model.to(device)
    
    # Loading data paths
    gpath_cell = opt.dataroot + "/cell/"
    opath = opt.dataroot + "/image/"
    
    # Load validation data
    _, valid_data_LD, _ = data_loader(opath, fold)
    
    # Use label_idx for training data
    label_idx = opt.label_idx
    batch_size = opt.batch_size
    
    import torch.utils.data as Data
    train_data_LD = Data.DataLoader(
        dataset=label_idx, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )

    # Loss function and optimizer
    loss_func = L.BinaryDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)

    train_epoch = opt.epoch
    valid_mdice = np.zeros(train_epoch)
    
    height = 512
    width = 384
    
    # Training loop (simplified, no verbose output for ensemble)
    for EPOCH in range(train_epoch):
        model.train()
        
        # Training phase
        for t_batch_num in train_data_LD:
            img_sub = o_data(opath, t_batch_num, width, height)
            t_gim_sub = g_data_cell_binary(gpath_cell, t_batch_num, width, height)

            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            target = torch.from_numpy(t_gim_sub.astype(np.float32)).to(device=device, dtype=torch.float)

            OUTPUT = model(INPUT)
            loss = loss_func(OUTPUT, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        valid_dice_list = []
        
        with torch.no_grad():
            for v_batch_num in valid_data_LD:
                val_sub = o_data(opath, v_batch_num, width, height)
                v_gim_sub = g_data_cell_binary(gpath_cell, v_batch_num, width, height)
                
                INPUT = torch.from_numpy(val_sub.astype(np.float32)).to(device=device, dtype=torch.float)
                target = torch.from_numpy(v_gim_sub.astype(np.float32)).to(device=device, dtype=torch.float)

                OUTPUT = model(INPUT)
                pred_mask = (OUTPUT > 0.5).float()
                dice = compute_dice_binary(pred_mask.cpu().detach().numpy(), v_gim_sub)
                valid_dice_list.append(dice)
        
        scheduler.step()
        valid_mdice[EPOCH] = np.mean(valid_dice_list)
        
        # Print progress every 5 epochs
        if (EPOCH + 1) % 5 == 0:
            print(f"  Model #{seed} - Epoch {EPOCH+1}/{train_epoch} - Val Dice: {valid_mdice[EPOCH]:.4f}")
    
    final_dice = valid_mdice[-1]
    print(f"✓ Model #{seed} completed - Final Dice: {final_dice:.4f}")
    
    return model, final_dice


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # AL strategy
    parser.add_argument('--AL_strategy', type=str, 
                        choices=['bald_ensemble_mean', 'bald_ensemble_nuclei', 'variance_ensemble'], 
                        required=True,
                        help='Active Learning strategy to use')
    
    # AL related setup
    parser.add_argument('--portion_start', type=float, required=True,
                        help='Starting percentage of training data')
    parser.add_argument('--portion_end', type=float, required=True,
                        help='Ending percentage of training data')
    parser.add_argument('--portion_interval', type=float, required=True,
                        help='Interval for incrementing portion')
    parser.add_argument('--initial_seed', type=int, required=True,
                        help='Random seed for initial data sampling (first portion)')
    parser.add_argument('--seed_start', type=int, default=1,
                        help='Starting seed for ensemble training (default: 1)')
    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of models in ensemble (default: 5)')
    
    # Data and experiment settings
    parser.add_argument('--dataroot', required=True, 
                        help='path to image and label dataset')
    parser.add_argument('--fold', type=int, default=0, 
                        help='which fold to use for cross-validation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device to use for training')
    
    # Model settings
    parser.add_argument('--input_nc', type=int, default=1, 
                        help='# of input image channels (1 for grayscale)')
    parser.add_argument('--output_nc', type=int, default=1, 
                        help='# of output channels (1 for binary segmentation)')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='learning rate (FIXED across all portions)')
    parser.add_argument('--step', type=int, default=10, 
                        help='step size of scheduler')
    parser.add_argument('--epoch', type=int, default=25, 
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size')
    
    # Result saving
    parser.add_argument('--exp_path', type=str, default='./segmentation/exp_results', 
                        help='path to save experiment results')
    
    return parser.parse_args()


def save_compact_json(data, file_path):
    """Save JSON with compact list formatting"""
    def format_dict(d, indent=0):
        lines = []
        items = list(d.items())
        for i, (key, value) in enumerate(items):
            is_last = (i == len(items) - 1)
            comma = '' if is_last else ','
            
            if isinstance(value, dict):
                lines.append('  ' * indent + f'"{key}": {{')
                lines.append(format_dict(value, indent + 1))
                lines.append('  ' * indent + '}' + comma)
            elif isinstance(value, list):
                # Keep list on single line
                list_str = '[' + ', '.join(str(x) for x in value) + ']'
                lines.append('  ' * indent + f'"{key}": {list_str}' + comma)
            else:
                lines.append('  ' * indent + f'"{key}": {json.dumps(value)}' + comma)
        
        return '\n'.join(lines)
    
    json_str = '{\n' + format_dict(data, 1) + '\n}'
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json_str)


def main():
    args = parse_arguments()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Generate file name
    file_name = f"{args.AL_strategy}_seed{args.initial_seed}_ensemble{args.n_models}_bs{args.batch_size}.json"
    
    print('=' * 80)
    print('ENSEMBLE BALD ACTIVE LEARNING FOR BINARY NUCLEI SEGMENTATION')
    print('=' * 80)
    print(f'Exp name: {file_name}')
    print(f'AL Strategy: {args.AL_strategy}')
    print(f'Initial Seed (data sampling): {args.initial_seed}')
    print(f'Ensemble Size: {args.n_models} models')
    print(f'Ensemble Seeds: {args.seed_start} to {args.seed_start + args.n_models - 1}')
    print(f'Portion range: {args.portion_start}% to {args.portion_end}% (interval: {args.portion_interval}%)')
    print(f'Learning Rate: {args.lr} (FIXED across all portions)')
    print(f'Batch Size: {args.batch_size}')
    print(f'Fold: {args.fold}')
    print(f'Device: {device}')
    print('=' * 80)
    
    # Get all training image names
    opath = args.dataroot + "/image/"
    train_data_LD, _, _ = data_loader(opath, args.fold)
    
    all_train_files = []
    for batch in train_data_LD:
        all_train_files.extend(batch)
    
    tot_num_train = len(all_train_files)
    print(f'Total Number of Training Images: {tot_num_train}')
    print('=' * 80)
    
    # Initialize label/unlabel indices
    label_idx = []
    unlabeled_idx = all_train_files.copy()
    
    # Prepare save path
    save_path = os.path.join(args.exp_path, "nuclei", f"AL_ensemble")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    # Load or create results file
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error in {file_path}: {e}")
            data = {}
    else:
        data = {}
    
    lr_key = str(args.lr)
    gpath_cell = args.dataroot + "/cell/"
    
    # Active Learning Loop
    ensemble_models = []
    
    for portion in np.arange(args.portion_start, args.portion_end + args.portion_interval/2, args.portion_interval):
        print('\n' + '=' * 80)
        print(f'PORTION: {portion}%')
        print('=' * 80)
        
        portion_key = str(float(portion))
        
        # Initialize portion if not exists
        if portion_key not in data:
            data[portion_key] = {}
        
        if lr_key not in data[portion_key]:
            data[portion_key][lr_key] = {
                "dice": [],
                "labeled_idx": []
            }
        
        # ===== Select Data to Label =====
        target_num = round(tot_num_train * portion / 100)
        num_to_label = target_num - len(label_idx)
        
        if portion == args.portion_start:
            # First iteration: random sampling with initial_seed
            random.seed(args.initial_seed)
            print(f'===== First Iteration: Random Sample {num_to_label} samples (seed={args.initial_seed}) =====')
            to_label_idx = random.sample(unlabeled_idx, num_to_label)
        else:
            # Subsequent iterations: use ensemble BALD
            print(f'===== AL Strategy: {args.AL_strategy} =====')
            print(f'Using ensemble of {len(ensemble_models)} models')
            
            if args.AL_strategy == 'bald_ensemble_mean':
                to_label_idx, bald_dict = bald_ensemble_mean(
                    ensemble_models, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            elif args.AL_strategy == 'bald_ensemble_nuclei':
                to_label_idx, bald_dict = bald_ensemble_nuclei(
                    ensemble_models, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            elif args.AL_strategy == 'variance_ensemble':
                to_label_idx, bald_dict = variance_ensemble(
                    ensemble_models, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            else:
                raise NotImplementedError(f"AL strategy {args.AL_strategy} not implemented")
        
        # Update label and unlabeled indices
        label_idx.extend(to_label_idx)
        unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
        
        print(f"\nSelected {len(to_label_idx)} samples")
        print(f"Total labeled: {len(label_idx)} | Remaining unlabeled: {len(unlabeled_idx)}")
        
        # Sanity check
        if len(label_idx) != len(set(label_idx)):
            raise ValueError("Duplicate indices in label_idx!")
        
        # ===== Train Ensemble of Models =====
        print(f'\n{"="*80}')
        print(f'Training Ensemble of {args.n_models} Models')
        print(f'{"="*80}')
        
        # Create training args
        class TrainArgs:
            pass
        
        train_args = TrainArgs()
        train_args.dataroot = args.dataroot
        train_args.fold = args.fold
        train_args.input_nc = args.input_nc
        train_args.output_nc = args.output_nc
        train_args.lr = args.lr
        train_args.step = args.step
        train_args.epoch = args.epoch
        train_args.batch_size = args.batch_size
        train_args.label_idx = label_idx
        
        # Train multiple models with different seeds
        ensemble_models = []
        ensemble_dice_scores = []
        
        for model_idx in range(args.n_models):
            seed = args.seed_start + model_idx
            model, dice = train_single_model(train_args, device, seed)
            ensemble_models.append(model)
            ensemble_dice_scores.append(dice)
        
        # Average dice from ensemble
        avg_dice = np.mean(ensemble_dice_scores)
        std_dice = np.std(ensemble_dice_scores)
        
        print(f'\n{"="*80}')
        print(f'ENSEMBLE TRAINING COMPLETED')
        print(f'{"="*80}')
        print(f'Individual Dice Scores: {[f"{d:.4f}" for d in ensemble_dice_scores]}')
        print(f'Average Dice: {avg_dice:.4f} ± {std_dice:.4f}')
        print(f'{"="*80}\n')
        
        # ===== Save Results =====
        data[portion_key][lr_key]["dice"].append(round(avg_dice, 4))
        data[portion_key][lr_key]["labeled_idx"].append(label_idx.copy())
        
        # Sort and save
        sorted_data = {}
        for portion_k in sorted(data.keys(), key=float):
            sorted_lrs = {}
            for lr_k in sorted(data[portion_k].keys(), key=float):
                sorted_lrs[lr_k] = data[portion_k][lr_k]
            sorted_data[portion_k] = sorted_lrs
        
        save_compact_json(sorted_data, file_path)
        print(f'✓ Result saved to {file_path}!')
    
    print('\n' + '=' * 80)
    print('ENSEMBLE ACTIVE LEARNING COMPLETED!')
    print('=' * 80)


if __name__ == "__main__":
    main()