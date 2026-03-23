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
from segmentation.AL_strategy.uncertainty import mean_entropy, nuclei_entropy
from segmentation.AL_strategy.hybrid import mean_entropy_clustering, nuclei_entropy_clustering

from segmentation.utils.train_NONE import train_none_AL

"""
=================================================================================
IMPORTANT NOTE: FIXED LEARNING RATE ACROSS ALL PORTIONS
=================================================================================
This version uses a SINGLE, FIXED learning rate across all active learning iterations.
The model is NOT re-tuned with different learning rates at each portion.

If you want to find the optimal learning rate for each portion separately, you would
need to run hyperparameter tuning at each portion before selecting the next batch
of data. This current implementation prioritizes efficiency by using one lr throughout.
=================================================================================
"""




def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # AL strategy
    parser.add_argument('--AL_strategy', type=str, 
                        choices=['mean_entropy', 'nuclei_entropy', 
                                 'mean_entropy_clustering', 'nuclei_entropy_clustering'], 
                        required=True,
                        help='Active Learning strategy to use')
    
    # AL related setup
    parser.add_argument('--portion_start', type=float, required=True,
                        help='Starting percentage of training data')
    parser.add_argument('--portion_end', type=float, required=True,
                        help='Ending percentage of training data')
    parser.add_argument('--portion_interval', type=float, required=True,
                        help='Interval for incrementing portion')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    
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
    
    # Set random seed for initial labeled pool selection
    random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Generate file name
    file_name = f"{args.AL_strategy}_{args.portion_start}_{args.portion_end}_{args.portion_interval}_seed{args.seed}_bs{args.batch_size}.json"
    
    print('=' * 80)
    print('ACTIVE LEARNING FOR BINARY NUCLEI SEGMENTATION')
    print('=' * 80)
    print(f'Exp name: {file_name}')
    print(f'AL Strategy: {args.AL_strategy}')
    print(f'Portion range: {args.portion_start}% to {args.portion_end}% (interval: {args.portion_interval}%)')
    print(f'Learning Rate: {args.lr} (FIXED across all portions)')
    print(f'Batch Size: {args.batch_size}')
    print(f'Fold: {args.fold}')
    print(f'Device: {device}')
    print('-' * 80)
    
    # Get all training image names
    opath = args.dataroot + "/image/"
    train_data_LD, _, _ = data_loader(opath, args.fold)
    
    # Extract all training file names
    all_train_files = []
    for batch in train_data_LD:
        all_train_files.extend(batch)
    
    tot_num_train = len(all_train_files)
    print(f'Total Number of Training Images: {tot_num_train}')
    print('=' * 80)
    
    # Initialize label/unlabel indices (using file names)
    label_idx = []
    unlabeled_idx = all_train_files.copy()
    
    # Prepare save path
    save_path = os.path.join(args.exp_path, "nuclei", f"AL_random")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    # Load or create results file
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error in {file_path}: {e}")
            print("Creating new data structure...")
            data = {}
    else:
        data = {}
    
    lr_key = str(args.lr)
    
    # Active Learning Loop
    last_trained_model = None
    gpath_cell = args.dataroot + "/cell/"
    
    for portion in np.arange(args.portion_start, args.portion_end + args.portion_interval/2, args.portion_interval):
        print('\n' + '=' * 80)
        print(f'PORTION: {portion}%')
        print('=' * 80)
        
        portion_key = str(float(portion))
        
        # Initialize portion if not exists
        if portion_key not in data:
            data[portion_key] = {}
        
        # Initialize lr if not exists
        if lr_key not in data[portion_key]:
            data[portion_key][lr_key] = {
                "dice": [],
                "labeled_idx": []
            }
        
        # ===== Select Data to Label =====
        target_num = round(tot_num_train * portion / 100)
        num_to_label = target_num - len(label_idx)
        
        if portion == args.portion_start:
            # First iteration: random sampling
            print(f'===== First Iteration: Random Sample {num_to_label} samples =====')
            to_label_idx = random.sample(unlabeled_idx, num_to_label)
            uncertainty_dict = None
        else:
            # Subsequent iterations: use AL strategy
            print(f'===== AL Strategy: {args.AL_strategy} - Select {num_to_label} samples =====')
            
            if args.AL_strategy == 'random':
                raise NotImplementedError()
                # to_label_idx, uncertainty_dict = random_sampling(
                #     last_trained_model, opath, gpath_cell, 
                #     unlabeled_idx, num_to_label, device,
                #     random_seed=args.seed
                # )
            elif args.AL_strategy == 'mean_entropy':
                to_label_idx, uncertainty_dict = mean_entropy(
                    last_trained_model, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            elif args.AL_strategy == 'nuclei_entropy':
                to_label_idx, uncertainty_dict = nuclei_entropy(
                    last_trained_model, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            elif args.AL_strategy == 'mean_entropy_clustering':
                to_label_idx, info_dict = mean_entropy_clustering(
                    last_trained_model, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            elif args.AL_strategy == 'nuclei_entropy_clustering':
                to_label_idx, info_dict = nuclei_entropy_clustering(
                    last_trained_model, opath, gpath_cell,
                    unlabeled_idx, num_to_label, device
                )
            else:
                raise NotImplementedError(f"AL strategy {args.AL_strategy} not implemented")
        
        # Update label and unlabeled indices
        label_idx.extend(to_label_idx)
        unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
        
        print(f"Selected {len(to_label_idx)} samples")
        print(f"Total labeled: {len(label_idx)} | Remaining unlabeled: {len(unlabeled_idx)}")
        
        # Sanity check
        if len(label_idx) != len(set(label_idx)):
            raise ValueError("Duplicate indices in label_idx!")
        if len(unlabeled_idx) != len(set(unlabeled_idx)):
            raise ValueError("Duplicate indices in unlabeled_idx!")
        
        # ===== Train Model =====
        print(f'\n===== Train Model with {portion}% data ({len(label_idx)} samples) =====')
        print('-' * 50)
        print(f'{"Batch Size":<20}: {args.batch_size}')
        print(f'{"Learning Rate":<20}: {args.lr}')
        print(f'{"Total Samples":<20}: {len(label_idx)}')
        print(f'{"Batches per Epoch":<20}: {max(1, len(label_idx) // args.batch_size)}')
        print('-' * 50)
        
        # Create a simple args object for train_none_AL
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
        
        # Train and get model + final dice
        last_trained_model, final_dice = train_none_AL(train_args, device)
        
        # Round to 4 decimal places
        final_dice = round(final_dice, 4)
        print(f"\nFinal Dice: {final_dice}")
        
        # ===== Save Results =====
        # Add results to the nested structure
        data[portion_key][lr_key]["dice"].append(final_dice)
        data[portion_key][lr_key]["labeled_idx"].append(label_idx.copy())
        
        # Sort portions and lrs
        sorted_data = {}
        for portion_k in sorted(data.keys(), key=float):
            sorted_lrs = {}
            for lr_k in sorted(data[portion_k].keys(), key=float):
                sorted_lrs[lr_k] = data[portion_k][lr_k]
            sorted_data[portion_k] = sorted_lrs
        
        # Save with compact list format
        save_compact_json(sorted_data, file_path)
        print(f'\n✓ Result saved to {file_path}!')
    
    print('\n' + '=' * 80)
    print('ACTIVE LEARNING COMPLETED!')
    print('=' * 80)


if __name__ == "__main__":
    main()