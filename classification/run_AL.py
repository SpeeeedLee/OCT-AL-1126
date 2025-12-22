import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR
from classification.utils.train_eval import train_model

from classification.AL_strategy.uncertainty import conf, entropy, margin
from classification.AL_strategy.diversity import coreset
from classification.AL_strategy.hybrid import badge


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
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--AL_strategy', type=str, 
                        choices=['conf', 'entropy', 'margin', 'coreset', 'badge'], 
                        required=True)

    # AL related setup
    parser.add_argument('--portion_start', type=float, required=True)
    parser.add_argument('--portion_end', type=float, required=True)
    parser.add_argument('--portion_interval', type=float, required=True)   
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)

    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, 
                        choices=['random', 'imagenet', 'simclr', 'auto_encoder'], 
                        required=True)   
    parser.add_argument('--simclr_path', type=str, default=None)
    
    # Training hyperparameters (aligned with run_first_iter.py)
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false', default=True)
    parser.add_argument('--aug_factor', type=int, default=4)   
    parser.add_argument('--flip_type', type=str, default='horizontal')
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained):
    model = get_resnet18_classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def initialize_simclr_model(num_classes, simclr_path):
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
    return model


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
    random.seed(args.seed) # this will be realted to only how the initial labeled pool be selected
    
    # Determine task configuration
    task_config = {
        'easy': (2, './ds/classification/two_class'),
        'medium': (4, './ds/classification/four_class'),
        'hard': (7, './ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    # Fix batch size to 16 (aligned with run_first_iter.py)
    batch_size = 16
    
    # Generate file name (aligned with run_first_iter.py format)
    file_name = f"{args.AL_strategy}_seed{args.seed}_bs{batch_size}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
    
    print(f'Exp name: {file_name}')
    print(f'AL Strategy: {args.AL_strategy}')
    print(f'Portion range: {args.portion_start}% to {args.portion_end}% (interval: {args.portion_interval}%)')
    print(f'Learning Rate: {args.lr} (FIXED across all portions)')
    print(f'Batch Size: {batch_size}')
    print('-' * 60)
    
    # Generate aug key (aligned with run_first_iter.py)
    if not args.data_aug:
        aug_key = "no_aug"
    elif args.aug_factor == 2:
        aug_key = f"aug{args.aug_factor}_{args.flip_type}"
    else:
        aug_key = f"aug{args.aug_factor}"
    
    print(f'Aug config key: {aug_key}')
    
    # Initialize label/unlabel indices
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    label_idx = []
    unlabeled_idx = list(range(tot_num_train))
    
    # Prepare save path
    save_path = os.path.join(args.exp_path, 
                             f"classification_{args.task_type}", 
                             f"AL_{args.pretrained_weights}")
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
    
    # Initialize data structure
    if aug_key not in data:
        data[aug_key] = {}
    
    lr_key = str(args.lr)
    
    # Active Learning Loop
    last_trained_model = None
    
    for portion in np.arange(args.portion_start, args.portion_end, args.portion_interval):
        print('\n' + '=' * 60)
        print(f'PORTION: {portion}%')
        print('=' * 60)
        
        portion_key = str(float(portion))
        
        # Initialize portion if not exists
        if portion_key not in data[aug_key]:
            data[aug_key][portion_key] = {}
        
        # Initialize lr if not exists
        if lr_key not in data[aug_key][portion_key]:
            data[aug_key][portion_key][lr_key] = {
                "acc": [],
                "labeled_idx": []
            }
        
        # ===== Select Data to Label =====
        target_num = round(tot_num_train * portion / 100)
        num_to_label = target_num - len(label_idx)
        
        if portion == args.portion_start:
            # First iteration: random sampling
            print(f'===== First Iteration: Random Sample {num_to_label} samples =====')
            to_label_idx = random.sample(unlabeled_idx, num_to_label)
        else:
            # Subsequent iterations: use AL strategy
            print(f'===== AL Strategy: {args.AL_strategy} - Select {num_to_label} samples =====')
            
            # if args.AL_strategy == 'random':
            #     to_label_idx = random.sample(unlabeled_idx, num_to_label)
            if args.AL_strategy == 'conf':
                to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'margin':
                to_label_idx, _ = margin(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'entropy':
                to_label_idx = entropy(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)    
            elif args.AL_strategy == 'coreset':
                to_label_idx = coreset(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'badge':
                to_label_idx = badge(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
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
        
        # ===== Initialize Model =====
        if args.pretrained_weights == 'random':
            print('Initialize ResNet18 without pretrained weights')
            model = initialize_model(num_classes, False)
        elif args.pretrained_weights == 'simclr':
            print('Initialize ResNet18 using SimCLR weights')
            print(f'SimCLR model path: {args.simclr_path}')
            model = initialize_simclr_model(num_classes, args.simclr_path)
        elif args.pretrained_weights == 'auto_encoder':
            raise NotImplementedError("Auto encoder not implemented yet")
        elif args.pretrained_weights == 'imagenet':
            print('Initialize ResNet18 using ImageNet pretrained weights')
            model = initialize_model(num_classes, True)
        else:
            raise NotImplementedError(f"Pretrained weights {args.pretrained_weights} not implemented")
        
        # ===== Setup Training =====
        criterion = nn.CrossEntropyLoss()
        
        if args.weight_decay is not None:
            print(f"Set weight decay to {args.weight_decay}")
            optimizer_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer_ = optim.AdamW(model.parameters(), lr=args.lr)
        
        lr_scheduler_ = lr_scheduler.LinearLR(
            optimizer_, 
            start_factor=1.0,
            end_factor=0.0,
            total_iters=args.epoch
        )
        
        # ===== Load Data =====
        if args.data_aug:
            data_loaders, dataset_sizes = get_data(
                data_dir, label_idx, batch_size, 
                data_aug=True, aug_factor=args.aug_factor, flip_type=args.flip_type
            )
        else:
            data_loaders, dataset_sizes = get_data(data_dir, label_idx, batch_size, data_aug=False)
        
        print(f"Dataset sizes: {dataset_sizes}")
        
        # ===== Train Model =====
        print(f'===== Train Model with {portion}% data ({len(label_idx)} samples) =====')
        print('-' * 50)
        print(f'{"Batch Size":<20}: {batch_size}')
        print(f'{"Learning Rate":<20}: {args.lr}')
        print(f'{"Weight Decay":<20}: {args.weight_decay if args.weight_decay else "None"}')
        print(f'{"Total Samples":<20}: {len(label_idx)}')
        print(f'{"Batches per Epoch":<20}: {len(label_idx) // batch_size}')
        
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{"Trainable Params":<20}: {total_trainable_params / 1e6:.2f}M')
        print('-' * 50)
        
        last_trained_model, final_acc = train_model(
            model, args.device, data_loaders, dataset_sizes, 
            criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch
        )
        
        # Round to 4 decimal places
        final_acc = round(final_acc, 4)
        print(f"Final Acc: {final_acc}")
        
        # ===== Save Results =====
        # Add results to the nested structure
        data[aug_key][portion_key][lr_key]["acc"].append(final_acc)
        data[aug_key][portion_key][lr_key]["labeled_idx"].append(label_idx.copy())
        
        # Sort portions AND lrs within each aug configuration
        sorted_data = {}
        for aug_k in sorted(data.keys()):
            sorted_portions = {}
            for portion_k in sorted(data[aug_k].keys(), key=float):
                sorted_lrs = {}
                for lr_k in sorted(data[aug_k][portion_k].keys(), key=float):
                    sorted_lrs[lr_k] = data[aug_k][portion_k][lr_k]
                sorted_portions[portion_k] = sorted_lrs
            sorted_data[aug_k] = sorted_portions
        
        # Save with compact list format
        save_compact_json(sorted_data, file_path)
        print(f'Result saved to {file_path}!')
    
    print('\n' + '=' * 60)
    print('Active Learning Completed!')
    print('=' * 60)


if __name__ == "__main__":
    main()