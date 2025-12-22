import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np
import torch
from segmentation.utils.train_NONE import train_none, test_none


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Data and experiment settings
    parser.add_argument('--dataroot', required=True, help='path to image and label dataset')
    parser.add_argument('--phase', type=str, default='train', help='[train | test]')
    parser.add_argument('--fold', type=int, default=0, help='which fold to use for cross-validation')
    parser.add_argument('--device', type=str, default='cuda:0')
    
    # Model settings
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels (1 for grayscale)')
    parser.add_argument('--output_nc', type=int, default=1, help='# of output channels (1 for binary segmentation)')
    
    # Training settings
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step', type=int, default=10, help='step size of scheduler')
    parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    
    # Portion and reproducibility
    parser.add_argument('--portion', type=float, required=True, help='percentage of training data to use (1-100)')
    parser.add_argument('--seed', type=int, required=True, help='random seed for data sampling')
    
    # Model loading (for testing)
    parser.add_argument('--load_model', action='store_true', default=False, help='load pre-trained model')
    parser.add_argument('--modelpath', type=str, default=None, help='path to load model from')
    
    # Result saving
    parser.add_argument('--exp_path', type=str, default='./segmentation/exp_results', help='path to save experiment results')
    
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


def check_existing_results(file_path, portion_key, lr_key, max_runs=5):
    """
    Check if the experiment has already been run enough times.
    Raises an error if the result list already has max_runs or more entries.
    """
    if not os.path.isfile(file_path):
        print(f"No existing results file found at {file_path}")
        return
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {file_path}: {e}")
        return
    
    # Navigate through the nested structure
    if portion_key in data:
        if lr_key in data[portion_key]:
            existing_results = data[portion_key][lr_key]
            num_existing = len(existing_results)
            
            if num_existing >= max_runs:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"Experiment already completed!\n"
                    f"Configuration: portion={portion_key}%, lr={lr_key}\n"
                    f"Existing results: {existing_results}\n"
                    f"Number of runs: {num_existing}/{max_runs}\n"
                    f"File: {file_path}\n"
                    f"{'='*80}\n"
                )
            else:
                print(f"Found {num_existing}/{max_runs} existing results for this configuration")
        else:
            print(f"No results found for lr={lr_key}")
    else:
        print(f"No results found for portion={portion_key}%")


def save_results_to_json(args, final_dice):
    """Save experiment results to JSON file"""
    # Generate file name: random_bs{batch_size}.json
    file_name = f"random_bs{args.batch_size}.json"
    
    # Construct save path: ./segmentation/exp_results/nuclei/cold_start_random/
    save_path = os.path.join(args.exp_path, "nuclei", "cold_start_random")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    # Generate keys
    portion_key = str(float(args.portion))
    lr_key = str(args.lr)
    
    print(f'\nSaving results to JSON...')
    print(f'Portion: {portion_key}%, LR: {lr_key}')
    
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
    
    # Initialize portion
    if portion_key not in data:
        data[portion_key] = {}
    
    # Initialize lr
    if lr_key not in data[portion_key]:
        data[portion_key][lr_key] = []
    
    # Add result (round to 4 decimal places)
    final_dice = round(final_dice, 4)
    data[portion_key][lr_key].append(final_dice)
    
    # Sort portions and lrs
    sorted_data = {}
    # Sort portions by float value
    for portion_k in sorted(data.keys(), key=float):
        # Sort lrs by float value within each portion
        sorted_lrs = {}
        for lr_k in sorted(data[portion_k].keys(), key=float):
            sorted_lrs[lr_k] = data[portion_k][lr_k]
        sorted_data[portion_k] = sorted_lrs
    
    # Save with compact list format
    save_compact_json(sorted_data, file_path)
    
    print(f'✓ Results saved to {file_path}')
    print(f'  Final Dice: {final_dice}')
    print(f'  Total runs for this config: {len(data[portion_key][lr_key])}')


def main():
    args = parse_arguments()
    
    # Validate portion
    if args.portion <= 0 or args.portion > 100:
        raise ValueError(f"Portion must be between 0 and 100, got {args.portion}")
    
    # Generate file name for checking
    file_name = f"random_bs{args.batch_size}.json"
    print(f'Exp name: {file_name}')
    
    # ===== CHECK IF EXPERIMENT ALREADY COMPLETED (for training) =====
    if args.phase == 'train':
        save_path = os.path.join(args.exp_path, "nuclei", "cold_start_random")
        file_path = os.path.join(save_path, file_name)
        
        portion_key = str(float(args.portion))
        lr_key = str(args.lr)
        
        print(f"\nChecking existing results...")
        check_existing_results(file_path, portion_key, lr_key, max_runs=5)
        print(f"Check passed. Proceeding with training...\n")
    # =================================================================
    
    if args.phase == 'train':
        print("="*80)
        print(f"BINARY NUCLEI SEGMENTATION - TRAINING (Fold {args.fold})")
        print("="*80)
        print(f"{'Portion':<20}: {args.portion}%")
        print(f"{'Seed':<20}: {args.seed}")
        print(f"{'Learning Rate':<20}: {args.lr}")
        print(f"{'Epochs':<20}: {args.epoch}")
        print(f"{'Batch Size':<20}: {args.batch_size}")
        print("="*80)
        
        # Train and get final dice
        final_dice = train_none(args)
        
        # Always save results to JSON
        save_results_to_json(args, final_dice)
        
    elif args.phase == 'test':
        print("="*80)
        print(f"BINARY NUCLEI SEGMENTATION - TESTING (Fold {args.fold})")
        print("="*80)
        if args.modelpath:
            print(f"{'Model Path':<20}: {args.modelpath}")
        else:
            raise ValueError("--modelpath is required for testing phase")
        print("="*80)
        
        args.load_model = True
        test_none(args)
        
    else:
        print(f"Error: Unknown phase '{args.phase}'. Use 'train' or 'test'.")


if __name__ == "__main__":
    main()