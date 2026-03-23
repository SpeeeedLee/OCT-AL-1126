import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from tqdm import tqdm

from segmentation.utils.data import data_loader, o_data, g_data_cell_binary
from segmentation.utils.model import Optim_U_Net
import segmentation.utils.loss as L
from segmentation.utils.tool import compute_dice_binary


def train_none(opt): 
    fold = opt.fold
    print('\n' + '='*80)
    print(f'U-Net Binary Segmentation Training (Nuclei Only) - Fold: {fold}')
    print('='*80)
    
    # Creating or loading the model
    model = Optim_U_Net(img_ch=opt.input_nc, output_ch=1, USE_DS=False, USE_DFS=False)
    if opt.load_model:
        model.load_state_dict(torch.load(opt.modelpath))
        print(f'✓ Model loaded from: {opt.modelpath}')
    model = model.to(opt.device)
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'✓ Number of parameters: {number_of_parameters:,}')
    print(f'✓ Device: {opt.device}')

    # Loading data
    gpath_cell = opt.dataroot + "/cell/"
    opath = opt.dataroot + "/image/"
    
    # Get batch size from args (default to 8 if not specified)
    batch_size = getattr(opt, 'batch_size', 8)
    
    train_data_LD, valid_data_LD, _ = data_loader(opath, fold)
    
    # ===== Apply portion sampling =====
    portion = getattr(opt, 'portion', 100.0)  # Default to 100% if not specified
    seed = getattr(opt, 'seed', 42)  # Default seed
    
    if portion < 100.0:
        print(f'\n{"="*80}')
        print(f'DATA SAMPLING: Using {portion}% of training data (seed={seed})')
        print(f'{"="*80}')
        
        # Get all training file names
        all_train_files = []
        for batch in train_data_LD:
            all_train_files.extend(batch)
        
        total_train = len(all_train_files)
        target_num = round(total_train * portion / 100)
        
        # Sample subset
        random.seed(seed)
        sampled_files = random.sample(all_train_files, target_num)
        
        print(f'✓ Total training images: {total_train}')
        print(f'✓ Sampled images: {target_num} ({portion}%)')
        print(f'{"="*80}')
        
        # Create new data loader with sampled data using batch_size from args
        import torch.utils.data as Data
        train_data_LD = Data.DataLoader(
            dataset=sampled_files, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )

    # Loss function and optimizer
    # Note: Model outputs probabilities (after sigmoid), so we use BinaryDiceLoss
    loss_func = L.BinaryDiceLoss()
    # Alternative: Combined BCE + Dice loss
    # loss_func = L.BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.1)
    
    print(f'✓ Loss function: BinaryDiceLoss')
    print(f'✓ Optimizer: Adam (lr={opt.lr})')
    print(f'✓ Scheduler: StepLR (step_size={opt.step}, gamma=0.1)')

    train_epoch = opt.epoch
    
    # Record the overall loss and dice
    train_loss = np.zeros(train_epoch)
    valid_loss = np.zeros(train_epoch)
    train_mdice = np.zeros(train_epoch)
    valid_mdice = np.zeros(train_epoch)
    
    height = 512
    width = 384
    
    for EPOCH in range(train_epoch):
        # ========== Epoch Header ==========
        print('\n' + '='*80)
        print(f'EPOCH: {EPOCH+1}/{train_epoch} | Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('='*80)
        
        start = time.time()
        model.train()
        
        # Record the training loss and dice
        train_loss_list = []
        train_dice_list = []
        
        # ========== Training Phase ==========
        print('Training...')
        train_progress = tqdm(train_data_LD, desc='Train', ncols=100, 
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch_idx, t_batch_num in enumerate(train_progress):
            # Loading training data
            img_sub = o_data(opath, t_batch_num, width, height)
            t_gim_sub = g_data_cell_binary(gpath_cell, t_batch_num, width, height)

            # Numpy to Tensor on GPU
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)
            target = torch.from_numpy(t_gim_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)

            # Model output and weight updating
            OUTPUT = model(INPUT)  # [batch, 1, H, W], already sigmoid-activated
            loss = loss_func(OUTPUT, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics recording
            train_loss_list.append(loss.item())
            
            # Get binary predictions (threshold at 0.5)
            pred_mask = (OUTPUT > 0.5).float()
            dice = compute_dice_binary(pred_mask.cpu().detach().numpy(), t_gim_sub)
            train_dice_list.append(dice)
            
            # Update progress bar
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        # ========== Validation Phase ==========
        model.eval()
        valid_loss_list = []
        valid_dice_list = []
        
        print('Validating...')
        valid_progress = tqdm(valid_data_LD, desc='Valid', ncols=100,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        with torch.no_grad():
            for batch_idx, v_batch_num in enumerate(valid_progress):
                # Loading validation data
                val_sub = o_data(opath, v_batch_num, width, height)
                v_gim_sub = g_data_cell_binary(gpath_cell, v_batch_num, width, height)
                
                # Numpy to Tensor on GPU
                INPUT = torch.from_numpy(val_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)
                target = torch.from_numpy(v_gim_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)

                # Computing loss
                OUTPUT = model(INPUT)  # [batch, 1, H, W], already sigmoid-activated
                loss_v = loss_func(OUTPUT, target)

                # Metrics recording
                valid_loss_list.append(loss_v.item())
                
                pred_mask = (OUTPUT > 0.5).float()
                dice = compute_dice_binary(pred_mask.cpu().detach().numpy(), v_gim_sub)
                valid_dice_list.append(dice)
                
                # Update progress bar
                valid_progress.set_postfix({
                    'Loss': f'{loss_v.item():.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        scheduler.step()

        # Record overall result for each epoch
        train_loss[EPOCH] = np.mean(train_loss_list)
        valid_loss[EPOCH] = np.mean(valid_loss_list)
        train_mdice[EPOCH] = np.mean(train_dice_list)
        valid_mdice[EPOCH] = np.mean(valid_dice_list)
        
        epoch_time = time.time() - start
        
        # ========== Epoch Summary ==========
        print('\n' + '-'*80)
        print('EPOCH SUMMARY:')
        print(f'  Training   -> Loss: {train_loss[EPOCH]:.4f} | Dice: {train_mdice[EPOCH]:.4f}')
        print(f'  Validation -> Loss: {valid_loss[EPOCH]:.4f} | Dice: {valid_mdice[EPOCH]:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print('-'*80)
    
    # ========== Training Completed ==========
    print('\n' + '='*80)
    print('TRAINING COMPLETED!')
    print('='*80)
    
    # Get final validation dice (last epoch)
    final_dice = valid_mdice[-1]
    
    print(f'FINAL RESULTS:')
    print(f'  Final Validation Dice: {final_dice:.4f}')
    print(f'  Best Validation Dice:  {np.max(valid_mdice):.4f} (Epoch {np.argmax(valid_mdice)+1})')
    print(f'  Final Training Dice:   {train_mdice[-1]:.4f}')
    print('='*80)
    
    return final_dice


def test_none(opt):
    fold = opt.fold
    print('\n' + '='*80)
    print(f'U-Net Binary Segmentation Testing (Nuclei Only) - Fold: {fold}')
    print('='*80)
    
    # Creating or loading the model
    model = Optim_U_Net(img_ch=opt.input_nc, output_ch=1, USE_DS=False, USE_DFS=False)
    if opt.load_model:
        model.load_state_dict(torch.load(opt.modelpath))
        print(f'✓ Model loaded from: {opt.modelpath}')
    model = model.to(opt.device)
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'✓ Number of parameters: {number_of_parameters:,}')

    # Loading data
    gpath_cell = opt.dataroot + "/cell/"
    opath = opt.dataroot + "/image/"
    _, _, test_data_LD = data_loader(opath, fold)
    
    height = 512
    width = 384
    model.eval()
    
    test_dice_list = []
    
    print('\nTesting...')
    test_progress = tqdm(test_data_LD, desc='Test', ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    with torch.no_grad():
        for batch_idx, v_batch_num in enumerate(test_progress):
            # Loading test data
            test_sub = o_data(opath, v_batch_num, width, height)
            t_gim_sub = g_data_cell_binary(gpath_cell, v_batch_num, width, height)
                
            # Numpy to Tensor on GPU
            INPUT = torch.from_numpy(test_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)
            target = torch.from_numpy(t_gim_sub.astype(np.float32)).to(device=opt.device, dtype=torch.float)

            # Model prediction
            OUTPUT = model(INPUT)  # [batch, 1, H, W], already sigmoid-activated

            # Metrics recording (per image)
            pred_mask = (OUTPUT > 0.5).float().cpu().detach().numpy()
            
            for idx in range(len(v_batch_num)):
                dice = compute_dice_binary(pred_mask[idx:idx+1], t_gim_sub[idx:idx+1])
                test_dice_list.append(dice)
            
            # Update progress bar
            current_dice = np.mean(test_dice_list)
            test_progress.set_postfix({'Avg Dice': f'{current_dice:.4f}'})
    
    # ========== Test Results ==========
    mean_dice = np.mean(test_dice_list)
    std_dice = np.std(test_dice_list)
    
    print('\n' + '='*80)
    print('TEST RESULTS:')
    print(f'  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}')
    print(f'  Min Dice:  {np.min(test_dice_list):.4f}')
    print(f'  Max Dice:  {np.max(test_dice_list):.4f}')
    print(f'  Total images tested: {len(test_dice_list)}')
    print('='*80 + '\n')
    
    return



def train_none_AL(opt, device):
    """
    訓練函數 for Active Learning
    
    Args:
        opt: 參數物件，必須包含：
            - dataroot: 資料路徑
            - fold: fold 編號
            - label_idx: 訓練資料的檔名列表
            - batch_size: batch size
            - lr: learning rate
            - step: scheduler step
            - epoch: 訓練 epochs
            - input_nc: 輸入 channels
            - output_nc: 輸出 channels
        device: 訓練設備
    
    Returns:
        model: 訓練完成的模型
        final_dice: 最終驗證 Dice score
    """
    fold = opt.fold
    print('\n' + '='*80)
    print(f'U-Net Binary Segmentation Training (Active Learning) - Fold: {fold}')
    print('='*80)
    
    # Creating model
    model = Optim_U_Net(img_ch=opt.input_nc, output_ch=1, USE_DS=False, USE_DFS=False)
    model = model.to(device)
    number_of_parameters = sum(p.numel() for p in model.parameters())
    print(f'✓ Number of parameters: {number_of_parameters:,}')
    print(f'✓ Device: {device}')

    # Loading data paths
    gpath_cell = opt.dataroot + "/cell/"
    opath = opt.dataroot + "/image/"
    
    # Load validation data using standard data_loader
    _, valid_data_LD, _ = data_loader(opath, fold)
    
    # Get batch size from args
    batch_size = opt.batch_size
    
    # Use label_idx for training data
    label_idx = opt.label_idx
    print(f'\n{"="*80}')
    print(f'ACTIVE LEARNING: Using {len(label_idx)} labeled samples')
    print(f'{"="*80}')
    
    # Create training data loader with label_idx
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
    
    print(f'✓ Loss function: BinaryDiceLoss')
    print(f'✓ Optimizer: Adam (lr={opt.lr})')
    print(f'✓ Scheduler: StepLR (step_size={opt.step}, gamma=0.1)')

    train_epoch = opt.epoch
    
    # Record the overall loss and dice
    train_loss = np.zeros(train_epoch)
    valid_loss = np.zeros(train_epoch)
    train_mdice = np.zeros(train_epoch)
    valid_mdice = np.zeros(train_epoch)
    
    height = 512
    width = 384
    
    for EPOCH in range(train_epoch):
        # ========== Epoch Header ==========
        print('\n' + '='*80)
        print(f'EPOCH: {EPOCH+1}/{train_epoch} | Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('='*80)
        
        start = time.time()
        model.train()
        
        # Record the training loss and dice
        train_loss_list = []
        train_dice_list = []
        
        # ========== Training Phase ==========
        print('Training...')
        train_progress = tqdm(train_data_LD, desc='Train', ncols=100, 
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch_idx, t_batch_num in enumerate(train_progress):
            # Loading training data
            img_sub = o_data(opath, t_batch_num, width, height)
            t_gim_sub = g_data_cell_binary(gpath_cell, t_batch_num, width, height)

            # Numpy to Tensor on GPU
            INPUT = torch.from_numpy(img_sub.astype(np.float32)).to(device=device, dtype=torch.float)
            target = torch.from_numpy(t_gim_sub.astype(np.float32)).to(device=device, dtype=torch.float)

            # Model output and weight updating
            OUTPUT = model(INPUT)
            loss = loss_func(OUTPUT, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics recording
            train_loss_list.append(loss.item())
            
            # Get binary predictions (threshold at 0.5)
            pred_mask = (OUTPUT > 0.5).float()
            dice = compute_dice_binary(pred_mask.cpu().detach().numpy(), t_gim_sub)
            train_dice_list.append(dice)
            
            # Update progress bar
            train_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        # ========== Validation Phase ==========
        model.eval()
        valid_loss_list = []
        valid_dice_list = []
        
        print('Validating...')
        valid_progress = tqdm(valid_data_LD, desc='Valid', ncols=100,
                             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        with torch.no_grad():
            for batch_idx, v_batch_num in enumerate(valid_progress):
                # Loading validation data
                val_sub = o_data(opath, v_batch_num, width, height)
                v_gim_sub = g_data_cell_binary(gpath_cell, v_batch_num, width, height)
                
                # Numpy to Tensor on GPU
                INPUT = torch.from_numpy(val_sub.astype(np.float32)).to(device=device, dtype=torch.float)
                target = torch.from_numpy(v_gim_sub.astype(np.float32)).to(device=device, dtype=torch.float)

                # Computing loss
                OUTPUT = model(INPUT)
                loss_v = loss_func(OUTPUT, target)

                # Metrics recording
                valid_loss_list.append(loss_v.item())
                
                pred_mask = (OUTPUT > 0.5).float()
                dice = compute_dice_binary(pred_mask.cpu().detach().numpy(), v_gim_sub)
                valid_dice_list.append(dice)
                
                # Update progress bar
                valid_progress.set_postfix({
                    'Loss': f'{loss_v.item():.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        scheduler.step()

        # Record overall result for each epoch
        train_loss[EPOCH] = np.mean(train_loss_list)
        valid_loss[EPOCH] = np.mean(valid_loss_list)
        train_mdice[EPOCH] = np.mean(train_dice_list)
        valid_mdice[EPOCH] = np.mean(valid_dice_list)
        
        epoch_time = time.time() - start
        
        # ========== Epoch Summary ==========
        print('\n' + '-'*80)
        print('EPOCH SUMMARY:')
        print(f'  Training   -> Loss: {train_loss[EPOCH]:.4f} | Dice: {train_mdice[EPOCH]:.4f}')
        print(f'  Validation -> Loss: {valid_loss[EPOCH]:.4f} | Dice: {valid_mdice[EPOCH]:.4f}')
        print(f'  Time: {epoch_time:.2f}s')
        print('-'*80)
    
    # ========== Training Completed ==========
    print('\n' + '='*80)
    print('TRAINING COMPLETED!')
    print('='*80)
    
    # Get final validation dice (last epoch)
    final_dice = valid_mdice[-1]
    
    print(f'FINAL RESULTS:')
    print(f'  Final Validation Dice: {final_dice:.4f}')
    print(f'  Best Validation Dice:  {np.max(valid_mdice):.4f} (Epoch {np.argmax(valid_mdice)+1})')
    print(f'  Final Training Dice:   {train_mdice[-1]:.4f}')
    print('='*80)
    
    # Return both model and final dice for Active Learning
    return model, final_dice