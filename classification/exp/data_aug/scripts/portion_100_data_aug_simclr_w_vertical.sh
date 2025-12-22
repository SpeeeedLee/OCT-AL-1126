#!/bin/bash

# 定義清理函數
cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught Ctrl+C! Exiting..."
    echo "=========================================="
    exit 1
}

# 設置 trap 來捕捉 Ctrl+C (SIGINT) 和 SIGTERM
trap cleanup SIGINT SIGTERM

######## custom configs ########
lrs=(1e-5 5e-5 1e-4 5e-4)
# lrs=(1e-3)
runs=3
seed=42
portion=100
device='cuda:2'
pretrained_weights='simclr'
simclr_path='./classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl'
#################################

for run in $(seq 1 $runs); do
    echo "=========================================="
    echo "Starting Run $run/$runs (Sequential execution)"
    echo "=========================================="
    
    for lr in "${lrs[@]}"; do
        echo "Running experiments with lr=$lr..."
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --simclr_path $simclr_path --device $device --portion $portion --seed $seed --no_data_aug
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --simclr_path $simclr_path --device $device --portion $portion --seed $seed --aug_factor 2 --flip_type 'horizontal'
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --simclr_path $simclr_path --device $device --portion $portion --seed $seed --aug_factor 2 --flip_type 'vertical'
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --simclr_path $simclr_path --device $device --portion $portion --seed $seed --aug_factor 3
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --simclr_path $simclr_path --device $device --portion $portion --seed $seed --aug_factor 4
        
        echo "Completed lr=$lr"
    done
        
    echo "Run $run/$runs completed"
done

echo "All experiments completed!"