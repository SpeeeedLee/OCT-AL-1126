#!/bin/bash

# 儲存所有進程 ID 的陣列
pids=()

# 定義清理函數
cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught Ctrl+C! Killing all processes..."
    echo "=========================================="
    
    # 殺掉所有收集到的進程
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # 等待所有進程結束
    wait 2>/dev/null
    
    echo "All processes killed. Exiting."
    exit 1
}

# 設置 trap 來捕捉 Ctrl+C (SIGINT) 和 SIGTERM
trap cleanup SIGINT SIGTERM

######## custom configs ########
# lrs=(1e-5 5e-5 1e-4 5e-4)
lrs=(1e-4 5e-4)
runs=3
seed=42
# portions=(5 10 15 20 25 30 35 45 50)
# portions=(40)
portions=(55 60)
device='cuda:2'
pretrained_weights='simclr'
simclr_path='./classification/model/simclr/ckpt_w_vertical_aug/resnet18_simclr_lr0.0004_bs256_ep300.pkl'
#################################

for run in $(seq 1 $runs); do
    echo "=========================================="
    echo "Starting Run $run/$runs"
    echo "=========================================="
    
    for portion in "${portions[@]}"; do
        echo "  Processing portion $portion..."
        
        # 對每個 portion，平行執行所有 lr
        for lr in "${lrs[@]}"; do
            python3 ./classification/run_first_iter.py \
                --task_type 'hard' \
                --lr $lr \
                --pretrained_weights $pretrained_weights \
                --simclr_path $simclr_path \
                --device $device \
                --portion $portion \
                --seed $seed \
                --aug_factor 4 &
            pids+=($!)
        done
        
        # 等待當前 portion 的所有 lr 完成
        wait
        
        # 清空 pids 陣列，準備下一個 portion
        pids=()
        
        echo "  Portion $portion completed"
    done
    
    echo "Run $run/$runs completed"
done

echo "All experiments completed!"