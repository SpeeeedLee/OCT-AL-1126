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
lrs=(5e-5 1e-4 5e-4)
# lrs=(1e-5 1e-3)
runs=3
seed=42
portion=10
device_1='cuda:2'
device_2='cuda:8'
pretrained_weights='imagenet'
#################################

for run in $(seq 1 $runs); do
    echo "=========================================="
    echo "Starting Run $run/$runs (All 5 experiments in parallel)"
    echo "=========================================="
    
    for lr in "${lrs[@]}"; do
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --device $device_1 --portion $portion --seed $seed --no_data_aug &
        pids+=($!)
        
        python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --device $device_1 --portion $portion --seed $seed --aug_factor 2 --flip_type 'horizontal' &
        pids+=($!)
        
        # python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --device $device_2 --portion $portion --seed $seed --aug_factor 2 --flip_type 'vertical' &
        # pids+=($!)
        
        # python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --device $device_2 --portion $portion --seed $seed --aug_factor 3 &
        # pids+=($!)
        
        # python3 ./classification/run_first_iter.py --task_type 'hard' --lr $lr --pretrained_weights $pretrained_weights --device $device_2 --portion $portion --seed $seed --aug_factor 4 &
        # pids+=($!)
    
        wait
    
        # 清空 pids 陣列，準備下一輪
        pids=()
    done
        
    echo "Run $run/$runs completed"
done

echo "All experiments completed!"