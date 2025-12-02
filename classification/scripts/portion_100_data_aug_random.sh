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

for run in {1..5}; do
    echo "=========================================="
    echo "Starting Run $run/5 (All 5 experiments in parallel)"
    echo "=========================================="
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --no_use_pretrained --device 'cuda:1' --portion 100 --seed 42 --no_data_aug --exp_path "./exp_results_temp_random_no_aug" &
    pids+=($!)
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --no_use_pretrained --device 'cuda:1' --portion 100 --seed 42 --aug_factor 2 --flip_type 'horizontal' --exp_path "./exp_results_temp_random_aug2_horizontal" &
    pids+=($!)
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --no_use_pretrained --device 'cuda:1' --portion 100 --seed 42 --aug_factor 2 --flip_type 'vertical' --exp_path "./exp_results_temp_random_aug2_vertical" &
    pids+=($!)
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --no_use_pretrained --device 'cuda:7' --portion 100 --seed 42 --aug_factor 3 --exp_path "./exp_results_random_temp_aug3" &
    pids+=($!)
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --no_use_pretrained --device 'cuda:7' --portion 100 --seed 42 --aug_factor 4 --exp_path "./exp_results_random_temp_aug4" &
    pids+=($!)
    
    wait
    
    # 清空 pids 陣列，準備下一輪
    pids=()
    
    echo "Run $run/5 completed"
done

echo "All experiments completed!"