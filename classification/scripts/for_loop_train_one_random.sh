#!/bin/bash

# ==================== 信号处理 ====================
# 用于存储所有后台进程的 PID
declare -a pids=()

# 清理函数：杀死所有后台进程
cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught Ctrl+C! Cleaning up..."
    echo "=========================================="
    
    # 杀死所有记录的后台进程
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    # 杀死所有当前 shell 的后台任务
    jobs -p | xargs -r kill -9 2>/dev/null
    
    echo "All processes killed. Exiting..."
    exit 1
}

# 捕获 Ctrl+C (SIGINT) 和 SIGTERM
trap cleanup SIGINT SIGTERM

# ===================================================

# ==================== 配置參數 ====================
# 定義不同的 portion 值
portions=(5 10 15 20 25 30 35 40 45 50)

# 定義 seed
seeds=(42)

# 每個配置重複執行的次數
num_runs=3

# ===================================================

# 遍歷每個 portion
for portion in "${portions[@]}"; do
    echo "=========================================="
    echo "Running experiments for portion=${portion}"
    echo "=========================================="
    
    # 遍歷每個 seed
    for seed in "${seeds[@]}"; do
        echo "Running with seed=${seed}, repeating ${num_runs} times (parallel)"
        
        # 清空当前批次的 PID 数组
        pids=()
        
        # 同時執行 num_runs 次（相同的參數）
        for run_id in $(seq 1 ${num_runs}); do
            echo "Starting run ${run_id}/${num_runs} for portion=${portion}, seed=${seed}"
            python3 ./classification/run_first_iter.py \
                --task_type 'hard' \
                --device 'cuda:1' \
                --portion ${portion} \
                --seed ${seed} \
                --no_use_pretrained &
            
            # 记录后台进程的 PID
            pids+=($!)
        done
        
        # 等待這 num_runs 個進程全部完成
        wait
        
        echo "Seed ${seed} completed!"
        echo ""
    done
    
    echo "Portion ${portion} completed!"
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="