#!/bin/bash

pids=()

cleanup() {
    echo ""
    echo "=========================================="
    echo "Caught Ctrl+C! Killing all processes..."
    echo "=========================================="
    for pid in "${pids[@]}"; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing process $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    wait 2>/dev/null
    echo "All processes killed. Exiting."
    exit 1
}

trap cleanup SIGINT SIGTERM

######## custom configs ########
lrs=(${LRS:-7e-5 1e-4 3e-4 5e-4 7e-4})
runs=${RUNS:-3}
seeds=(${SEEDS:-10})                             # 支援多個 seed
portions=(${PORTIONS:-40})
device=${DEVICE:-'cuda:1'}
pretrained_weights=${PRETRAINED:-'imagenet'}
epoch=${EPOCH:-20}
max_run=${MAX_RUN:-5}
augs=(${AUGS:-no_aug aug2_h aug2_v aug3 aug4})
#################################

# 驗證 augs 的值是否合法
valid_augs=(no_aug aug2_h aug2_v aug3 aug4)
for aug in "${augs[@]}"; do
    valid=false
    for v in "${valid_augs[@]}"; do
        [ "$aug" = "$v" ] && valid=true && break
    done
    if [ "$valid" = false ]; then
        echo "ERROR: invalid aug '$aug'. Valid options: ${valid_augs[*]}"
        exit 1
    fi
done

echo "Seeds   : ${seeds[*]}"
echo "Augs    : ${augs[*]}"
echo "Portions: ${portions[*]}"
echo "LRs     : ${lrs[*]}"

# ---- worker pool 函式 ----
wait_for_slot() {
    while true; do
        local alive=()
        for pid in "${pids[@]}"; do
            if ps -p $pid > /dev/null 2>&1; then
                alive+=($pid)
            fi
        done
        pids=("${alive[@]}")
        if [ ${#pids[@]} -lt $max_run ]; then
            return
        fi
        sleep 1
    done
}

wait_all() {
    wait 2>/dev/null
    pids=()
}

# ---- 輔助：把 aug 名稱轉成對應的 python args ----
aug_to_args() {
    case "$1" in
        no_aug)  echo "--no_data_aug" ;;
        aug2_h)  echo "--aug_factor 2 --flip_type horizontal" ;;
        aug2_v)  echo "--aug_factor 2 --flip_type vertical" ;;
        aug3)    echo "--aug_factor 3" ;;
        aug4)    echo "--aug_factor 4" ;;
    esac
}

# ---- 建立 job list ----
# 順序：run → seed → portion → aug → lr
declare -a job_queue

for run in $(seq 1 $runs); do
    for seed in "${seeds[@]}"; do
        for portion in "${portions[@]}"; do
            for aug in "${augs[@]}"; do
                aug_args=$(aug_to_args "$aug")
                for lr in "${lrs[@]}"; do
                    job_queue+=("--task_type hard --lr $lr --pretrained_weights $pretrained_weights --device $device --portion $portion --seed $seed --epoch $epoch $aug_args | run=$run seed=$seed portion=$portion aug=$aug lr=$lr")
                done
            done
        done
    done
done

echo "Total jobs: ${#job_queue[@]}, max_run: $max_run"
echo "=========================================="

total=${#job_queue[@]}

for job_entry in "${job_queue[@]}"; do
    python_args="${job_entry%% | *}"
    label="${job_entry##* | }"

    wait_for_slot

    echo "  [LAUNCH] $label  (running: ${#pids[@]}/$max_run)"
    python3 ./run_first_iter.py $python_args &
    pids+=($!)
done

wait_all
echo "=========================================="
echo "All $total jobs dispatched and completed!"
echo "=========================================="