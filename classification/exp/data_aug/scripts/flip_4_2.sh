#!/bin/bash

# 執行順序：
# Run 1
#   Portion 10%
#     no_aug:        4 LRs 並行 → wait
#     aug2 horizontal: 4 LRs 並行 → wait
#     aug2 vertical:   4 LRs 並行 → wait
#     aug3:            4 LRs 並行 → wait
#     aug4:            4 LRs 並行 → wait
#   Portion 30%  ...
#   Portion 50%  ...
# Run 2 ...
# Run 3 ...

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
lrs=(7e-5 1e-4 3e-4 5e-4 7e-4)
runs=3
# seed=42
# seed=24
seed=10
portions=(5)
device='cuda:0'
pretrained_weights='imagenet'
epoch=20
#################################

total_portions=${#portions[@]}

for run in $(seq 1 $runs); do
    echo "=========================================="
    echo "Run $run/$runs"
    echo "=========================================="

    current_portion=0
    for portion in "${portions[@]}"; do
        current_portion=$((current_portion + 1))
        echo "  --- Portion $portion% [$current_portion/$total_portions] ---"

        # # ---- no_aug ----
        # echo "    [no_aug] launching ${#lrs[@]} LRs in parallel..."
        # for lr in "${lrs[@]}"; do
        #     python3 ./run_first_iter.py \
        #         --task_type 'hard' --lr $lr \
        #         --pretrained_weights $pretrained_weights \
        #         --device $device \
        #         --portion $portion --seed $seed \
        #         --epoch $epoch \
        #         --no_data_aug &
        #     pids+=($!)
        # done
        # wait; pids=()
        # echo "    [no_aug] done"

        # # ---- aug2 horizontal ----
        # echo "    [aug2 horizontal] launching ${#lrs[@]} LRs in parallel..."
        # for lr in "${lrs[@]}"; do
        #     python3 ./run_first_iter.py \
        #         --task_type 'hard' --lr $lr \
        #         --pretrained_weights $pretrained_weights \
        #         --device $device \
        #         --portion $portion --seed $seed \
        #         --epoch $epoch \
        #         --aug_factor 2 --flip_type 'horizontal' &
        #     pids+=($!)
        # done
        # wait; pids=()
        # echo "    [aug2 horizontal] done"

        # # ---- aug2 vertical ----
        # echo "    [aug2 vertical] launching ${#lrs[@]} LRs in parallel..."
        # for lr in "${lrs[@]}"; do
        #     python3 ./run_first_iter.py \
        #         --task_type 'hard' --lr $lr \
        #         --pretrained_weights $pretrained_weights \
        #         --device $device \
        #         --portion $portion --seed $seed \
        #         --epoch $epoch \
        #         --aug_factor 2 --flip_type 'vertical' &
        #     pids+=($!)
        # done
        # wait; pids=()
        # echo "    [aug2 vertical] done"

        # # ---- aug3 ----
        # echo "    [aug3] launching ${#lrs[@]} LRs in parallel..."
        # for lr in "${lrs[@]}"; do
        #     python3 ./run_first_iter.py \
        #         --task_type 'hard' --lr $lr \
        #         --pretrained_weights $pretrained_weights \
        #         --device $device \
        #         --portion $portion --seed $seed \
        #         --epoch $epoch \
        #         --aug_factor 3 &
        #     pids+=($!)
        # done
        # wait; pids=()
        # echo "    [aug3] done"

        # ---- aug4 ----
        echo "    [aug4] launching ${#lrs[@]} LRs in parallel..."
        for lr in "${lrs[@]}"; do
            python3 ./run_first_iter.py \
                --task_type 'hard' --lr $lr \
                --pretrained_weights $pretrained_weights \
                --device $device \
                --portion $portion --seed $seed \
                --epoch $epoch \
                --aug_factor 4 &
            pids+=($!)
        done
        wait; pids=()
        echo "    [aug4] done"

        echo "  Portion $portion% completed"
    done

    echo "Run $run/$runs completed"
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

