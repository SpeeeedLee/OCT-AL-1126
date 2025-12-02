#!/bin/bash

for run in {1..5}; do
    echo "=========================================="
    echo "Starting Run $run/5 (All 5 experiments in parallel)"
    echo "=========================================="
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --device 'cuda:5' --portion 100 --seed 42 --no_data_aug --exp_path "./exp_results_temp_no_aug" &
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --device 'cuda:6' --portion 100 --seed 42 --aug_factor 2 --flip_type 'horizontal' --exp_path "./exp_results_temp_aug2_horizontal" &
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --device 'cuda:8' --portion 100 --seed 42 --aug_factor 2 --flip_type 'vertical' --exp_path "./exp_results_temp_aug2_vertical" &
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --device 'cuda:5' --portion 100 --seed 42 --aug_factor 3 --exp_path "./exp_results_temp_aug3" &
    
    python3 ./classification/run_first_iter.py --task_type 'hard' --device 'cuda:6' --portion 100 --seed 42 --aug_factor 4 --exp_path "./exp_results_temp_aug4" &
    
    wait
    echo "Run $run/5 completed"
done

echo "All experiments completed!"