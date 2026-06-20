#!/usr/bin/env bash
# 5.3.3　把 TypiClust AL 軌跡延伸跑到 ρ=90%（seed 42），以取得 bottom-10% leftover。
# 從 4.4 既有軌跡 cumulative@60% 接續，b=5 跑到 90%。
# 結果寫到隔離樹，不污染 4.4 主線。
#   DEVICE=cuda:X ./thesis/chapter_5/run_5_3_3_extend_typiclust.sh
set -eu
cd "$(git rev-parse --show-toplevel)"
DEVICE="${DEVICE:-cuda:8}"
EXP="${EXP:-./classification/exp_results/ch5_5_3_3_extend}"
SIMCLR="SSL/simclr/ckpt/resnet18_simclr_lr0.0002_bs256_ep500.pkl"
PORTIONS="${PORTIONS:-2.5 5 7.5 10 12.5 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5 55 57.5 60 65 70 75 80 85 90}"
RESUME_FROM="${RESUME_FROM:-60}"
RESUME_SRC="${RESUME_SRC:-./classification/exp_results/classification_hard/AL_simclr/labeled_ids/typiclust_seed42_bs16.json}"

echo "TypiClust AL → ρ=90%  seed=42  device=$DEVICE  exp_path=$EXP  (resume@${RESUME_FROM}%)"
python3 classification/run_AL.py \
  --task_type hard --AL_strategy typiclust \
  --pretrained_weights simclr --simclr_path "$SIMCLR" \
  --portions "$PORTIONS" \
  --resume_labeled_ids "$RESUME_SRC" --resume_from "$RESUME_FROM" \
  --portion_start 2.5 --portion_end 92.5 --portion_interval 2.5 \
  --seed 42 --device "$DEVICE" --exp_path "$EXP" \
  --lr_schedule sweep --parallel_lr \
  --coldstart_lr_path ./classification/exp_results
echo "DONE typiclust. leftover-10% = 全 train 池 \ cumulative@90%"
