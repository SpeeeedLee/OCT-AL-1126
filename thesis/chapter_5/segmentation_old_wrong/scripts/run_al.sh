#!/usr/bin/env bash
# Active-learning sweep for Ch5 nuclei segmentation.
# Fixed lr across portions; varies seed over the 5 thesis seeds.
#
#   DEVICE=cuda:0 STRATEGIES="margin coreset cluster_margin random" \
#     ./thesis/chapter_5/segmentation/scripts/run_al.sh
#
# Env overrides: STRATEGIES, SEEDS, START, END, INTERVAL, LR, EPOCH, AUG, BS, INIT, DEVICE
set -e
cd "$(git rev-parse --show-toplevel)"

DEVICE=${DEVICE:-cuda:0}
STRATEGIES=${STRATEGIES:-"random margin coreset cluster_margin"}
SEEDS=${SEEDS:-"10 24 38 42 57"}
START=${START:-5}
END=${END:-60}
INTERVAL=${INTERVAL:-2.5}
LR=${LR:-0.001}
EPOCH=${EPOCH:-25}
AUG=${AUG:-4}
BS=${BS:-8}
INIT=${INIT:-random}

for strat in $STRATEGIES; do
  for seed in $SEEDS; do
    echo "=== AL strategy=$strat seed=$seed ==="
    python3 thesis/chapter_5/segmentation/run_AL.py \
        --AL_strategy "$strat" --dataroot ./ds/segmentation \
        --portion_start "$START" --portion_end "$END" --portion_interval "$INTERVAL" \
        --seed "$seed" --aug_factor "$AUG" --lr "$LR" --epoch "$EPOCH" \
        --batch_size "$BS" --init "$INIT" --device "$DEVICE" || true
  done
done
