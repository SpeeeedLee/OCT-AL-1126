#!/usr/bin/env bash
# Cold-start baseline (passive, random subset) for Ch5 nuclei segmentation.
# Fixed lr; one JSON per seed; runs the 5 thesis seeds.
#
#   DEVICE=cuda:0 ./thesis/chapter_5/segmentation/scripts/run_coldstart.sh
#
# Env overrides: SEEDS, PORTIONS, LR, EPOCH, AUG, BS, INIT, DEVICE
set -e
cd "$(git rev-parse --show-toplevel)"

DEVICE=${DEVICE:-cuda:0}
SEEDS=${SEEDS:-"10 24 38 42 57"}
PORTIONS=${PORTIONS:-"5 10 20 30 40 50 60 100"}
LR=${LR:-0.001}
EPOCH=${EPOCH:-25}
AUG=${AUG:-4}
BS=${BS:-8}
INIT=${INIT:-random}

for seed in $SEEDS; do
  for p in $PORTIONS; do
    echo "=== cold-start portion=$p seed=$seed ==="
    python3 thesis/chapter_5/segmentation/run_first_iter.py \
        --dataroot ./ds/segmentation --portion "$p" --seed "$seed" \
        --aug_factor "$AUG" --lr "$LR" --epoch "$EPOCH" --batch_size "$BS" \
        --init "$INIT" --device "$DEVICE" || true
  done
done
