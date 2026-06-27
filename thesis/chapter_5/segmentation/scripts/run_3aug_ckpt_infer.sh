#!/usr/bin/env bash
# Train the 3 cold-start (random-init) nuclei U-Nets used for the qualitative
# 4-panel figure: seed 42, portion 10%, three aug settings, each at ITS OWN best lr
# (seed42/10% best-lr = 0.001 for all three, read from the aug_curve sweep).
# Saves a named checkpoint for each, then runs inference over the whole test split.
#
# Usage:  bash run_3aug_ckpt_infer.sh [cuda:N] [EPOCH]
#   e.g.  bash run_3aug_ckpt_infer.sh cuda:0          # default 25 epochs
#         bash run_3aug_ckpt_infer.sh cuda:3 25
# Runs the 3 trainings SEQUENTIALLY on one GPU (each ~10-15 min at 10%).
set -u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate oct-env
cd "$(git rev-parse --show-toplevel)" || exit 1

cleanup() { trap - INT TERM; echo; echo "[run_3aug] interrupted -> killing child..."; \
            kill $(jobs -p) 2>/dev/null; pkill -P $$ 2>/dev/null; exit 130; }
trap cleanup INT TERM

DEV="${1:-cuda:0}"
EPOCH="${2:-25}"
SEED=42
PORTION=10
RI="python3 thesis/chapter_5/segmentation/train_save_infer.py \
    --dataroot ./ds/segmentation_correct --seed $SEED --portion $PORTION \
    --epoch $EPOCH --device $DEV"
LOGD=thesis/chapter_5/segmentation/logs/ckpt_infer; mkdir -p "$LOGD"

# "aug  tag   lr"  (best lr per setting at seed42/10%)
SETTINGS=(
  "1  noaug  0.001"
  "2  hf     0.001"
  "4  4x     0.001"
)

echo "=== train + save ckpt + infer test | seed=$SEED portion=$PORTION% | $DEV | epoch=$EPOCH ==="
for S in "${SETTINGS[@]}"; do
  read -r AUG TAG LR <<< "$S"
  lf="$LOGD/${TAG}.log"
  echo ">>> [$TAG] aug=$AUG lr=$LR  (log: $lf)"
  $RI --aug "$AUG" --tag "$TAG" --lr "$LR" 2>&1 | tee "$lf"
  echo ">>> [$TAG] done."
done
echo "=== ALL DONE -> ckpts in thesis/chapter_5/segmentation/ckpts/ , preds in .../preds/ ==="
ls -1 thesis/chapter_5/segmentation/ckpts/unet_nuclei_seed${SEED}_p${PORTION}_*.pkl 2>/dev/null