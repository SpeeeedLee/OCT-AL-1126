#!/usr/bin/env bash
set -euo pipefail

EXTRACTORS=(
  "dinov2_base"
  "dinov2_small"
  "dinov2_large"
  "dinov2_giant"
)

for ext in "${EXTRACTORS[@]}"; do
  for trial in 0 1 2; do
    for run in {1..5}; do
      echo "===== extractor=${ext}, trial_id=${trial}, run=${run} ====="
      python3 ./Classification/run_first_iter.py \
        --task_type 'medium' \
        --pretrained_weights 'simclr' \
        --assign_initial './Classification/label_idx/resnet18_simclr_initial.json' \
        --device 'cuda:2' \
        --batch_size 2 \
        --extractor "${ext}" \
        --trial_id "${trial}" \
        --portion 1.0
    done
  done
done
