#!/usr/bin/env bash
# VFHV (3x) cold-start sweep, rho=30..100.
# Runs a concurrency-limited POOL on ONE gpu: up to PAR trainings at once.
# Usage:  bash <this>.sh cuda:N [PAR=2]      (PAR = how many runs at once on this GPU)
#   rho<100 : 1 run/(seed,lr), --max_runs 1 (rerun-safe). seeds/lrs match w/o/HF/4x.
#   rho=100 : single seed 42, 6 runs/lr (matches aug1/2/4).
#   VRAM: each run ~7GB -> PAR=2 ~14GB, PAR=3 ~21GB. Pick PAR to fit your GPU.
set -u
DEV="${1:?usage: bash <script> cuda:N [PAR=2]}"
PAR="${2:-2}"
RI="python3 thesis/chapter_5/segmentation/run_first_iter.py --dataroot ./ds/segmentation --epoch 25 --batch_size 8"
FLIP=vfhv
EXP=./thesis/chapter_5/segmentation/exp_results/aug_curve/vfhv3
LOGD=thesis/chapter_5/segmentation/logs/aug_sweep; mkdir -p "$LOGD"

# ---- build job list: each entry = "MAXRUNS PORTION SEED LR" ----
JOBS=()
for S in 10 24 38 42 57; do for LR in 0.0003 0.001 0.003; do JOBS+=("1 30 $S $LR"); done; done
for S in 10 24 38 42 57; do for LR in 0.0003 0.001 0.003; do JOBS+=("1 40 $S $LR"); done; done
for S in 10 24 38 42 57; do for LR in 0.0003 0.0005 0.001; do JOBS+=("1 50 $S $LR"); done; done
for S in 10 24 38 42 57; do for LR in 0.0003 0.0005 0.001; do JOBS+=("1 60 $S $LR"); done; done
for S in 10 24 42; do for LR in 0.0003 0.001; do JOBS+=("1 70 $S $LR"); done; done
for S in 10 24 42; do for LR in 0.0003 0.001; do JOBS+=("1 80 $S $LR"); done; done
for S in 10 24 42; do for LR in 0.0003 0.001; do JOBS+=("1 90 $S $LR"); done; done
for LR in 0.0003 0.0005 0.001; do for i in $(seq 1 6); do JOBS+=("6 100 42 $LR"); done; done

echo "=== $FLIP rho=30..100 on $DEV | ${#JOBS[@]} jobs | PAR=$PAR concurrent ==="
running=0; n=0
for J in "${JOBS[@]}"; do
  read -r MR P S LR <<< "$J"
  n=$((n+1))
  lf="$LOGD/${FLIP}_p${P}_s${S}_lr${LR}_$n.log"
  echo "[dispatch $n/${#JOBS[@]}] rho=$P seed=$S lr=$LR (max_runs=$MR) -> $lf"
  $RI --max_runs $MR --device $DEV --flip_set $FLIP --portion $P --seed $S --lr $LR --exp_path $EXP > "$lf" 2>&1 &
  running=$((running+1))
  if (( running >= PAR )); then wait -n; running=$((running-1)); fi
done
wait
echo "=== DONE $FLIP on $DEV ==="
