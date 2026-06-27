#!/usr/bin/env bash
# Cold-start data-augmentation sweep on the CORRECT (leakage-free) dataset
# (ds/segmentation_correct).  w/o-Aug = 1, HF = 2, 4x = 4.
# Runs a concurrency-limited POOL on ONE gpu: up to PAR trainings at once.
#
# Usage:  bash run_aug_sweep.sh <AUG:1|2|4> cuda:N [PAR=2] ["PORTIONS"]
#   - 4th arg = which portions to run (space-separated, QUOTE it). Omit = run all.
#       e.g.  bash run_aug_sweep.sh 1 cuda:5 2 "2.5 5 10"      # just these 3
#             bash run_aug_sweep.sh 4 cuda:7 2 "100"           # just full-data
#             bash run_aug_sweep.sh 2 cuda:6 2                  # all portions
#   - per-portion lr grid (sweep), pick best-lr per seed downstream.
#   - rho<100 : 5 seeds (10 24 38 42 57), 1 run/(seed,lr)  (--max_runs 1 -> rerun-safe)
#   - rho=100 : single seed 42, 6 runs/lr (seed-independent -> std from reps)
#   - results -> exp_results/aug_curve/aug<AUG>/ (the aug-curve tree; plots read it)
#   - VRAM: each U-Net job ~7GB -> PAR=2 ~14GB, PAR=3 ~21GB. Pick PAR to fit your GPU.
set -u
# always run inside oct-env from repo root (so `import cv2` etc. work regardless of
# the calling shell, and relative paths resolve).
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate oct-env
cd "$(git rev-parse --show-toplevel)" || exit 1

# Ctrl-C / TERM -> kill ALL child trainings, not just the dispatcher.
cleanup() { trap - INT TERM; echo; echo "[run_aug_sweep] interrupted -> killing child runs..."; \
            kill $(jobs -p) 2>/dev/null; pkill -P $$ 2>/dev/null; exit 130; }
trap cleanup INT TERM
AUG="${1:?usage: bash run_aug_sweep.sh <1|2|4|vf|vfhv> cuda:N [PAR=2] [\"PORTIONS\"]}"
DEV="${2:?need a device, e.g. cuda:0}"
PAR="${3:-2}"
DR=./ds/segmentation_correct
# AUG selects the flip composition + which aug-curve tree to write:
#   1 = w/o (aug1) | 2 = HF (aug2) | 4 = 4x (aug4) | vf = VF-only (vf2) | vfhv = VF+HV (vfhv3)
case "$AUG" in
  1)    FLAG="--aug_factor 1" ;             TREE="aug1"  ;;
  2)    FLAG="--aug_factor 2" ;             TREE="aug2"  ;;
  4)    FLAG="--aug_factor 4" ;             TREE="aug4"  ;;
  vf)   FLAG="--aug_factor 2 --flip_set vf" ;   TREE="vf2"   ;;
  vfhv) FLAG="--aug_factor 3 --flip_set vfhv" ; TREE="vfhv3" ;;
  *)    echo "AUG must be one of: 1 (w/o) | 2 (HF) | 4 (4x) | vf (VF only) | vfhv (VF+HV)"; exit 1 ;;
esac
RI="python3 thesis/chapter_5/segmentation/run_first_iter.py --dataroot $DR --epoch 25 --batch_size 8 $FLAG"
EXP=./thesis/chapter_5/segmentation/exp_results/aug_curve/${TREE}
LOGD=thesis/chapter_5/segmentation/logs/aug_sweep; mkdir -p "$LOGD"

SEEDS="10 24 38 42 57"
PORTIONS="${4:-2.5 5 10 20 30 40 50 60 70 80 90 100}"   # 4th arg overrides (quote it)
lrgrid () {                       # per-portion lr grid (mirrors lr_grid_for)
  case "$1" in
    2.5|5|10)  echo "0.0005 0.001 0.003" ;;   # <15
    20|30|40)  echo "0.0003 0.001 0.003" ;;   # 15-49
    *)         echo "0.0003 0.0005 0.001" ;;  # >=50 (incl 100)
  esac
}

# ---- build job list: "MAXRUNS PORTION SEED LR" ----
#   rho=100 is seed-independent -> single seed 42, 6 runs/lr; others -> 5 seeds, 1 run.
JOBS=()
for P in $PORTIONS; do
  if [ "$P" = "100" ]; then
    for LR in $(lrgrid 100); do for i in 1 2 3 4 5 6; do JOBS+=("6 100 42 $LR"); done; done
  else
    for S in $SEEDS; do for LR in $(lrgrid "$P"); do JOBS+=("1 $P $S $LR"); done; done
  fi
done

echo "=== aug=$AUG sweep on $DEV | portions: $PORTIONS | ${#JOBS[@]} jobs | PAR=$PAR | -> aug_curve/$TREE ==="
running=0; n=0
for J in "${JOBS[@]}"; do
  read -r MR P S LR <<< "$J"; n=$((n+1))
  lf="$LOGD/aug${AUG}_p${P}_s${S}_lr${LR}_$n.log"
  echo "[dispatch $n/${#JOBS[@]}] aug$AUG rho=$P seed=$S lr=$LR (max_runs=$MR) -> $lf"
  $RI --max_runs "$MR" --device "$DEV" --portion "$P" --seed "$S" --lr "$LR" --exp_path "$EXP" > "$lf" 2>&1 &
  running=$((running+1))
  if (( running >= PAR )); then wait -n; running=$((running-1)); fi
done
wait
echo "=== DONE aug$AUG on $DEV ==="
