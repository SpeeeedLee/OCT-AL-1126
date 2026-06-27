#!/usr/bin/env bash
# Downstream nuclei-seg finetune initialised from an AE checkpoint:
#   ENCODER (Conv1..Conv6) <- AE weights ; DECODER (Up*) random  (load strict=False).
#
# Usage (named flags; --portions / --seeds accept MULTIPLE values, looped over):
#   bash run_ae_finetune.sh --ae-epoch 2000 --portions 10 30 --seeds 42 24 \
#        --aug 4 --device cuda:1 --max-runs 3
#
#   --ae-epoch   AE epoch (1,3,10,30,50,100,200,500,1000,1500,2000) OR a full ckpt path
#   --portions   one or more portions (e.g. 2.5 5 10 100). rho=100 -> single seed 42, 6 runs/lr
#   --seeds      one or more selection seeds (10 24 38 42 57)
#   --aug        1|noaug | 2|hf | 4|4x|4aug | vf | vfhv
#   --device     cuda:N
#   --max-runs   how many trainings to run AT ONCE on this GPU (concurrency; default 2)
#   --warmup     frozen-ENCODER warm-up epochs (decoder-only first N ep, then unfreeze;
#                default 0). >0 -> results go to a separate tree aug<...>_wu<N>.
#   --lrs        OPTIONAL space-separated LRs that OVERRIDE the per-portion default grid
#                (applies to every portion this run). rerun-safe, so use it to top-up extra
#                LRs without touching the default grid, e.g. --lrs 0.0075 0.01
#
# lr is SWEPT over the per-portion grid (best-lr picked downstream). rerun-safe.
# results -> exp_results/ae_init/<eptag>/<augtree>/nuclei/cold_start_ae/random_<seed>_bs8.json
set -u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate oct-env
cd "$(git rev-parse --show-toplevel)" || exit 1

# Ctrl-C / TERM -> kill ALL child trainings, not just the dispatcher.
cleanup() { trap - INT TERM; echo; echo "[run_ae_finetune] interrupted -> killing child runs..."; \
            kill $(jobs -p) 2>/dev/null; pkill -P $$ 2>/dev/null; exit 130; }
trap cleanup INT TERM

AECK=""; PORTIONS=""; SEEDS="10 24 38 42 57"; AUG=""; DEV=""; PAR=2; WARMUP=0; LRS=""   # seeds default = 5 thesis seeds
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ae-epoch) AECK="$2"; shift 2 ;;
    --aug)      AUG="$2";  shift 2 ;;
    --device)   DEV="$2";  shift 2 ;;
    --max-runs) PAR="$2";  shift 2 ;;
    --warmup)   WARMUP="$2"; shift 2 ;;
    --portions) PORTIONS=""; shift; while [[ $# -gt 0 && "$1" != --* ]]; do PORTIONS="$PORTIONS $1"; shift; done ;;
    --seeds)    SEEDS="";    shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS="$SEEDS $1"; shift; done ;;
    --lrs)      LRS="";      shift; while [[ $# -gt 0 && "$1" != --* ]]; do LRS="$LRS $1"; shift; done ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
: "${AECK:?--ae-epoch required}"; : "${AUG:?--aug required}"; : "${DEV:?--device required}"
[[ -n "${PORTIONS// }" ]] || { echo "--portions required"; exit 1; }
[[ -n "${SEEDS// }"    ]] || { echo "--seeds required"; exit 1; }

DR=./ds/segmentation_correct
CKDIR=thesis/chapter_5/segmentation/autoencoder/ckpt
if [[ "$AECK" =~ ^[0-9]+$ ]]; then
  CKPT="$CKDIR/unet_ae_lr1e-05_bs8_run2000_ep$(printf '%04d' "$AECK")_full.pkl"; EPTAG="ep$AECK"
else
  CKPT="$AECK"; EPTAG=$(basename "$AECK" .pkl)
fi
[ -f "$CKPT" ] || { echo "AE ckpt not found: $CKPT"; exit 1; }

case "$AUG" in
  1|noaug|none) FLAG="--aug_factor 1" ;             AT=aug1  ;;
  2|hf)         FLAG="--aug_factor 2" ;             AT=aug2  ;;
  4|4x|4aug)    FLAG="--aug_factor 4" ;             AT=aug4  ;;
  vf)           FLAG="--aug_factor 2 --flip_set vf" ;   AT=vf2   ;;
  vfhv)         FLAG="--aug_factor 3 --flip_set vfhv" ; AT=vfhv3 ;;
  *) echo "aug must be 1|2|4|vf|vfhv (aliases: noaug/hf/4aug)"; exit 1 ;;
esac
WUTAG=""; [ "$WARMUP" -gt 0 ] 2>/dev/null && WUTAG="_wu$WARMUP"   # isolate warmup variants
EXP=./thesis/chapter_5/segmentation/exp_results/ae_init/$EPTAG/${AT}${WUTAG}
LOGD=thesis/chapter_5/segmentation/logs/ae_init; mkdir -p "$LOGD"
# WIDER grid for AE-init: the random decoder wants a higher lr and the old top
# (3e-3) was still climbing at low rho -> widen each band upward.
lrgrid () { case "$1" in
  2.5|5|10) echo "0.0005 0.001 0.003 0.005 0.01" ;;   # <15
  20|30|40) echo "0.0003 0.001 0.003 0.005" ;;        # 15-49
  *)        echo "0.0003 0.0005 0.001 0.003" ;;       # >=50 (incl 100)
esac; }
# --lrs "<space-separated>" overrides the per-portion default grid above (for all portions this run)

# ---- build job list: "REPSGUARD PORTION SEED LR" ----
JOBS=()
for P in $PORTIONS; do
  GRID="${LRS:-$(lrgrid "$P")}"          # --lrs overrides the per-portion default grid
  if [[ "$P" == "100" ]]; then
    for LR in $GRID; do for i in 1 2 3 4 5 6; do JOBS+=("6 100 42 $LR"); done; done
  else
    for S in $SEEDS; do for LR in $GRID; do JOBS+=("1 $P $S $LR"); done; done
  fi
done

RI="python3 thesis/chapter_5/segmentation/run_first_iter.py --dataroot $DR --epoch 25 --batch_size 8 --init ae --simclr_path $CKPT $FLAG --warmup $WARMUP"
echo "=== AE-init finetune | $EPTAG | aug=$AUG | warmup=$WARMUP | portions:$PORTIONS | seeds:$SEEDS | ${#JOBS[@]} jobs | $DEV | concurrency=$PAR ==="
echo "    encoder <- $CKPT  (decoder random) | -> ae_init/$EPTAG/${AT}${WUTAG}"
running=0; n=0
for J in "${JOBS[@]}"; do
  read -r MR P S LR <<< "$J"; n=$((n+1))
  lf="$LOGD/${EPTAG}_${AT}_p${P}_s${S}_lr${LR}_$n.log"
  echo "[dispatch $n/${#JOBS[@]}] rho=$P seed=$S lr=$LR"
  $RI --max_runs "$MR" --portion "$P" --seed "$S" --lr "$LR" --exp_path "$EXP" --device "$DEV" > "$lf" 2>&1 &
  running=$((running+1))
  if (( running >= PAR )); then wait -n; running=$((running-1)); fi
done
wait
echo "=== DONE | $EPTAG aug=$AUG portions:$PORTIONS seeds:$SEEDS ==="
