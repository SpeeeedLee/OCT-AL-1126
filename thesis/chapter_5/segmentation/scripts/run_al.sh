#!/usr/bin/env bash
# Active-learning sweep for Ch5 nuclei segmentation on the CORRECT (leakage-free)
# data (ds/segmentation_correct). random init (theta_Rand, NO AE/SimCLR), 4x aug,
# per-portion lr SWEEP (option A: each portion sweeps lr_grid_for, lowest-val-loss
# model = selector). One python process per (strategy, seed) = one AL trajectory.
#
# Results -> exp_results/al_sweep/nuclei/AL_random/<strat>_seed<seed>_bs8.json
#   (the *_sweep tree the AL plots read). The Random baseline is NOT run here — it
#   comes from the aug4 cold-start curve the plots already use.
#
# Usage (env-overridable):
#   DEVICE=cuda:0 STRATEGIES="margin coreset cluster_margin" SEEDS="10 24 38 42 57" \
#     bash thesis/chapter_5/segmentation/scripts/run_al.sh
#
#   Split across GPUs by launching several with different STRATEGIES/SEEDS + DEVICE,
#   e.g. one strategy per card (each runs its 5 seeds sequentially):
#     DEVICE=cuda:1 STRATEGIES=margin         bash .../run_al.sh
#     DEVICE=cuda:2 STRATEGIES=coreset        bash .../run_al.sh
#     DEVICE=cuda:3 STRATEGIES=cluster_margin bash .../run_al.sh
#
# Env: STRATEGIES SEEDS START END INTERVAL EPOCH AUG BS DEVICE EXP
set -u
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate oct-env
cd "$(git rev-parse --show-toplevel)" || exit 1
cleanup(){ trap - INT TERM; echo; echo "[run_al] interrupted -> killing children..."; \
           kill $(jobs -p) 2>/dev/null; pkill -P $$ 2>/dev/null; exit 130; }
trap cleanup INT TERM

DEVICE=${DEVICE:-cuda:0}
STRATEGIES=${STRATEGIES:-"margin coreset cluster_margin"}   # random comes from cold-start curve
SEEDS=${SEEDS:-"10 24 38 42 57"}
START=${START:-2.5}; END=${END:-60}; INTERVAL=${INTERVAL:-2.5}
EPOCH=${EPOCH:-25}; AUG=${AUG:-4}; BS=${BS:-8}
EXP=${EXP:-./thesis/chapter_5/segmentation/exp_results/al_sweep}
LOGD=thesis/chapter_5/segmentation/logs/al_sweep; mkdir -p "$LOGD"

echo "=== AL sweep | data=segmentation_correct | aug=${AUG}x | init=random | lr=sweep"
echo "    strategies: $STRATEGIES | seeds: $SEEDS | rho ${START}->${END} step ${INTERVAL} | $DEVICE"
for strat in $STRATEGIES; do
  for seed in $SEEDS; do
    lf="$LOGD/${strat}_seed${seed}.log"
    echo ">>> AL strategy=$strat seed=$seed  (log: $lf)"
    python3 thesis/chapter_5/segmentation/run_AL.py \
        --AL_strategy "$strat" --dataroot ./ds/segmentation_correct \
        --portion_start "$START" --portion_end "$END" --portion_interval "$INTERVAL" \
        --seed "$seed" --aug_factor "$AUG" --lr_schedule sweep \
        --epoch "$EPOCH" --batch_size "$BS" --init random \
        --exp_path "$EXP" --device "$DEVICE" 2>&1 | tee "$lf" || true
  done
done
echo "=== DONE -> $EXP/nuclei/AL_random/ ==="