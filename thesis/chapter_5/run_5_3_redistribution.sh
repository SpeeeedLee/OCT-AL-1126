#!/usr/bin/env bash
# 5.3 類別重分布消融批次驅動。從 repo root 執行。
#
#   DEVICE=cuda:6 ./thesis/chapter_5/run_5_3_redistribution.sh
#
# 可覆寫 STRATEGIES / PORTIONS / SEEDS / DEVICE：
#   STRATEGIES="margin" PORTIONS="10 20 30" SEEDS="10 24 38 42 57" DEVICE=cuda:7 \
#       ./thesis/chapter_5/run_5_3_redistribution.sh
#
# 單卡循序跑（一個 seed 一個 portion 一次 = lr-grid × 3 runs）。要分卡：開多個 shell，
# 各自設不同 DEVICE 與 SEEDS 子集即可（結果各寫各的 {strategy}_seed{seed}.json，安全）。
# 重跑安全：每個 (strategy,seed,portion,lr) 已滿 3 runs 會自動 skip。
set -u

STRATEGIES="${STRATEGIES:-margin coreset cluster_margin}"
PORTIONS="${PORTIONS:-10 20 30}"
SEEDS="${SEEDS:-10 24 38 42 57}"
DEVICE="${DEVICE:-cuda:6}"

cd "$(git rev-parse --show-toplevel)" || exit 1
echo "STRATEGIES=[$STRATEGIES]  PORTIONS=[$PORTIONS]  SEEDS=[$SEEDS]  DEVICE=$DEVICE"

trap 'echo "Interrupted"; exit 130' INT

for strat in $STRATEGIES; do
  for p in $PORTIONS; do
    for s in $SEEDS; do
      echo ""
      echo "########## $strat  ρ=${p}%  seed=$s  ($DEVICE) ##########"
      python3 thesis/chapter_5/run_5_3_redistribution.py \
        --strategy "$strat" --portion "$p" --seed "$s" --device "$DEVICE" || true
    done
  done
done
echo "ALL DONE."
