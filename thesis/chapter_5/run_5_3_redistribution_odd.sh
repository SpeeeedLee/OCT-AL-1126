#!/usr/bin/env bash
# 5.3 類別重分布消融 — 額外 portion 集合 {15, 25, 35, 45, 55}。
# 薄包裝：固定 PORTIONS 後轉呼叫 run_5_3_redistribution.sh。
# SEEDS / DEVICE / STRATEGIES 一樣可指定，例：
#   SEEDS="10" DEVICE=cuda:0 ./thesis/chapter_5/run_5_3_redistribution_odd.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTIONS="15 25 35 45 55" exec "$HERE/run_5_3_redistribution.sh"
