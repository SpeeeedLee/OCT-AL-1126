#!/usr/bin/env bash
# 5.3 類別重分布消融 — 額外 portion 集合 {40, 50, 60}。
# 薄包裝：固定 PORTIONS 後轉呼叫 run_5_3_redistribution.sh。
# SEEDS / DEVICE / STRATEGIES 一樣可指定，例：
#   SEEDS="42" DEVICE=cuda:7 ./thesis/chapter_5/run_5_3_redistribution_high.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTIONS="40 50 60" exec "$HERE/run_5_3_redistribution.sh"
