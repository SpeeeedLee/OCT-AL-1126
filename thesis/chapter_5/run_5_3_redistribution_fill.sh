#!/usr/bin/env bash
# Fill missing redistribution ablation runs (coreset + cluster_margin).
# Uses 4 GPUs × 2 parallel slots = 8 concurrent jobs.
#
# Usage (from repo root):
#   GPU0=cuda:0 GPU1=cuda:1 GPU2=cuda:2 GPU3=cuda:3 \
#       bash thesis/chapter_5/run_5_3_redistribution_fill.sh
#
# Defaults to cuda:0–3 if not set.
set -eu
cd "$(git rev-parse --show-toplevel)"

GPU0="${GPU0:-cuda:0}"
GPU1="${GPU1:-cuda:1}"
GPU2="${GPU2:-cuda:2}"
GPU3="${GPU3:-cuda:3}"

# 4 GPUs × 2 slots each → round-robin over 8 "slots"
GPUS=("$GPU0" "$GPU0" "$GPU1" "$GPU1" "$GPU2" "$GPU2" "$GPU3" "$GPU3")
MAX_JOBS=8

PIDS=()
JOB_IDX=0

run_one() {
    local strategy="$1" portion="$2" seed="$3"
    local slot=$(( JOB_IDX % MAX_JOBS ))
    local device="${GPUS[$slot]}"
    JOB_IDX=$(( JOB_IDX + 1 ))

    echo "[$(date +%H:%M:%S)] LAUNCH  $strategy  ρ=$portion  seed=$seed  → $device"
    python3 thesis/chapter_5/run_5_3_redistribution.py \
        --strategy "$strategy" --portion "$portion" --seed "$seed" \
        --device "$device" &
    PIDS+=($!)

    # Once we have MAX_JOBS in flight, wait for ONE to finish before launching next
    if (( ${#PIDS[@]} >= MAX_JOBS )); then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")   # pop front
    fi
}

echo "========================================================"
echo " Redistribution fill — $(date)"
echo " GPUs: $GPU0 $GPU1 $GPU2 $GPU3  (2 slots each)"
echo "========================================================"

# ── coreset (17 missing combos) ──────────────────────────────
run_one coreset 25 10
run_one coreset 35 10
run_one coreset 45 10
run_one coreset 55 10

run_one coreset 15 24
run_one coreset 25 24
run_one coreset 35 24
run_one coreset 45 24
run_one coreset 55 24

run_one coreset 10 42
run_one coreset 15 42
run_one coreset 20 42
run_one coreset 25 42
run_one coreset 30 42
run_one coreset 40 42
run_one coreset 50 42
run_one coreset 60 42

# ── cluster_margin (24 missing combos) ──────────────────────
run_one cluster_margin 15 10
run_one cluster_margin 25 10
run_one cluster_margin 35 10
run_one cluster_margin 45 10
run_one cluster_margin 55 10

run_one cluster_margin 15 24
run_one cluster_margin 25 24
run_one cluster_margin 35 24
run_one cluster_margin 45 24
run_one cluster_margin 55 24

run_one cluster_margin 10 42
run_one cluster_margin 20 42
run_one cluster_margin 30 42
run_one cluster_margin 40 42
run_one cluster_margin 50 42

run_one cluster_margin 10 57
run_one cluster_margin 15 57
run_one cluster_margin 20 57
run_one cluster_margin 25 57
run_one cluster_margin 30 57
run_one cluster_margin 35 57
run_one cluster_margin 40 57
run_one cluster_margin 45 57
run_one cluster_margin 50 57

# Wait for all remaining jobs
echo "Waiting for last batch..."
for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "========================================================"
echo " ALL DONE — $(date)"
echo " Re-run plot: python3 thesis/chapter_5/plot_5_3_redistribution.py"
echo "========================================================"
