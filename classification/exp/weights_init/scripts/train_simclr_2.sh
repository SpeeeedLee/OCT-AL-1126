#!/bin/bash

EPOCHS=(20 200)
BATCHES=(16 32 64 128 256)
DEVICE="cuda:1"  # 預設

# 解析 named argument --DEVICE
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --DEVICE) DEVICE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

for EPOCH in "${EPOCHS[@]}"; do
    for B in "${BATCHES[@]}"; do
        echo "Running: epochs=$EPOCH, batch_size=$B, device=$DEVICE"
        python3 ./SSL/simclr/run.py --epochs $EPOCH -b $B --device $DEVICE
    done
done