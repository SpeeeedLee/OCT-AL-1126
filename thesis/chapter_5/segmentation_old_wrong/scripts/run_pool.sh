#!/usr/bin/env bash
# GPU-pool dispatcher: run a list of jobs across a fixed set of cuda devices,
# one job per device at a time. Each job line in JOBFILE must contain the
# placeholder __DEV__ (replaced by the assigned cuda:N).
#
#   ./run_pool.sh JOBFILE "cuda:0 cuda:1 cuda:2 ..."
set -u
JOBFILE="$1"; shift
read -r -a DEVICES <<< "$1"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oct-env
cd "$(git rev-parse --show-toplevel)"

mapfile -t JOBS < "$JOBFILE"
N=${#JOBS[@]}
declare -A PID_OF
i=0
echo "[pool] $N jobs over ${#DEVICES[@]} devices: ${DEVICES[*]}"
while (( i < N )); do
  for d in "${DEVICES[@]}"; do
    pid=${PID_OF[$d]:-}
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
      (( i < N )) || break
      cmd="${JOBS[$i]//__DEV__/$d}"
      logf="thesis/chapter_5/segmentation/logs/job_$(printf '%03d' "$i").log"
      echo "[dispatch $(date +%H:%M:%S)] dev=$d job=$i/$N -> $logf"
      bash -c "$cmd" > "$logf" 2>&1 &
      PID_OF[$d]=$!
      ((i++))
    fi
  done
  sleep 5
done
wait
echo "[pool] ALL $N JOBS DONE $(date +%H:%M:%S)"
