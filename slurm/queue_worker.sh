#!/bin/bash
# Pull commands from a shared queue file and run them on one GPU.
#   bash slurm/queue_worker.sh <queue_file> <gpu_index>
# Each line of the queue is one shell command, run from ~/morbius_eval with
# slurm/env.sh sourced. Claimed lines move to <queue>.done (with exit code) under
# flock, so any number of workers can share the queue and new lines can be
# appended while they run.
set -u
Q=$1
GPU=$2
LOGDIR=${LOGDIR:-$(dirname "$Q")/logs}
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
while true; do
  CMD=$(flock "$Q.lock" bash -c '
    line=$(grep -v "^\s*\(#\|$\)" "'"$Q"'" | head -n1)
    if [ -n "$line" ]; then
      grep -vxF -- "$line" "'"$Q"'" > "'"$Q"'.tmp" || true
      mv "'"$Q"'.tmp" "'"$Q"'"
      echo "$line"
    fi')
  [ -z "$CMD" ] && break
  TAG=$(date +%m%d_%H%M%S)_g$GPU
  echo "[$TAG] START $CMD" | tee -a "$Q.running"
  bash -c "$CMD" > "$LOGDIR/$TAG.log" 2>&1
  RC=$?
  echo "$RC $TAG $CMD" >> "$Q.done"
  echo "[$TAG] END rc=$RC"
done
echo "worker gpu=$GPU: queue empty"
