#!/bin/bash
# Same as queue_worker.sh, but waits for new lines instead of exiting when the queue is empty.
#   bash slurm/queue_worker_wait.sh <queue_file> <gpu_index>
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
  if [ -z "$CMD" ]; then
    sleep 60
    continue
  fi
  TAG=$(date +%m%d_%H%M%S)_g$GPU
  echo "[$TAG] START $CMD" | tee -a "$Q.running"
  bash -c "$CMD" > "$LOGDIR/$TAG.log" 2>&1
  RC=$?
  echo "$RC $TAG $CMD" >> "$Q.done"
  echo "[$TAG] END rc=$RC"
done
