#!/bin/bash
# Memory-fit check of every wave-1 config on this node's 4 GPUs.
#   bash slurm/fitcheck_all.sh <gpu_tag> "<micro batches>"     e.g. 2080ti "1 2"   or   3090 "2 4"
set -uo pipefail
TAG=$1; MBS=$2
cd ~/morbius
PY=$(ls -d /ssd_scratch/$USER/morbius/python/cpython-3.12*/bin/python3.12 | head -1)
OUT=/ssd_scratch/$USER/morbius/fitcheck
mkdir -p "$OUT"
for c in w1_mem_full w1_mem_cut_near w1_dense_full w1_dense_cut_near w1_mem_cut_far; do
  "$PY" -m ffnmem.fitcheck --config "configs/$c.yaml" --nproc 4 --micro_batch $MBS --seq_len 2048 \
    --zero1 off on --out "$OUT/$TAG.jsonl"
done
echo FITCHECK_DONE
