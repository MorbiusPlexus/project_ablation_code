#!/bin/bash
# Copy the Python env and the tokenized dataset from gnode066 to this node's
# /ssd_scratch/$USER/morbius (skipped when already present, or when on gnode066).
# Reaching gnode066 needs a job of $USER there (pam_slurm_adopt), e.g.
#   sbatch -A research -p u22 -w gnode066 -c 2 --mem=4G -t 1-00:00:00 -J morbius_hold066 --wrap "sleep 86400"
set -euo pipefail

# Source of env + data; override with env vars to stage from another node you have a job on.
SRC_NODE=${SRC_NODE:-gnode066}
SRC_ENV=${SRC_ENV:-/ssd_scratch/aditya.shankar/morbius/python}   # uv standalone CPython 3.12 + torch 2.11 cu128
SRC_DATA=${SRC_DATA:-/ssd_scratch/morbius/climbmix/climbmix}     # tokenized ClimbMix (meta.json, val.bin, shards, index)
B=/ssd_scratch/$USER/morbius
mkdir -p "$B"

if [ "$(hostname -s)" = "$SRC_NODE" ] && [ "$USER" = "aditya.shankar" ]; then
  ENV_DIR=$SRC_ENV; DATA_DIR=$SRC_DATA
else
  ENV_DIR=$B/python; DATA_DIR=$B/climbmix
  if [ ! -f "$ENV_DIR/.staged" ]; then
    echo "[stage] env $SRC_NODE:$SRC_ENV -> $ENV_DIR"
    rsync -a --delete "$SRC_NODE:$SRC_ENV/" "$ENV_DIR/"
    touch "$ENV_DIR/.staged"
  fi
  if [ ! -f "$DATA_DIR/.staged" ]; then
    echo "[stage] data $SRC_NODE:$SRC_DATA -> $DATA_DIR"
    rsync -a --exclude _parquet_cache "$SRC_NODE:$SRC_DATA/" "$DATA_DIR/"
    touch "$DATA_DIR/.staged"
  fi
fi
PY=$(ls -d "$ENV_DIR"/cpython-3.12*/bin/python3.12 | head -1)
echo "ENV_PY=$PY"
echo "DATA_DIR=$DATA_DIR"
