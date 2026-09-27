#!/bin/bash
# One training run (any config) on one node (4 GPUs; set NPROC=2 via --export=ALL,NPROC=2 for a 2-GPU node).
#   sbatch -A research -p u22 --constraint=2080ti --gres=gpu:4 -c 16 --mem=48G -t 4-00:00:00 \
#          -J w1_mem_full -o ~/morbius/joblogs/%x-%j.out --signal=B:USR1@600 \
#          ~/morbius/slurm/train_job.sh w1_mem_full [extra train.py args, e.g. --set train.ckpt_every=400]
# Checkpoints live on this node's /ssd_scratch; to resume, re-submit with -w <same node>.
set -uo pipefail
CFG=$1; shift
cd "${MORBIUS_DIR:-$HOME/morbius}"   # code dir (one codebase; variants are configs)
eval "$(bash slurm/stage.sh | grep -E '^(ENV_PY|DATA_DIR)=')"
[ -n "${ENV_PY:-}" ] && [ -n "${DATA_DIR:-}" ] || { echo "staging failed: is a holder job of $USER running on gnode066?"; exit 1; }
OUT=/ssd_scratch/$USER/morbius/runs/$CFG
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_DIR="$OUT/logs" WANDB_CACHE_DIR=/ssd_scratch/$USER/morbius/wandb_cache WANDB_CONFIG_DIR=/ssd_scratch/$USER/morbius/wandb_config
echo "job ${SLURM_JOB_ID} $CFG on $(hostname) at $(date); out=$OUT data=$DATA_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Slurm sends USR1 600 s before the time limit: ranks checkpoint at the next step and exit.
trap 'echo "USR1 at $(date)"; pkill -USR1 -P "$PID" || true; wait "$PID"; exit 0' USR1
"$ENV_PY" -m torch.distributed.run --standalone --nproc_per_node="${NPROC:-4}" -m ffnmem.train \
  --config "configs/$CFG.yaml" --data_dir "$DATA_DIR" --out_dir "$OUT" --resume auto --wandb "$@" &
PID=$!
wait "$PID"
rc=$?
echo "exit $rc at $(date)"
exit $rc
