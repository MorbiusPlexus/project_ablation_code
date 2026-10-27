#!/bin/bash
# ==============================================================================
# Project Morbius | Idea 2: Delayed Memory Injection (Delta=4) on Plafnet (3090)
# Submits w2_mem_delayed4 to Ada plafnet2 partition.
# ==============================================================================
# Usage:
#   # 4-GPU run on gnode118 (Default):
#   bash check/plafnet_scripts/run_train_delayed4_plafnet.sh
#
#   # 2-GPU run (e.g. gnode114 / gnode115):
#   bash check/plafnet_scripts/run_train_delayed4_plafnet.sh --2gpu
#
#   # Fast sanity / smoke-test run:
#   bash check/plafnet_scripts/run_train_delayed4_plafnet.sh --set train.tokens_total=10000000 --set train.ckpt_every=100
# ==============================================================================

set -euo pipefail

CFG="w2_mem_delayed4"
NODE="gnode118"
GPUS=4
CPUS=16
MEM="80G"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --2gpu)
      NODE=""
      GPUS=2
      CPUS=8
      MEM="40G"
      shift
      ;;
    --1gpu)
      NODE="gnode118"
      GPUS=1
      CPUS=4
      MEM="20G"
      shift
      ;;
    --node)
      NODE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

NODE_ARG=""
if [ -n "$NODE" ]; then
  NODE_ARG="-w $NODE"
fi

EXPORT_ARG="ALL"
if [ "$GPUS" -eq 2 ]; then
  EXPORT_ARG="ALL,NPROC=2"
elif [ "$GPUS" -eq 1 ]; then
  EXPORT_ARG="ALL,NPROC=1"
fi

mkdir -p "$HOME/morbius/joblogs"

echo "=================================================================="
echo " Submitting IDEA 2 (w2_mem_delayed4) on Ada Plafnet"
echo " Partition: plafnet2 | Account: plafnet2 | GPUs: $GPUS x RTX 3090"
[ -n "$NODE" ] && echo " Target Node: $NODE"
echo " Export: $EXPORT_ARG"
[ ${#EXTRA_ARGS[@]} -gt 0 ] && echo " Extra Args: ${EXTRA_ARGS[*]}"
echo "=================================================================="

CMD="sbatch -A plafnet2 -p plafnet2 $NODE_ARG -x gnode113,gnode066 \
  --gres=gpu:$GPUS -c $CPUS --mem=$MEM -t 4-00:00:00 \
  -J $CFG -o $HOME/morbius/joblogs/%x-%j.out --signal=B:USR1@600 \
  --export=$EXPORT_ARG \
  $HOME/morbius/slurm/train_job.sh $CFG --set train.micro_batch=4 ${EXTRA_ARGS[*]:-}"

echo "Executing:"
echo "  $CMD"
echo "------------------------------------------------------------------"
eval "$CMD"
echo "Job submitted! Monitor with: squeue -u ${USER:-sparsh.bhartia}"
