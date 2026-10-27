#!/bin/bash
# ==============================================================================
# Project Morbius | Script to Obtain / Allocate a Plafnet GPU Node on Ada
# Account: sparsh.bhartia | Partition: plafnet2 | Target GPUs: NVIDIA RTX 3090
# ==============================================================================
# Usage:
#   bash check/plafnet_scripts/get_plafnet_node.sh [options]
# Options:
#   --4gpu      Request the dedicated 4x RTX 3090 node (gnode118) [Default]
#   --2gpu      Request a 2x RTX 3090 node (gnode114/115/116, excludes faulty gnode113)
#   --1gpu      Request 1x RTX 3090 for fast debugging / testing
#   --holder    Submit a long-running background holder job (sleep) for pam_slurm_adopt
#   --status    Check status of all jobs on Ada for sparsh.bhartia
#   --help      Show this help message
# ==============================================================================

set -euo pipefail

USER_NAME="${USER:-sparsh.bhartia}"
ACCOUNT="plafnet2"
PARTITION="plafnet2"
EXCLUDE_NODES="gnode113,gnode066"  # gnode113 has broken GPU ordinal; gnode066 GPU0 faulty

MODE="interactive"
GPU_COUNT=4
TARGET_NODE="gnode118"
CPUS=16
MEM="80G"
TIME="08:00:00"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --4gpu)
      GPU_COUNT=4
      TARGET_NODE="gnode118"
      CPUS=16
      MEM="80G"
      shift
      ;;
    --2gpu)
      GPU_COUNT=2
      TARGET_NODE=""
      CPUS=8
      MEM="40G"
      shift
      ;;
    --1gpu)
      GPU_COUNT=1
      TARGET_NODE=""
      CPUS=4
      MEM="20G"
      shift
      ;;
    --holder)
      MODE="holder"
      shift
      ;;
    --status)
      echo "=== Active Jobs for $USER_NAME on Ada ==="
      squeue -u "$USER_NAME" -o "%.10i %.12P %.15j %.8u %.2t %.10M %.6D %R"
      exit 0
      ;;
    --help|-h)
      sed -ne 's/^# //p' "$0" | head -15
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Run with --help for usage."
      exit 1
      ;;
  esac
done

echo "=================================================================="
echo " Ada Cluster Plafnet Node Allocator (User: $USER_NAME)"
echo " Partition: $PARTITION | Account: $ACCOUNT"
echo " Target GPUs: $GPU_COUNT x RTX 3090 | CPUs: $CPUS | Memory: $MEM"
[ -n "$TARGET_NODE" ] && echo " Preferred Node: $TARGET_NODE"
echo " Excluded Nodes: $EXCLUDE_NODES"
echo "=================================================================="

if [ "$MODE" = "holder" ]; then
  JOB_NAME="morbius_plafnet_hold"
  echo "Submitting 1-day background holder job on plafnet2..."
  
  NODE_ARG=""
  if [ -n "$TARGET_NODE" ]; then
    NODE_ARG="-w $TARGET_NODE"
  fi

  CMD="sbatch -A $ACCOUNT -p $PARTITION $NODE_ARG -x $EXCLUDE_NODES --gres=gpu:$GPU_COUNT -c $CPUS --mem=$MEM -t 1-00:00:00 -J $JOB_NAME -o ~/morbius/joblogs/%x-%j.out --wrap 'echo Holder job started on \$(hostname) at \$(date); sleep 86400'"
  echo "Executing: $CMD"
  eval "$CMD"
  echo ""
  echo "Holder job submitted successfully! Run with --status to monitor."
  exit 0
fi

# Interactive mode
echo "Launching interactive session via srun..."
echo "Tip: Once in the shell, run 'nvidia-smi' to verify your RTX 3090 GPUs."
echo ""

NODE_ARG=""
if [ -n "$TARGET_NODE" ]; then
  NODE_ARG="-w $TARGET_NODE"
fi

SRUN_CMD="srun --pty -A $ACCOUNT -p $PARTITION $NODE_ARG -x $EXCLUDE_NODES --gres=gpu:$GPU_COUNT -c $CPUS --mem=$MEM -t $TIME /bin/bash"

echo "Executing:"
echo "  $SRUN_CMD"
echo "=================================================================="

exec $SRUN_CMD
