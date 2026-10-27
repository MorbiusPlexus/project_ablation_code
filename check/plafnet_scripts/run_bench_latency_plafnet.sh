#!/bin/bash
# ==============================================================================
# Project Morbius | Sub-Task 2A: Host-RAM Latency Harness Execution on Plafnet
# ==============================================================================
# Usage:
#   # 1. When already inside an interactive plafnet node shell:
#   bash check/plafnet_scripts/run_bench_latency_plafnet.sh
#
#   # 2. Or to submit as an autonomous Slurm batch job on plafnet2:
#   bash check/plafnet_scripts/run_bench_latency_plafnet.sh --sbatch
#
#   # Optional arguments:
#   bash check/plafnet_scripts/run_bench_latency_plafnet.sh --tokens 512 --warmup 50
# ==============================================================================

set -euo pipefail

ROOT_DIR="${MORBIUS_DIR:-$HOME/morbius}"
if [ ! -d "$ROOT_DIR" ]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

SBATCH_MODE=0
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch)
      SBATCH_MODE=1
      shift
      ;;
    *)
      BENCH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$SBATCH_MODE" -eq 1 ]; then
  mkdir -p "$HOME/morbius/joblogs"
  echo "Submitting Host-RAM Latency Benchmark to Ada Plafnet partition..."
  sbatch -A plafnet2 -p plafnet2 -x gnode113,gnode066 \
    --gres=gpu:1 -c 8 --mem=40G -t 01:00:00 \
    -J bench_offload -o "$HOME/morbius/joblogs/%x-%j.out" \
    --wrap "bash $ROOT_DIR/check/plafnet_scripts/run_bench_latency_plafnet.sh ${BENCH_ARGS[*]:-}"
  echo "Batch benchmark job submitted! Check $HOME/morbius/joblogs/ for results."
  exit 0
fi

# Direct / interactive execution on node
echo "=================================================================="
echo " Running Host-RAM Latency Harness (Sub-Task 2A)"
echo " Host: $(hostname) | Date: $(date)"
echo "=================================================================="

# Check Python environment
if [ -f "/ssd_scratch/$USER/morbius/python/bin/python3" ]; then
  PY="/ssd_scratch/$USER/morbius/python/bin/python3"
elif [ -n "${ENV_PY:-}" ] && [ -x "$ENV_PY" ]; then
  PY="$ENV_PY"
else
  PY="$(command -v python3 || echo "python3")"
fi

echo "Using Python: $PY"
if command -v nvidia-smi &> /dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi
echo "=================================================================="

cd "$ROOT_DIR"
"$PY" "$ROOT_DIR/check/idea2/bench_offload_latency.py" "${BENCH_ARGS[@]:-}"
echo "=================================================================="
echo "Latency benchmark completed successfully!"
