#!/bin/bash
# One evaluation command on one GPU. Stages the training env, then the eval venv.
#   sbatch ... ~/morbius_eval/slurm/eval_job.sh python -m ffnmem.run_lm_eval --ckpt ...
set -uo pipefail
cd "${MORBIUS_DIR:-$HOME/morbius_eval}"
# shellcheck disable=SC1091
eval "$(bash slurm/stage.sh | grep -E '^(ENV_PY|DATA_DIR|EVAL_PY)=')"
PY="${EVAL_PY:-$ENV_PY}"
[ -n "${PY:-}" ] || { echo "staging failed"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) py=$PY"
"$PY" "$@"
