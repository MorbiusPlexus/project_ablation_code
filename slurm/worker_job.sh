#!/bin/bash
# One-GPU queue worker. Submit as many as the GPU quota allows; they share the queue.
#   sbatch slurm/worker_job.sh ~/morbius_eval_runs/queue.txt
#SBATCH -J morbius_worker
#SBATCH -A research
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH -x gnode066
#SBATCH --gres=gpu:1
#SBATCH -c 9
#SBATCH --mem=26G
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/%u/morbius_eval_runs/worker_%j.out
set -u
Q=${1:-$HOME/morbius_eval_runs/queue.txt}
nvidia-smi -L
export LOGDIR=$HOME/morbius_eval_runs/logs
# Slurm already restricts the job to its GPU; index 0 inside the allocation.
bash $HOME/morbius_eval/slurm/queue_worker.sh "$Q" "${CUDA_VISIBLE_DEVICES:-0}"
