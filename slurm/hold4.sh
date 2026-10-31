#!/bin/bash
# Holds one 4x 2080 Ti node for the full time limit. Queue workers run in the background,
# one per GPU; whatever they do, this script keeps the allocation. To add work, append
# lines to queue.txt; if the workers already exited, start one with
#   srun --jobid=<id> --overlap -n1 --gres=gpu:1 bash ~/morbius_eval/slurm/queue_worker.sh <queue> 0
#SBATCH -J morbius_hold4
#SBATCH -A research
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH -x gnode066
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 30
#SBATCH --mem=90G
#SBATCH -t 2-00:00:00
#SBATCH -o /home2/anangappara/morbius_eval_runs/hold4_%j.out
hostname; nvidia-smi -L; date
Q=$HOME/morbius_eval_runs/queue.txt
mkdir -p $HOME/morbius_eval_runs/logs
for g in 0 1 2 3; do
  LOGDIR=$HOME/morbius_eval_runs/logs nohup bash $HOME/morbius_eval/slurm/queue_worker.sh "$Q" $g \
    > $HOME/morbius_eval_runs/worker_g$g.out 2>&1 &
  sleep 20
done
while true; do sleep 3600; done
