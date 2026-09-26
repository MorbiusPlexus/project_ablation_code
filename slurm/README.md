# Running on ada (IIIT)

Plain scripts, used by hand. The job log is `cluster_jobs.md` at the repo root.

## Layout
- **Code:** `~/morbius` on both accounts. Home NFS quota is small, so nothing large goes in home.
- **Env:** a uv standalone CPython 3.12 with torch 2.11.0+cu128. Built once on gnode066 at `/ssd_scratch/aditya.shankar/morbius/python`.
- **Data:** tokenized ClimbMix (10B tokens, 174 shards of ~57M, `val.bin` 57.5M) at `gnode066:/ssd_scratch/morbius/climbmix/climbmix`. The fixed-order index `index_seq2048_seed1234.npy` is built there once:
  `python -m ffnmem.data.build_index --data /ssd_scratch/morbius/climbmix/climbmix`
- **Staging:** `slurm/stage.sh` copies env and data to `/ssd_scratch/$USER/morbius/` on the job's node. It needs a holder job of the same user on gnode066, because pam_slurm_adopt only lets you ssh to nodes where you have a job.
- **Runs:** `/ssd_scratch/$USER/morbius/runs/<config>/` holds checkpoints and logs, on node-local ssd. To resume, re-submit pinned with `-w <node>`.
- **End of a run:** copy the final checkpoint back to home by hand.

## Accounts and placement
- `aditya.shankar`: `-A research -p u22` (QOS medium: 4 GPU / 40 CPU / 125 GB, 4 days). Uses 2080 Ti nodes (`--constraint=2080ti`). Avoid gnode066: GPU0 errors.
- `sparsh.bhartia`: `-A research -p u22` (2080 Ti), and `-A plafnet2 -p plafnet2`. gnode118 is the only 4× 3090 node there; gnode113–116 have 2× 3090.

## Submit a run
u22 caps memory at 3000 MB/CPU (plafnet2: 5000), so use `-c 16 --mem=48G` on u22 (and at most `--mem=80G` on plafnet2). Asking for more inflates the CPU count against the 40-CPU QOS limit.
Build the index on gnode066 **before** staging anywhere: `stage.sh` never re-syncs a staged node.

```bash
sbatch -A research -p u22 --constraint=2080ti -x gnode066 --gres=gpu:4 -c 16 --mem=48G -t 4-00:00:00 \
  -J w1_mem_full -o ~/morbius/joblogs/%x-%j.out --signal=B:USR1@600 \
  ~/morbius/slurm/train_job.sh w1_mem_full --set train.ckpt_every=<steps for ~3h>
```

## Memory-fit check (per GPU type, before training)
```bash
$PY -m ffnmem.fitcheck --config configs/w1_mem_full.yaml --nproc 4 --micro_batch 1 2 --seq_len 2048 --zero1 off on \
  --out /ssd_scratch/$USER/morbius/fitcheck/<gpu>.jsonl
```

## Optional flash-attn
Default is SDPA with the memory-efficient kernel pinned: the same kernel on sm75 and sm86, and no build needed. Enable flash-attn only if both hold:
- `tests/test_model.py::test_flash_matches_sdpa_fp16` passes.
- A 50-step fitcheck shows more than 15% end-to-end gain.

Builds:
- **2080 Ti:** the fork https://github.com/KLM-corporation/flash-attention-2080ti (fp16 only).
- **3090:** upstream flash-attn 2.

Drop it if the build costs more than a day.
