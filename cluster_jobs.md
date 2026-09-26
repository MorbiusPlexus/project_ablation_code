# Cluster job log (ada.iiit.ac.in)

Every job launched for this project, which account, partition and node, and why.

## Layout
- **Code:** `~/morbius` on both accounts. Small; only code lives on home NFS.
- **Node-local data/env/runs:** `/ssd_scratch/morbius/` on each compute node.
  - `climbmix/climbmix/` holds the tokenized dataset. The master copy is on gnode066.
  - `env/` holds the Python env. Built on gnode066 and copied to other nodes at the same path.
  - `runs/<config>/` holds checkpoints and logs for a run.
- **Access:** other nodes are reachable only while the same user has a job there (pam_slurm_adopt). Copies from gnode066 need a holder job on gnode066 for that user.

## Accounts
| user | slurm account | partition | GPUs | limit |
|---|---|---|---|---|
| aditya.shankar | research (qos medium) | u22 | 4× 2080 Ti | 4 GPU / 40 CPU / 125 GB, 4 days |
| sparsh.bhartia | research (qos medium) | u22 | 4× 2080 Ti | same |
| sparsh.bhartia | plafnet2 | plafnet2 | 4× 3090 (gnode118 is the only 4-GPU node) | 4 days |

## Run placement plan
- Runs 1 and 2 are the key pair, so they run on the same GPU type (2080 Ti).
- Runs 4 and 5 go on the 3090.

| run | config | account | GPU | status |
|---|---|---|---|---|
| 1 | w1_mem_full | aditya / u22 gnode078 | 2080 Ti | running (2717512) |
| 2 | w1_mem_cut_near | sparsh / u22 gnode082 | 2080 Ti | running (2717520) |
| 4 | w1_dense_full | sparsh / plafnet2 gnode118 | 3090 | running (2717513) |
| 3 | w1_mem_cut_far | aditya / u22 gnode078 | 2080 Ti | queued (2717537, after run 1) |
| 5 | w1_dense_cut_near | sparsh / plafnet2 gnode114 (2x3090) | 3090 | running (2717521) |

## Jobs
Times are cluster clock (IST).

| date (IST) | job id | user | partition | node | name | purpose | status |
|---|---|---|---|---|---|---|---|
| 2026-09-26 18:35 | 2717343 | aditya | u22 | gnode066 | morbius_setup | 4-GPU holder for setup | cancelled 18:50 (gnode066 GPU0 faulty) |
| 2026-09-26 18:51 | 2717345 | aditya | u22 | gnode066 | morbius_hold066 | CPU holder (4 CPU, 1 day): access to gnode066 for env build, index, staging | cancelled ~20:50 (staging done) |
| 2026-09-26 18:51 | 2717346 | sparsh | u22 | gnode066 | morbius_hold066 | CPU holder (2 CPU, 1 day): lets sparsh jobs pull env/data from gnode066 | cancelled ~20:50 (staging done) |
| 2026-09-26 19:20 | 2717435 | aditya | u22 | gnode078 | morbius_fit2080 | 4x2080Ti holder: stage env+data, 2080 Ti fitcheck, real-path smoke + resume test | cancelled 20:33 (handed node to 2717512) |
| 2026-09-26 19:20 | 2717436 | sparsh | plafnet2 | gnode118 | morbius_fit3090 | 4x3090 holder: stage env+data, 3090 fitcheck | cancelled 20:33 (handed node to 2717513) |
| 2026-09-26 20:33 | 2717512 | aditya | u22 | gnode078 | w1_mem_full | **run 1**; 4x2080Ti, zero1, micro_batch 2, ckpt every 500 | running |
| 2026-09-26 20:33 | 2717513 | sparsh | plafnet2 | gnode118 | w1_dense_full | **run 4**; 4x3090, micro_batch 4, ckpt every 1000 | running |
| 2026-09-26 20:41 | 2717520 | sparsh | u22 | gnode082 | w1_mem_cut_near | **run 2**; 4x2080Ti, zero1, micro_batch 2, ckpt every 500 | running |
| 2026-09-26 20:41 | 2717521 | sparsh | plafnet2 | gnode114 | w1_dense_cut_near | **run 5**; 2x3090 (NPROC=2), micro_batch 4, ckpt every 500 | running |
| 2026-09-26 20:52 | 2717537 | aditya | u22 | gnode078 (pinned) | w1_mem_cut_far | **run 3**; starts after 2717512 succeeds (afterok); env and data already on gnode078, so no gnode066 holder needed | pending (dependency) |

Not ours:
- 2717256 (aditya, interactive on gnode066) was ended by the user.
- 2717341 (sparsh, `anlp_a2_part3`, u22 gnode042, 1 GPU) ended by 19:20, so sparsh's u22 quota is free.

gnode066 is the data/env source only: GPU0 reports "Unknown Error".

## Where things are
- **Run dirs:** `<node>:/ssd_scratch/<user>/morbius/runs/<config>/` holds checkpoints and `logs/train.jsonl`.
- **Slurm output:** `~/morbius/joblogs/<config>-<jobid>.out`.
- **wandb:** https://wandb.ai/ad1tya2-iiit-hyderabad/morbius-wave1. `w1_mem_full-3a5981e5` there is the 4-step smoke test, not run 1.
- **Resume after a failure:** re-submit the same command with `-w <same node>`; `--resume auto` picks the latest checkpoint.

## Throughput and ETA at launch
| run | tok/s | step | ETA (2B tokens + eval) |
|---|---|---|---|
| 1 | 23.0k | 22.8 s | ~25 h (≈ 27 Sep 21:30) |
| 2 | 21.2k | 24.8 s | ~27 h (≈ 27 Sep 23:40) |
| 4 | 42.0k | 12.5 s | ~14 h (≈ 27 Sep 10:30) |
| 5 | 19.6k | 26.8 s | ~28 h (≈ 28 Sep 00:40) |
| 3 | ~21k expected | | starts when run 1 ends, ~27 h after that |
