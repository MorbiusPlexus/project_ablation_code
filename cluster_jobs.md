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
| 1 | w1_mem_full | aditya / u22 gnode078 | 2080 Ti | **finished** 27 Sep 22:28 (final val 2.805) |
| 2 | w1_mem_cut_near | sparsh / u22 gnode082 | 2080 Ti | running (2717520) |
| 4 | w1_dense_full | sparsh / plafnet2 gnode118 | 3090 | **finished** 27 Sep 10:29 (final val 2.802) |
| 3 | w1_mem_cut_far | aditya / u22 gnode078 | 2080 Ti | running since 27 Sep 22:28 (2717537) |
| 5 | w1_dense_cut_near | sparsh / plafnet2 gnode114 (2x3090) | 3090 | running (2717521) |
| A | w2_mem_rank1 | sparsh / plafnet2 gnode115 (2x3090) | 3090 | running (2719140) |

## Jobs
Times are cluster clock (IST).

| date (IST) | job id | user | partition | node | name | purpose | status |
|---|---|---|---|---|---|---|---|
| 2026-09-26 18:35 | 2717343 | aditya | u22 | gnode066 | morbius_setup | 4-GPU holder for setup | cancelled 18:50 (gnode066 GPU0 faulty) |
| 2026-09-26 18:51 | 2717345 | aditya | u22 | gnode066 | morbius_hold066 | CPU holder (4 CPU, 1 day): access to gnode066 for env build, index, staging | cancelled ~20:50 (staging done) |
| 2026-09-26 18:51 | 2717346 | sparsh | u22 | gnode066 | morbius_hold066 | CPU holder (2 CPU, 1 day): lets sparsh jobs pull env/data from gnode066 | cancelled ~20:50 (staging done) |
| 2026-09-26 19:20 | 2717435 | aditya | u22 | gnode078 | morbius_fit2080 | 4x2080Ti holder: stage env+data, 2080 Ti fitcheck, real-path smoke + resume test | cancelled 20:33 (handed node to 2717512) |
| 2026-09-26 19:20 | 2717436 | sparsh | plafnet2 | gnode118 | morbius_fit3090 | 4x3090 holder: stage env+data, 3090 fitcheck | cancelled 20:33 (handed node to 2717513) |
| 2026-09-26 20:33 | 2717512 | aditya | u22 | gnode078 | w1_mem_full | **run 1**; 4x2080Ti, zero1, micro_batch 2, ckpt every 500 | completed 27 Sep 22:28 |
| 2026-09-26 20:33 | 2717513 | sparsh | plafnet2 | gnode118 | w1_dense_full | **run 4**; 4x3090, micro_batch 4, ckpt every 1000 | completed 27 Sep 10:29 |
| 2026-09-26 20:41 | 2717520 | sparsh | u22 | gnode082 | w1_mem_cut_near | **run 2**; 4x2080Ti, zero1, micro_batch 2, ckpt every 500 | running |
| 2026-09-26 20:41 | 2717521 | sparsh | plafnet2 | gnode114 | w1_dense_cut_near | **run 5**; 2x3090 (NPROC=2), micro_batch 4, ckpt every 500 | running |
| 2026-09-26 20:52 | 2717537 | aditya | u22 | gnode078 (pinned) | w1_mem_cut_far | **run 3**; starts after 2717512 succeeds (afterok); env and data already on gnode078, so no gnode066 holder needed | started 27 Sep 22:28 (running) |
| 2026-09-27 14:27 | 2718727 | sparsh | plafnet2 | gnode118 | morbius_probe | 4x3090: memory zero-out probe + per-token dumps on run 1/2 step-2000 checkpoints (run 1 file relayed via laptop) | done, cancelled ~15:15 |
| 2026-09-27 19:35 | 2719088 | sparsh | plafnet2 | gnode113 | w2_mem_rank1 | idea A (rank-1 add-on) | **failed**: gnode113 exposes only 1 of 2 allocated GPUs (`invalid device ordinal`); cancelled, gnode113 excluded |
| 2026-09-27 19:44 | 2719140 | sparsh | plafnet2 | gnode115 | w2_mem_rank1 | **wave 2 idea A**: 2x3090 (NPROC=2), micro_batch 4, ckpt 500; staged from gnode114; log in `~/morbius/joblogs` | running, ETA ~29 Sep 01:00 |

Not ours:
- 2717256 (aditya, interactive on gnode066) was ended by the user.
- 2717341 (sparsh, `anlp_a2_part3`, u22 gnode042, 1 GPU) ended by 19:20, so sparsh's u22 quota is free.

gnode066 is the data/env source only: GPU0 reports "Unknown Error".
gnode113 (plafnet2) is faulty: it shows only 1 of the 2 allocated GPUs. Exclude it.

## Where things are
- **Code:** one codebase, deployed as `~/morbius` on both accounts at the same git commit (29d6c7f). Wave 1 and wave 2 differ only by config. `~/morbius_w2` (sparsh) is a symlink to `~/morbius`, kept because job 2719140 was started from that path. For a 2-GPU plafnet2 run staged from gnode114, add `--export=ALL,NPROC=2,SRC_NODE=gnode114,SRC_ENV=/ssd_scratch/sparsh.bhartia/morbius/python,SRC_DATA=/ssd_scratch/sparsh.bhartia/morbius/climbmix`.
- **Run dirs:** `<node>:/ssd_scratch/<user>/morbius/runs/<config>/` holds checkpoints and `logs/train.jsonl`.
- **Slurm output:** `~/morbius/joblogs/<config>-<jobid>.out`.
- **wandb:** https://wandb.ai/ad1tya2-iiit-hyderabad/morbius-wave1. `w1_mem_full-3a5981e5` there is the 4-step smoke test, not run 1. Run ids:
  - run 1 `w1_mem_full-4c044a60`
  - run 2 `w1_mem_cut_near-66a228d6`
  - run 3 `w1_mem_cut_far-0846dc74`
  - run 4 `w1_dense_full-143ce477`
  - run 5 `w1_dense_cut_near-caf53436`
  - idea A `w2_mem_rank1-150f542d`
- **Final checkpoints:** copy to home only at the end; see `important_notes.md` §9 (home quota caveat).
- **Resume after a failure:** re-submit the same command with `-w <same node>`; `--resume auto` picks the latest checkpoint.

## Throughput and ETA at launch
| run | tok/s | step | ETA (2B tokens + eval) |
|---|---|---|---|
| 1 | 23.0k | 22.8 s | ~25 h (≈ 27 Sep 21:30) |
| 2 | 21.2k | 24.8 s | ~27 h (≈ 27 Sep 23:40) |
| 4 | 42.0k | 12.5 s | ~14 h (≈ 27 Sep 10:30) |
| 5 | 19.6k | 26.8 s | ~28 h (≈ 28 Sep 00:40) |
| 3 | ~21k expected | | starts when run 1 ends, ~27 h after that |
