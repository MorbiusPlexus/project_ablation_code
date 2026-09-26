# Wave 1 build report

Status as of 2026-09-26 (IST). This covers what was built, the decisions taken, what was tested, what was measured, and what is still open. The spec is `wave1_plan.md` (then `notes_and_updates.md`, then the proposal). The cluster job log is `cluster_jobs.md`.

## 1. What was built

| Path | What |
|---|---|
| `ffnmem/model.py` | Llama-style decoder: RMSNorm, RoPE, SwiGLU, GQA, tied embeddings. Init std 0.02; output projections scaled by 1/√(2L). Per-layer `d_ff` (by original layer id), block deletion (`layers`), memory insertion (`memory_layers`, kept-layer numbering). Memory is added before attention. `count_params` reports counts by component. |
| `ffnmem/attention.py` | Attention backends `sdpa \| naive \| flash`. SDPA pins the memory-efficient kernel on CUDA; `sdpa_kernel` raises if that kernel is unavailable. CPU uses the math kernel. |
| `ffnmem/memory/lngram_v2.py` | Vendored byte-identical from github.com/zyaaa-ux/Lngram at commit `c8a53ca1…` (MIT). See `VENDORED.md`. |
| `ffnmem/memory/wrapper.py` | Owns the router. Applies the Appendix E init; runs routing logits in fp32; defines param groups; collects diagnostics (Sink mass, effective codes per route, dead codes). |
| `ffnmem/params.py` | Component table for many configs, match assertion, `solve_routes`, `solve_dense_dff`. |
| `ffnmem/data/{loader,build_index,sanity}.py` | Memmapped shard loader, one-time fixed-order index, sanity script. |
| `ffnmem/train.py`, `checkpoint.py`, `eval.py` | DDP loop (fp16 AMP with GradScaler, clipping, cosine schedule, optional ZeRO-1, activation checkpointing). Resume is exact and atomic; retention keeps the last K plus every Nth. Eval, logging (JSONL, TensorBoard, wandb), and an abort on scaler collapse or NaN. |
| `ffnmem/fitcheck.py` | Memory-fit gate: peak memory and step time of the real training step, one isolated launch per combination. OOM is recorded as a result. |
| `ffnmem/distlaunch.py` | Local multi-rank launcher. On Windows, torch 2.11's torchrun agent ignores `USE_LIBUV=0`, so ranks are spawned directly. The cluster uses torchrun. |
| `configs/` | `base.yaml`, the 5 wave-1 runs, `local_gpu_sanity{,_cut,_dense}`, `tiny_cpu`. |
| `slurm/stage.sh`, `train_job.sh`, `fitcheck_all.sh`, `README.md` | Plain cluster scripts. The earlier YAML/Jinja launcher was removed at the user's request. |
| `tests/` | 36 CPU tests, 2 GPU tests, 1 optional flash test, 3 multi-process tests. |

## 2. Data findings (the user's tokenization code)

- **Tokenizer:** `data/tokenizer/`, byte-level BPE, vocab **32,768**. Special ids: `<unk>`=0, `<s>`=1, `</s>`=2 (EOS), `<pad>`=3.
- **Special tokens in the data:** no BOS; EOS is appended after each document.
- **Shard format:** raw little-endian **uint16** token ids with `meta.json`. Shards start at a document boundary and end on EOS.
- **Cluster dataset** (`gnode066:/ssd_scratch/morbius/climbmix/climbmix`): produced by a different version of the download script than the one in the repo.
  - Source is `karpathy/climbmix-400b-shuffle`.
  - One shard per source parquet: 175 shards of ~57M tokens, 10B tokens in total.
  - `val.bin` has 57.5M tokens, taken from a separate parquet.
  - Same on-disk format, so the loader works unchanged.
  - Index: 4,904,017 windows of 2048+1 tokens, enough for 19,156 steps at global batch 256. A wave-1 run uses 3,815.
- **Local slice:** 100M train + 20M val tokens, made with the repo script run unchanged (`PYTHONUTF8=1` was needed on Windows because the script prints "→").
  - The script crashed in tqdm's final print after every shard was written, so `meta.json` says `complete: false`.
  - `verify_dataset.py` passed every file check. It works because transformers 5 resolves `TokenizersBackend`.
- **Sanity script** (`ffnmem.data.sanity`) passed on both datasets: decoded text is clean, no window crosses a shard, ids < vocab.

## 3. Decisions and corrections to the plan documents

**Parameter matching and model shape**
- **Readout size:** ≈2.6M per memory layer including the router, not 3M. So run 1 has **369.77M** non-embedding parameters.
- **Dense `d_ff`:** 4256, not 4160 (multiple of 32; +0.187% vs run 1). The dense cut goes 4256 → 2976, which is 30% of `d_ff`.
- **Cut = widening exactly:** both are 31,457,280 parameters. Runs 1–3 match to the parameter.
- **Active parameters differ:** dense runs activate ~370M parameters per token vs ~270M for memory runs. That is inherent to iso-total matching and should be reported next to FLOPs.
- **Tied embedding:** 33.55M (vocab 32,768), not 32.8M.

**Lngram v2 integration**
- `q_surrogate_accum_fp32` exists only in v1; v2 already accumulates in fp32.
- `readout_head_dim` must be passed as 64; the default is 128.
- The router is not part of the vendored module, so the wrapper owns it. It reads the pre-attention norm output, as in v1.
- Document-boundary masking is off (`segment_ids=None`).
- The Sink starts at 50% mass (`sink_initial_active_mass=0.5`).

**Data and training**
- **Validation:** the first 20M tokens of `val.bin`, not a slice of the permutation. `val.bin` is document-aligned and already disjoint from training.
- **Data order:** step *s* uses permutation entries `[256s, 256(s+1))`, so the order is identical for any world size or micro-batch. This was tested: 2-rank DDP and ZeRO-1 losses match single-process.
- **Activation checkpointing** is non-reentrant. Reentrant mode would silently turn off the Lngram surrogate. There is a test for this.

**Stale text and superseded items**
- Stale text in `wave1_plan`: "hidden 768", `w1_cut_mid`, "runs 1–4".
- The notes' 200M-token smoke test is superseded by the plan.

## 4. Parameter table (`python -m ffnmem.params configs/w1_*.yaml`)

```
reference: w1_mem_full (369,767,936 non-embedding params); tolerance 0.5%

| component                      | w1_mem_full           | w1_mem_cut_near                             | w1_mem_cut_far                              | w1_dense_full | w1_dense_cut_near                           |
|--------------------------------|-----------------------|---------------------------------------------|---------------------------------------------|---------------|---------------------------------------------|
| layers                         | 24                    | 24                                          | 24                                          | 24            | 24                                          |
| d_ff                           | 2816                  | 2816 / 1536 in [2, 3, 4, 5, 12, 13, 14, 15] | 2816 / 1536 in [6, 7, 8, 9, 16, 17, 18, 19] | 4256          | 4256 / 2976 in [2, 3, 4, 5, 12, 13, 14, 15] |
| attention                      | 16h/2kv               | 24h/4kv                                     | 24h/4kv                                     | 16h/2kv       | 24h/4kv                                     |
| memory                         | R=90 dm=128 @ [1, 11] | R=90 dm=128 @ [1, 11]                       | R=90 dm=128 @ [1, 11]                       | none          | none                                        |
| params.embedding               | 33.554M               | 33.554M                                     | 33.554M                                     | 33.554M       | 33.554M                                     |
| params.attention               | 56.623M               | 88.080M                                     | 88.080M                                     | 56.623M       | 88.080M                                     |
| params.ffn                     | 207.618M              | 176.161M                                    | 176.161M                                    | 313.786M      | 282.329M                                    |
| params.norms                   | 0.050M                | 0.050M                                      | 0.050M                                      | 0.050M        | 0.050M                                      |
| params.memory_tables           | 100.270M              | 100.270M                                    | 100.270M                                    | 0.000M        | 0.000M                                      |
| params.memory_readout          | 5.207M                | 5.207M                                      | 5.207M                                      | 0.000M        | 0.000M                                      |
| non_embedding                  | 369.768M              | 369.768M                                    | 369.768M                                    | 370.460M      | 370.460M                                    |
| total                          | 403.322M              | 403.322M                                    | 403.322M                                    | 404.014M      | 404.014M                                    |
| active_non_embedding_per_token | 269.544M              | 269.544M                                    | 269.544M                                    | 370.460M      | 370.460M                                    |
| non_embedding exact            | 369,767,936           | 369,767,936                                 | 369,767,936                                 | 370,459,648   | 370,459,648                                 |
| vs reference                   | +0.000%               | +0.000%                                     | +0.000%                                     | +0.187%       | +0.187%                                     |

all matched configs within tolerance
```

## 5. Tests

`python -m pytest tests -q` (local Windows box, GTX 1650 Super, torch 2.11 cu128): **all pass**.

- **Skipped:** only `test_flash_matches_sdpa_fp16`, because flash-attn is not installed. It is optional and cluster-only.
- **Covered:**
  - parameter hand-counts and matching, per-layer widths and deletion (19 blocks)
  - causality, with and without memory
  - memory block equals the plain block at init
  - memory gradient flow to tables, router, readout and conv, with and without activation checkpointing
  - activation checkpointing gives exact gradients
  - param groups (tables at 4× LR with wd 0; router wd 0), including under ZeRO-1
  - SDPA equals naive attention in fp32 (1e-5) and fp16 on GPU (<1e-2)
  - fp16 memory forward/backward on GPU
  - loader reproducibility, rank disjointness and coverage, world-size invariance, held-out disjointness
  - schedule shape
  - resume 10 → 20 bit-matches the uninterrupted run
  - checkpoint retention
  - `--init_from --new_schedule`
  - logs and diagnostics are written
  - overfit a 512-token sequence (loss < 0.1), with and without memory
  - 2-process DDP matches single-process; ZeRO-1 matches single-process; ZeRO-1 resume

## 6. Local GPU sanity (GTX 1650 Super, `local_gpu_sanity`: 6 layers, config-S width, memory at 1 and 4, R=16, seq 1024)

The run was stopped at step ~100 of 300 by Claude Code's low-memory reaper on the Windows host. That was not a training failure.

| check | result |
|---|---|
| NaN / inf | none |
| GradScaler | stable at 65,536 |
| loss | train 10.61 → 6.36 by step 90; val 6.17 at step 100 |
| Sink mass (starts at 0.50) | 0.44 (layer 1), 0.22 (layer 4) at step 100: moved |
| effective codes per route | 8.6 of 16; no routes below 4 or below 2; 10% dead codes |
| peak memory | 2.41 GB allocated |
| SDPA memory-efficient kernel on sm75 | used; the kernel is pinned and `sdpa_kernel` raises if unavailable |
| throughput | ~500 tok/s (a 1650 has no tensor cores; not representative) |

Not run yet: the 100-step no-memory run and the widened-cut variant.

## 7. Memory-fit gate

Measured with `ffnmem.fitcheck` on the cluster. Each point is a full training step: DDP, activation checkpointing, fp16 + GradScaler, clip, optimizer, two micro-batches. Seq is 2048, on 4 GPUs. Numbers are peak GB allocated / reserved. tok/s is conservative, because the optimizer runs once per 2 micro-batches instead of once per 64/micro-batch.

**4× RTX 2080 Ti** (gnode078, 10.57 GB usable per card)

| config | mb1, ZeRO off | mb1, ZeRO on | mb2, ZeRO off | **mb2, ZeRO on** |
|---|---|---|---|---|
| w1_mem_full | 9.04 / 9.57, 18.3k | 6.61 / 6.95, 15.9k | OOM | **7.46 / 8.36, 20.3k** |
| w1_mem_cut_near | 8.98 / 9.60, 17.5k | 6.56 / 7.03, 15.2k | OOM | **7.42 / 8.94, 18.8k** |
| w1_mem_cut_far | 8.99 / 9.35, 17.8k | 6.56 / 7.11, 15.3k | OOM | **7.42 / 8.29, 18.9k** |
| w1_dense_full | 9.14 / 9.69, 22.5k | 6.88 / 7.35, 18.1k | OOM | **7.72 / 8.82, 25.0k** |
| w1_dense_cut_near | 9.18 / 9.75, 21.1k | 6.91 / 7.44, 17.9k | OOM | **7.74 / 8.83, 22.8k** |

**4× RTX 3090** (gnode118, 23.56 GB usable per card)

| config | mb2, ZeRO off | mb2, ZeRO on | **mb4, ZeRO off** | mb4, ZeRO on |
|---|---|---|---|---|
| w1_mem_full | 9.87 / 10.80, 31.4k | 7.46 / 8.36, 27.9k | **11.56 / 14.68, 34.7k** | 9.14 / 11.94, 32.4k |
| w1_mem_cut_near | 9.83 / 11.07, 28.9k | 7.42 / 8.79, 25.9k | **11.49 / 15.10, 31.6k** | 9.07 / 12.02, 29.8k |
| w1_mem_cut_far | 9.84 / 10.92, 28.7k | 7.41 / 8.94, 25.8k | **11.50 / 14.21, 31.5k** | 9.08 / 11.43, 29.7k |
| w1_dense_full | 9.97 / 10.85, 36.8k | 7.72 / 8.82, 32.2k | **11.64 / 13.57, 41.2k** | 9.38 / 11.53, 38.1k |
| w1_dense_cut_near | 10.00 / 10.91, 33.7k | 7.74 / 8.84, 29.7k | **11.64 / 13.60, 36.9k** | 9.39 / 11.54, 34.4k |

**Chosen settings (bold above)**

| GPU | micro-batch | ZeRO-1 | headroom |
|---|---|---|---|
| 2080 Ti | 2 | on | ~1.7 GB |
| 3090 | 4 | off | ~9 GB |

Config S fits at seq 2048 on both GPU types, so no mitigation (shorter sequences etc.) is needed.

**What differs between GPU types**
- Micro-batch and ZeRO differ by GPU type. The effective batch (256 × 2048) and the data order are identical everywhere.
- The paired comparisons (1↔2, 1↔3 and 2↔3 on 2080 Ti; 4↔5 on 3090) stay within one GPU type.
- 1↔4 (memory vs dense) carries a GPU-type confound. Both use fp16 with the same pinned SDPA kernel.
- Run 5 uses 2× 3090 (32 accumulation steps instead of 16). World-size invariance is tested.

**Real-job smoke test on gnode078** (3 steps plus a resume to step 4, ZeRO-1, wandb)
- Throughput is 22.9k tok/s, better than the conservative fit-check estimate.
- One consolidated checkpoint is 4.84 GB and takes ~3 min to save. At one save per ~3.2 h that is ~1.5% overhead.
- The resume found and fixed a GPU-only bug: fused CUDA AdamW rejects the CPU `step` tensors left by ZeRO-1's `load_state_dict`. There is now a local GPU resume test (non-ZeRO; ZeRO on Windows gloo+CUDA crashes, so ZeRO resume was verified on the cluster).
- After a resume, reserved memory rises to 9.6 GB. It is mitigated with `empty_cache()` after load and `expandable_segments`.

## 8. Cluster status and open items

Launched 2026-09-26 ~20:33–20:41 IST. See `cluster_jobs.md` for job ids, nodes and ETAs.

| run | job | node | status |
|---|---|---|---|
| 1 `w1_mem_full` | 2717512 | gnode078, 4× 2080 Ti | running |
| 2 `w1_mem_cut_near` | 2717520 | gnode082, 4× 2080 Ti | running |
| 3 `w1_mem_cut_far` | 2717537 | gnode078, starts after run 1 succeeds | queued |
| 4 `w1_dense_full` | 2717513 | gnode118, 4× 3090 | running |
| 5 `w1_dense_cut_near` | 2717521 | gnode114, 2× 3090 | running |

- wandb project: https://wandb.ai/ad1tya2-iiit-hyderabad/morbius-wave1
- The first eval, and the first post-init Sink-mass reading, comes at step 200: ~1.3 h in on 2080 Ti, ~40 min on 3090.
- Checkpoints are on node-local ssd. Resuming needs the same node; if a node dies, that run restarts from scratch unless a checkpoint was copied off. At the end, copy the final checkpoints to home by hand.

- The auto-mode classifier refused to make `/home2/sparsh.bhartia` traversable. The aim was to share Sparsh's existing miniconda (torch 2.6) with Aditya's jobs. Instead, one standalone env (CPython 3.12, torch 2.11 cu128) was built on gnode066 and copied to training nodes. Its files are world-readable by default, so no permission changes were needed.
- gnode066 GPU0 reports "Unknown Error", so that node serves as the data and env source only.
- wandb credentials: Aditya's `api.wandb.ai` entry was copied into Sparsh's `~/.netrc` at the user's request. The backup is `~/.netrc.bak_morbius`.
- The repo has no git commit yet, so checkpoints record `git: unknown`. Make a first commit before the runs if a hash is wanted.
