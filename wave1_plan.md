# wave1\_agent\_plan

Sep 26, 2026 · @Aditya

Instructions for a coding agent to build everything needed for wave 1 of the FFN-reallocation study, up to (not including) launching runs. Read alongside the proposal (Morbius-Proposal) and notes\_and\_updates.

## Scope and hard rules

Build a complete, tested training stack for the four wave-1 runs; stop before launching anything.

**In scope**

- Llama-style dense decoder with per-layer FFN width, block deletion and Lngram v2 memory insertion.
- Loader for the already-tokenized dataset (the user's tokenization code is in the folder; reuse its output format, do not re-tokenize).
- DDP training loop: fp16 AMP, cosine schedule, frequent resumable checkpoints, validation loss, logging.
- Config files for the five wave-1 runs, a local GPU sanity config, and a parameter-matching tool.
- Slurm sbatch templates for a multi-account cluster.
- Unit tests, a tiny CPU end-to-end test, and a written test report.

**Hard rules**

- Do not launch any run on the cluster without the user's explicit go-ahead; launching is the final step. There is no separate cluster smoke test: the real runs are launched and watched (see §First launch).
- Do not modify the user's tokenization code.
- All code must run on Linux (the cluster) and be developed on Windows: use `pathlib`, no shell-specific commands in Python, line endings LF, no Windows-only dependencies. Local tests run on CPU (or a local GPU if present).
- Every run must be exactly parameter-matched (see §Configs). Print and assert parameter counts at startup.
- Same seed, same data order and same precision (fp16) for every run.
- Never delete or skip anything that would block extending a run later: keep optimizer, scheduler, scaler, RNG and data-position state in every checkpoint.
- Read the proposal and notes\_and\_updates before starting; precedence when they conflict: this plan, then notes\_and\_updates, then the proposal.
- Ask the user when something in §Open items is missing. Otherwise make the decision, state it in the report, and proceed.

## Decisions taken

Defaults chosen to keep wave 1 simple; the user can override any of them.

| Item | Decision | Reason |
| --- | --- | --- |
| Model size | Config S: \~365M non-embedding (265M backbone + 100M memory), \~398M with the 33M tied embedding | User's range (340–370M); smaller models show the effects poorly. \~5.5 tokens/param at 2B is accepted undertraining |
| Architecture | Llama-style: RMSNorm, RoPE, SwiGLU, GQA; tied embeddings | What the proposal specifies; nothing to defend |
| Memory module | Lngram v2 (`Lngram_v2.py` from the Lngram repo), vendored unchanged | See notes\_and\_updates |
| Tokens per run | 2B, cosine schedule set to 2B | Time budget; extension later needs a re-warm, which the checkpoints allow |
| Sequence length | 2048 | Fits 11 GB cards in fp16 with activation checkpointing |
| Global batch | 256 × 2048 = 524k tokens/step → 3,815 steps | Standard for this size; same on every machine via gradient accumulation |
| Precision | fp16 AMP with fp32 master weights and GradScaler; Lngram routing logits, Sink softmax and surrogate accumulation in fp32 | 2080 Ti has no bf16; same precision on all machines |
| Attention | Backend switch: default torch SDPA (memory-efficient kernel); optional flash-attn (the 2080 Ti fork on Turing, upstream FA2 on Ampere) behind a flag | See §Attention |
| Optimizer | AdamW β=(0.9, 0.95), wd 0.1, clip 1.0; peak LR 6e-4, warmup 2% (min 100 steps), cosine to 6e-5 | Standard 125M–350M recipe |
| Memory LRs | Tables: 4× base LR, no weight decay. Router/readout: base LR, no weight decay on router | Lngram v2 Appendix E |
| Lngram init | Tables normal (std scale 1.0), output projection zero, conv zero, Sink μ₀ = 0.5, approximate surrogate | Lngram v2 Appendix E; overrides the code defaults |
| Insertion layers | 0-indexed 1 and 11 (the papers' "2 and 12") | Both Lngram text setups |
| Checkpoints | Every 200 steps (\~100M tokens); keep last 3 plus every 1,000th; full resumable state | Frequent checkpointing requested |
| Validation | Every 200 steps on a fixed 20M-token held-out slice; report loss and perplexity | Primary metric |
| Logging | JSONL + TensorBoard by default; wandb optional via flag | Works offline on the cluster |
| Data order | One fixed shuffle of shards, seed 1234, identical for all runs; exact resume from data position | Paired comparisons |

## Wave-1 configs

Base backbone (config S): hidden 1024, 24 layers, 16 heads, 2 KV heads (head dim 64), d\_ff 2816, vocab from the tokenized dataset (assumed 32k). Layers 0-indexed; memory at layers 1 and 11.

Per-layer counts at this shape: attention 2.36M, FFN 8.65M, block 11.0M. Backbone 264.5M non-embedding; tied embedding 32.8M at 32k vocab.

Memory: R=90 routes, dm=128, 4 KV readout heads → tables \~100.3M, readout \~3M per layer. Run 1 total ≈ 365M non-embedding. Every other run matches run 1's total within 0.5%.

**Cut** = d\_ff reduced by 1,280 in 8 named layers: 8 × 3 × 1024 × 1280 = 31.5M freed.

**Attention widening** (destination of every cut): inner dim 1024 → 1536 (24 query heads × 64), KV heads 2 → 4, in all 24 layers: +1.31M per layer = +31.5M. Cut and widening are equal by construction; the tool confirms the residual and adjusts R by at most 1 if needed.

| Run | Config name | d\_ff base | Cut layers | Attention | Memory | Non-emb. total |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `w1_mem_full` | 2816 | none | 16 heads, 2 KV | 100M | \~365M |
| 2 | `w1_mem_cut_near` | 2816 → 1536 in cut layers | 2–5, 12–15 | 24 heads, 4 KV | 100M | \~365M |
| 3 | `w1_mem_cut_far` | 2816 → 1536 in cut layers | 6–9, 16–19 | 24 heads, 4 KV | 100M | \~365M |
| 4 | `w1_dense_full` | \~4160 everywhere (tool sets the exact value to hit run 1's total) | none | 16 heads, 2 KV | none | \~365M |
| 5 | `w1_dense_cut_near` | run 4 base → base − 1280 in cut layers | 2–5, 12–15 | 24 heads, 4 KV | none | \~365M |

Notes:

- Runs 1, 2, 4, 5 form a cut × memory 2×2; run 3 is the location control. Runs 1 and 2 are the key pair and should run on the same GPU type.
- Runs 4 and 5 cut from a wider base, so the cut fraction differs (31% vs 45%); the moved amount and destination are identical.
- Insertion layers 1 and 11 and their FFNs are never cut.
- Launch order = rank; runs can be submitted as machines free up.

## Repo layout and deliverables

One Python package, plain PyTorch, no training framework. Pin versions in `requirements.txt`.

```
repo/
  configs/            YAML per run: w1_mem_full, w1_mem_cut_near, w1_mem_cut_far, w1_dense_full, w1_dense_cut_near, local_gpu_sanity
  data/               user's tokenization code (do not modify) + new loader
  ffnmem/
    model.py          Llama-style decoder; per-layer d_ff list; layer list (deletion); memory-layer set
    attention.py      backend switch: sdpa | flash
    memory/lngram_v2.py   vendored from github.com/zyaaa-ux/Lngram (commit hash recorded)
    memory/wrapper.py     config → module, init overrides, fp32 guards, param groups
    params.py         count by component; solve R for a target total
    train.py          DDP loop, AMP, schedule, checkpoint/resume, eval
    eval.py           validation loss on the held-out slice from a checkpoint
    checkpoint.py     save/load: model, optimizer, scheduler, scaler, RNG, data position
  slurm/              sbatch templates + launcher script (writes commands, does not submit)
  tests/              unit tests + tiny CPU end-to-end run
  REPORT.md           what was built, decisions, test results, parameter tables, open issues
```

Deliverables, in order:

1. `params.py` with a CLI that prints the component table for every config and asserts matching.
2. Model, attention backend and memory wrapper, with tests.
3. Loader over the existing tokenized output.
4. Training loop with resume, tested on CPU with a tiny config.
5. Configs for the six runs above.
6. Slurm templates and launcher.
7. `REPORT.md`.
8. Local GPU sanity run, then first cluster launch only after the user says go.

## Data pipeline

The dataset is already tokenized by the user's code. Build only the loader.

- Read the user's tokenization code first. Record in `REPORT.md`: tokenizer name, vocab size, EOS/BOS handling, shard format, dtype, token count. If any of these can't be determined from the code, ask the user.
- Loader: memory-mapped shards; contiguous 2048-token windows with next-token targets; documents concatenated with EOS between them (standard GPT-style, no document masking).
- Order: one fixed permutation of windows from seed 1234, materialized once as an index file so every run and every machine reads the same sequence. DDP ranks take disjoint strided slices of that order.
- Resume: the checkpoint stores the global step; the loader recomputes its position from it. No sampler state to serialize.
- Validation slice: the last 20M tokens of the permutation, held out from training, used by every run.
- Sanity script: decode 3 random windows back to text with the tokenizer and print them; check no window crosses a shard boundary incorrectly; assert token ids < vocab size.
- Do not depend on the tokenizer at training time; only the loader and the sanity script need it.

**Local data (Windows box)**

- Only the download/tokenization scripts are local; the full dataset is not. Download a small slice (\~50–100M tokens is enough) and tokenize it with the user's script, unchanged. Use it for loader tests, the data sanity script and the local GPU sanity run.
- Keep all data paths in the config; nothing hard-coded. The full dataset will be downloaded and tokenized on the cluster, and the fixed-order index and held-out slice built there from the full data.
- The index-building step must be a standalone command (`python -m ffnmem.data.build_index --data <dir>`), run once on the cluster before any training job; training jobs only read the index.

## Model and Lngram v2 integration

### `model.py`

- Base it on a minimal Llama reference (e.g. llama2.c `model.py`): RMSNorm, RoPE, SwiGLU, GQA, pre-norm blocks, tied embeddings, weight init std 0.02 with output-projection scaling 1/sqrt(2L).
- Config fields: `d_model`, `n_heads`, `n_kv_heads` (attention inner dim = n\_heads × 64 and may exceed `d_model`; the output projection maps back to `d_model`), `vocab`, `seq_len`, `d_ff: list[int]` (one per layer), `layers: list[int]` (original layer ids kept; deletion = omit ids), `memory_layers: list[int]` (0-indexed, in the kept-layer numbering), `memory: {routes, mem_dim, kv_heads, ...}`.
- Block forward: `h = h + mem(h)` (if this layer has memory) → `h = h + attn(norm(h))` → `h = h + ffn(norm(h))`. Memory is added before attention, as in both Lngram papers.
- Attention through `attention.py` only; never call a kernel directly from the model.
- Provide `count_params(model)` returning a dict by component: embeddings, attention, ffn, memory tables, memory readout (router + Q/KV/O + conv), norms.

### `memory/lngram_v2.py`

- Vendor the file unchanged from the Lngram repo; record the commit hash. Put any adaptation in `wrapper.py`.
- Wrapper sets the Appendix E init explicitly: `table_init_std_scale=1.0`, `output_proj_init_std_scale=0.0`, `conv_zero_init=True`, `q_surrogate_mode="approximate"`, `q_surrogate_accum_fp32=True`, Sink initial mass 0.5, 2/3-grams, 4 bits per route.
- Readout heads: 16 Q heads × 64 = 1024 (must equal `d_model`), 4 KV heads, head dim 64. Independent of the backbone's attention widening.
- fp16 guards: routing logits, Sink softmax and the surrogate computed in fp32 (wrap the readout softmax in `autocast(enabled=False)` if the vendored code doesn't already). Verify no NaN/inf in the local GPU sanity run (the Windows box has a CUDA GPU).
- Param groups: tables (4× LR, wd 0), router projection (base LR, wd 0), readout and conv (base LR, wd 0.1), backbone (base LR, wd 0.1).
- Packed-sequence masking: the vendored code masks n-grams across `cu_seqlens` boundaries. Since windows are plain 2048-token chunks, pass a single segment per window; document that document-boundary masking is deliberately off.
- Do not use the incremental-decode or context-parallel paths in training; leave them intact for later latency work.

### `params.py`

- `solve_routes(config, target_total)`: given backbone widths and memory settings, find the integer R whose total is closest to the target; print the component table and the residual. Fail if the residual exceeds 0.5%.
- CLI: `python -m ffnmem.params configs/*.yaml` prints one table for all configs side by side. Its output goes into `REPORT.md`.

## Training loop

- `torchrun` DDP, one process per GPU. Global batch 256 sequences fixed in the config; per-GPU micro-batch and gradient-accumulation steps are derived from `world_size` and a `micro_batch` field, so 4×3090 and 4×2080 Ti runs see identical effective batches.
- fp16 AMP: `torch.autocast(dtype=float16)` + `GradScaler`; fp32 master weights; gradient clipping after unscale. Log the scaler scale; abort if it collapses below 1.0 for 100 consecutive steps.
- Activation checkpointing per block, switchable in the config (needed on 11 GB cards).
- Schedule: linear warmup then cosine to 10% of peak, total length = `tokens_total / tokens_per_step`. Length is a config value, never inferred, so extension runs can set a new schedule.
- Checkpoint every `ckpt_every` steps (default 200): model, optimizer, scheduler, scaler, Python/NumPy/Torch RNG per rank, global step, config, git hash. Keep the last 3 and every 1,000th (config). Atomic writes (temp file then rename).
- `--resume <ckpt>` restores everything and continues from the exact data position; `--init_from <ckpt> --new_schedule` loads weights only (for later extensions). Both paths tested.
- Validation every `eval_every` steps (default 200) on the fixed held-out slice; log loss, perplexity, tokens seen, wall-clock, throughput (tokens/s), peak GPU memory.
- Memory diagnostics every eval: mean Sink mass per memory layer, fraction of routes with effective codes < 4, table gradient norm. Cheap and needed later.
- Logging: JSONL per rank 0 plus TensorBoard; wandb optional via `--wandb`. Never require network access.
- Determinism: set all seeds from the config; log `torch.backends.cudnn` flags. Full bitwise determinism is not required.
- `eval.py` loads any checkpoint and reports validation loss, for use on saved checkpoints during training.

## Attention backend

Default to PyTorch's built-in SDPA memory-efficient kernel; treat flash-attn as an optional speed-up to be earned by a benchmark.

- **Default (`backend: sdpa`)**: `F.scaled_dot_product_attention` with `is_causal=True` under `sdpa_kernel([MEM_EFFICIENT])`. Works on Turing (sm75) and Ampere, needs no build, fp16-safe. Pin the same kernel on all machines so numerics don't differ by GPU type.
- **Optional (`backend: flash`)**: on 2080 Ti use [KLM-corporation/flash-attention-2080ti](https://github.com/KLM-corporation/flash-attention-2080ti), a FlashAttention-2 fork with sm75 support, fp16 only; on 3090 use upstream flash-attn 2. The fork's README lists a build environment of PyTorch 2.9 / CUDA 13 and 76 failing varlen tests. You don't use varlen (fixed 2048 windows), and head dim 64 is its best case.
- Equivalence test: on any available GPU, compare `flash` vs `sdpa` outputs and gradients on random inputs in fp16 (tolerance 1e-2 relative). Only enable `flash` for real runs if it passes and a 50-step throughput benchmark on that machine type shows more than 15% end-to-end gain. Record both numbers in `REPORT.md`.
- Do not attempt to build flash-attn on Windows. Build instructions go into `slurm/README.md` for the cluster.
- At 2048 tokens and hidden 768, the FFN dominates compute; if the flash build costs more than a day, drop it.

## Slurm launch (write, do not submit)

The runs will be launched by the user on a cluster with several accounts. Prepare everything so launching is one command per run.

- `slurm/cluster.yaml`: one entry per account/partition with fields `account`, `partition`, `gpu_type` (3090 | 2080ti), `gpus_per_node`, `time_limit`, `modules` (list), `env_activate` (command), `scratch_dir`. Ship it with placeholders; the user fills it in.
- `slurm/train.sbatch.j2`: Jinja template. Requests one node, 4 GPUs, sets `--account`/`--partition`, loads modules, activates the env, runs `torchrun --nproc_per_node=4 -m ffnmem.train --config <cfg> --resume auto`. `--resume auto` picks the latest checkpoint in the run's directory, so a re-submitted job continues; a `--requeue`-safe trap re-submits on preemption or time-limit.
- `slurm/launch.py --run w1_cut_mid --account <name>`: renders the sbatch file into `slurm/rendered/` and prints the `sbatch` command. It never calls `sbatch` unless `--submit` is passed, and `--submit` is documented as user-only.
- Per-run directory: `runs/<config>/` holding rendered sbatch, config copy, logs, checkpoints. Checkpoint paths must live on the scratch filesystem given in `cluster.yaml`.
- Machine-type note in `REPORT.md`: which runs should share a GPU type (runs 1 and 2 on the same type), and the expected time per run on each type from the throughput measured in the first cluster hour.
- `slurm/README.md`: environment setup on the cluster (Python version, CUDA/PyTorch pins, optional flash-attn build), and the launch procedure.

## Tests and acceptance criteria

All tests run on CPU with tiny configs (d\_model 64, 4 layers, vocab 256, seq 32) unless marked GPU.

| Test | Passes when |
| --- | --- |
| Parameter table | `params.py` output for runs 1–4 matches within 0.5%; component counts equal hand computation for the base shape (attention 2.36M, FFN 8.65M per layer at config S; widened attention 3.67M) |
| Per-layer widths | A config with `d_ff` list produces layers with those exact shapes; deletion config produces 19 blocks with the right original ids |
| Causality | Changing token t does not change logits at positions < t |
| Memory insertion | Memory layers present only where configured; output of a memory block equals the plain block at init (zero output projection) |
| Memory gradient flow | After 5 steps on random data, tables, router and readout all have non-zero gradients (surrogate active) |
| Attention equivalence | `sdpa` equals a naive softmax attention on random inputs (fp32, 1e-5) |
| Loader | Fixed order reproducible across two constructions; ranks' slices disjoint and covering; held-out slice disjoint from training |
| Resume | Train 20 steps, checkpoint at 10, resume from 10: losses at steps 11–20 match the uninterrupted run to fp32 tolerance |
| Init-from | `--init_from` loads weights and starts a fresh schedule without error |
| Overfit | Tiny model memorizes a 512-token sequence (loss < 0.1) within 300 steps, with and without memory |
| Slurm render | `launch.py` renders all six configs; rendered files pass `bash -n` |
| fp16 (GPU only) | Local GPU sanity run passes (see §First launch); sdpa run on the local GPU matches naive attention in fp16 (1e-2 relative) |

Acceptance: every test above passes or is explicitly marked skipped with a reason in `REPORT.md`; `REPORT.md` contains the parameter tables, the data-format findings, decisions made, and anything the agent could not verify.

## First launch (needs the user's go-ahead)

No separate 200M-token smoke test. Instead: a short local GPU sanity run, then launch the real runs and watch the first hours. If anything is wrong, fix and relaunch from scratch.

**Local GPU sanity run (Windows box, before any cluster launch)**

- Hardware: GTX 1650 Super, 4 GB, Turing sm75 (same architecture as the 2080 Ti, but no tensor cores). Good proxy for kernel and fp16 behaviour on Turing; not representative for speed. Full config S does not fit (weights + grads + Adam states alone ≈ 6.4 GB).
- Sanity config `local_gpu_sanity`: same width as config S (d\_model 1024, 16 heads, 2 KV heads, d\_ff 2816, memory readout identical) but 6 layers, memory at layers 1 and 4, R=16, seq 1024, micro-batch 1–2, activation checkpointing. Plus a matching 6-layer widened-attention variant (24 heads, 4 KV, cut in layers 2–3) to exercise the run-2 code path.
- 300 steps on the local data slice with memory; 100 steps without memory.
- Pass: no NaN/inf; GradScaler scale stable (not collapsing); loss falls clearly in both; memory tables, router and readout get non-zero gradients; Sink mass moves off its initial value; checkpoint → resume reproduces the next step's loss; SDPA memory-efficient kernel selected on sm75.
- Record peak memory, but take throughput estimates from the first cluster hour, not from this GPU.
- Use the SDPA backend. Do not try to build flash-attn on Windows.

**Cluster launch order**

- Launch runs 1 and 4 first, together (memory vs dense, same GPU type if possible). Launch 2, 3, 5 once they look healthy, or immediately if the user prefers.
- Watch the first \~400 steps (\~200M tokens) of each run for: NaN/inf or loss spikes; scaler collapse; throughput far below the local estimate; Sink mass stuck at \~1.0 or \~0.0 (memory unused or saturated); dead routes (effective codes < 2).
- Early losses of run 1 vs run 4 are only a sanity signal (memory runs can lag early); do not tune on them. If a run fails a check: stop it, report, and relaunch after a fix. Fallback to Lngram v1 single-table is the user's call.

## Open items for the user

- [x] Config S at \~365M non-embedding, the 8-layer cut, attention as destination: confirmed.
- [x] Proposal reframing (freed FFN → attention, widened in all 24 layers): confirmed.
- [x] Windows box has a GTX 1650 Super (4 GB, sm75): local fp16, attention and code-path tests run there on a reduced-depth config; full config S is checked only on the cluster. Only scripts are local; download a small dataset slice for testing.
- [ ] Fill in `slurm/cluster.yaml` (accounts, partitions, GPU types, modules, env activation, scratch path) before any launch.
- [ ] Tokenizer name and vocab size, if the agent can't determine them from the tokenization code.
- [ ] Go/no-go for the first cluster launch after reviewing `REPORT.md` and the local sanity run.
