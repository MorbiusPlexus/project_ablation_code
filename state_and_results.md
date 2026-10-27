# State and results: wave 1

Current job state, measured numbers and diagnostics. Things to look at later are in `important_notes.md`. The job log is in `cluster_jobs.md`.

## FINAL RESULTS: all runs complete (2026-09-29 02:51 IST)

Val loss on the fixed 20M-token held-out slice after 2.0B tokens (step 3815). Single seed each.

| run | config | GPUs | final val | vs its baseline |
|---|---|---|---|---|
| 1 | w1_mem_full | 4× 2080 Ti | 2.805 | +0.003 vs run 4 (memory, uncut) |
| 2 | w1_mem_cut_near | 4× 2080 Ti | 2.791 | −0.014 vs run 1 (cut 2–5, 12–15) |
| 3 | w1_mem_cut_far | 4× 2080 Ti | 2.790 | −0.015 vs run 1 (cut 6–9, 16–19) |
| 4 | w1_dense_full | 4× 3090 | 2.802 | – (dense baseline) |
| 5 | w1_dense_cut_near | 2× 3090 | **2.774** | −0.028 vs run 4 |
| A | w2_mem_rank1 (idea A) | 2× 3090 | 2.807 | +0.002 vs run 1 |
| **Super1** | **w2_mem_super1 (Idea 1)** | **2× 3090** | **2.7875** | **−0.0035 vs run 2 (best memory run)** |

**Conclusions (single seed; treat differences below ~0.005 as noise)**

0. **Idea 1 (Single-Layer Super-Memory, w2_mem_super1) succeeds**: Setting R=184 at Layer 1 and eliminating Layer 11 drops final validation loss to **2.7875 (2.788)**, beating all prior memory configurations (run 1: 2.805, run 2: 2.791, run 3: 2.790, idea A: 2.807). Memory output/residual norm ratio jumps to **0.3535** (up from 0.28). This confirms that eliminating deep-layer address drift and doubling early routing capacity is strictly superior.

1. **Cut + attention widening helps**, in every configuration: −0.014 / −0.015 with memory, −0.028 without. Moving 31.5M parameters from FFN into wider attention (24 q / 4 kv heads) is the better allocation at this scale.
2. **Cut location does not matter with memory.** Run 3 (far) ≈ run 2 (near): 2.790 vs 2.791. This is the decision point in `important_notes.md` §5a: "location isn't the lever".
3. **Memory ties dense at equal total parameters** (2.805 vs 2.802), using 27% fewer active parameters per token. In the cut configuration memory is worse: 2.791 vs 2.774 (+0.017).
4. **The cut × memory interaction goes against the hypothesis:** the cut helps dense twice as much as it helps memory. The probe suggests why: wider attention and memory do overlapping local-pattern work (run 2 depends on memory half as much as run 1).
5. **Idea A (rank-1 add-on at layer 1, paid for by halving layer 11) gives no gain:** 2.807 vs 2.805. The rank-1 path is used, but stays small (~7% of residual norm vs ~32% for v2 at layer 1, stable from step 400 on).

## Job state (history; snapshot 2026-09-27 22:41 IST, all finished by 2026-09-29 02:16)

| run | config | job | node | GPUs | state | ETA | wandb run |
|---|---|---|---|---|---|---|---|
| 1 | w1_mem_full | 2717512 | gnode078 | 4× 2080 Ti | **finished** 27 Sep 22:28, final val **2.805** | done | `w1_mem_full-4c044a60` |
| 2 | w1_mem_cut_near | 2717520 | gnode082 | 4× 2080 Ti | step 3360 / 3815 | ~28 Sep 01:30 | `w1_mem_cut_near-66a228d6` |
| 3 | w1_mem_cut_far | 2717537 | gnode078 | 4× 2080 Ti | running since 27 Sep 22:28 (step 20) | ~29 Sep 00:00 | `w1_mem_cut_far-0846dc74` |
| 4 | w1_dense_full | 2717513 | gnode118 | 4× 3090 | **finished** 27 Sep 10:29, final val **2.802** | done | `w1_dense_full-143ce477` |
| 5 | w1_dense_cut_near | 2717521 | gnode114 | 2× 3090 | step 3370 / 3815 | ~28 Sep 02:00 | `w1_dense_cut_near-caf53436` |
| A | w2_mem_rank1 (idea A) | 2719140 | gnode115 | 2× 3090 | step 370 / 3815 | ~29 Sep 01:00 | `w2_mem_rank1-150f542d` |
| probe | memory zero-out | 2718727 | gnode118 | 4× 3090 | done 27 Sep ~15:10, job cancelled | – | – |

- wandb project: https://wandb.ai/ad1tya2-iiit-hyderabad/morbius-wave1. `w1_mem_full-3a5981e5` there is the 4-step smoke test.
- No crashes, NaNs or OOMs in any training run. The fp16 scale has grown to 131072 in every wave-1 run.
- **Final weights are saved to home** as `~/morbius/final_ckpts/<cfg>_step3815.pt`: fp32 model + config, no optimizer (resuming has no value after the schedule ends).
  - Done: run 1 (aditya) and run 4 (sparsh).
  - Runs 2, 3, 5 and A are saved automatically when they finish.
  - Full resumable checkpoints stay on node ssd.

## Val loss (fixed 20M-token held-out slice)

| step | tokens | run 1 mem_full | run 2 mem_cut_near | run 4 dense_full | run 5 dense_cut_near |
|---|---|---|---|---|---|
| 200 | 105M | 4.888 | 4.984 | 4.906 | 5.268 |
| 400 | 210M | 3.931 | 4.011 | 3.932 | 4.085 |
| 600 | 315M | 3.517 | 3.538 | 3.534 | 3.556 |
| 800 | 419M | 3.341 | 3.341 | 3.362 | 3.347 |
| 1000 | 524M | 3.234 | 3.222 | 3.244 | 3.223 |
| 1200 | 629M | 3.156 | 3.143 | 3.163 | 3.142 |
| 1400 | 734M | 3.097 | 3.082 | 3.101 | 3.077 |
| 1600 | 839M | 3.047 | 3.030 | 3.050 | 3.024 |
| 1800 | 944M | 3.005 | 2.990 | 3.008 | 2.981 |
| 2000 | 1.05B | 2.970 | 2.954 | 2.970 | 2.944 |
| 2200 | 1.15B | 2.938 | 2.923 | 2.938 | 2.911 |
| 2400 | 1.26B | 2.911 | 2.895 | 2.910 | 2.882 |
| 2600 | 1.36B | 2.886 | 2.871 | 2.885 | 2.857 |
| 2800 | 1.47B | 2.865 | 2.850 | 2.864 | 2.836 |
| 3000 | 1.57B | 2.847 | 2.833 | 2.845 | 2.818 |
| 3200 | 1.68B | 2.833 | 2.818 | 2.830 | 2.802 |
| 3400 | 1.78B | 2.821 | 2.807 | 2.818 | 2.791 |
| 3600 | 1.89B | 2.812 | 2.798 | 2.809 | 2.781 |
| 3815 | 2.00B | **2.805** | **2.791** | **2.802** | **2.774** |

**Differences at matched steps** (negative means the cut or memory helps):

| step | cut, with memory (2 − 1) | cut, dense (5 − 4) | memory, uncut (1 − 4) | memory, cut (2 − 5) |
|---|---|---|---|---|
| 400 | +0.080 | +0.153 (spike) | −0.001 | −0.074 |
| 1000 | −0.012 | −0.021 | −0.010 | −0.001 |
| 2000 | −0.016 | −0.026 | 0.000 | +0.010 |
| 2200 | −0.015 | −0.027 | 0.000 | +0.012 |
| 3200 | −0.015 | −0.028 | +0.003 | +0.016 |
| 3815 (final) | **−0.014** | **−0.028** | **+0.003** | **+0.017** |

**Reading (step 3200, ~1.68B tokens; final for runs 1 and 4):**
- **The cut plus attention widening helps**, with or without memory. It helps more without memory (−0.028 vs −0.015), and both gaps have been stable since step ~2000.
- **Memory adds nothing over dense at equal total parameters.** The final values are 2.805 vs 2.802. In the cut configuration memory is slightly worse (+0.016). It does use 27% fewer active parameters per token (see the probe).
- **The cut × memory interaction** is +0.013 at step 3200, against the hypothesis. It is small and single-seed.

## Memory diagnostics

Values at each eval. "Sink" is the mass given to the zero-value "nothing useful" slot; it starts at 0.50. "Codes" is the effective number of codes per route, out of 16.

| step | run 1 Sink, layer 1 | run 1 Sink, layer 11 | run 1 codes, layer 1 / 11 | run 1 table grad norm | run 2 Sink, layer 1 | run 2 Sink, layer 11 | run 2 codes, layer 1 / 11 |
|---|---|---|---|---|---|---|---|
| 200 | 0.461 | 0.434 | 14.6 / 13.0 | 0.0111 | 0.423 | 0.437 | 13.7 / 12.2 |
| 400 | 0.439 | 0.446 | 15.0 / 14.2 | 0.0046 | 0.388 | 0.436 | 14.6 / 13.5 |
| 1000 | 0.453 | 0.453 | 15.0 / 14.6 | 0.0020 | 0.410 | 0.447 | 15.0 / 14.2 |
| 1800 | 0.460 | 0.450 | 14.9 / 14.6 | 0.0013 | 0.416 | 0.445 | 15.0 / 14.2 |
| 2200 | 0.463 | 0.451 | 14.9 / 14.6 | 0.0013 | 0.418 | 0.445 | 15.0 / 14.2 |
| 2600 | 0.464 | 0.450 | 14.8 / 14.6 | 0.0012 | | | |
| 3815 | 0.470 | 0.451 | 14.8 / – | – | | | |

- Sink mass drifts back toward 0.5 after step 400, and routing codes stay near uniform (~15 of 16; the paper reports ~8).
- **These two diagnostics do not mean the memory is unused.** The zero-out probe below shows the models depend on it heavily. Sink mass near 0.5 and near-uniform codes are compatible with a large memory contribution.
- The cut model (run 2) uses memory a little more at layer 1 (0.42 vs 0.46).
- No dead routes in either run.

## Run 5 warmup spike (steps 60–80)

| step | loss | grad norm | LR |
|---|---|---|---|
| 60 | 6.488 | 3.37 | 3.6e-4 |
| 70 | 6.657 | 14.52 | 4.2e-4 |
| 80 | 6.963 | 2.29 | 4.8e-4 |
| 90 | 6.582 | 1.19 | 5.4e-4 |
| 110 | 6.350 | 0.86 | 6.0e-4 |

Train-loss gap to run 4: +0.38 (step 200), +0.25 (300), +0.15 (400). By step 1000 it had reversed to −0.02. Fully recovered.

## Memory zero-out probe (step 2000 checkpoints, gnode118)

Full 20M-token val slice. The listed memory layers have their output replaced by zeros; everything else is unchanged. The last column is the mean ‖memory output‖ / ‖residual‖.

| model (step 2000) | memory on | layer 1 zeroed | layer 11 zeroed | both zeroed | output/residual (layer 1 / 11) |
|---|---|---|---|---|---|
| run 1 mem_full | 2.970 | 3.182 (**+0.212**) | 2.992 (+0.022) | 3.299 (**+0.328**) | 0.33 / 0.14 |
| run 2 mem_cut_near | 2.954 | 3.027 (+0.073) | 2.969 (+0.015) | 3.098 (+0.144) | 0.28 / 0.11 |
| run 4 dense_full (reference) | 2.970 | | | | |

**Findings**

1. **The models rely on memory heavily.** Zeroing it costs +0.33 nats in run 1, far more than the whole memory-vs-dense difference. Layer 1 carries most of it (+0.21); layer 11 matters little (+0.02). This matches Lngram v2 Appendix B, where the deeper insertion layer is the weaker, less stable one.
2. **The net gain is still zero.** Run 1 ties dense at equal total parameters (2.970 vs 2.970).
   - Memory acts as a **substitute** for FFN capacity, not an addition: 105M memory parameters buy about what 106M extra FFN parameters buy.
   - Because the tables are sparse, run 1 reaches that loss with **27% fewer active parameters per token** (270M vs 370M) and correspondingly fewer FLOPs. That is a positive result for the efficiency framing.
3. **The cut model depends on memory less** (+0.14 vs +0.33). The wider attention (24 query / 4 KV heads) appears to take over some of the local-pattern work the memory does in run 1.

**Failed-lookup test (H1 vs H2, `important_notes.md` §4)**
- Method: per-token loss gap (run 2 − run 1) on the first 2.05M val tokens, bucketed by Sink mass.
- Result: the gap is flat. Across Sink-mass deciles it ranges from −0.012 to −0.022 at both memory layers, whether bucketed by run 1's or run 2's Sink. corr(Sink, gap) = 0.000.
- **Neither H1 (gap in high-Sink tokens) nor H2 (gap in low-Sink tokens) is supported at step 2000.** The cut-plus-widen effect is uniform, unrelated to lookup success.
- **Caveat:** at step 2000 the cut is a net gain. The hypotheses were about the early penalty (steps ≤ 400), and those checkpoints were pruned (the earliest kept is step 1000).
- Bucketing by one model's own loss shows a strong but meaningless pattern (regression to the mean), so it is not used.

Data: `gnode118:/ssd_scratch/sparsh.bhartia/morbius/probe/` holds `probe_results.jsonl`, `dump_run{1,2}/` (per-token loss and Sink) and the model-only step-2000 files. Code: `ffnmem/probe_memory.py`, `slurm/run_probe.sh`, `slurm/analyze_dump.py`.

## Wave 2: idea A (`w2_mem_rank1`), launched 2026-09-27 19:44

**Design** (iso-parameter with run 1: 369,767,808 vs 369,767,936 non-embedding)
- **Layer 1:** Lngram v2 unchanged (R=90, dm=128), plus a per-entry rank-1 path on the same n-gram addresses.
  - Each row stores w, u ∈ ℝ³².
  - Output = U_out · Σ u · SiLU(w · P_in h) / √(R·orders), added to the v2 output.
  - U_out is zero-initialised, so at init this equals v2.
  - Rank-1 tables use the table recipe (4× LR, wd 0). They give no router gradient; the router trains through the v2 surrogate.
- **Layer 11:** v2 with dm=64 (half the table). This pays for the rank-1 tables. Probe: layer 11 is worth +0.02 nats vs +0.21 for layer 1.

**Comparison caveats**
- The comparison is to run 1, which ran on 2080 Ti with ZeRO-1 at micro-batch 2. This run is on 2× 3090 at micro-batch 4. Same data order and recipe otherwise.
- If it is clearly better, a later ablation separates "rank-1 added" from "layer 11 halved".

**New eval diagnostics:** `mem1/rank1_out_to_residual`, `mem1/v2_out_to_residual`.

**Progress (complete)**

| step | idea A val | run 1 (2080 Ti) | run 3 mem_cut_far | rank-1 out / residual | v2 out / residual (layer 1) | Sink layer 1 / 11 |
|---|---|---|---|---|---|---|
| 200 | 4.866 | 4.888 | 4.940 | 0.058 | 0.279 | 0.462 / 0.466 |
| 400 | 3.952 | 3.931 | 3.959 | 0.076 | 0.321 | 0.442 / 0.455 |
| 1000 | 3.235 | 3.234 | 3.215 | 0.078 | 0.313 | 0.458 / 0.457 |
| 2000 | 2.972 | 2.970 | 2.952 | 0.073 | 0.306 | 0.465 / 0.452 |
| 3000 | 2.850 | 2.847 | 2.832 | 0.069 | 0.314 | 0.469 / 0.451 |
| 3815 | **2.807** | **2.805** | **2.790** | 0.068 | 0.316 | 0.470 / 0.450 |

- Idea A tracks run 1 within ±0.003 from step 1000 on; the early lead (−0.022 at step 200) vanished.
- **Verdict:** no gain at this budget. Possible reasons, all untested:
  1. The rank-1 path is too small (32-dim) to matter.
  2. Its SiLU gate is not tied to v2's route weights.
  3. Halving layer 11 cost about as much as rank-1 gave. Ablation (a) in `important_notes.md` §5c would tell.
- Run 3's full curve is in the table above for reference: it tracks run 2 closely (2.952 vs 2.954 at step 2000; 2.790 vs 2.791 final).

The rank-1 path is in use from early on (6% of the residual norm, vs 28% for v2). The early lead is −0.022 vs run 1, but early numbers are noisy and cross GPU type.

**Launch notes**
- The first attempt (job 2719088) landed on gnode113, which exposes only 1 of its 2 allocated GPUs (`invalid device ordinal`). It was cancelled and gnode113 excluded.
- Env and data were staged from gnode114 (Sparsh's run 5 node).
- **Open decision:** a matched-GPU baseline (run 1 config on 2× 3090, e.g. gnode116) would remove the GPU-type confound and give a first noise estimate. It has not been launched.

Same codebase as wave 1 (`~/morbius`); the variant is just `configs/w2_mem_rank1.yaml`.
