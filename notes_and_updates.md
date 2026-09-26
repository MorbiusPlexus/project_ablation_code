# notes\_and\_updates

Sep 26, 2026 · @Aditya

## Summary

- Core idea survives: shrink FFN width where memory adds depth, move the parameters into a RAM-servable table.
- Switch the module from Lngram v1 to v2: same loss, 82.6% fewer module parameters, 95.2% fewer active; fixes the readout coupling.
- MMLU motivation is at chance level; reframe around validation loss and scale-appropriate benchmarks.
- "1.4% GPU memory" and "<3% offload" claims don't transfer to latent keys; offload is unmeasured and becomes your contribution.
- v2 already scores retrieved entries; narrow the scored-readout extension to hard top-k or drop it.
- From-scratch training is the least stable setting for discrete addresses; add a staged-insertion check.
- Init: random tables + zero output projection (v2 Appendix E). Drop the UltraMemV2 variance rule.
- Freed FFN parameters go to attention, not the table (the table would overshoot Lngram's \~25% memory optimum). Mid-submission: 5 runs in wave 1 forming a cut × memory 2×2 plus a location control, cosine schedule at 2–3B tokens, hypothesis-driven cut region. Rank-1 per-entry idea is the first post-midsem run.

## Sources read

All six core papers were read in full, plus the Lngram v1 and v2 code.

| Paper | What was read | Role for us |
| --- | --- | --- |
| [Lngram v2](https://arxiv.org/abs/2609.03426) (arXiv v2, 22 Sep 2026) | Full PDF incl. appendices | Module to build on; v1-vs-v2 text result; init and LR recipe |
| [Lngram v1](https://arxiv.org/abs/2605.24869) | Full PDF + [code](https://github.com/zyaaa-ux/Lngram) | Effective-depth evidence; 23-layer result; baselines |
| [Engram](https://arxiv.org/abs/2601.07372) | Full PDF | Hashed-key baseline; offload numbers; sensitivity probe |
| [X-GRAM](https://arxiv.org/abs/2604.21724) | Full HTML | Long-tail under-training; Engram at sub-1B scale |
| [UltraMemV2](https://arxiv.org/abs/2508.18756) | Full PDF text | Per-slot mini-FFN (PEER); activation density |
| [UltraMem](https://arxiv.org/abs/2411.12364) | Full HTML | Skip-layer (delayed) injection |

The Lngram v1 PDF from the GitHub repo is the May 2026 preprint. Only the arXiv v2 of the Lngram v2 paper was read; whether its appendices changed from the 3 Sep version is unknown.

## Framing-level changes

### 1. MMLU motivation

- Lngram v1 Table 1 MMLU: 0.2435 base, 0.2619 Lngram, 0.2300 23-layer. All near the 25% chance level.
- X-GRAM 0.73B–1.15B: 23–27. UltraMem, even 6.5B dense: 24–28.
- PIQA drop 0.6926 → 0.6740: 1.9 points on \~1,800 questions, under 2 SE. No paper reports seeds.
- Motivate with the LogitLens/CKA depth gain and "deletion removes attention too", not MMLU.
- Metrics: validation loss primary; ARC-Easy, SciQ, HellaSwag, cloze MMLU (MMLU-var), WinoGrande, PIQA secondary.

### 2. Lngram v2 as the module

- v1: R = d/4 fixed (256 at d=1024); readout ≈ 47% of table size, GPU-resident, active every token.
- v2: R free; W\_KV shared across routes; readout stays fixed as the table grows.
- Present the readout as inherited. Cite v2. Keep v1 for depth evidence: v2 has no LogitLens/CKA.

### 3. Offload and GPU-memory claims

- v1's 1.4%: parameter-matched MoE, tables on GPU, one H200. Not offload.
- Engram's <3%: throughput over 512 batched sequences on an H800, prefetch during block 1. Not batch-1 decode.
- Latent keys need the layer-2 hidden state, so rows can't be prefetched. No paper measures offload for latent memory.
- Rewrite as: unmeasured; measuring it is part of the contribution. Delayed injection becomes core, not optional.

### 4. Readout claims

- v2's GQA readout with zero-value Sink already scores every route.
- Scored-readout extension → hard top-k over routes only, or drop.
- Retitle: "Ranked Latent Memory" names the extension, not the contribution.

### 5. Where the freed FFN parameters go: attention, not the table

- Proposal Sec. 5.3 says "whatever width we remove goes into the table". Change this: freed parameters widen attention (inner dim 1024 → 1536, i.e. 24 query heads × 64; KV heads 2 → 4; same in every layer).
- Pooling into the table overshoots Lngram's own optimum: \~25% of sparse capacity in memory was best, \~35% was worse. A 100M table is already 27% of 365M; adding the freed parameters would push it to 37%.
- The claim becomes "memory shifts the optimal FFN:attention ratio", which matches both papers' argument that lookup frees attention for global context.
- Consequence: the reallocated parameters stay on the GPU. The RAM-offload result (Contribution 3) is still reported for the 100M table, but it is no longer the point of the reallocation. Reword Sec. 5.3 and Contribution 1.
- A table-destination twin of run 2 can be a wave-2 run if the original claim should also be tested.

## Technical corrections

Each item is a specific claim in the proposal that the papers or code contradict or qualify.

| Proposal claim | What the sources show | Fix |
| --- | --- | --- |
| Zero-init convolution keeps the branch from disturbing the backbone | The branch output is V + SiLU(conv(·)), so a zero conv zeroes only the conv sub-path. v1's code silences the branch by zero-initializing the tables | Use the v2 paper recipe (see next section) |
| Use UltraMemV2's variance-matching init | Conflicts with any rule that starts the branch silent. v2 zero-initializes the output projection | Drop the UltraMemV2 rule |
| Every token updates entries evenly, unlike hashed tables | Not guaranteed. v2 reports about 8 effective codes of 16 per route, top code about 30% | Log code utilization (effective codes, entropy, dead codes) |
| X-GRAM shows Engram worsening as its table grows at sub-1B | True at 0.73B (47.2, 47.7, 45.6 at 1×/2×/4×) but not at 1.15B, where 4× is best. Tables were 2.3–15B, many times the backbone | Soften, and note the different regime from a 100M table |
| Per-slot network uses a nonlinearity (as in UltraMemV2) | UltraMemV2 removed the activation; its PEER ablation also cut keys and top-m to hold bytes fixed | Drop the nonlinearity; hold bytes fixed by halving the memory dimension |
| Memory at layers 2 and 12; keep layers 0–2 full width | Code uses `target_layers=(1, 11)`, zero-indexed, which is the paper's 2 and 12 | Pick one indexing convention throughout |
| Lngram measures 1.4% extra GPU memory for its module | See framing change 3 | Rewrite |

Two further details:

- **Use the counterfactual surrogate gradient.** In the v2 paper, a straight-through estimator stalls at 7.33 validation loss versus 2.86 with the surrogate.
- **Engram's sensitivity test is a free probe.** Zeroing Engram at inference cut factual tasks to 29–44% of their score while reading comprehension kept 81–93%. Running the same test on your models complements the per-layer FFN-importance profile at no training cost.

## What the full Lngram v2 paper adds

### Table 4: v1 vs v2, text LM from scratch

| Model | Routes | Val loss |
| --- | --- | --- |
| Baseline (no memory) | – | 2.8899 |
| Lngram v1 | 192 (= d/4) | 2.8609 |
| Lngram v2 | 192 | 2.8527 |
| Lngram v2 | 64 | 2.8578 |
| Lngram v2 | 16 | 2.8584 |

- Setup (App. D.3): 0.6B DeepSeek-V2-style MoE, 24 layers, hidden 768, 8 routed + 1 shared experts; memory at layers 2 and 12; 10,000 steps ≈ 1.31B FineWeb-Edu tokens; same seed and data order.
- Supported: memory helps (≈0.03 nats); v2 at R=16 matches v1 at a fraction of the cost.
- Not supported: v2 beating v1. Gaps 0.008 and 0.0025, single seed; R=16 vs R=64 differ by 0.0006.
- Likely measured early (inferred): A.5's matching run stops at step 10,000 of a 61,036-step cosine schedule and reports 2.8585 ≈ R=16's 2.8584. LR was still high.

### Appendix B: unstable addresses while representations shift

- Mature Keye2B: +0.70 avg after 5B tokens. Freshly assembled SigLIP2+Qwen3-1.7B: +0.17 after 40B tokens.
- Cause: same pattern gets different codes over training. Layer-13 3-gram address survival 0.48 (new) vs 0.79 (mature). Memory zero-out changed top-1 in 0/48 docs (new) vs 6/48 (mature).
- Text addresses far more stable than visual (layer-13 code agreement 0.897 vs 0.916). Deeper insertion layer is the less stable one.
- Authors now insert v2 only mid-training. Evidence is two model families on 48 docs: suggestive, not conclusive.

### Appendix E: recipe (Keye2B)

| Setting | Value |
| --- | --- |
| Routes / memory dim | 121 / 128 |
| Readout Q heads / KV heads / head dim | 16 / 16 / 128 |
| Sink initial memory mass μ₀ | 0.5 |
| Table init | normal, std scale 1.0 |
| Output projection init | zero |
| Short conv | kernel 4, dilation 3, no bias, zero-init |
| Non-table LR peak / min | 1e-4 / 2e-6 |
| Table LR peak / min | 4e-4 / 1e-5 |
| Weight decay, tables and router | 0 |
| Surrogate gradient | approximate (one-bit) |
| Module size | 155M total, 21M active/token |

- Code defaults are the opposite init (`table_init_std_scale=0.0`, output proj std scale 1.0). Set both explicitly.
- STE instead of the surrogate: 7.33 val loss vs 2.86. Don't simplify the routing gradient.
- On-GPU decode overhead on H800: +5.6% (R=16) to +8.0% (R=64). Reference for your latency numbers.

## Run plan for mid-submission

**Updated 26 Sep with wave1\_agent\_plan:** config S stays (\~365M non-embedding: 265M backbone + 100M table). The cut is now 8 layers, not 14; freed parameters go to attention, not the table; the no-memory baselines are iso-parameter at 365M; block deletion moves to wave 2.

### Sizing (config S, d=1024, 24 layers)

- Table rows per layer = R × (16² + 16³) = R × 4,352. R=90, dm=128 over two layers ≈ 100M table. R=180, dm=64 is the same size.
- Start at R=90, dm=128 (closest to the tested recipe). v2 ablations: routes beat memory dim; Table 4: route count barely matters at 0.6B.
- Readout ≈ 3M/layer with 16 × 64 heads (estimate from the architecture, not a paper number).
- Insertion layers 2 and 12 (1-indexed), as in both Lngram papers' 24-layer text setups.

### Required before wave 1 (≈ 1 day)

1. Data pipeline (ClimbMix tokenized, fixed order).
2. Instantiate the v2 module at config S and count parameters; set R so every run is exactly iso-parameter.
3. Smoke test: v2 for \~200M tokens, loss must drop below a memory-free run. Fallback: v1 single-table + approximate gradient.

### Rules for all runs

- Same seed and data order everywhere (paired comparison, as Lngram v2 did). Small single-seed differences are then interpretable.
- Budget: 2–3B tokens per run (\~5–7 tokens/param); the proposal's \~8B is deferred. Precedent: Lngram v2 Table 4 used 1.31B tokens on a 0.6B model.
- Cosine schedule, length set to the planned budget, identical for every run. If runs must be cut short, stop all at the same step and compare there.
- Caveat: undertrained models are data-limited, so FFN cuts cost less than at convergence. Label results "at N tokens". The wrong-region cut is affected equally, so that comparison stays fair.
- Cut region from v1's evidence now (layers 3–16); the importance profile validates later. "Measurement decides" becomes "measurement confirms".
- Same precision for every run (fp16 everywhere, or at least key pairs on the same hardware type).
- Report total params, active params/token, FLOPs, GPU memory for every run.

### Wave 1: five runs, about a day each at 2B tokens, all ≈365M non-embedding

Layers 0-indexed; memory at layers 1 and 11. "Cut" = d\_ff reduced by 1,280 in the named layers (2816 → 1536 for memory runs; 4160 → 2880 for the dense runs). The 31.5M freed always goes to the same attention widening (inner dim 1024 → 1536, KV heads 2 → 4, all layers), so the destination is identical in every cut run.

| Rank | Run | Cut layers | Memory | Why |
| --- | --- | --- | --- | --- |
| 1 | `mem_full` | none | 100M | Baseline to beat |
| 2 | `mem_cut_near` | 2–5, 12–15 (the 4 layers after each insertion) | 100M | Main claim: memory frees FFN capacity right after injection |
| 3 | `mem_cut_far` | 6–9, 16–19 (same cut, 4 layers later) | 100M | Location control: does *where* matter |
| 4 | `dense_full` | none (d\_ff 4160 everywhere to reach 365M) | none | Iso-param dense baseline; also the checkpoint for the per-layer importance profile |
| 5 | `dense_cut_near` | 2–5, 12–15 from the 4160 base | none | With 1, 2, 4: cut × memory 2×2. Interaction = (5 − 4) vs (2 − 1), unconfounded by intrinsic layer importance |

Notes: runs 4 and 5 cut from a wider base (31% vs 45% of d\_ff), so the cut fraction differs; the moved amount and destination are the same. Runs 1 and 2 are the key pair; keep them on the same GPU type.

### Wave 2 (if time before mid-submission)

| Rank | Run | Why |
| --- | --- | --- |
| 6 | Block deletion, iso-param: remove \~3 whole blocks (\~11M each) after the insertions, freed → attention | Proposal Sec. 7 core run ("uniform layer deletion"), the alternative argued against. Pick blocks using run 4's importance profile |
| 7 | `mem_cut_near` with freed → table (R ≈ 118) | Tests the proposal's original destination; expected to overshoot the memory sweet spot |
| 8 | Location control matched on dense importance: cut 8 layers whose FFN importance in run 4 equals that of 2–5, 12–15 | Cleaner than run 3 if run 3 is ambiguous |
| 9 | Second cut depth (Δ = 640 or 1,920 in the same 8 layers) | Sensitivity |

FFN-size controls are runs 3, 5 and 8. Engram baseline: skip for mid-submission; hashed-vs-learned keys is a separate question, and X-GRAM plus Lngram v1 give citable comparisons.

### After mid-submission, in order

**Before planning post-midsem work, read the full original proposal (Morbius-Proposal) alongside this doc.** The list below is a starting order, not the whole scope. Re-check every proposal idea against mid-submission results:

- Targeted FFN reallocation and the cut-location measurement (pre-run profile on an existing model, post-run importance profile).
- GPU memory and batch-1 decode latency reporting, with the table in host RAM.
- Optional extensions: per-slot networks (the rank-1 idea), delayed injection, scored readout.
- Sweeps: cut amount, cut profile (even vs deeper near insertion points), memory share of the budget.
- Baselines: dense at same parameters and at same validation loss; Engram at same parameters.
- Analyses: LogitLens and CKA, as in Lngram.
- Frozen-backbone domain fine-tuning (PubMed → MedQA/PubMedQA).
- Longer-term: hybrid table with a hash-keyed region for editable facts and a latent-keyed region for patterns.
- Stretch config M (d=1280, 28 layers).

1. **Rank-1 per-entry memory (high priority).** Two stored vectors per entry, key and value, instead of one vector through shared W\_KV. Equal bytes: dm 64+64 vs shared 128. No nonlinearity (UltraMemV2 removed it). Novelty over v2 = separate learned key per entry.
2. Delayed injection Δ sweep, with the offload prototype.
3. Staged insertion: memory from step 0 vs after 10–20% of tokens.
4. v1 vs v2 at config S, full schedule.
5. Engram baseline.

## Static experiments (parallel to training)

Only the first two rows block a run.

| Experiment | Needs | When |
| --- | --- | --- |
| Module parameter and FLOP count | Code | Before wave 1 (required) |
| Smoke test, \~200M tokens | Code + data | Before wave 1 (required) |
| Offload latency prototype: batch-1 decode, table on GPU vs host RAM, with/without Δ, on 3090 and 2080 Ti | Architecture, random weights | Any time |
| Eval harness incl. cloze MMLU, resampling | None | Before mid-submission eval |
| Address stability (code agreement, 2/3-gram survival) and code utilization | Saved checkpoints | During training |
| Per-layer FFN importance (ablate FFN per layer, measure loss) | Checkpoints | During training, on run 1 |
| Engram-style memory zero-out, by task and layer | Checkpoints | End |
| LogitLens and CKA vs dense | Two checkpoints | End |
| Per-token loss split by n-gram frequency (head vs tail) | Checkpoints + n-gram counts | End |

## Risk: memory miss with a narrowed FFN

- Width reduction keeps attention and a narrower FFN in every layer; insertion layers 2 and 12 stay full width. The "no FFN nearby" failure belongs to deletion, not this design.
- Real risk: memory covers head patterns; narrowed FFNs have less capacity for the tail. Expect losses concentrated on rare n-grams.
- Lngram degrades more gracefully than Engram: 16 codes per route are coarse, so unseen patterns share codes with seen ones; keys carry attention context; the Sink makes "nothing useful retrieved" explicit. Engram's hash gives a miss or a collision on unseen n-grams.
- Layers 3–11 also consume the layer-2 memory output. Narrowing them cuts both reconstruction and use.
- Diagnostics: Sink mass per token (how often memory is effectively off); head/tail loss split; per-layer importance.
- Mitigation if it bites: 25% cut, or keep 3–4 more layers full around each insertion.

## Smaller inconsistencies

These are internal to the proposal and quick to fix.

- **Title.** "Ranked Latent Memory" refers to the scored readout, an optional extension that v2 now largely covers. Name the main contribution, FFN reallocation.
- **Dense baseline priority.** The evaluation plan lists the plain dense Transformer as "as time allows", but the timeline schedules dense reference runs in weeks 3–4.
- **Compute accounting.** The FLOP estimate uses about 265M non-embedding parameters, while Table 1 gives S a total of about 400M including a 100M table. State whether "20 tokens per parameter" counts the table, and use the same count for the FLOP estimate.
- **Table 1 memory size.** With v2, report table and readout parameters separately. The readout is GPU-resident and active per token; the table is not.
- **Related work.** Add Lngram v2 to Section 4; it is now load-bearing. UltraMem is already cited, and the delayed-injection citation checks out (it takes memory input at one layer and adds it back a few layers later).

## Open questions

- [ ] Did Table 4's runs stop at step 10,000 of a 61,036-step schedule? Inferred from the A.5 ablation's matching setup and loss; ask the authors or check the code if it matters.
- [ ] Does v2 preserve v1's effective-depth profile (middle-layer CKA gain)? Unmeasured; your per-layer importance profile will test it.
- [ ] Is my count of v1's readout right? From the code it is about 151M activated parameters in the 2B model, but v1's paper says about 40M extra. Instantiate and count.
- [ ] How much does host-RAM serving cost at batch-1 decode on the 3090 and 2080 Ti machines, with and without delayed injection?
- [ ] How long does the backbone need before addresses stabilize in a from-scratch text model? The staged-insertion check answers this.
- [ ] Did the appendices change between Lngram v2's arXiv v1 (3 Sep) and v2 (22 Sep)? Only v2 was read.
