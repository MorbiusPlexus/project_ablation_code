# Important notes: wave 1 in progress

Running observations and open questions. **Decision for now: no changes to any running job.** Let wave 1 finish, then act on what follows.

## 1. Loss spike in run 5 (`w1_dense_cut_near`), 2026-09-26

**What happened**
- Train loss fell to 6.49 at step 60, then rose to 6.66 (step 70) and 6.96 (step 80). The other runs kept falling over the same steps.
- Grad norm went from ~1 to 3.4 (step 60), then 14.5 (step 70), and was back to 1.2 by step 90. Clipping at 1.0 held the update size.
- Loss was back on trend by step ~110.

**Why it is not a bug**
- It happened in warmup, while the LR rose from 3.6e-4 to 4.8e-4.
- The fp16 scale stayed at 65536, so it was not an overflow.
- All runs see the same data at every step, and runs 1, 2 and 4 went through those steps cleanly. Run 4 had a milder grad-norm bump (6.4 at step 30) with no loss bump.
- Run 2 has the same widened attention (24 query heads, 4 KV heads) and did not spike.
- Assessment: ordinary single-seed warmup instability.

**The damage is washing out.** Train loss gap, run 5 minus run 4:

| step | gap |
|---|---|
| 200 | +0.38 |
| 300 | +0.25 |
| 400 | +0.15 |

**Decision:** observe. At steps 1000–1500, check whether the run 5 − run 4 gap has flattened.
**Option:** after run 4 finishes on gnode118 (~27 Sep 10:30), replicate run 5 there (4× 3090, ~15 h). It won't be bit-exact because the world size differs; that is fine for a second sample.

## 2. Early numbers (val loss on the fixed 20M-token slice)

| step (tokens) | run 1 mem_full | run 2 mem_cut_near | run 4 dense_full | run 5 dense_cut_near |
|---|---|---|---|---|
| 200 (105M) | 4.888 | 4.984 | 4.906 | 5.268 |
| 400 (210M) | 3.931 | 4.011 | 3.932 | 4.085 |

- **Cut cost with memory (run 2 − run 1):** +0.096 at step 200 and +0.080 at step 400. This is clean and shrinking slowly.
- **Cut cost without memory (run 5 − run 4):** +0.153 at step 400, but spike-contaminated.
- **Interaction (cut × memory): not readable yet.** "Memory halves the cut cost" is exactly the artifact the run-5 spike would produce. Wait until run 5's gap to run 4 is stable.
- **Memory vs dense at equal params (run 1 vs run 4):** 3.931 vs 3.932, no difference yet. That is expected this early; the Lngram v2 gain was only ~0.03 nats at 1.3B tokens.
- **Memory diagnostics at step 400:**
  - Effective codes are 14–15 of 16 in both memory runs. That is near uniform, so routing has not specialised yet; the paper reports ~8 later.
  - Sink mass at layer 1: run 2 0.39, run 1 0.44. The cut model sends more mass to real routes, so it leans on memory harder. That confirms the memory is compensating, but it does not discriminate between the hypotheses below.

## 3. Why the cut looks weak so far (and why an early gap is expected)

1. **Fixed cost vs growing benefit.** The cut removes capacity from step 0. The memory benefit only grows as the tables fill and the addresses settle.
   - At 210M tokens (5% of the run) the codes are near uniform. The tables are being written at moving addresses, which is the unstable-address problem Lngram v2 Appendix B describes for from-scratch training.
2. **Scale of the trade.** The cut takes 31.5M FFN parameters, ~12% of the backbone. For the thesis to hold, the memory must be worth more than the FFN it replaces in those specific layers. The whole-model memory gain in the paper is only ~0.03 nats, so this can only be judged late in training.
3. **Cut and destination are confounded.** The freed parameters went to attention widening, so runs 4 vs 5 measure cut + widen together. We can't yet tell whether the cut hurts or the destination is a poor trade at this scale. A "cut, don't widen" arm would separate them; it is not iso-parameter, but it is a useful diagnostic.
4. **Cut depth.** 45% of `d_ff` in each of 8 layers is a deep, concentrated cut. Losses from narrowing are usually convex in the cut fraction, so a milder cut spread over more layers may cost less for the same parameters.

## 4. Hypotheses and a test that tells them apart

**H1 (user's hypothesis): failed lookups.** Tokens whose lookup fails (high Sink mass: novel or rare n-grams) depend on the FFNs right after injection to rebuild local patterns. The cut takes that capacity away exactly where they need it.

**H2: integration.** The FFNs right after injection are needed to decode and integrate the memory output. Lngram's LogitLens shows layer 1 briefly worse than baseline before pulling ahead. The cut hurts tokens where memory did fire.

**Test** (post hoc, cheap, from checkpoints):
- Bucket val tokens by Sink mass at memory layer 1 (and 11).
- Compute the per-token loss gap (run 2 − run 1) per bucket.
- Predictions:
  - H1: the gap concentrates in the **high**-Sink buckets.
  - H2: the gap concentrates in the **low**-Sink buckets.
- Complement: bucket by training-set n-gram frequency (head vs tail).

## 5. Proposed variant: two fat FFNs after injection, then cut the next three

Keep the FFNs right after each insertion at full width, then cut the following 3 layers, e.g.:
- full width at 2–3 and 12–13
- cut at 4–6 and 14–16

This protects both the tokens that need rebuilding (H1) and the integration step (H2). Keep the cut parameter count equal by adjusting Δ: 6 layers instead of 8 means a larger Δ, or cut 4 layers per block.

**Decision rule, from run 3** (cut 6–9 and 16–19, queued behind run 1):
- If run 3 costs materially less than run 2, location matters. Give this in-between variant a wave-2 slot.
- If run 3 ≈ run 2, location isn't the lever; look at cut depth or destination instead.
- Don't finalise the variant until run 3 reports.

## 6. Wave-2 candidates (pick after wave 1)

- The two-fat-then-cut variant above (§5).
- **Milder cut:** Δ = 640 in the same layers (sensitivity), or spread over more layers.
- **Table destination:** same cut, freed parameters to the memory table (R ≈ 118). This is `notes_and_updates` run 7.
- **Cut without widening:** diagnostic, not iso-parameter. Separates the cut effect from the destination effect.
- **Cut profile from measurement:** use the per-layer FFN importance profile (§7).
- **Staged insertion:** add memory after 10–20% of tokens, to avoid the unstable-address phase (notes, post-midsem item 3).
- Optionally, a seed replicate of runs 2 and 5, since every comparison so far is single-seed.

## 7. Post-hoc analyses (after runs finish, no running job touched)

- **Per-layer FFN importance** (Contribution 2 of the proposal): ablate one FFN at a time and measure val loss, on run 1 and run 4 checkpoints at the same step. This shows where memory made FFN capacity redundant, and should set the wave-2 cut profile and size.
- **Sink-mass / n-gram-frequency bucketed loss gaps** (§4).
- **Address stability and code utilisation** across saved checkpoints.

**Logistics**
- Checkpoints are node-local:
  - run 1: gnode078, aditya
  - run 2: gnode082, sparsh
  - run 3: gnode078, aditya
  - run 4: gnode118, sparsh
  - run 5: gnode114, sparsh
- Comparing across runs needs a small staging job to bring checkpoints onto one GPU node. The first realistic window is gnode118 after run 4 finishes.
- Checkpoints to compare: steps 1000, 2000, 3000 (kept by retention) and the final one (3815).

## 8. Standing decision

No changes to running jobs (runs 1, 2, 4, 5 running; run 3 queued). All of the above is for after wave 1.
