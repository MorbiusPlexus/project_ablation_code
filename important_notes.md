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

**The damage washed out.** By step 1000 run 5 was ahead of run 4 (numbers in `state_and_results.md`).

**Decision:** observe. Resolved at step 1000: run 5 recovered fully.
**Option:** after run 4 finishes on gnode118 (~27 Sep 10:30), replicate run 5 there (4× 3090, ~15 h). It won't be bit-exact because the world size differs; that is fine for a second sample.

## 2. Numbers

All losses, diagnostics and probe results live in `state_and_results.md`. This file is for things to look at later.

## 3. Why the cut looked weak early (step ≤ 400)

> **Update (step 1000+):** the cut runs overtook their uncut baselines, and memory has so far added nothing over dense (see `state_and_results.md`). Keep the points below for the write-up: they explain the early penalty.

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

## 5a. DECISION POINT: run 3 vs run 2 (recorded 2026-09-27)

**When:** run 3 (`w1_mem_cut_far`, job 2717537) starts on gnode078 after run 1 finishes (~27 Sep 21:30). Its first comparable evals follow; the final call is at matched steps (1000 / 2000 / final 3815).

**What is compared:** val loss at matched steps.
- run 3 − run 1: cut 4 layers later (6–9, 16–19)
- run 2 − run 1: cut right after injection (2–5, 12–15)

Both are on the 2080 Ti with identical recipes, so only the cut location differs.

**Rule:**

| outcome | meaning | action |
|---|---|---|
| run 3's penalty clearly smaller than run 2's (difference well above ~0.01–0.02 run-to-run noise) | location matters | wave 2 includes the user's architecture (below) as the main candidate |
| run 3 ≈ run 2 | location isn't the lever | wave 2 moves to cut depth (Δ=640 / spread cut) or destination (freed → table) instead |
| run 3's penalty larger than run 2's | layers right after injection are the *cheaper* place to cut, so the near-cut hypothesis is supported | tune the near cut (depth/profile) rather than move it |

**Also read before deciding:** the Sink-mass bucketed loss gap (§4) and the run 5 − run 4 gap once it has stabilised (§1).

## 5b. User's proposed architecture: "two fat FFNs after injection, then cut" (recorded 2026-09-27)

**Idea:** after each memory insertion, keep the next two FFNs at full width, as a safety net for tokens whose lookup failed (H1) and to integrate the memory output (H2). Then cut the FFN in the following three layers.

**Layer map** (24 layers, 0-indexed, memory at 1 and 11):

| layers | FFN |
|---|---|
| 0–1 | full (1 carries memory) |
| 2–3 | **full**: the two fat FFNs after injection 1 |
| 4–6 | **cut** |
| 7–11 | full (11 carries memory) |
| 12–13 | **full**: the two fat FFNs after injection 2 |
| 14–16 | **cut** |
| 17–23 | full |

**Parameter matching:** same total cut as wave 1 (31,457,280) with the same destination (attention widening to 24 q / 4 kv heads), so it stays iso-parameter with runs 1–5.
- **3 layers per block (as proposed):** 6 cut layers, so Δ ≈ 1707. Rounded to a multiple of 32: **Δ = 1696**, d_ff 2816 → **1120** (60% cut per layer). It frees 196,608 fewer parameters than wave 1, so the total is +0.053% vs run 1 (checked with `ffnmem.params`), well inside the 0.5% tolerance.
- **Caveat:** 60% per layer is much deeper than wave 1's 45%. If §3.4 (narrowing losses are convex in depth) holds, this could cost more than the location saves.
- **Alternative at wave-1 depth:** 4 layers per block, cut 4–7 and 14–17 with Δ = 1280 (d_ff 1536). Exact parameter match. This isolates "location" from "depth" relative to run 2.

**Config sketch** (not created yet; add in wave 2):
```yaml
# configs/w2_mem_fat2_cut3.yaml
extends: w1_mem_full.yaml
name: w2_mem_fat2_cut3
params: {reference: false}
model:
  n_heads: 24
  n_kv_heads: 4
  ffn_cut: {layers: [4, 5, 6, 14, 15, 16], delta: 1696}
```
Its dense twin (same cut on the 4256 base, no memory) is needed to read the cut × memory interaction, as with runs 4 and 5.

## 5c. User's memory-readout ideas (recorded 2026-09-27)

### Idea A: per-entry rank-1 matrix *added to* the v2 readout, not replacing it

Keep the Lngram v2 readout unchanged: shared W_KV, GQA over routes, Sink, conv.
- Each table row additionally stores two vectors, u_i and w_i.
- Together they act as a tiny rank-1 matrix A_i = u_i w_iᵀ applied to the hidden state: out_i = u_i · φ(w_iᵀ h).
- The layer output becomes: v2 readout + Σ_i g_i · u_i φ(w_iᵀ h).
- Here g_i is the v2 route weight (reusing its softmax, so the Sink still switches it off), or a separate gate.

**Why add it rather than replace v2:**
- The shared W_KV keeps its role as a common, well-trained decoder.
- The rank-1 term only has to carry what the shared projection can't: a key/value decoupled per entry, and a per-entry transformation of h rather than a fixed vector.
- It starts as a pure add-on. Zero-init u_i, so the model at init equals v2.

**Rank ladder:** A_i = U_i W_iᵀ with k columns (rank k) is a k-hidden-unit mini-FFN per entry (PEER / UltraMemV2 style). Try k = 1 first, then raise it if it helps.

**Bytes:** 2·r·d per entry, with r = rank and d = the dimension u and w live in (e.g. 64 in a per-head space). At equal total bytes this must be paid for by a smaller table (fewer routes or smaller mem_dim), or reported as extra parameters.

**Nonlinearity φ:** SiLU on w_iᵀ h, or none. UltraMemV2 found no activation acceptable. With g_i from the softmax, the extra term is already gated, so φ is optional.

### Idea B: shared low-rank matrix per group of nearby entries, combined with a per-entry rank-1 matrix

**Structure:**
- A rank-~10 matrix S_g is shared by a small group g of "nearby" entries.
- Each entry also has its own rank-1 part u_i w_iᵀ.
- The shared part captures what the neighbourhood has in common; the rank-1 part specialises each entry. This is a hierarchical "shared base + per-entry delta", like LoRA per entry on top of a group basis.

**Two readings of "multiplied":**
1. **Additive:** A_i = S_g + u_i w_iᵀ. The effective rank is up to 11; group features plus an entry-specific direction.
2. **Composed:** out_i = u_i · (w_iᵀ S_g h). The shared S_g first maps h into a group-specific feature space, then the entry's rank-1 reads one direction out of it. The effective rank per entry stays 1, but it lives in a learned group subspace.
   - Note: a plain product of a rank-10 and a rank-1 matrix has rank ≤ 1. So the composed form gains expressiveness through S_g's input transform, not through rank.

**What "nearby" can mean** (all natural in Lngram addressing):
- Same route and same (n−1)-gram prefix: siblings that differ only in the newest token's code.
- Codes within Hamming distance 1. These are exactly the neighbours the counterfactual surrogate compares, so sharing may also smooth the router's gradient.
- The same code across the 2-gram and 3-gram tables of a route.

**Bytes per entry:** 2·10·d / G for the shared part (G = group size) + 2·d for the rank-1 part. Grouping is what makes rank 10 affordable.

**Why it might help:**
- Rarely hit entries borrow statistical strength from their group through S_g.
- Frequent entries still get a private direction.
- This addresses the main risk of pure per-entry parameters (noisy, under-trained rows) while keeping specialisation.

**Order to test:**
1. Idea A with rank 1, added to v2 (the cheapest clean arm).
2. Idea A with rank k.
3. Idea B, additive form, grouping by (n−1)-gram prefix.

Each needs an equal-bytes or clearly labelled extra-parameter comparison against v2.

## 6. Wave-2 candidates (pick after wave 1)

- The two-fat-then-cut architecture (§5b), gated by the run 3 decision point (§5a).
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
