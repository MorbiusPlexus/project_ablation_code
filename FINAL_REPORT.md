# Morbius: evaluation, fine-tuning and memory interpretability of seven 400M-parameter models

This report covers everything that was run on the seven final Morbius checkpoints: general language-model evaluation, long-context and factual-recall probes, task fine-tuning under different trainable-parameter regimes, PubMed continued pre-training, and a set of interpretability analyses of the Lngram v2 memory. For each part it states the aim, how it was done, the variants, the results, what they mean, and the limitations.

Raw tables: `RESULTS_ADA.md` (every number) and `STATS_ADA.md` (every statistical test). Both are generated from the result JSON files by `eval/aggregate_ada.py` and `eval/stats_ada.py`.

---

## 0. Main conclusions

1. **At equal total parameters, adding the Lngram memory neither helps nor hurts general ability.** Across 14 zero-shot tasks, 8 few-shot tasks run with the Engram paper's shot settings, and 2 five-shot tasks, the memory models and the dense baselines are within about 1 point of each other on average. The only differences that survive multiple-comparison correction are small: memory models are better on BLiMP grammaticality (+0.6 to +1.7 points), the cut models are better on LAMBADA (+1.8 to +2.0), and most models differ on zero-shot BoolQ, where the dense baseline sits at the majority-class rate.
2. **The models depend heavily on their memory, almost entirely at layer 1.** Replacing the memory output with zeros raises validation loss by 0.18 to 0.99 nats; replacing it with its mean raises it by 0.15 to 0.33. Layer 1 accounts for nearly all of it, and layer 11 costs only 0.01 to 0.02 nats. The models that keep their full FFN width (R1 and A) depend on memory about twice as much as the models whose FFNs were cut to widen attention (R2, R3 and S1).
3. **What the memory does is local and lexical.** Its route codes are an almost deterministic function of the current token. Its effect on predictions is concentrated at the current position and the next 1–2 positions, it matters most for very frequent n-grams, newlines and word continuations, and its busiest table rows store formatting n-grams such as `.\n\n` and `\n\nQuestion`. Activation patching shows that it moves the identity of a token forward by 1–2 positions without going through attention, which is exactly what n-gram addressing implies.
4. **The memory also carries some semantic information, but no more than the router that addresses it.** A nearest-neighbour search on route codes alone recovers about 25% (layer 1) to 45% (layer 11) of the entity-type information in the hidden state above a position baseline (permutation p = 0.005). Several hundred code↔entity-type associations replicate on held-out data. The codes do as well as the continuous router logits, so quantizing to codes loses almost nothing.
5. **Memory supports factual recall at the subject.** On CounterFact, zeroing memory loses 12 points of recall in R1 and A and 3 to 4 points in R2 and S1 (McNemar p < 0.001), and nothing significant in R3. Zeroing memory only at the subject tokens accounts for most of the loss. Zeroing it only at the last subject token or the last prompt token does almost nothing, so the contribution is spread over all subject tokens.
6. **Memory-only fine-tuning is a weaker parameter-efficient method than tuning an equally sized FFN subset.** With 105.5M trainable memory parameters at its best learning rate, it trails full fine-tuning by 1.9 points (SST-2), 2.7 (BoolQ) and 3.7 (NER F1). An FFN subset of the same size, at its own best learning rate, is within 0.4 to 0.8 points of full fine-tuning. Some of the gap seen in the first single-seed runs came from the learning rate, and the controls separate that out.
7. **For domain adaptation, memory-only continued pre-training is the least effective option.** On PubMed it lowers in-domain loss the least (−0.09 to −0.16 nats, versus −0.25 to −0.32 for an FFN subset of similar size and −0.43 for full training). Its forgetting of general text is low, but not lower than the FFN subset's, so it sits below the FFN subsets on the trade-off between domain gain and forgetting.
8. **Long context:** with memory intact, all seven models are similar on mini-RULER (means 0.31 to 0.38, with 30 items per cell). Zeroing memory destroys long-context retrieval in R1 (0.32 → 0.02) and damages R3 and S1 badly. R2 is barely affected (0.38 → 0.33), even though R2 and R3 lose similar amounts of validation loss under the same ablation.

---

## 1. Models and setup

### 1.1 The seven checkpoints

All models share one recipe: a Llama-style decoder with 24 layers, d_model 1024 and RoPE, trained on 2.0B ClimbMix tokens (3815 steps of 256 × 2048 tokens, AdamW at lr 6e-4 with cosine decay, fp16). The tokenizer is byte-level BPE with 32,768 tokens. Every model has about 370M non-embedding parameters, so all comparisons are iso-parameter.

| id | config | memory | FFN | attention | what it tests |
|---|---|---|---|---|---|
| R1 | `w1_mem_full` | layers 1 and 11, 90 routes, 128-dim rows | d_ff 2816 everywhere | 16 q / 2 kv heads | memory added; memory reference model |
| R2 | `w1_mem_cut_near` | as R1 | d_ff cut by 1280 in layers 2–5 and 12–15 (just after each memory layer) | widened to 24 q / 4 kv heads | memory plus FFN cut, with the freed parameters moved into attention |
| R3 | `w1_mem_cut_far` | as R1 | cut in layers 6–9 and 16–19 (far from memory) | 24 q / 4 kv | whether the cut location matters |
| R4 | `w1_dense_full` | none | d_ff 4256 everywhere (widened to match R1's parameter count) | 16 q / 2 kv | iso-parameter dense baseline |
| R5 | `w1_dense_cut_near` | none | R4 with R2's cut | 24 q / 4 kv | dense version of R2, isolating the effect of memory under the cut |
| A | `w2_mem_rank1` | R1 plus a rank-1 path at layer 1; layer-11 table halved | as R1 | as R1 | "idea A": a richer layer-1 memory |
| S1 | `w2_mem_super1` | layer 1 only, 184 routes | as R2 | 24 q / 4 kv | "super-memory": all memory capacity at layer 1 |

Final ClimbMix validation losses from pre-training: R1 2.805, R2 2.791, R3 2.790, R4 2.802, R5 2.774 (best), A 2.807, S1 2.788 (best memory model).

### 1.2 What the memory is (Lngram v2)

The memory sits before attention in its block: `h = h + mem(h, norm(h))`. Inside it:

1. **Router.** A linear router reads the normalized hidden state and outputs R routes × 4 bits. Each route's bits are hard-quantized into a code from 0 to 15, so every token gets R discrete codes.
2. **Addressing.** For each route, the codes of the last 2 and last 3 positions (n-gram orders 2 and 3) form an address into a table: 16² or 16³ rows per route, each row a 128-dim vector. A token's memory content therefore depends on the current token and the 1–2 tokens before it, but only through their codes.
3. **Readout.** The looked-up rows are read by a small grouped-query attention over routes. A fixed zero-valued "sink" competes with the real rows, so the memory can effectively say "nothing useful here". Sink mass starts at 0.5, and a short causal depthwise convolution adds local context.
4. **Gradients.** The tables train with 4× the base learning rate and no weight decay. The router trains through a straight-through-style surrogate.

Memory adds about 105M parameters in total, but because each token reads only a few rows, a memory model uses about 27% fewer active parameters per token than R4.

**Lngram, not Engram.** Every memory model here uses Lngram v2. The Engram paper (conditional memory through hashed n-gram embeddings) is used only as a source of evaluation protocol: its Table 1 shot settings (the `paper` track) and its sensitivity analysis, which suppresses the memory output (the `zero` ablation). Engram makes several claims about n-gram memory in general: that it helps long-context retrieval, stores factual knowledge and frees depth for reasoning. This report tests those claims on Lngram; none of the results are about Engram's own architecture.

### 1.3 Interventions used throughout

Every memory analysis compares the model with memory intact (`none`) against an intervention on the memory output, implemented in `ffnmem/hooks.py` (`Intervention`):

- `zero`: replace the memory output with zeros at every memory layer. This is the Engram paper's "sensitivity" test. It is off-distribution, because downstream layers never saw an all-zero memory.
- `mean`: replace the memory output with its average over held-out validation tokens, per layer. This removes token-specific information but keeps the average contribution, so it is the cleaner measure of what the memory *knows*.
- `zero@i` / `mean@i`: the same, but only at memory layer i (1 or 11).
- Position-restricted versions: replace the output only at chosen positions. CounterFact localisation and the locality analysis use these.

### 1.4 Infrastructure

Everything ran on the ada cluster (Slurm), on RTX 2080 Ti, RTX 3090 and RTX 4070 GPUs spread across 6 nodes and 4 accounts. A file-based job queue with one worker per GPU ran about 330 jobs, roughly 80–90 GPU-hours in total. Results were merged into one tree and the tables regenerated by script. All lm-eval tasks use lm-evaluation-harness 0.4.9 through a custom adapter (`ffnmem/lm_eval_adapter.py`) that supports the interventions.

---

## 2. General language-model evaluation

### 2.1 Aim

To find out whether adding the memory (at equal parameters) changes what the model can do, and how much the model relies on the memory for each kind of ability.

### 2.2 How it was done

Five tracks, each run for every model with memory intact, and for the memory models also under `zero` and `mean`:

| track | content | items | metric |
|---|---|---|---|
| `val` | ClimbMix held-out validation windows (2048 tokens) | fixed val slice | loss, bits per byte, next-token accuracy |
| `synthetic` (`ffnmem/synthetic.py`) | **induction**: random tokens, then a 64-token random span repeated; score = accuracy on the second copy. **MQAR**: 16 random key→value pairs, then 8 queried keys. **passkey**: a 5-digit key hidden in filler text at depth 10%, 50% or 90%, for lengths 256, 1024 and 2000 tokens | 64 / 128 / 20 per cell | accuracy |
| `zeroshot` | HellaSwag, PIQA, WinoGrande, COPA, Social-IQA, ARC-e, ARC-c, SciQ, OpenBookQA, MMLU (continuation format), LAMBADA, BoolQ, RACE, BLiMP, WikiText | full test sets | acc / acc_norm; WikiText bits per byte |
| `paper` | shot settings from the Engram paper's Table 1, applied to our models: MMLU 5-shot, ARC-e/ARC-c 25-shot, TriviaQA 5-shot, BBH 3-shot, WinoGrande 5-shot, DROP 1-shot, RACE 5-shot; prompts left-truncated to 2048 tokens | 300 per task | acc / exact match / F1 |
| `fewshot` | ARC-e and HellaSwag 5-shot | 300 per task | acc_norm |

Tasks a 400M model scores at chance on (GSM8K, MATH, HumanEval, MBPP, MMLU-Pro, Chinese benchmarks) are listed as `floor` in `eval/tasks.py` and were not run.

**Statistics.** Per-item scores are aligned across runs, and the difference between two runs is bootstrapped (2000 resamples) for a 95% CI. The p-value comes from the bootstrap standard error, Holm-corrected within each table (about 230 tests in the zero-shot table). A first version took p from the empirical bootstrap tail. That floors p at 1/2000, which after Holm correction makes nothing significant, so it was replaced.

### 2.3 Results with memory intact

Mean over the 14 zero-shot accuracy tasks (WikiText excluded):

| R1 | R2 | R3 | R4 | R5 | A | S1 |
|---|---|---|---|---|---|---|
| 0.456 | 0.450 | 0.460 | 0.460 | 0.461 | 0.460 | 0.453 |

These differences of under 1 point are not meaningful. Of 154 model-versus-model comparisons (11 pairs × 14 tasks), these survive Holm correction:

| finding | comparisons | size |
|---|---|---|
| Memory models better on BLiMP (syntax) | R1, R2, R3, A, S1 vs R4; R2, R3 vs R5; S1 vs R2 | +0.6 to +1.7 points |
| BoolQ differs | every memory model is below R4 (−1.5 to −10.4); S1 and A are above R2 | R4 scores 0.622, which is the "yes" rate of BoolQ dev |
| Cut models better on LAMBADA | R2 vs R4 +2.0, R5 vs R4 +1.8; R1 and A below R2 (−1.9, −2.0) | about 2 points |
| R5 better on SciQ than R4 | | +4.3 |

On the paper protocol, the only corrected differences are that R4 beats R2 and R5 on 5-shot MMLU (+2.0 and +2.1 points; all models are at 0.26–0.28, close to the 0.25 chance level), and R5 beats R3 on BBH (+2.3). TriviaQA (0.02–0.04) and DROP (0.02–0.03) are at the floor for every model. Five-shot ARC-e and HellaSwag show no corrected differences.

**Interpretation.** At this scale and budget, the memory is a substitute for FFN capacity, not an addition. That matches the pre-training losses, where R1 is 2.805 and R4 2.802. The consistent BLiMP advantage is the one ability-level signal. It fits the memory being a store of local n-gram statistics, because BLiMP tests local grammatical agreement. BoolQ zero-shot at this scale mostly measures a yes/no bias: no model is far above the 0.62 majority rate, so its differences should not be read as reading-comprehension differences.

### 2.4 How much the models depend on memory (sensitivity analysis)

ClimbMix validation loss increase under each ablation (nats):

| model | zero (both layers) | mean (both) | zero@1 | mean@1 | zero@11 | mean@11 |
|---|---|---|---|---|---|---|
| R1 | +0.985 | +0.325 | +0.635 | +0.227 | +0.020 | +0.020 |
| A | +0.829 | +0.334 | +0.561 | +0.221 | +0.022 | +0.022 |
| S1 (layer 1 only) | +0.311 | +0.145 | – | – | – | – |
| R3 | +0.259 | +0.212 | +0.184 | +0.147 | +0.014 | +0.014 |
| R2 | +0.184 | +0.146 | +0.113 | +0.087 | +0.014 | +0.013 |

Zero-shot mean (14 tasks), intact → zero → mean: R1 0.456 → 0.395 → 0.437; A 0.460 → 0.394 → 0.434; R3 0.460 → 0.442 → 0.450; S1 0.453 → 0.429 → 0.450; R2 0.450 → 0.436 → 0.444.

Tasks hit hardest by zeroing (accuracy points; all Holm-significant):
- **LAMBADA**: −20 (R1, A), −11 (S1), −8 (R2), −7 (R3).
- **BLiMP**: −18 (A), −17.5 (R1), −11 (S1), −8 (R2), −7 (R3).
- **SciQ**: −11 (A), −9 (R1).
- **Few-shot BBH** (generation): −12 (R2) to −23.5 (R3, S1). R3 falls from 0.243 to 0.008, S1 from 0.258 to 0.023.
- **DROP**: −2.6 to −3.0 (R1, A, S1).

Knowledge-heavy multiple-choice tasks (MMLU, ARC-c, WinoGrande) barely move. In R2 and R3, zero-shot BoolQ even *improves* under zeroing (+6.8 and +3.2), which most likely shifts the yes/no bias rather than showing real comprehension.

Synthetic tasks (intact → zero → mean):
- **Induction**: R1 0.87 → 0.00 → 0.27; R2 0.85 → 0.01 → 0.63; R3 0.85 → 0.19 → 0.66; A 0.88 → 0.00 → 0.08; S1 0.86 → 0.00 → 0.37. Induction is an attention mechanism, so this shows that the attention heads doing induction read features the layer-1 memory writes. The mean ablation recovers much of it in the cut models, much less in R1 and A.
- **MQAR**: R1 0.62 → 0.02; A 0.61 → 0.01; S1 0.55 → 0.12; R2 0.45 → 0.25; R3 0.56 → 0.48. Intact, R4 scores 0.50 and R5 0.58.

**Interpretation.**
1. Layer 1 carries almost everything; layer 11 can be removed at inference for 0.01–0.02 nats. This matches the earlier step-2000 probe and the Lngram v2 appendix.
2. Widening attention (R2, R3, S1) roughly halves the dependence on memory. Wider attention seems to take over part of the local-pattern work memory does in R1, which also explains why the FFN cut helps dense models twice as much as memory models in pre-training.
3. The gap between zero and mean ablation is large for R1 and A (0.99 vs 0.33) and small for R2 and R3 (0.18 vs 0.15). So R1 and A have learned to rely on the memory's *average* output as a bias: removing it entirely is much worse than removing only its token-specific part.
4. Memory loss hurts generation and cloze tasks (LAMBADA, BBH, DROP) and syntax (BLiMP) far more than multiple-choice knowledge tasks.

---

## 3. Long context

### 3.1 Passkey (within the synthetic track)

At 256 and 1024 tokens most models retrieve the key (0.73 to 1.00). At 2000 tokens nearly every model and condition scores exactly 0.67, or 0.65 for R4, meaning one of the three depths always fails. Because it is identical across architectures and ablations, it is most likely a prompt-construction or truncation artifact at 2000 tokens, not a model property. R5 is the exception (0.75 at 1024, 0.33 at 2000). Zeroing memory does not reduce passkey accuracy and sometimes raises it at 256 tokens, so passkey is not memory-dependent.

### 3.2 Mini-RULER

**Aim.** The Engram paper reports that n-gram memory improves long-context retrieval (RULER). This tests whether the same holds for Lngram, at the scale these models can handle.

**How it was done** (`eval/mini_ruler.py`). RULER (Hsieh et al. 2024) scaled to a 2048-token model and built with the project tokenizer. There are 7 tasks:
- `niah_single`: one key–value "needle" in filler text.
- `niah_multikey`: the needle plus 3 distractor needles with other keys.
- `niah_multivalue`: one key with 2 values, both of which must be produced.
- `niah_multiquery`: 2 keys asked at once.
- `vt` (variable tracking): chains of assignments; list every variable equal to a given value.
- `cwe` (common-word extraction): the 3 most frequent words in a list.
- `qa_squad`: a SQuAD question whose gold paragraph is hidden among distractors.

Each task runs at 512, 1024 and 2000 tokens with 30 items per cell. Answers are greedy decodes, and the score is RULER's recall: the fraction of expected strings that appear in the output. Memory models were run intact, zeroed and mean-ablated.

**Results** (mean over the 21 cells):

| model | intact | zero | mean |
|---|---|---|---|
| R2 | 0.376 | 0.334 | 0.359 |
| R5 | 0.357 | – | – |
| R3 | 0.344 | 0.133 | 0.294 |
| S1 | 0.338 | 0.162 | 0.311 |
| A | 0.333 | 0.249 | 0.314 |
| R1 | 0.323 | 0.024 | 0.286 |
| R4 | 0.313 | – | – |

- With memory intact, every model solves single-needle retrieval (0.83 to 1.00 at all lengths). All are moderate on multi-key, multi-value and multi-query (0.2 to 0.5), near zero on variable tracking, and at zero on common words and SQuAD at 2000 tokens.
- The cut models with wider attention (R2, R5, R3) have the highest intact means. The 0.06-point spread between models is small for 30 items per cell, though.
- Zeroing memory collapses R1 to 0.02, failing even single-needle retrieval (0.97 → 0.10 at 512 tokens). R3 and S1 halve, A loses a quarter, and R2 barely changes. Mean ablation recovers most of it in every model.

**Interpretation.**
1. No evidence that memory improves long-context retrieval at this scale. R2 has the highest mean, but R5 (dense, same cut) is close behind, and R1 (memory, no cut) is second-lowest. Attention width, not memory, is what separates the models.
2. The zero-ablation collapse is the same off-distribution effect as with induction: the retrieval heads read memory features, so a missing memory breaks them. It does not mean retrieval information is stored in the memory.
3. R2 being much more robust than R3 under zeroing, although their validation-loss sensitivity is similar (+0.18 vs +0.26), is an unexplained difference between the two cut placements. Cutting the FFNs right after the memory layers (R2) apparently leaves attention less dependent on the memory's specific features.

---

## 4. Factual recall: CounterFact

### 4.1 Aim

The Engram paper argues that n-gram memory takes over factual-knowledge storage. CounterFact tests whether the Lngram memory contributes to recalling facts and, if so, at which positions.

### 4.2 How it was done

The script is `eval/counterfact.py` on the `azhx/counterfact` dataset, using 2000 items (seed 0). Each item has a prompt ("The mother tongue of Danielle Darrieux is"), the true object (" French") and a counterfactual object (" English"). The model "knows" the fact if log p(true) > log p(counterfactual), each summed over the object's tokens; the margin is the difference.

Two measurements:
1. **Recall** under `none`, `zero`, `mean`, `zero@1`, `zero@11`, `mean@1` and `mean@11`.
2. **Localisation**, on the items the model gets right with memory intact (about 1300 per model). The memory is ablated only at (a) all subject tokens, (b) the last subject token, (c) the last prompt token, (d) every prompt token except the subject, or (e) every prompt token. The drop in margin shows where memory contributes.

Statistics: McNemar's exact test for intact versus zeroed, and a paired bootstrap between models.

### 4.3 Results

| model | intact | zero | mean | zero@1 | zero@11 | McNemar p | items lost / gained |
|---|---|---|---|---|---|---|---|
| R1 | 0.663 | 0.542 | 0.608 | 0.553 | 0.658 | < 0.001 | 359 / 118 |
| A | 0.658 | 0.534 | 0.621 | 0.544 | 0.657 | < 0.001 | 381 / 134 |
| S1 | 0.675 | 0.634 | 0.657 | – | – | < 0.001 | 168 / 86 |
| R2 | 0.663 | 0.632 | 0.655 | 0.634 | 0.661 | < 0.001 | 134 / 73 |
| R3 | 0.655 | 0.642 | 0.641 | 0.643 | 0.655 | 0.064 | 112 / 85 |
| R5 | 0.659 | – | – | – | – | – | – |
| R4 | 0.645 | – | – | – | – | – | – |

Margin drop (nats) on known items when memory is **zeroed** only at:

| model | subject | last subject token | last token | non-subject tokens | whole prompt |
|---|---|---|---|---|---|
| R1 | 0.98 | 0.04 | 0.00 | 0.15 | 1.35 |
| A | 0.91 | 0.09 | 0.01 | 0.23 | 1.46 |
| S1 | 0.28 | 0.02 | 0.01 | 0.10 | 0.49 |
| R2 | 0.25 | 0.02 | 0.01 | 0.13 | 0.51 |
| R3 | 0.13 | 0.02 | 0.00 | 0.03 | 0.17 |

Mean ablation gives the same ordering at about a third of the size (R1: subject 0.42, prompt 0.28).

### 4.4 Interpretation

1. **Memory models know slightly more facts than the dense baseline.** S1 is +2.95 points over R4 (bootstrap p < 0.001); R1 and R2 are +1.75 (p ≈ 0.03, uncorrected). That points the same way as the Engram paper's knowledge claim, but the effect is small. R5 (dense, cut) sits in between at +1.35, which is not significant.
2. **Recall depends on memory in proportion to how much the model depends on memory overall.** Ordered by dependence, the models go R1 ≈ A, then S1 ≈ R2, then R3 (not significant), the same order as the validation-loss sensitivity. Layer 1 accounts for nearly all of it; zeroing layer 11 alone changes nothing.
3. **The memory's contribution to a fact is located at the subject tokens and spread across them.** Ablating the whole subject removes about 70% of the whole-prompt effect. Ablating only the last subject token, the position that ROME-style analyses find critical for MLP fact storage, removes almost nothing. Since the memory is addressed by 2–3-token n-grams, every subject token contributes a piece of the subject's identity. That is an n-gram lexicon of entity names, not a key–value store keyed on the completed subject.
4. Memory lost at the subject cannot be compensated for elsewhere: ablating everything except the subject costs much less (0.15 for R1, versus 0.98 for the subject alone).

---

## 5. Fine-tuning regimes

### 5.1 Aim

To test whether the memory is a good place to put task adaptation: is training only the memory parameters an efficient parameter-efficient fine-tuning method, and how does it compare with training an equally large dense subset or the whole model?

### 5.2 How it was done

The script is `ffnmem/finetune.py`, one GPU per run, fp16 autocast, with 10% warmup and then linear decay.

| task | data | setup | metric |
|---|---|---|---|
| SST-2 | GLUE SST-2 train / dev | prompt plus verbalizer (" negative" / " positive"); cross-entropy on the label token over the full vocabulary; prediction = argmax over the two verbalizer tokens. 1 epoch, batch 32, max length 128 | dev accuracy |
| BoolQ | SuperGLUE BoolQ train / dev | passage plus question, verbalizer " no" / " yes". 1 epoch, batch 16 (2 micro-batches), max length 384 | dev accuracy |
| NER | CoNLL-2003 | linear head on the final hidden state, labels on each word's first sub-token. 3 epochs, batch 32, max length 160 | entity-level F1 (seqeval) |

Regimes are set by `ffnmem/trainable.py`, which freezes everything except:

| regime | what trains | size |
|---|---|---|
| `none` | nothing (SST-2 and BoolQ: zero-shot verbalizer; NER: linear probe on frozen states) | 0 |
| `all` | everything | 403–404M |
| `backbone` | everything except memory | 297.8M |
| `memory` | memory only: tables, router, readout and conv | 105.5M |
| `ffn8` (R4) | FFN weights of layers 2–5 and 12–15 | 104.6M, matched to memory |
| `ffn12` (R2) | FFN weights of R2's 12 uncut layers (0, 1, 6–11, 16–19) | 103.8M, matched to memory on the same model |

The default learning rate is 5e-5 for `all`, `backbone` and FFN subsets, and 5e-4 for `memory`, with tables at 4× as in pre-training.

**Experiments:**
1. Main grid: every model × task × applicable regime, seed 0.
2. Extra seeds 1 and 2 for the headline regimes (R1, R2 and S1 `memory` and `all`; R4 `ffn8` and `all`).
3. **Learning-rate and parameter-matched controls**, each with 3 seeds:
   - R2 `memory` at 5e-5, the dense learning rate;
   - R2 `ffn12` at 5e-5 and 5e-4;
   - R4 `ffn8` at 5e-4, the memory learning rate.

Comparisons use Welch t-tests over seeds.

### 5.3 Results

Means over 3 seeds (sd 0.001–0.015); "none" and `backbone` are single runs:

| regime (lr) | SST-2 | BoolQ | NER F1 |
|---|---|---|---|
| R2 `all` (5e-5) | 0.902 | 0.700 | 0.715 |
| R4 `all` (5e-5) | 0.903 | 0.696 | 0.719 |
| R2 `ffn12` (5e-5) | **0.911** | **0.708** | **0.716** |
| R4 `ffn8` (5e-5) | 0.899 | 0.686 | 0.711 |
| R2 `memory` (5e-4) | 0.883 | 0.673 | 0.679 |
| R1 `memory` (5e-4) | 0.876 | 0.672 | 0.682 |
| S1 `memory` (5e-4) | 0.878 | 0.671 | 0.676 |
| R2 `memory` (5e-5) | 0.853 | 0.628 | 0.658 |
| R2 `ffn12` (5e-4) | 0.827 | 0.635 | 0.697 |
| R4 `ffn8` (5e-4) | 0.869 | 0.661 | 0.705 |
| `backbone` (R1 / R2 / R3 / S1) | 0.899–0.904 | 0.695–0.710 | 0.712–0.718 |
| `none` (range over models) | 0.52–0.66 | 0.53–0.62 | 0.63–0.64 |

Key tests (Welch, 3 vs 3 seeds):

| comparison | SST-2 | BoolQ | NER |
|---|---|---|---|
| memory (5e-4) vs full fine-tuning, R2 | −1.9 (p = 0.016) | −2.7 (p = 0.020) | −3.7 (p < 0.001) |
| memory vs FFN8, each at its own default LR | −1.6 (p = 0.024) | −1.2 (p = 0.12) | −3.3 (p < 0.001) |
| memory vs FFN12 on R2, both at 5e-5 | −5.8 | −8.0 | −5.8 (all p ≤ 0.002) |
| memory vs FFN12 on R2, both at 5e-4 | +5.6 (p < 0.001) | +3.8 (p = 0.012) | −1.8 (p = 0.004) |
| memory: 5e-4 vs 5e-5 | +3.0 (p = 0.006) | +4.5 (p = 0.006) | +2.1 (p < 0.001) |
| full fine-tuning R2 vs R4 | −0.1 (n.s.) | +0.4 (n.s.) | −0.4 (n.s.) |
| memory-only S1 vs R2, R1 vs R2 | all n.s. | n.s. | n.s. |

### 5.4 Interpretation

1. **Full fine-tuning gives the same result on every architecture.** The pre-trained memory neither helps nor hurts downstream fine-tuning.
2. **Memory-only tuning is clearly worse than tuning an equally large FFN subset, when each gets its best learning rate.** The fair comparison is each regime at its own best learning rate: memory at 5e-4 and FFN subsets at 5e-5. There, FFN subsets match or beat full fine-tuning, and memory trails by 1.2 to 3.7 points. Comparing at one shared learning rate is misleading in either direction: at 5e-4 memory "wins" because the FFN subset is over-driven, and at 5e-5 memory loses by 6 to 8 points because it is under-driven.
3. **Memory needs a 10× higher learning rate,** which fits its sparse updates: each step touches only the rows addressed by the batch.
4. **Token tagging (NER) is where memory-only tuning is weakest** (−3.3 to −3.7). Memory outputs are a function of local n-gram codes, so tuning them cannot change how context is combined, which tagging needs.
5. The type of memory model (R1, R2, S1) makes no difference to memory-only tuning.

---

## 6. Domain adaptation: PubMed continued pre-training

### 6.1 Aim

A natural hypothesis for a lexical memory is that it is a cheap domain adapter: new domain n-grams could be stored in the tables while the backbone, and so general ability, stays untouched. This experiment tests that.

### 6.2 How it was done

The script is `eval/cpt.py`. Data is PubMed abstracts, tokenized with the project tokenizer (`data/prepare_domain.py`), with a held-out PubMed validation split.

Each run continues training from the final checkpoint for 20M tokens: 306 steps of 32 × 2048, a fresh cosine schedule with peak lr 1e-4 and 5% warmup, on one GPU. Optimizer and data loader state were reset.

Evaluation before and after training covers:
- PubMed validation loss, the in-domain gain (100 batches);
- ClimbMix validation loss, i.e. forgetting of general text;
- the zero-shot track, i.e. forgetting of abilities;
- a domain track: PubMedQA plus 7 MMLU medical and biology subjects (anatomy, clinical knowledge, college biology, college medicine, high-school biology, medical genetics, professional medicine), in continuation format.

| run | trains | size |
|---|---|---|
| CPT_R2_memory | R2 memory only | 105.5M |
| CPT_R2_memory_lr5e-4 | same, at 5× the learning rate (memory's fine-tuning learning rate is 10× the dense one) | 105.5M |
| CPT_R2_ffn12 | R2's 12 uncut FFN layers | 103.8M |
| CPT_R2_all | everything | 403.3M |
| CPT_R4_ffn8 | R4's FFN layers 2–5 and 12–15 | 104.6M |
| CPT_R4_all | everything | 404.0M |
| CPT_S1_memory | S1 memory only | 105.5M |
| CPT_S1_all | everything | 403.3M |

### 6.3 Results

Base → after (validation losses in nats, lower is better):

| run | PubMed loss | ClimbMix loss | zero-shot mean | domain mean | PubMedQA |
|---|---|---|---|---|---|
| CPT_R2_all | 2.863 → 2.429 (−0.43) | 2.788 → 3.022 (+0.23) | 0.490 → 0.481 | 0.294 → 0.277 | 0.460 → 0.354 |
| CPT_R4_all | 2.879 → 2.441 (−0.44) | 2.798 → 3.034 (+0.24) | 0.500 → 0.484 | 0.303 → 0.283 | 0.528 → 0.414 |
| CPT_R2_ffn12 | 2.863 → 2.548 (−0.32) | 2.788 → 2.876 (+0.09) | 0.490 → 0.484 | 0.294 → 0.285 | 0.460 → 0.380 |
| CPT_R4_ffn8 | 2.879 → 2.633 (−0.25) | 2.798 → 2.853 (+0.06) | 0.500 → 0.495 | 0.303 → 0.291 | 0.528 → 0.526 |
| CPT_R2_memory_lr5e-4 | 2.863 → 2.704 (−0.16) | 2.788 → 2.953 (+0.17) | 0.490 → 0.476 | 0.294 → 0.272 | 0.460 → 0.354 |
| CPT_R2_memory | 2.863 → 2.768 (−0.10) | 2.788 → 2.855 (+0.07) | 0.490 → 0.480 | 0.294 → 0.278 | 0.460 → 0.366 |
| CPT_S1_memory | 2.864 → 2.778 (−0.09) | 2.784 → 2.836 (+0.05) | 0.493 → 0.494 | 0.291 → 0.288 | 0.518 → 0.512 |
| CPT_S1_all | 2.864 → *pending* | 2.784 → *pending* | | | |

CPT_S1_all is still running and finishes around 21:35. Its row is added in `RESULTS_ADA.md` once done.

The zero-shot mean in this table is the CPT section's own average over the zero-shot track, computed the same way before and after, so it is not identical to the 14-task mean in §2.

PubMed gain per nat of ClimbMix forgetting: R4_ffn8 4.5, R2_ffn12 3.6, R2_all 1.9, R4_all 1.9, S1_memory 1.7, R2_memory 1.4, R2_memory_lr5e-4 1.0.

### 6.4 Interpretation

1. **The "memory as a domain adapter" hypothesis is not supported.** At a matched parameter budget (about 105M), memory-only training lowers PubMed loss 2.5 to 3.3× less than an FFN subset (−0.10 vs −0.25 to −0.32). Raising its learning rate buys more domain gain (−0.16) but costs proportionally more forgetting (+0.17), so it does not reach the FFN subsets' trade-off.
2. **Memory-only training does forget little** (+0.05 to +0.07 on ClimbMix). But dense FFN subsets forget about as little (+0.06 to +0.09) while learning 2.5 to 3× more, so memory is not on the efficient frontier.
3. **The domain benchmarks are at the floor and do not track loss.** MMLU-medical averages 0.27–0.30, against 0.25 chance. PubMedQA *drops* after almost every run, most likely because 20M tokens of abstracts shift the model's yes/no/maybe answer prior. The two runs that change the least (R4_ffn8 and S1_memory) keep PubMedQA. In-domain loss is the reliable measure here; the benchmark numbers should not be read as "CPT hurts medical knowledge".
4. Full training forgets the most (+0.23 to +0.24) and loses about 1 to 1.6 zero-shot points. That is expected at this learning rate without replay.

---

## 7. Memory interpretability

These analyses were run on R1, R2 and S1, which are representative of the memory designs, with R4 as the dense reference where applicable. Code: `eval/run_mech.py`, `eval/run_interp.py`, `eval/cka.py`; results in `results/<model>/{mech,interp}/`.

### 7.1 Propagation: how far the memory's effect reaches

**Aim.** To find out how much of the final representation and prediction depends on each memory layer.

**Method.** Mean-ablate one memory layer (or both) and measure at the output:
- R = ‖Δ final hidden‖ / ‖final hidden‖, and the same for logits;
- Jensen–Shannon divergence between the output distributions;
- top-1 agreement between the intact and ablated predictions.

**Results:**

| model | layer | hidden R | logit R | JS | top-1 agree |
|---|---|---|---|---|---|
| R1 | 1 | 0.50 | 0.22 | 0.111 | 0.67 |
| R1 | 11 | 0.11 | 0.05 | 0.004 | 0.93 |
| R1 | 1+11 | 0.62 | 0.28 | 0.169 | 0.58 |
| R2 | 1 | 0.23 | 0.10 | 0.025 | 0.86 |
| R2 | 11 | 0.09 | 0.04 | 0.003 | 0.95 |
| R3 | 1 | 0.29 | 0.13 | 0.038 | 0.83 |
| A | 1 | 0.49 | 0.22 | 0.111 | 0.68 |
| S1 | 1 | 0.34 | 0.15 | 0.054 | 0.79 |

**Conclusion.** The token-specific part of layer-1 memory changes half of R1's final hidden state and a third of its top-1 predictions. Layer 11 changes 5–7% of predictions. The effects of the two layers add up almost linearly.

### 7.2 Code utilisation and the Sink

**Aim.** The Lngram papers use sink mass and the number of effective codes as health diagnostics. This checks them on the final models.

**Method.** On validation text, measure the average readout mass on the zero-valued sink, and the effective number of codes per route (the exponential of the entropy of the code distribution, out of 16).

**Results.** Sink mass 0.42–0.47 and 14.3–15.0 effective codes, for every model and layer.

**Conclusion.** Codes are used almost uniformly (the Lngram paper reports about 8 of 16), and the sink takes close to its initial 0.5. Together with §2.4, this shows these diagnostics do **not** indicate how much the model depends on memory: R1, whose memory carries 1 nat, looks the same as R2, whose memory carries 0.18.

### 7.3 Logit lens

**Aim.** To find out whether memory makes predictions form earlier in depth.

**Method.** Project each layer's residual stream through the final norm and unembedding, and measure KL(final ‖ layer) per token.

**Results (nats):**

| model | L0 | L2 | L5 | L11 | L17 | L22 |
|---|---|---|---|---|---|---|
| R4 (dense) | 6.13 | 5.81 | 5.29 | 3.65 | 2.18 | 0.58 |
| R5 (dense, cut) | 5.53 | 4.86 | 4.36 | 3.55 | 2.48 | 0.33 |
| R1 | 5.84 | 4.65 | 4.19 | 3.02 | 2.09 | 0.33 |
| R2 | 5.95 | 4.70 | 4.29 | 3.34 | 2.23 | 0.29 |
| S1 | 6.03 | 4.54 | 3.92 | 3.00 | 1.81 | 0.27 |

**Conclusion.** After layer 1, every memory model is about 1.1–1.3 nats closer to its final prediction than R4, and stays closer through the middle layers. The memory front-loads the prediction: the Lngram lookup hands the network a decent next-token guess at layer 1, which is the Engram paper's "memory frees depth" argument, observed here with Lngram. R5 shows part of the same effect without memory, so wider attention contributes as well.

### 7.4 Token attribution: which tokens need memory

**Aim.** To find out which tokens the memory matters for.

**Method.** On 16 validation windows (about 33k tokens), compute each token's Δloss, KL and top-1 flip rate under mean and zero ablation. Bucket them by:
- token class;
- training-set frequency of the preceding bigram and trigram (counted on 60M training tokens);
- whether the target is "copyable" (the same bigram appeared earlier in the window);
- position in the window.

**Results.** Mean-ablation Δloss: R1 0.33, R2 0.14, S1 0.14 nats per token; top-1 flips 25%, 17% and 16%.

| bucket | R1 | R2 | S1 |
|---|---|---|---|
| newline tokens | 1.01 | 1.04 | 0.95 |
| word-continuation sub-tokens | 0.64 | 0.15 | 0.17 |
| word starts | 0.19 | 0.09 | 0.09 |
| trigram seen ≥ 10k times | 0.96 | 0.71 | 0.72 |
| trigram never seen | 0.27 | 0.11 | 0.10 |
| positions 0–15 in the window | 0.80 | 0.33 | 0.23 |
| copyable vs novel (zero ablation) | 1.71 vs 0.99 | 0.22 vs 0.17 | 0.49 vs 0.29 |

**Conclusion.**
1. The memory matters most where an n-gram table should: very frequent n-grams, formatting (newlines), completing a word from its pieces, and the start of a window where attention has little context.
2. R1 additionally uses memory to complete multi-token words (0.64 vs 0.15). In the cut models that job has moved to the wider attention.
3. Under zero ablation, copyable tokens suffer more than novel ones. That is the induction breakage from §2.4 again: zeroing memory disturbs the attention heads that copy.

### 7.5 What the tables store (entries)

**Aim.** To find out what is in the table rows and how the codes relate to tokens.

**Method.**
- Count row hits per n-gram order, with concentration and a Zipf slope.
- Measure purity: the share of a busy row's hits taken by its single most common token n-gram.
- Compute normalized, shuffle-corrected mutual information between codes and the current or previous token.
- Measure lexical determinism: for frequent tokens, the share of occurrences that get the token's modal code.
- Direct logit attribution: project the memory output alone through the final norm and unembedding, and find the rank of the true next token, against a norm-matched random vector.

**Results** (R1 / R2 / S1, layer 1):
- **Code–token MI:** 0.66 / 0.64 / 0.64 for the current token, versus 0.14–0.15 for the previous token. At layer 11 it falls to 0.41.
- **Lexical determinism:** 0.94 / 0.94 / 0.95, meaning a given token gets the same code on 94% of routes. At layer 11 it is 0.62.
- **Row use:** every order-2 row and 93% of order-3 rows are used. Use is spread out (the top 1% of rows take only 5–9% of hits; Zipf slope −0.3).
- **Purity:** busy rows are shared by many n-grams (layer-1 purity 0.22–0.40, layer 11 0.07–0.11).
  - Layer 1's busiest rows hold formatting n-grams: `\n\n`, `.\n`, `.\n\n`, `\n\nQuestion`, ` of the`, ` to the`.
  - Layer 11's rows hold sub-word pieces of content words, e.g. `Australop`/`alopit`/`opithe` together, ` headlights`, ` flooring` and ` recycling`.
- **Direct logits:** the memory output alone ranks the true next token about 4× better than random (median rank 3,400–9,800 vs about 16,400) and puts it in the top 10 for 2.6–8.4% of tokens, versus 0.1% for random. Example: after "challenge the status qu", S1 and R2's memory contributes to predicting "o".

**Conclusion.** Layer-1 codes are essentially a hash of the current token, so layer-1 memory is a learned n-gram table over (previous 1–2 tokens, current token). Layer 11 operates on contextual states and stores sub-word and content n-grams. Its codes depend less on the token, which also explains why it is less stable and less useful. The direct-logit result shows the memory writes partly into the output space directly, much like a bigram/trigram language model mixed into the residual stream.

### 7.6 ID semantics: do the codes carry meaning?

**Aim.** Following Lngram §3, to find out whether the discrete route codes alone encode semantic categories beyond lexical identity.

**Method.**
1. Collect route codes, hidden states and router logits for every token of Few-NERD entities (8 coarse types: art, building, event, location, organization, person, product, other).
2. Split documents 70/30 into reference and query.
3. Classify each query token by its 10 nearest reference neighbours, from other documents only, using three representations:
   - Hamming distance on codes (**AP_ID**);
   - cosine on the final hidden state (**AP_H**);
   - cosine on the continuous router logits (**AP_Q**).
4. Compare against a nuisance predictor using position bucket × word-initial flag (**AP_nuisance**).
5. Compute ρ = (AP_ID − AP_n) / (AP_H − AP_n), the fraction of the hidden state's excess information that the codes retain.
6. Test significance with a stratified label permutation (200 permutations).
7. Test single (route, code) → entity-type associations against the nuisance baseline, then check that each replicates on held-out data with Benjamini–Hochberg correction.
8. Repeat the classification using only the top-m routes, to see whether semantics is concentrated or distributed.

**Results:**

| model | layer | AP_ID | AP_Q | AP_H | AP_n | ρ | p | associations tested / replicated |
|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 0.152 | 0.161 | 0.443 | 0.066 | 0.23 | 0.005 | 520 / 497 |
| R1 | 11 | 0.237 | 0.249 | 0.443 | 0.066 | 0.45 | 0.005 | 1097 / 1074 |
| R2 | 1 | 0.155 | 0.161 | 0.439 | 0.066 | 0.24 | 0.005 | 588 / 569 |
| R2 | 11 | 0.218 | 0.245 | 0.439 | 0.066 | 0.41 | 0.005 | 1130 / 1100 |
| S1 | 1 | 0.160 | 0.166 | 0.439 | 0.066 | 0.25 | 0.005 | 1180 / 1149 |

The best single routes have codes that raise the probability of "organization" or "location" from about 6% (nuisance) to 22–31% on held-out data.

The top-m curve shows the information is distributed. At layer 1, a single route gives AP 0.05–0.06, 8 routes 0.14, and 16–32 routes reach the plateau (0.15). S1's extra routes beyond 64 add nothing. At layer 11, AP is still rising at 90 routes.

**Conclusion.**
1. The codes carry real, replicable semantic information: about 2.3× the nuisance baseline. Layer 11 carries about twice as much as layer 1, consistent with it reading contextual states.
2. Codes lose almost nothing relative to the continuous router logits (AP_ID ≈ AP_Q), so binarization is not a bottleneck.
3. The codes hold only a quarter to a half of what the hidden state knows about entity type, so the memory is addressed by a coarse, partly semantic hash.
4. Doubling the routes (S1: 184 at layer 1) does not add semantic content. It adds capacity for the same lexical signal.

### 7.7 Locality: how far does one token's memory reach?

**Method.** Mean-ablate the memory at a single position t (48 sampled positions) and measure KL(intact ‖ ablated) at positions t−1 … t+16.

**Results (nats):**

| offset | 0 | +1 | +2 | +3 | +4 | +8 | +16 |
|---|---|---|---|---|---|---|---|
| R1 | 0.109 | 0.009 | 0.009 | 0.003 | 0.002 | 0.001 | 0.0005 |
| R2 | 0.030 | 0.007 | 0.003 | 0.001 | 0.002 | 0.001 | 0.0003 |
| S1 | 0.022 | 0.007 | 0.003 | 0.001 | 0.001 | 0.001 | 0.0002 |

Offset −1 is exactly 0, which confirms causality.

**Conclusion.** The memory's effect is almost entirely local: at offset 0, with a short tail over the next 2 tokens. That is what 2- and 3-gram addressing implies. The memory is not used as a long-range channel.

### 7.8 Activation patching on entities

**Aim.** To trace how information about an entity flows to the prediction, and what role memory plays in that.

**Method.**
1. Take 150 Few-NERD sentences that end inside a multi-token entity. The task is to predict the entity's last token.
2. Build a corrupt input by replacing the entity's first token with a random word-start token, and keep pairs where corruption drops log p(target) by at least 1 nat (63 to 70 pairs per model).
3. Denoising: patch clean activations into the corrupt run, one layer and one site at a time. Sites are the residual stream, attention output, FFN output and memory output, at the corrupted position or the last position. Recovery is normalized so that 0 = corrupt and 1 = clean.
4. Patch the memory output at each offset from the corrupted token, and at all positions together.
5. Noising: the reverse, patching corrupt activations into the clean run, for the residual grid.

**Results:**
- **Residual stream at the corrupted position:**
  - In R4, patching it restores the prediction fully through layer 5 (recovery ≥ 1.0). The information stays at the corrupted position until attention moves it.
  - In R1, patching the residual after layer 1 recovers only 0.81, and in R2 and S1 0.91 to 0.94. Part of the information has already left that position by layer 1, through the memory.
- **Memory output, per offset from the corrupted token:** patching the clean layer-1 memory output only at the **next** position recovers 0.28 (R1), 0.10 (R2) and 0.11 (S1) of the gap, and at offset +2 recovers 0.18, 0.09 and 0.04. Offset 0 recovers nothing.
- **Memory output at all positions:** layer 1 recovers 0.32 (R1), 0.13 (R2) and 0.12 (S1) of the gap; layer 11 recovers 0.10 (R1) and 0.07 (R2).
- **Attention output at the last position** has its biggest effect at layer 9 in R4 (0.25), layer 4 in R2 (0.29), layer 5 in S1 (0.31), and is spread over layers 12–19 in R1 (at most 0.13).
- **FFN at the last position** matters in the final layers (layers 20–23, 0.17–0.29) in every model.

**Conclusion.**
1. Through its n-gram addresses, the memory is a second information path that carries a token's identity to the next 1–2 positions without attention. That is why patching the residual at the corrupted position no longer restores everything in memory models.
2. The memory at the corrupted position itself does nothing. Its use is to tell the *following* positions which n-gram they are in.
3. In R1 this path is about 2.5× stronger than in R2 and S1, consistent with the other measures of dependence.
4. In the wide-attention models, the attention step that moves the entity to the prediction position happens earlier (layers 4–5, versus 9 in R4), which matches the earlier convergence in the logit lens.

### 7.9 Representation similarity with the dense model (CKA)

**Method.** Linear CKA between the layer-l hidden states of a memory model and of R4, at the final token of 7,796 Few-NERD entities.

**Results.** The diagonal CKA (same layer in both) is 0.78–0.80 for layers 0–8. It dips to 0.70–0.74 around layers 10–12, then rises to 0.82–0.84 for layers 17–23, with the same profile for R1, R2, R3 and S1.

**Conclusion.** The memory models and the dense model learn similar representations, most similar at the end. They differ most in the middle of the network. The dip is present in S1, which has no layer-11 memory, so it is not caused by the layer-11 memory. It reflects a different division of labour in the middle layers.

### 7.10 Visualisation

HTML heatmaps (`results/<model>/interp/html/`) show, for 8 validation passages, the per-token gate (1 − sink mass) at each memory layer and the per-token Δloss under mean ablation. They are qualitative illustrations of §7.4: high-Δloss tokens cluster on newlines, word continuations and frequent phrases.

---

## 8. Overall picture

- **The memory is a learned, lexically addressed n-gram model embedded at layer 1.** It is consulted on every token, it front-loads next-token predictions, and it supports syntax, cloze, generation and part of entity recall. Its contents are mostly lexical, with some recoverable semantic structure.
- **It substitutes for FFN capacity instead of adding to it.** At 400M parameters and 2B tokens it gives equal quality with 27% fewer active parameters per token, which makes the case for it about efficiency, not quality.
- **Widening attention reduces reliance on memory.** The cut, wider-attention models are less dependent on memory and more robust to its removal. They are also the best models overall (R5 best in loss, R2 best on mini-RULER).
- **Training only the memory is not a strong adaptation method.** For both task fine-tuning and domain pre-training, an equally large FFN subset adapts more for the same or less forgetting.
- **Layer 11 is close to dead weight** (0.01–0.02 nats, 5–7% of predictions), which supports moving all capacity to layer 1. S1 did that and is the best memory model in pre-training and on CounterFact.

---

## 9. Limitations

1. **One pre-training run per architecture.** Differences between models below about 0.005 nats in loss, or a point or two on benchmarks, may be run-to-run noise. Only fine-tuning has seeds. Wave-1 runs also differ in GPU type (R1–R3 on 2080 Ti, R4–R5 on 3090).
2. **Scale.** At 400M parameters and 2B tokens many benchmarks are near chance: MMLU, ARC-c, TriviaQA, DROP, BBH (intact), and the medical domain track. Null results there say little.
3. **Zero ablation is off-distribution.** Large zero-ablation effects on induction, retrieval and BBH partly measure that downstream components break when memory is missing, not what the memory stores. Mean-ablation numbers are reported alongside and are the better estimate of memory's information content.
4. **Sample sizes:** 300 items per task for the paper and few-shot tracks, 30 per cell for mini-RULER (no significance tests), 150 entity pairs for patching (63–70 kept), 2000 CounterFact items.
5. **Passkey at 2000 tokens** appears to be an artifact (identical 0.67 across models and conditions) and should not be cited.
6. **Fine-tuning:** 3 seeds only for the headline regimes; R3, R5 and `backbone` have 1 seed. Only two learning rates were tried per regime, so the "best LR" comparison is coarse. Epoch counts are fixed (1 epoch for SST-2/BoolQ, 3 for NER).
7. **Domain pre-training:** one run per configuration, 20M tokens and one learning rate (plus one higher-LR memory variant); no replay data. Domain benchmarks are at the floor and PubMedQA seems driven by the answer prior, so in-domain loss is the reliable measure.
8. **Interpretability** was run on R1, R2 and S1 (and R4 for patching and CKA), not on all seven models. Few-NERD entity types are a coarse notion of semantics.
9. **Statistics:** p-values come from a normal approximation to the bootstrap distribution (fine for means over hundreds of items), Holm-corrected per table, which is conservative. The CounterFact between-model p-values are uncorrected.

---

## 10. Where things are

| what | where |
|---|---|
| all numbers | `RESULTS_ADA.md` (from `eval/aggregate_ada.py`) |
| all tests | `STATS_ADA.md` (from `eval/stats_ada.py`) |
| raw results | `anangappara@ada:~/morbius_eval_runs/results/<model>/{none,zero,mean,...}`, `.../ft/`, `.../cpt*/`, `.../cka/` |
| task definitions | `eval/tasks.py`, `ffnmem/synthetic.py`, `eval/mini_ruler.py`, `eval/counterfact.py` |
| interventions | `ffnmem/hooks.py` |
| fine-tuning | `ffnmem/finetune.py`, `ffnmem/trainable.py`, `eval/make_ft_extra.py` |
| domain pre-training | `eval/cpt.py`, `eval/make_cpt_jobs.py`, `data/prepare_domain.py` |
| interpretability | `eval/run_mech.py`, `eval/run_interp.py`, `eval/cka.py`, `ffnmem/interp/` |
| pre-training history | `state_and_results.md` |
