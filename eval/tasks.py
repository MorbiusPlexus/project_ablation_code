"""Evaluation protocol aligned with Engram Table 1 and Lngram's diagnostics.

Tasks a 400M / 2B-token English model scores at chance on are listed as ``floor``
rather than dropped. Shots follow the Engram paper and are cut when the prompt
would exceed the 2048-token training context.
"""

from __future__ import annotations

# name, shots, metric, group, status
PAPER_TASKS = [
    ("pile_bpb", 0, "bpb", "lm", "run"),
    ("climbmix_val", 0, "loss", "lm", "run"),
    ("mmlu", 5, "acc", "knowledge", "run"),
    ("arc_easy", 25, "acc", "knowledge", "run"),
    ("arc_challenge", 25, "acc", "knowledge", "run"),
    ("triviaqa", 5, "em", "knowledge", "run"),
    ("popqa", 15, "em", "knowledge", "run"),
    ("sciq", 0, "acc", "knowledge", "run"),
    ("openbookqa", 0, "acc", "knowledge", "run"),
    ("bbh", 3, "em", "reasoning", "run"),
    ("hellaswag", 0, "acc_norm", "commonsense", "run"),
    ("piqa", 0, "acc_norm", "commonsense", "run"),
    ("winogrande", 5, "acc", "commonsense", "run"),
    ("copa", 0, "acc", "commonsense", "run"),
    ("siqa", 0, "acc", "commonsense", "run"),
    ("lambada", 0, "acc", "reading", "run"),
    ("drop", 1, "f1", "reading", "run"),
    ("race", 5, "acc", "reading", "run"),
    ("boolq", 0, "acc", "reading", "run"),
    ("blimp", 0, "acc", "syntax", "run"),
    ("ruler", 0, "acc", "reading", "run"),
    ("gsm8k", 5, "em", "math", "floor"),
    ("math", 4, "em", "math", "floor"),
    ("humaneval", 0, "pass@1", "code", "floor"),
    ("mbpp", 3, "pass@1", "code", "floor"),
    ("mmlu_pro", 5, "acc", "knowledge", "floor"),
    ("cmmlu", 5, "acc", "knowledge", "floor"),
    ("ceval", 5, "acc", "knowledge", "floor"),
    ("agieval", 0, "acc", "knowledge", "floor"),
    ("c3", 0, "acc", "reading", "floor"),
]

# lm-eval 0.4.9 task names. (task, shots, primary metric, group)
ZEROSHOT_TRACK = [
    ("hellaswag", 0, "acc_norm", "commonsense"),
    ("piqa", 0, "acc_norm", "commonsense"),
    ("winogrande", 0, "acc", "commonsense"),
    ("copa", 0, "acc", "commonsense"),
    ("social_iqa", 0, "acc", "commonsense"),
    ("arc_easy", 0, "acc_norm", "knowledge"),
    ("arc_challenge", 0, "acc_norm", "knowledge"),
    ("sciq", 0, "acc_norm", "knowledge"),
    ("openbookqa", 0, "acc_norm", "knowledge"),
    ("mmlu_continuation", 0, "acc", "knowledge"),
    ("lambada_openai", 0, "acc", "reading"),
    ("boolq", 0, "acc", "reading"),
    ("race", 0, "acc", "reading"),
    ("blimp", 0, "acc", "syntax"),
    ("wikitext", 0, "bits_per_byte", "lm"),
]

# Engram Table 1 shots. Prompts longer than 2048 tokens are left-truncated by the
# adapter, which logs the truncated fraction per task.
PAPER_TRACK = [
    ("mmlu", 5, "acc", "knowledge"),
    ("arc_easy", 25, "acc_norm", "knowledge"),
    ("arc_challenge", 25, "acc_norm", "knowledge"),
    ("triviaqa", 5, "exact_match", "knowledge"),
    ("bbh_fewshot", 3, "exact_match", "reasoning"),
    ("winogrande", 5, "acc", "commonsense"),
    ("drop", 1, "f1", "reading"),
    ("race", 5, "acc", "reading"),
]

FEWSHOT_TRACK = [
    ("arc_easy", 5, "acc_norm", "knowledge"),
    ("hellaswag", 5, "acc_norm", "commonsense"),
]

SIQA_TRACK = [("social_iqa", 0, "acc", "commonsense")]

# Biomedical knowledge for the PubMed continued-pretraining study.
DOMAIN_TRACK = [
    ("pubmedqa", 0, "acc", "domain"),
    ("mmlu_continuation_anatomy", 0, "acc", "domain"),
    ("mmlu_continuation_clinical_knowledge", 0, "acc", "domain"),
    ("mmlu_continuation_college_biology", 0, "acc", "domain"),
    ("mmlu_continuation_college_medicine", 0, "acc", "domain"),
    ("mmlu_continuation_high_school_biology", 0, "acc", "domain"),
    ("mmlu_continuation_medical_genetics", 0, "acc", "domain"),
    ("mmlu_continuation_professional_medicine", 0, "acc", "domain"),
]

TRACKS = {"zeroshot": ZEROSHOT_TRACK, "paper": PAPER_TRACK, "fewshot": FEWSHOT_TRACK, "siqa": SIQA_TRACK,
          "domain": DOMAIN_TRACK}

RULER_SUBTASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe", "qa_squad",
]
RULER_LENGTHS = [1024, 2048]

FT_TASKS = ["sst2", "boolq", "ner"]
FT_REGIMES_MEMORY = ["backbone", "memory", "all"]
FT_REGIMES_DENSE = ["all", "ffn_layers:2,3,4,5,12,13,14,15"]

MODELS = {
    "w1_mem_full": {"account": "aditya", "kind": "memory", "layers": [1, 11]},
    "w1_mem_cut_near": {"account": "sparsh", "kind": "memory", "layers": [1, 11]},
    "w1_mem_cut_far": {"account": "aditya", "kind": "memory", "layers": [1, 11]},
    "w1_dense_full": {"account": "sparsh", "kind": "dense", "layers": []},
    "w1_dense_cut_near": {"account": "sparsh", "kind": "dense", "layers": []},
    "w2_mem_rank1": {"account": "sparsh", "kind": "memory", "layers": [1, 11]},
    "w2_mem_super1": {"account": "sparsh", "kind": "memory", "layers": [1]},
}

SHORT = {
    "w1_mem_full": "R1", "w1_mem_cut_near": "R2", "w1_mem_cut_far": "R3",
    "w1_dense_full": "R4", "w1_dense_cut_near": "R5", "w2_mem_rank1": "A", "w2_mem_super1": "S1",
}

CHANCE = {
    "hellaswag": 0.25, "piqa": 0.5, "winogrande": 0.5, "arc_easy": 0.25,
    "arc_challenge": 0.25, "boolq": 0.5, "blimp": 0.5, "sciq": 0.25,
    "openbookqa": 0.25, "copa": 0.5, "siqa": 0.333, "social_iqa": 0.333, "mmlu": 0.25, "race": 0.25,
    "pubmedqa": 0.333,
}


def shots_that_fit(shots: int, seq_len: int = 2048, est_prompt: int = 80) -> int:
    if shots <= 0:
        return 0
    return max(0, min(shots, (seq_len - est_prompt) // max(est_prompt, 1)))
