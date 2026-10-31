"""Mini-RULER: synthetic long-context tasks built with the project tokenizer.

    python eval/mini_ruler.py --ckpt <ckpt> --name R2 --out_dir results \
        --interventions none,zero,mean --lengths 512,1024,2000 --n 30

Tasks follow RULER (Hsieh et al. 2024), scaled to a 2048-token model:
  niah_single     one key-value needle in essay filler
  niah_multikey   one needle plus 3 distractor needles with other keys
  niah_multivalue one key with 2 values; both must be produced
  niah_multiquery 2 keys asked at once; both values must be produced
  vt              variable tracking: chains of assignments, list every variable equal to a value
  cwe             common-word extraction: the 3 most frequent words in a word list
  qa_squad        SQuAD question with its gold paragraph hidden among distractor paragraphs
Score: fraction of expected strings found in the greedy continuation (RULER's recall metric).
Writes <out_dir>/<name>/<intervention>/ruler.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import Tokenizer

from eval.means import DATA, load_or_make_means
from ffnmem.hooks import Intervention, load_model, parse_intervention
from ffnmem.scoring import greedy_generate

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "Rivers run to the sea and the hills stay where they are. ")
WORDS = ("apple river stone cloud window garden silver candle forest mirror ladder harbor violin "
         "pepper rocket saddle thunder velvet walnut yellow zebra anchor bamboo cactus dolphin "
         "engine falcon glacier hammer island jacket kettle lantern meadow needle orchard pillow "
         "quartz ribbon saffron tunnel umbrella vessel whistle").split()
TASKS = ["niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "vt", "cwe", "qa_squad"]


def _num(rng):
    return str(rng.randint(1000000, 9999999))


def _key(rng):
    return "-".join(rng.sample(WORDS, 2))


class Builder:
    def __init__(self, tok: Tokenizer, squad=None):
        self.tok = tok
        self.squad = squad

    def n(self, text: str) -> int:
        return len(self.tok.encode(text).ids)

    def _fill(self, budget: int) -> list[str]:
        unit = FILLER.split(". ")
        out, used, i = [], 0, 0
        while used < budget:
            s = unit[i % len(unit)].strip().rstrip(".") + ". "
            out.append(s)
            used += self.n(s)
            i += 1
        return out

    def _haystack(self, length: int, needles: list[str], question: str, rng) -> str:
        head = "Some special magic numbers are hidden within the following text. Make sure to memorize them.\n"
        room = length - self.n(head) - self.n(question) - sum(self.n(x) for x in needles) - 8
        sents = self._fill(max(room, 0))
        for nd in needles:
            sents.insert(rng.randint(0, len(sents)), nd + " ")
        return head + "".join(sents) + "\n" + question

    def build(self, task: str, length: int, rng) -> tuple[str, list[str], int]:
        if task == "niah_single":
            k, v = _key(rng), _num(rng)
            q = f"What is the special magic number for {k}? The special magic number for {k} is"
            return self._haystack(length, [f"One of the special magic numbers for {k} is: {v}."], q, rng), [v], 12
        if task == "niah_multikey":
            keys = [_key(rng) for _ in range(4)]
            vals = [_num(rng) for _ in keys]
            nds = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
            q = f"What is the special magic number for {keys[0]}? The special magic number for {keys[0]} is"
            return self._haystack(length, nds, q, rng), [vals[0]], 12
        if task == "niah_multivalue":
            k, vals = _key(rng), [_num(rng), _num(rng)]
            nds = [f"One of the special magic numbers for {k} is: {v}." for v in vals]
            q = f"What are all the special magic numbers for {k}? The special magic numbers for {k} are"
            return self._haystack(length, nds, q, rng), vals, 24
        if task == "niah_multiquery":
            keys = [_key(rng), _key(rng)]
            vals = [_num(rng), _num(rng)]
            nds = [f"One of the special magic numbers for {k} is: {v}." for k, v in zip(keys, vals)]
            q = (f"What are the special magic numbers for {keys[0]} and {keys[1]}? "
                 f"The special magic numbers for {keys[0]} and {keys[1]} are")
            return self._haystack(length, nds, q, rng), vals, 24
        if task == "vt":
            names = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(5)) for _ in range(5)]
            val = str(rng.randint(10000, 99999))
            chain = [f"VAR {names[0]} = {val}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, 5)]
            q = f"Find all variables that are assigned the value {val}. Answer: the variables are"
            return self._haystack(length, chain, q, rng), names, 32
        if task == "cwe":
            common = rng.sample(WORDS, 3)
            rest = [w for w in WORDS if w not in common]
            head = "Below is a numbered list of words. In these words, some appear more often than others.\n"
            q = "Question: What are the 3 most common words in the above list? Answer: The top 3 words are:"
            items, i = [], 1
            budget = length - self.n(head) - self.n(q) - 10
            pool = common * 30
            used = 0
            while used < budget:
                w = rng.choice(pool) if rng.random() < 0.3 else rng.choice(rest)
                s = f"{i}. {w} "
                items.append(s)
                used += self.n(s)
                i += 1
            return head + "".join(items) + "\n" + q, common, 16
        if task == "qa_squad":
            row = self.squad[rng.randrange(len(self.squad))]
            gold = row["context"]
            paras = [gold]
            used = self.n(gold)
            q = f"Answer the question based on the documents above.\nQuestion: {row['question']}\nAnswer:"
            budget = length - self.n(q) - 20
            while True:
                d = self.squad[rng.randrange(len(self.squad))]["context"]
                if d in paras:
                    continue
                if used + self.n(d) > budget:
                    break
                paras.insert(rng.randint(0, len(paras)), d)
                used += self.n(d)
            docs = "".join(f"Document {i + 1}:\n{p}\n\n" for i, p in enumerate(paras))
            return docs + q, row["answers"]["text"][:1], 24
        raise ValueError(task)


def score(task: str, out: str, refs: list[str]) -> float:
    o = out.lower()
    return sum(r.lower() in o for r in refs) / len(refs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--out_dir", default="results/eval")
    p.add_argument("--interventions", default="none")
    p.add_argument("--lengths", default="512,1024,2000")
    p.add_argument("--tasks", default=",".join(TASKS))
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_dir", default=DATA, help="ClimbMix shards for the mean vectors")
    a = p.parse_args()
    device = torch.device(a.device)
    tok = Tokenizer.from_file(a.tokenizer)
    model, state = load_model(a.ckpt, device)
    seq_len = int(state["config"]["model"]["seq_len"])
    del state
    has_memory = any(b.memory is not None for b in model.blocks)
    tasks = [t for t in a.tasks.split(",") if t]
    squad = None
    if "qa_squad" in tasks:
        from datasets import load_dataset

        squad = [r for r in load_dataset("rajpurkar/squad", split="validation") if r["answers"]["text"]]
    builder = Builder(tok, squad)
    lengths = [int(x) for x in a.lengths.split(",")]
    root = Path(a.out_dir) / a.name
    for spec in [s for s in a.interventions.split(",") if s]:
        mode, layers = parse_intervention(spec)
        if mode is not None and not has_memory:
            continue
        out_path = root / spec / "ruler.json"
        if out_path.exists():
            print(f"exists {out_path}", flush=True)
            continue
        means = load_or_make_means(model, root, seq_len, device, a.data_dir) if mode == "mean" else None
        ctx = Intervention(model, memory=mode, layers=layers, mean=means) if mode else nullcontext()
        res, examples, t0 = {}, {}, time.time()
        with ctx, torch.no_grad():
            for task in tasks:
                res[task] = {}
                for L in lengths:
                    rng = random.Random(f"{a.seed}-{task}-{L}")
                    scores = []
                    for i in range(a.n):
                        text, refs, max_new = builder.build(task, L, rng)
                        ids = [2] + tok.encode(text).ids
                        new = greedy_generate(model, ids, max_new, device, seq_len, use_cache=mode is None)
                        gen = tok.decode([t for t in new if t != 2])
                        scores.append(score(task, gen, refs))
                        if i == 0:
                            examples[f"{task}@{L}"] = {"refs": refs, "gen": gen, "prompt_tokens": len(ids)}
                    res[task][str(L)] = sum(scores) / len(scores)
                print(f"[{a.name} {spec}] {task}: {res[task]}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"name": a.name, "intervention": spec, "n": a.n, "result": res,
                                        "examples": examples, "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
