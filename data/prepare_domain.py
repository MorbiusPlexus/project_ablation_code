"""Tokenize a text corpus into the uint16 shard format the trainer already reads.

Pass ``--text`` a utf-8 file with one document per line to build a domain
without contacting Hugging Face. ``--hf ncbi/pubmed`` streams abstracts when
the dataset library and a network are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def tokenize_docs(docs: list[str], tokenizer_path: str, eos_id: int = 2) -> list[np.ndarray]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    out = []
    for doc in docs:
        ids = tok.encode(doc).ids
        if not ids:
            continue
        arr = np.asarray(ids + [eos_id], dtype="<u2")
        out.append(arr)
    return out


def write_shards(docs: list[np.ndarray], out_dir: Path, val_tokens: int, vocab: int, eos_id: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = np.concatenate(docs) if docs else np.zeros(0, dtype="<u2")
    val_n = min(val_tokens, max(0, len(stream) // 5))
    val = stream[:val_n]
    train = stream[val_n:]
    val.astype("<u2").tofile(out_dir / "val.bin")
    # One shard is enough for a domain slice; split only so a shard stays near 1M tokens.
    shards = []
    chunk = 1_000_000
    if len(train) == 0:
        train = np.asarray([eos_id], dtype="<u2")
    for i in range(0, len(train), chunk):
        name = f"shard_{i // chunk:05d}.bin"
        piece = train[i : i + chunk]
        piece.astype("<u2").tofile(out_dir / name)
        shards.append({"file": name, "tokens": int(len(piece))})
    meta = {
        "dataset": "domain",
        "vocab_size": vocab,
        "eos_id": eos_id,
        "dtype": "uint16",
        "byte_order": "little",
        "val_file": "val.bin",
        "val_tokens": int(len(val)),
        "shards": shards,
        "total_tokens": int(sum(s["tokens"] for s in shards)),
        "complete": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def docs_from_text(path: Path, limit: int | None) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:limit] if limit else lines


def docs_from_hf(name: str, limit: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True)
    docs = []
    for row in ds:
        text = row.get("abstract") or row.get("text") or row.get("content") or ""
        if text:
            docs.append(text)
        if len(docs) >= limit:
            break
    return docs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--text")
    p.add_argument("--hf")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--vocab", type=int, default=32768)
    p.add_argument("--val_tokens", type=int, default=2000)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)
    if args.text:
        docs = docs_from_text(Path(args.text), args.limit)
    elif args.hf:
        docs = docs_from_hf(args.hf, args.limit)
    else:
        docs = [f"pubmed abstract number {i} about mitral valve protein kinase" for i in range(32)]
    arrays = tokenize_docs(docs, args.tokenizer)
    meta = write_shards(arrays, Path(args.out), args.val_tokens, args.vocab)
    from ffnmem.data.loader import build_index

    build_index(args.out, args.seq_len, args.seed, overwrite=True)
    print(json.dumps({"out": args.out, "tokens": meta["total_tokens"], "val": meta["val_tokens"]}))
    return meta


if __name__ == "__main__":
    main()
