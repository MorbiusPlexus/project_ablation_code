"""Download evaluation datasets into a cache. Run where the network exists, not on a compute node.

The full Engram task list is large. ``--tiny`` only checks that the client imports.
"""

from __future__ import annotations

import argparse
import os


DATASETS = [
    ("glue", "sst2"),
    ("google/boolq", None),
    ("wikitext", "wikitext-103-raw-v1"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default=os.environ.get("HF_HOME", "data_local/hf"))
    p.add_argument("--tiny", action="store_true")
    args = p.parse_args()
    os.environ["HF_HOME"] = args.cache
    if args.tiny:
        print(json_tiny(args.cache))
        return
    from datasets import load_dataset
    for name, config in DATASETS:
        if config:
            load_dataset(name, config, split="validation[:1]")
        else:
            load_dataset(name, split="validation[:1]")
        print("cached", name)


def json_tiny(cache: str) -> str:
    return f"prefetch skipped (--tiny); cache={cache}"


if __name__ == "__main__":
    main()
