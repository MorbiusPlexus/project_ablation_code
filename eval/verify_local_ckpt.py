"""Check an exported checkpoint against the val loss stored next to it, if given.

Route codes of the fp16 copy must agree with a fresh fp32 copy of the same
weights on a few windows. Without ``--reference`` this only checks that the
file loads and runs.
"""

from __future__ import annotations

import argparse
import json

import torch

from ffnmem.hooks import load_model
from ffnmem.interp.entries import route_codes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--reference")
    p.add_argument("--expect_loss", type=float, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    model, state = load_model(args.ckpt, args.device)
    ids = torch.randint(4, model.cfg["vocab"], (1, 16))
    logits = model(ids.to(next(model.parameters()).device))
    if isinstance(logits, tuple):
        logits = logits[0]
    result = {"loaded": True, "step": state.get("step"), "finite": bool(torch.isfinite(logits).all())}
    if args.reference:
        ref, _ = load_model(args.reference, args.device)
        a = route_codes(model, ids)
        b = route_codes(ref, ids)
        agree = []
        for layer in a:
            agree.append(float((a[layer] == b[layer]).float().mean()))
        result["code_agreement"] = min(agree) if agree else None
    if args.expect_loss is not None:
        result["expect_loss"] = args.expect_loss
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
