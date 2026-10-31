#!/bin/bash
# CPU job on ada. Write an fp16 copy of a final checkpoint, keeping the router
# and RMSNorm weights in fp32 (sign bits and the norm scale are precision-sensitive).
#   bash slurm/export_local.sh /path/to/w1_mem_full_step3815.pt /path/to/out.pt
set -euo pipefail
IN=$1
OUT=$2
PY="${ENV_PY:-python3}"
"$PY" - "$IN" "$OUT" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
state = torch.load(src, map_location="cpu", weights_only=False)
model = state["model"]
for name, tensor in list(model.items()):
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        continue
    keep = ("router" in name) or name.endswith("norm.weight") or name.endswith("attn_norm.weight") or name.endswith("ffn_norm.weight")
    if tensor.dtype == torch.float32 and not keep:
        model[name] = tensor.half()
keep_keys = ("model", "config", "step", "params")
small = {k: state[k] for k in keep_keys if k in state}
torch.save(small, dst)
print("wrote", dst)
PY
