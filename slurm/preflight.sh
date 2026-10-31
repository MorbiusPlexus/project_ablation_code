#!/bin/bash
# First job on a node. Fails before any eval job if the env or a checkpoint is wrong.
set -euo pipefail
cd "${MORBIUS_DIR:-$HOME/morbius_eval}"
eval "$(bash slurm/stage.sh | grep -E '^(ENV_PY|DATA_DIR|EVAL_PY)=')"
PY="${EVAL_PY:-$ENV_PY}"
echo "host $(hostname) py $PY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
"$PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA missing")
print("gpu", torch.cuda.get_device_name(0))
x = torch.randn(2, 4, 8, 16, device="cuda")
y = torch.nn.functional.scaled_dot_product_attention(x, x, x, is_causal=True)
print("sdpa", float(y.float().abs().mean()))
PY
echo "preflight ok"
