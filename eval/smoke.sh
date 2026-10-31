#!/bin/bash
# Tiered smoke. Tier 1 is pytest. Tier 2 adds the local suite (includes decode and offload).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
TIER="${1:-1}"
python -m pytest tests/test_eval_infra.py -q
if [ "$TIER" -ge 2 ]; then
  python eval/make_tiny_ckpt.py
  python eval/inventory_ckpts.py
  python eval/run_local_suite.py
  python analysis/aggregate.py
  python eval/submit.py --dry-run | head -n 5
fi
echo "smoke tier $TIER ok"
