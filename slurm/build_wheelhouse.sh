#!/bin/bash
# Run on the ada login node (it has internet). Compute nodes install from this directory.
set -euo pipefail
ROOT="${MORBIUS_DIR:-$HOME/morbius_eval}"
DEST="${ROOT}/wheels"
mkdir -p "$DEST"
python3 -m pip download -d "$DEST" -r "$ROOT/requirements-eval.txt"
echo "wheels in $DEST"
