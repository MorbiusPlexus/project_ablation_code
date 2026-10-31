#!/bin/bash
# Rsync this checkout to ~/morbius_eval. Never touches ~/morbius (a running job may be using it).
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${MORBIUS_EVAL_DEST:-$HOME/morbius_eval}"
if grep -R -l $'\r' "$SRC/slurm" "$SRC/ffnmem" >/dev/null 2>&1; then
  echo "CRLF found; convert to LF before syncing (sbatch rejects CRLF)" >&2
  exit 1
fi
mkdir -p "$DEST"
rsync -a --delete --exclude .git --exclude .venv-eval --exclude results --exclude runs \
  "$SRC/" "$DEST/"
if command -v git >/dev/null && git -C "$SRC" rev-parse HEAD >/dev/null 2>&1; then
  git -C "$SRC" rev-parse HEAD > "$DEST/GIT_REV"
fi
echo "synced $SRC -> $DEST"
