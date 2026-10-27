#!/bin/bash
# ==============================================================================
# Project Morbius | Autonomous GPU Auto-Upgrade Daemon
# Continuously monitors Ada Plafnet and upgrades w2_mem_delayed4 from 1 GPU -> 2/4 GPUs
# whenever resources become free.
# ==============================================================================
# Usage:
#   # Run in foreground:
#   bash check/plafnet_scripts/run_auto_upgrade.sh
#
#   # Run in background with nohup (keeps running even if you disconnect from Ada):
#   nohup bash check/plafnet_scripts/run_auto_upgrade.sh > ~/morbius/joblogs/auto_upgrade.log 2>&1 &
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PY="$(command -v python3 || echo "python3")"

mkdir -p "$HOME/morbius/joblogs"

echo "=================================================================="
echo " Starting Morbius GPU Auto-Upgrade Daemon"
echo " Time: $(date)"
echo " Root: $ROOT_DIR"
echo "=================================================================="

exec "$PY" "$SCRIPT_DIR/auto_upgrade_gpu.py" "$@"
