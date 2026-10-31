#!/bin/bash
# Merge results from the borrowed accounts (home dirs and node-local /ssd_scratch) into
# anangappara's results tree, relayed through this machine (the accounts cannot write to
# each other's homes, and scratch is only reachable from inside the hold job via srun).
#   bash slurm/sync_results.sh
set -eu
STAGE=${STAGE:-/tmp/morbius_merge}
ANA="ssh -o ControlPath=/tmp/ada2-%r@%h"
# name  ssh-wrapper  hold-job (empty = login node)  results dir
SOURCES=(
  "adi_home  /tmp/adissh.sh  -      /home2/aditya.shankar/mrun/results"
  "sp_home   /tmp/spssh.sh   -      /home2/sparsh.bhartia/mrun/results"
  "adi68     /tmp/adissh.sh  50629  /ssd_scratch/aditya.shankar/mrun/results"
  "sp56      /tmp/spssh.sh   50688  /ssd_scratch/sparsh.bhartia/mrun/results"
  "sp57      /tmp/spssh.sh   50908  /ssd_scratch/sparsh.bhartia/mrun57/results"
  "af44      /tmp/afssh.sh   50883  /ssd_scratch/afoolishman/mrun/results"
  "af57      /tmp/afssh.sh   50903  /ssd_scratch/afoolishman/mrun57/results"
)
for s in "${SOURCES[@]}"; do
  read -r name ssh job dir <<< "$s"
  mkdir -p "$STAGE/$name"
  if [ "$job" = - ]; then pre=""; else pre="srun --jobid $job --overlap "; fi
  $ssh "${pre}bash -c 'test -d $dir && tar -C $dir --exclude=*.pt -cf - . || tar -cf - -T /dev/null'" | tar -C "$STAGE/$name" -xf -
  echo "$name: $(find "$STAGE/$name" -name '*.json' | wc -l) json"
done
for s in "${SOURCES[@]}"; do
  read -r name _ <<< "$s"
  rsync -az --ignore-existing --exclude '*.pt' -e "$ANA" "$STAGE/$name/" anangappara@10.4.24.29:morbius_eval_runs/results/
done
echo "merged $(find "$STAGE" -name '*.json' | wc -l) json files"
