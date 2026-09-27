#!/bin/bash
# Memory zero-out probe on gnode118 (run 1 + run 2 at step 2000; run 4 as dense reference).
cd ~/morbius
B=/ssd_scratch/$USER/morbius
PY=$(ls -d $B/python/cpython-3.12*/bin/python3.12 | head -1)
P=$B/probe; D=$B/climbmix; OUT=$P/probe_results.jsonl
run() { CUDA_VISIBLE_DEVICES=$1 $PY -m ffnmem.probe_memory --ckpt $2 --data_dir $D --zero $3 --label $4 --out $OUT ${@:5} > $P/log_$4_$3.txt 2>&1; }
R4=$B/runs/w1_dense_full/checkpoints/step_0002000.pt
( run 0 $P/run1_step2000_model.pt none run1 --dump 1000 --dump_dir $P/dump_run1; run 0 $P/run1_step2000_model.pt all run1 ) &
( run 1 $P/run1_step2000_model.pt 1 run1; run 1 $P/run1_step2000_model.pt 11 run1; run 1 $R4 none run4 ) &
( run 2 $P/run2_step2000_model.pt none run2 --dump 1000 --dump_dir $P/dump_run2; run 2 $P/run2_step2000_model.pt all run2 ) &
( run 3 $P/run2_step2000_model.pt 1 run2; run 3 $P/run2_step2000_model.pt 11 run2 ) &
wait
echo PROBE_DONE
