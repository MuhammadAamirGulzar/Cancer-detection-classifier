#!/usr/bin/env bash
# Patch-importance MIL for the remaining encoders, one after another (the HDD is the bottleneck).
# Bag building is retried, since it resumes from finished slides.
cd "$(dirname "$0")"
PY=/c/Users/datainsight/anaconda3/envs/exaonepath/python.exe
export PYTHONWARNINGS=ignore
for m in Virchow2 Conch1_5 ConchV1; do
  ok=0
  for i in 1 2 3 4; do
    "$PY" 01_build_bags.py --model "$m" > "logs/bags_${m}_${i}.log" 2>&1 && { ok=1; break; }
  done
  [ "$ok" = 1 ] || { echo "$m: bag build failed" >> logs/run_remaining.log; continue; }
  for meth in abmil topk; do
    "$PY" 02_train_abmil.py --model "$m" --method "$meth" > "logs/train_${m}_${meth}.log" 2>&1 || echo "$m $meth: train failed" >> logs/run_remaining.log
  done
  "$PY" 05_patchsum_baseline.py --model "$m" > "logs/train_${m}_patchsum.log" 2>&1 || echo "$m patchsum failed" >> logs/run_remaining.log
  for meth in abmil topk patchsum; do
    "$PY" 03_compare.py --model "$m" --method "$meth" > "logs/compare_${m}_${meth}.log" 2>&1 || echo "$m $meth: compare failed" >> logs/run_remaining.log
  done
  for meth in abmil topk; do
    "$PY" 04_explain.py --model "$m" --method "$meth" > "logs/explain_${m}_${meth}.log" 2>&1 || echo "$m $meth: explain failed" >> logs/run_remaining.log
  done
  echo "$m: done $(date +%H:%M)" >> logs/run_remaining.log
done
"$PY" 06_mlflow_table.py > logs/mlflow_table.log 2>&1
echo "all done $(date +%H:%M)" >> logs/run_remaining.log
