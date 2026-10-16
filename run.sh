#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
DATASET=${DATASET:-squad}
SEEDS=${SEEDS:-42}
EPOCHS=${EPOCHS:-2}
DATA="data/$DATASET"
PYTHON=${PYTHON:-python3}
"$PYTHON" "$SCRIPT_DIR/bench.py" prepare --dataset "$DATASET" --model "$MODEL" --data "$DATA" --train-groups "${TRAIN_GROUPS:-1000}" --eval-groups "${EVAL_GROUPS:-100}"
mkdir -p results
"$PYTHON" "$SCRIPT_DIR/bench.py" eval --model "$MODEL" --data "$DATA" --output "results/${DATASET}_base.jsonl"
for seed in $SEEDS; do
  for format in single multi; do
    for loss in assistant full; do
      out="runs/${DATASET}_${format}_${loss}_${seed}"
      "$PYTHON" "$SCRIPT_DIR/bench.py" train --model "$MODEL" --data "$DATA" --format "$format" --loss "$loss" --seed "$seed" --epochs "$EPOCHS" --output "$out"
      "$PYTHON" "$SCRIPT_DIR/bench.py" eval --model "$MODEL" --data "$DATA" --adapter "$out" --output "results/${DATASET}_${format}_${loss}_${seed}.jsonl"
    done
  done
  "$PYTHON" "$SCRIPT_DIR/bench.py" compare --left "results/${DATASET}_multi_assistant_${seed}.jsonl" --right "results/${DATASET}_single_assistant_${seed}.jsonl" --output "results/${DATASET}_turn_effect_${seed}.json"
  for format in single multi; do
      "$PYTHON" "$SCRIPT_DIR/bench.py" compare --left "results/${DATASET}_${format}_full_${seed}.jsonl" --right "results/${DATASET}_${format}_assistant_${seed}.jsonl" --output "results/${DATASET}_${format}_prompt_effect_${seed}.json"
  done
done
