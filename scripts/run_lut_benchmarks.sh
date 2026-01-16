#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
results_file="acc_results.txt"

  {
  echo "=== Baseline ==="
  python scripts/run_benchmark_eval.py \
    --model meta-llama/Llama-2-7b-hf \
    --tasks siqa \
    --linear_layer Linear
  } >> "${results_file}" 2>&1

for n in {5..7}; do
  {
    echo "=== lut-size ${n} ==="
    python scripts/run_benchmark_eval.py \
      --model meta-llama/Llama-2-7b-hf \
      --tasks siqa \
      --linear_layer LinearNF4Compute \
      --lut-size "${n}"
    echo
  } >> "${results_file}" 2>&1
done
