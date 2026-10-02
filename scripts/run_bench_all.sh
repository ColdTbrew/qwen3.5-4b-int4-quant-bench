#!/usr/bin/env bash
# Serving benchmark for each model in $MODELS, serially, with the GPU otherwise idle.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-eval/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs/bench
for m in ${MODELS:?set MODELS}; do
  [ -f results/raw/bench/$m/summary.json ] && { echo "skip $m (exists)"; continue; }
  echo "[$(date '+%F %T')] bench $m"
  python bench/serve_bench.py $m > logs/bench/$m.log 2>&1
  echo "[$(date '+%F %T')] done $m rc=$?"
done
