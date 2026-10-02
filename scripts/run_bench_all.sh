#!/usr/bin/env bash
# Serving benchmark, one model at a time. Waits until no quality-eval process of
# this project is running so the GPU is otherwise idle (apart from other tenants).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-eval/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs/bench
for m in ${MODELS:?set MODELS}; do
  [ -f results/raw/bench/$m/summary.json ] && { echo "skip $m (exists)"; continue; }
  while pgrep -u "$(id -u)" -f "python (lm_eval_run|ppl_wikitext)\.py" > /dev/null; do sleep 60; done
  for try in 1 2; do
    echo "[$(date '+%F %T')] bench $m (try $try)"
    python bench/serve_bench.py $m > logs/bench/$m.log 2>&1
    [ -f results/raw/bench/$m/summary.json ] && break; sleep 30
  done
  echo "[$(date '+%F %T')] done $m ok=$([ -f results/raw/bench/$m/summary.json ] && echo yes || echo no)"
done
