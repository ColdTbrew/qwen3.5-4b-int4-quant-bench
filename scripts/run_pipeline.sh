#!/usr/bin/env bash
# End-to-end: quality evals (2 parallel runners on disjoint model lists), then serving bench (serial, GPU otherwise idle), then aggregate + plots.
set -uo pipefail
cd "$(dirname "$0")/.."
log=logs/pipeline.log
echo "[$(date "+%F %T")] pipeline start" >> $log
MODELS="bf16 rtn-w4a16-g128 gptq-w4a16-g128-actorder awq-w4a16-g128" bash scripts/run_eval_all.sh > logs/eval/runner_a.log 2>&1 &
A=$!
MODELS="int8-w8a8-smoothquant-gptq hqq-w4a16-g128 autoround-w4a16-g128" bash scripts/run_eval_all.sh > logs/eval/runner_b.log 2>&1 &
B=$!
wait $A $B
echo "[$(date "+%F %T")] evals done" >> $log
MODELS="bf16 int8-w8a8-smoothquant-gptq rtn-w4a16-g128 gptq-w4a16-g128-actorder awq-w4a16-g128 hqq-w4a16-g128 autoround-w4a16-g128" bash scripts/run_bench_all.sh > logs/bench/runner.log 2>&1
echo "[$(date "+%F %T")] bench done" >> $log
.venv-eval/bin/python scripts/aggregate.py >> $log 2>&1 && .venv-eval/bin/python scripts/plots.py >> $log 2>&1
echo "[$(date "+%F %T")] PIPELINE_DONE rc=$?" >> $log
