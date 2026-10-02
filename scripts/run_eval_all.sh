#!/usr/bin/env bash
# Quality evals (WikiText-2 PPL + lm-eval) for each model in $MODELS, serially.
# Quality numbers do not depend on GPU contention, so several of these runners may
# be used in parallel on disjoint model lists (vLLM start-up is serialized by a
# file lock in eval/common.py). Each step is skipped when its results exist and
# retried up to 3 times (transient start-up failures on the shared GB10).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-eval/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs/eval
TASKS="mmlu gsm8k arc_challenge hellaswag"
lm_done() { for t in $TASKS; do [ -f results/raw/lm_eval/$1/$t.json ] || return 1; done; }
for m in ${MODELS:?set MODELS}; do
  for try in 1 2 3; do
    [ -f results/raw/ppl/$m.json ] && break
    echo "[$(date '+%F %T')] ppl $m (try $try)"; (cd eval && python ppl_wikitext.py $m > ../logs/eval/ppl_$m.log 2>&1); sleep 20
  done
  for try in 1 2 3; do
    lm_done $m && break
    echo "[$(date '+%F %T')] lm-eval $m (try $try)"; (cd eval && python lm_eval_run.py $m > ../logs/eval/lmeval_$m.log 2>&1); sleep 20
  done
  ok=yes; [ -f results/raw/ppl/$m.json ] || ok=no; lm_done $m || ok=no
  echo "[$(date '+%F %T')] done $m complete=$ok"
done
