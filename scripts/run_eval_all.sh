#!/usr/bin/env bash
# Quality evals (WikiText-2 PPL + lm-eval) for each model in $MODELS, serially.
# Quality numbers do not depend on GPU contention, so two of these runners may
# be used in parallel on disjoint model lists.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-eval/bin/activate
export PATH=/usr/local/cuda/bin:$PATH
mkdir -p logs/eval
for m in ${MODELS:?set MODELS}; do
  if [ ! -f results/raw/ppl/$m.json ]; then
    echo "[$(date '+%F %T')] ppl $m"; (cd eval && python ppl_wikitext.py $m > ../logs/eval/ppl_$m.log 2>&1)
  fi
  echo "[$(date '+%F %T')] lm-eval $m"; (cd eval && python lm_eval_run.py $m > ../logs/eval/lmeval_$m.log 2>&1)
  echo "[$(date '+%F %T')] done $m rc=$?"
done
