#!/usr/bin/env bash
# Runs every quantization method serially with the GPU otherwise idle, so the
# recorded quantization wall times are not inflated by other jobs.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-quant/bin/activate
mkdir -p logs/quant
for m in ${METHODS:-rtn hqq_quant gptq awq autoround int8_sq_gptq}; do
  echo "[$(date '+%F %T')] start $m"
  python quant/$m.py > logs/quant/$m.log 2>&1
  echo "[$(date '+%F %T')] end $m rc=$?"
done
