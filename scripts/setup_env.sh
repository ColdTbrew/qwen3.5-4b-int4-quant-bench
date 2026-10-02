#!/usr/bin/env bash
# Creates two isolated uv venvs:
#   .venv-eval  : vLLM serving/benchmark + lm-evaluation-harness (vLLM backend)
#   .venv-quant : quantization toolchains (llm-compressor, auto-round, hqq)
# Two venvs are used because vLLM pins torch/transformers versions that the
# quantization libraries do not always agree with.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=3.12

uv venv --python $PY .venv-eval
uv pip install --python .venv-eval/bin/python \
  "vllm==0.30.0" "lm-eval[vllm]==0.4.13" "datasets>=3" pandas matplotlib pyyaml aiohttp

uv venv --allow-existing --python $PY .venv-quant
uv pip install --python .venv-quant/bin/python \
  "torch==2.13.0" "llmcompressor==0.14.0" "auto-round==0.15.1" "hqq==0.2.8.post1" \
  "datasets>=3" pyyaml accelerate
