#!/usr/bin/env bash
set -euo pipefail

# Recipe: tp1-vllm-fp8-mtp4 (DP=8 x TP=1)
# Container: vllm/vllm-openai:v0.34.0
# Prerequisite: install the skinny Triton in_proj_ba plugin (qwen_g4_kernels) in the vLLM environment:
#   pip install git+https://github.com/jawadaminGOOG/g4-recipes.git#subdirectory=plugins/qwen_g4_kernels

export HF_XET_HIGH_PERFORMANCE=1

# NOTE: Prefix caching is ENABLED by default for production.
# Pass --no-enable-prefix-caching only to reproduce uncached stress benchmarks.
exec vllm serve Qwen/Qwen3.8-27B-FP8 \
  --host=0.0.0.0 \
  --port=8000 \
  --trust-remote-code \
  --language-model-only \
  --max-model-len=16384 \
  --kv-cache-dtype=fp8 \
  --mamba-ssm-cache-dtype=bfloat16 \
  --max-num-batched-tokens=4096 \
  --max-num-seqs=256 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":4,"draft_lm_head_quantization":"fp8"}' \
  "$@"
