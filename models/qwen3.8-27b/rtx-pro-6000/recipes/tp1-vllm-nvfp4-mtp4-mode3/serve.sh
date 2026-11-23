#!/usr/bin/env bash
set -euo pipefail

# Recipe: tp1-vllm-nvfp4-mtp4-mode3 (DP=8 x TP=1)
# Container: vllm/vllm-openai:v0.34.0
export HF_XET_HIGH_PERFORMANCE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH="0"

# NOTE: Prefix caching is ENABLED by default for production.
# Pass --no-enable-prefix-caching only to reproduce uncached stress benchmarks.
exec vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --host=0.0.0.0 \
  --port=8000 \
  --trust-remote-code \
  --language-model-only \
  --max-model-len=16384 \
  --kv-cache-dtype=fp8 \
  --mamba-ssm-cache-dtype=bfloat16 \
  --max-num-batched-tokens=4096 \
  --max-num-seqs=256 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":4}' \
  '--compilation-config={"mode":3,"pass_config":{"fuse_qk_norm_rope_kvcache":true}}' \
  "$@"
