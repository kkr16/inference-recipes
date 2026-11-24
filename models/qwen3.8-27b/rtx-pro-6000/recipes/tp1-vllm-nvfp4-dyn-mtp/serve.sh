#!/usr/bin/env bash
set -euo pipefail

# Recipe: tp1-vllm-nvfp4-dyn-mtp (DP=8 x TP=1 — 1k/8k & 8k/1k Champion)
# Container: vllm/vllm-openai:v0.34.0
# Key levers: Dynamic MTP [[1,16,6],[17,256,3]], Tokio Rust frontend (vllm-rs),
# Extensible KV cache (omit --gpu-memory-utilization), fp8 KV + bf16 SSM.
# Do NOT enable qwen_g4_kernels (QWEN_G4_SKINNY_BA=0) on TP=1 NVFP4.

export HF_XET_HIGH_PERFORMANCE=1
export VLLM_USE_RUST_FRONTEND="1"
export QWEN_G4_SKINNY_BA="0"

# NOTE: Prefix caching is ENABLED by default for production.
# Pass --no-enable-prefix-caching only to reproduce uncached stress benchmarks.
exec vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --host=0.0.0.0 \
  --port=8000 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size=1 \
  --max-model-len=16384 \
  --kv-cache-dtype=fp8 \
  --mamba-ssm-cache-dtype=bfloat16 \
  --max-num-batched-tokens=4096 \
  --max-num-seqs=256 \
  '--speculative-config={"method":"qwen3_next_mtp","num_speculative_tokens":6,"num_speculative_tokens_per_batch_size":[[1,16,6],[17,256,3]]}' \
  "$@"
