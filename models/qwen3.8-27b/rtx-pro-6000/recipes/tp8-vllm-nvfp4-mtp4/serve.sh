#!/usr/bin/env bash
set -euo pipefail

# Recipe: tp8-vllm-nvfp4-mtp4 (1 x TP=8)
# Container: vllm/vllm-openai:v0.34.0
# Prerequisite: install the skinny Triton in_proj_ba plugin (qwen_g4_kernels) in the vLLM environment:
#   pip install git+https://github.com/jawadaminGOOG/g4-recipes.git#subdirectory=plugins/qwen_g4_kernels

export HF_XET_HIGH_PERFORMANCE=1
export NCCL_P2P_LEVEL="SYS"
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="1"

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
  --tensor-parallel-size=8 \
  --gpu-memory-utilization=0.85 \
  '--speculative-config={"method":"mtp","num_speculative_tokens":4}' \
  "$@"
