#!/usr/bin/env bash
set -euo pipefail

# Recipe: tp2-vllm-nvfp4-dyn-mtp (DP=4 x TP=2 PIX — 1k/8k & 8k/1k Champion)
# Container: vllm/vllm-openai:v0.34.0
# Pin each TP=2 replica to a PCIe-switch-local PIX pair: (0,1), (2,3), (4,5), or (6,7).
# Optional for c=1 1k/8k (+7.1%): install qwen_g4_kernels plugin:
#   pip install git+https://github.com/jawadaminGOOG/g4-recipes.git#subdirectory=plugins/qwen_g4_kernels

export HF_XET_HIGH_PERFORMANCE=1
export NCCL_P2P_LEVEL="SYS"
export NCCL_CUMEM_ENABLE="0"
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="1"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:-/hf-cache/flashinfer_autotune_cache}"
export VLLM_USE_RUST_FRONTEND="1"
export QWEN_G4_SKINNY_BA="${QWEN_G4_SKINNY_BA:-1}"

# NOTE: Prefix caching is ENABLED by default for production.
# Pass --no-enable-prefix-caching only to reproduce uncached stress benchmarks.
exec vllm serve nvidia/Qwen3.8-27B-NVFP4 \
  --host=0.0.0.0 \
  --port=8000 \
  --trust-remote-code \
  --language-model-only \
  --tensor-parallel-size=2 \
  --gpu-memory-utilization=0.85 \
  --max-model-len=16384 \
  --kv-cache-dtype=fp8 \
  --mamba-ssm-cache-dtype=bfloat16 \
  --max-num-batched-tokens=4096 \
  --max-num-seqs=256 \
  '--speculative-config={"method":"qwen3_next_mtp","num_speculative_tokens":6,"num_speculative_tokens_per_batch_size":[[1,4,6],[5,16,4],[17,256,3]],"use_local_argmax_reduction":true}' \
  "$@"
