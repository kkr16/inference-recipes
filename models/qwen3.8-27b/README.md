# Qwen3.8-27B Serving Recipes

Tuned serving recipes and scaling benchmarks for **Qwen3.8-27B** (hybrid Gated DeltaNet / Mamba + Attention dense 27B architecture).

## Available Hardware Guides

| Hardware | Cloud Instance | Topologies Tested | Best `c=1` E2E p90 (`10k/500`) | Peak 8-GPU Node Throughput | Guide |
|---|---|---|---:|---:|---|
| **NVIDIA RTX PRO 6000 Blackwell** (`SM120`, 96 GB) | GCP `g4-standard-*` | `TP=1` (`DP=8`), `TP=2`, `TP=4`, `TP=8` | **1.65 s** (`TP=8`, NVFP4 + MTP-5) | **4,172 tok/s** (`DP=8 × TP=1`, NVFP4 + MTP-4) | [RTX PRO 6000 Guide](rtx-pro-6000/README.md) |
