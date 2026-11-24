# Qwen3.8-27B Serving Recipes

Tuned serving recipes and scaling benchmarks for **Qwen3.8-27B** (hybrid Gated DeltaNet / Mamba + Attention dense 27B architecture) across three canonical workloads: **`10k/500`** (long-context enterprise), **`1k/8k`** (decode-heavy reasoning), and **`8k/1k`** (prefill-heavy RAG).

## Available Hardware Guides

| Hardware | Cloud Instance | Topologies Tested | `10k/500` Best `c=1` / Node Peak | `1k/8k` Best `c=1` / Node Peak | `8k/1k` Best `c=1` / Node Peak | Guide |
|---|---|---|---:|---:|---:|---|
| **NVIDIA RTX PRO 6000 Blackwell** (`SM120`, 96 GB) | GCP `g4-standard-*` | `TP=1` (`DP=8`), `TP=2`, `TP=4`, `TP=8` | **1.65 s** p90 / **4,172 tok/s** | **544 tok/s/u** (`15.2s`) / **29,699 tok/s** | **635 tok/s/u** (`2.21s`) / **8,583 tok/s** | [RTX PRO 6000 Guide](rtx-pro-6000/README.md) |
