# Inference Recipes

Reproducible, tuned LLM serving recipes, controlled ablations, and multi-GPU scaling benchmarks across accelerators and serving engines.

📖 **[Live Documentation & Pareto Explorer → kkr16.github.io/inference-recipes](https://kkr16.github.io/inference-recipes/)**

## Recipe Catalog

<!-- CATALOG:START -->
| Model | Hardware | Topology | Engine & Stack | Workload | `c=1` E2E p90 | Peak Throughput | Date | Recipe |
|---|---|---|---|---|---:|---:|---|---|
| **Qwen3.8-27B** | `rtx-pro-6000` | `DP=8 x TP=1` | vLLM 0.34.0 · FP8 (MTP-4, fp8-draft-head, qwen_g4_kernels, extensible-kv) | 10k/500 (uncached) | **4.91 s** | **2,550.3 tok/s** (DP=8 @ c=8/GPU) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp1-vllm-fp8-mtp4) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `DP=8 x TP=1` | vLLM 0.34.0 · NVFP4 (MTP-4, mode=3, extensible-kv) | 10k/500 (uncached) | **3.75 s** | **4,106.9 tok/s** (DP=8 @ c=16/GPU) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp1-vllm-nvfp4-mtp4-mode3) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `DP=8 x TP=1` | vLLM 0.34.0 · NVFP4 (MTP-4, bf16-ssm, fp8-kv) | 10k/500 (uncached) | **3.86 s** | **4,172.4 tok/s** (DP=8 @ c=24/GPU) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp1-vllm-nvfp4-mtp4) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `DP=4 x TP=2` | vLLM 0.34.0 · NVFP4 (MTP-4, PCIE_IPC, P2P=SYS) | 10k/500 (uncached) | **2.62 s** | **2,090.9 tok/s** (DP=4 @ c=16/rep) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp2-vllm-nvfp4-mtp4) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `DP=2 x TP=4` | vLLM 0.34.0 · NVFP4 (MTP-4, PCIE_IPC, P2P=SYS) | 10k/500 (uncached) | **2.13 s** | **1,137.2 tok/s** (DP=2 @ c=16/rep) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp4-vllm-nvfp4-mtp4) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `1 x TP=8` | vLLM 0.34.0 · NVFP4 (MTP-4, PCIE_IPC, P2P=SYS, qwen_g4_kernels) | 10k/500 (uncached) | **1.76 s** | **578.9 tok/s** (TP=8 @ c=16) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp8-vllm-nvfp4-mtp4) |
| **Qwen3.8-27B** | `rtx-pro-6000` | `1 x TP=8` | vLLM 0.34.0 · NVFP4 (MTP-5, PCIE_IPC, P2P=SYS, qwen_g4_kernels) | 10k/500 (uncached) | **1.65 s** | **400.5 tok/s** (TP=8 @ c=2) | 2026-11-23 | [Guide](models/qwen3.8-27b/rtx-pro-6000/README.md) · [Files](models/qwen3.8-27b/rtx-pro-6000/recipes/tp8-vllm-nvfp4-mtp5) |
<!-- CATALOG:END -->

## Repository Structure

```text
inference-recipes/
├── README.md                    # Master catalog & quick start
├── CONTRIBUTING.md              # Naming conventions & recipe checklist
├── hardware/                    # Accelerator specs, interconnect topology & cloud VM mappings
├── infra/                       # Shared GKE / Kubernetes deployment templates
├── bench/                       # Shared dataset generator, async benchmark client & catalog builder
├── plugins/                     # Reusable custom kernels / out-of-tree engine plugins
└── models/                      # Model recipes (lowercase kebab-case)
    └── <model-name>/
        ├── README.md            # Cross-hardware summary
        └── <accelerator>/       # Hardware-specific tuning guide, recipes/ & results/
```

## Quick Start

1. Browse the catalog above and open the model's hardware guide (e.g., [Qwen3.8-27B on NVIDIA RTX PRO 6000](models/qwen3.8-27b/rtx-pro-6000/README.md)).
2. Run a recipe directly on a GPU host (`serve.sh`) or deploy on Kubernetes / GKE (`deploy-gke.yaml`).
3. Reproduce benchmarks using the shared harness in [`bench/`](bench/README.md).
