# Contributing & Repository Conventions

This guide keeps `inference-recipes` consistent as new models, accelerators, and serving engines are added.

## Directory Layout

```text
models/
└── <model-slug>/                  # lowercase kebab-case (e.g., qwen3.8-27b, deepseek-v4-flash)
    ├── README.md                  # cross-hardware summary & links
    └── <accelerator-slug>/        # accelerator-first slug (e.g., rtx-pro-6000, b200, gb200, h200, tpu-v6e)
        ├── README.md              # tuning story, ablations, scaling tables & charts
        ├── recipes/
        │   └── <topology>-<engine>-<quant>[-<feature>]/
        │       ├── serve.sh       # bare-metal / container launch script
        │       ├── deploy-gke.yaml # (optional) Kubernetes / GKE manifest
        │       └── metadata.yaml  # structured summary for auto-building the root catalog
        └── results/               # compact summary CSV/JSON and PNG charts
```

## Naming Rules

- **Models**: Lowercase `kebab-case`, matching the canonical model name (`qwen3.8-27b`, `glm-5.3-flash`).
- **Accelerators**: Hardware-first slugs (`rtx-pro-6000`, `b200`, `gb200`, `h200`, `tpu-v6e`). Map cloud instance types (e.g., GCP G4, A4, A4X) inside `hardware/<accelerator>.md` and `metadata.yaml`.
- **Recipes**: `<topology>-<engine>-<quant>[-<feature>]` (e.g., `tp1-vllm-nvfp4-mtp4`, `tp8-vllm-nvfp4-mtp5`).

## Recipe Checklist

1. **Reproducible launch**: Include a runnable `serve.sh` (and `deploy-gke.yaml` where applicable) with pinned container image tags and exact flags/env vars.
2. **Structured metadata**: Include `metadata.yaml` so `python3 bench/scripts/build_catalog.py` can regenerate the root `README.md` table.
3. **Keep Git lightweight**: Do not commit multi-MB prompt datasets or raw server logs. Use `bench/scripts/` to generate datasets and commit only compact summaries and charts.
4. **Sanitize before pushing**: Never include customer names, private GCP project IDs, internal storage buckets, or internal links.
