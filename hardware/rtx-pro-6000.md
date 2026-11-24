# NVIDIA RTX PRO 6000 Blackwell Server Edition

## Hardware Specifications

| Spec | Value |
|---|---|
| **Architecture** | NVIDIA Blackwell (`SM120`) |
| **VRAM** | 96 GB GDDR7 (`~94.97 GiB` allocatable per GPU) |
| **Memory Bandwidth** | `~1,598 GB/s` (measured) / `1,792 GB/s` (datasheet) |
| **Interconnect** | PCIe Gen5 x16 (no NVLink) |
| **Native Low Precision** | FP8, NVFP4 (FP4 Tensor Cores) |

## Cloud VM Mappings

| Cloud | Instance Family | GPUs | vCPUs | Host RAM | Notes |
|---|---|---:|---:|---:|---|
| **Google Cloud** | `g4-standard-48` | 1x | 48 | 192 GB | Single-GPU serving |
| **Google Cloud** | `g4-standard-96` | 2x | 96 | 384 GB | Single PCIe bridge (`PIX`) |
| **Google Cloud** | `g4-standard-192` | 4x | 192 | 768 GB | Single NUMA socket (`PIX` + `NODE`) |
| **Google Cloud** | `g4-standard-384` | 8x | 384 | 1,536 GB | Dual-socket PCIe (`PIX` + `NODE` + `SYS`) |

## Multi-GPU PCIe Topology (`8x` Dual-Socket Node)

On an 8-GPU `g4-standard-384` node (`nvidia-smi topo -m`):
- **NUMA Socket 0 (GPUs 0–3)**: `(0,1)` and `(2,3)` share a PCIe bridge (`PIX`); cross-pair `(0,2)` / `(1,3)` traverse the host bridge (`NODE`).
- **NUMA Socket 1 (GPUs 4–7)**: Symmetric layout on Socket 1.
- **Cross-Socket (GPUs 0–3 ↔ 4–7)**: Connected over the inter-socket fabric (`SYS`).

## Essential Multi-GPU Tuning Knobs (`TP >= 2`)

Because RTX PRO 6000 uses PCIe Gen5 without NVLink, default NCCL settings fall back to CPU shared-memory (`SHM`) staging for `TP >= 4`:

1. **`NCCL_P2P_LEVEL=SYS`**: Unlocks direct PCIe P2P DMA inside NCCL across `NODE` and `SYS` links (`~2x` lower small-message AllReduce latency and `38–40 GB/s` bus bandwidth on large prefill chunks).
2. **`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` + `NCCL_CUMEM_ENABLE=0`** *(vLLM)*: Activates FlashInfer's auto-tuned CUDA-IPC AllReduce for decode steps (`<= 512` tokens) while keeping `NCCL_P2P_LEVEL=SYS` for prefill chunks. Setting `NCCL_CUMEM_ENABLE=0` pairs cleanly with CUDA IPC and cuts CUDA Graph VRAM capture overhead in half (`3.73 → 1.80 GiB` on `TP=8`). Pair with `--gpu-memory-utilization=0.85` and persist `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` on local SSD so restarts skip the ~40 s autotuning warmup.
3. **`use_local_argmax_reduction: true`** *(vLLM speculative decoding on `TP >= 2`)*: Replaces the per-draft-step `O(vocab_size)` PCIe AllGather (`31.8 MB` at `c=64` on a 248k vocabulary) with an `O(2 * TP)` scalar reduction (`+4.6%` to `+6.2%` high-batch throughput on `TP=2/4/8`).
4. **`VLLM_USE_RUST_FRONTEND=1`** *(vLLM 0.34+)*: Launches the precompiled Tokio `vllm-rs` HTTP frontend, removing Python/uvicorn SSE overhead on long-generation decode bursts (`1k/8k`).
5. **Prefer `DP` over `TP` for throughput**: When a model fits on 1 GPU (e.g., quantized 27B–32B models), `DP=8 × TP=1` delivers `3.6x–7.1x` higher node throughput than `TP=4` or `TP=8` on `10k/500`, `6.56x` on `8k/1k`, and `4.67x` on `1k/8k`. Use `TP >= 2` only to hit strict single-request latency or interactivity SLAs.
