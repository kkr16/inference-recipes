# Benchmarking Harness

Shared scripts for generating exact-token synthetic workloads, running async OpenAI-compatible streaming benchmarks, and regenerating the root catalog table.

## Scripts

- [`scripts/make_dataset.py`](scripts/make_dataset.py): Generates unique, realistic multi-turn enterprise prompts verified to tokenize to an exact input sequence length (default: `10,000` tokens, `add_special_tokens=False`).
- [`scripts/bench_client.py`](scripts/bench_client.py): Async streaming client for `/v1/completions` supporting both closed-loop concurrency sweeps and open-loop Poisson arrival rates. Collects client-side `TTFT`, `TPOT`, `ITL`, `E2E` percentiles and scrapes Prometheus `/metrics` deltas (including speculative decoding acceptance length for both vLLM and SGLang).
- [`scripts/build_catalog.py`](scripts/build_catalog.py): Reads all `models/**/metadata.yaml` files and prints/updates the Markdown leaderboard in the root `README.md`.

## Standard Methodology

1. **Exact token counts**: Prompts are verified to exact ISL (`add_special_tokens=False`), and requests pass `max_tokens=<OSL>`, `ignore_eos=True`, `temperature=0.0`, `stream=True`.
2. **Uncached stress baseline**: Unless a recipe explicitly benchmarks prefix caching, disable prefix/radix caching (`--no-enable-prefix-caching` on vLLM, `--disable-radix-cache` on SGLang) during measurement so every request pays full prefill cost. *(Re-enable prefix caching in production!)*
3. **Warmup**: Every concurrency/RPS point executes un-timed warmup requests before collecting timed metrics.
4. **Zero-error gate**: All published numbers require `0` failed requests.

## Quick Usage

```bash
# 1. Generate 128 exact 10k-token prompts (requires transformers)
python3 bench/scripts/make_dataset.py \
  --tokenizer nvidia/Qwen3.8-27B-NVFP4 \
  --target-tokens 10000 \
  --num-prompts 128 \
  --output /tmp/prompts-10k.jsonl

# 2. Run a closed-loop concurrency benchmark
python3 bench/scripts/bench_client.py \
  --url http://127.0.0.1:8000 \
  --model nvidia/Qwen3.8-27B-NVFP4 \
  --dataset /tmp/prompts-10k.jsonl \
  --max-tokens 500 \
  --concurrency 4 \
  --num-requests 16 \
  --warmup 2 \
  --output /tmp/result_c4.json

# 3. Rebuild the root README catalog table from metadata.yaml files
python3 bench/scripts/build_catalog.py --write
```
