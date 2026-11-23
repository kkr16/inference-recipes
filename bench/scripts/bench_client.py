#!/usr/bin/env python3
"""Unified async streaming benchmark harness for SGLang and vLLM OpenAI /v1/completions."""

import argparse
import asyncio
import json
import math
import random
import statistics
import time
import urllib.request
from typing import Any, Dict, List, Optional

import aiohttp


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "mean": round(statistics.mean(s), 3),
        "p50": round(percentile(s, 50), 3),
        "p90": round(percentile(s, 90), 3),
        "p99": round(percentile(s, 99), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
    }


def fetch_text(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def parse_prom_metrics(text: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not text:
        return out
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        metric_key, val_str = parts
        metric_name = metric_key.split("{", 1)[0]
        try:
            val = float(val_str)
        except ValueError:
            continue
        out[metric_name] = out.get(metric_name, 0.0) + val
    return out


def extract_spec_deltas(m_before: Dict[str, float], m_after: Dict[str, float]) -> Dict[str, Any]:
    deltas: Dict[str, float] = {}
    all_keys = set(m_before.keys()) | set(m_after.keys())
    for k in sorted(all_keys):
        if any(x in k for x in ["spec", "prefix", "cache_hit", "radix", "accept", "draft", "preempt"]):
            deltas[k] = round(m_after.get(k, 0.0) - m_before.get(k, 0.0), 4)
    # Compute derived accept length / rate where counters exist
    derived: Dict[str, Any] = {"raw_deltas": deltas}
    # vLLM spec decode counters
    drafts = deltas.get("vllm:spec_decode_num_drafts_total", deltas.get("vllm:spec_decode_num_drafts", 0.0))
    draft_toks = deltas.get("vllm:spec_decode_num_draft_tokens_total", deltas.get("vllm:spec_decode_num_draft_tokens", 0.0))
    acc_toks = deltas.get("vllm:spec_decode_num_accepted_tokens_total", deltas.get("vllm:spec_decode_num_accepted_tokens", 0.0))
    if drafts > 0:
        derived["vllm_avg_accepted_tokens_per_draft"] = round(acc_toks / drafts, 4)
        derived["vllm_mean_tokens_per_step"] = round(1.0 + (acc_toks / drafts), 4)
    if draft_toks > 0:
        derived["vllm_acceptance_rate"] = round(acc_toks / draft_toks, 4)
    # SGLang spec accept length (can be gauge or sum/count histogram)
    if "sglang:spec_accept_length" in m_after:
        derived["sglang_spec_accept_length_gauge"] = round(m_after["sglang:spec_accept_length"], 4)
    if deltas.get("sglang:spec_accept_length_count", 0.0) > 0:
        derived["sglang_spec_accept_length_delta"] = round(
            deltas["sglang:spec_accept_length_sum"] / deltas["sglang:spec_accept_length_count"], 4
        )
    # SGLang total spec tokens/steps if present
    for k in m_after:
        if "accept" in k or "spec" in k:
            derived.setdefault("after_gauges", {})[k] = round(m_after[k], 4)
    return derived


async def send_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt_item: Dict[str, Any],
    max_tokens: int,
    use_min_tokens: bool,
    capture_sample: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt_item["prompt"],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if use_min_tokens:
        payload["min_tokens"] = max_tokens

    t_send = time.perf_counter()
    t_first: Optional[float] = None
    t_last_text: Optional[float] = None
    prev_chunk_t: Optional[float] = None
    itl_list: List[float] = []
    num_chunks = 0
    prompt_tokens = 0
    completion_tokens = 0
    chunks_text: List[str] = []
    error_msg: Optional[str] = None

    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "ok": False,
                    "error": f"HTTP {resp.status}: {body[:500]}",
                    "prompt_id": prompt_item["id"],
                }
            buf = b""
            async for raw_chunk in resp.content.iter_any():
                buf += raw_chunk
                while b"\n" in buf:
                    line_bytes, buf = buf.split(b"\n", 1)
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                    choices = obj.get("choices") or []
                    if choices:
                        txt = choices[0].get("text")
                        if txt is not None and txt != "":
                            now = time.perf_counter()
                            num_chunks += 1
                            if t_first is None:
                                t_first = now
                            else:
                                if prev_chunk_t is not None:
                                    itl_list.append((now - prev_chunk_t) * 1000.0)
                            prev_chunk_t = now
                            t_last_text = now
                            if capture_sample:
                                chunks_text.append(txt)
            t_end = time.perf_counter()
            if t_last_text is None:
                t_last_text = t_end
            if t_first is None:
                return {
                    "ok": False,
                    "error": "No token chunks received",
                    "prompt_id": prompt_item["id"],
                }
            ttft_ms = (t_first - t_send) * 1000.0
            e2e_ms = (t_end - t_send) * 1000.0
            if completion_tokens == 0:
                completion_tokens = num_chunks
            tpot_ms = ((e2e_ms - ttft_ms) / max(1, completion_tokens - 1))
            return {
                "ok": True,
                "prompt_id": prompt_item["id"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_ms": ttft_ms,
                "e2e_ms": e2e_ms,
                "tpot_ms": tpot_ms,
                "itl_ms": itl_list,
                "num_chunks": num_chunks,
                "sample_text": "".join(chunks_text) if capture_sample else None,
            }
    except Exception as exc:
        error_msg = repr(exc)
        return {"ok": False, "error": error_msg, "prompt_id": prompt_item["id"]}


async def run_phase(
    url: str,
    model: str,
    prompts: List[Dict[str, Any]],
    num_requests: int,
    concurrency: Optional[int],
    request_rate: Optional[float],
    max_tokens: int,
    use_min_tokens: bool,
    start_idx: int = 0,
    dump_first: bool = False,
) -> Dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)
    sem = asyncio.Semaphore(concurrency) if concurrency else None

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker(req_idx: int, delay_s: float) -> Dict[str, Any]:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            p = prompts[(start_idx + req_idx) % len(prompts)]
            cap = dump_first and req_idx == 0
            if sem is not None:
                async with sem:
                    return await send_one_request(
                        session, url, model, p, max_tokens, use_min_tokens, capture_sample=cap
                    )
            return await send_one_request(
                session, url, model, p, max_tokens, use_min_tokens, capture_sample=cap
            )

        delays: List[float] = []
        if request_rate is not None and request_rate > 0:
            rng = random.Random(42 + start_idx)
            cur = 0.0
            for _ in range(num_requests):
                delays.append(cur)
                cur += rng.expovariate(request_rate)
        else:
            delays = [0.0] * num_requests

        t0 = time.perf_counter()
        results = await asyncio.gather(*(worker(i, delays[i]) for i in range(num_requests)))
        wall_s = time.perf_counter() - t0

    return {"wall_s": wall_s, "results": results}


async def main_async(args):
    prompts: List[Dict[str, Any]] = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    assert len(prompts) > 0, "Empty dataset"

    # Warmup
    if args.warmup > 0:
        w_conc = min(args.concurrency or 4, args.warmup)
        print(f"[{args.label}] Running warmup: {args.warmup} prompts at concurrency={w_conc} ...", flush=True)
        w_res = await run_phase(
            args.url, args.model, prompts, args.warmup, w_conc, None,
            args.max_tokens, args.use_min_tokens, start_idx=0, dump_first=args.dump_sample
        )
        w_ok = [r for r in w_res["results"] if r["ok"]]
        w_fail = [r for r in w_res["results"] if not r["ok"]]
        if w_fail:
            print(f"[{args.label}] Warmup had {len(w_fail)} failures, first error: {w_fail[0]['error']}", flush=True)
            if len(w_ok) == 0:
                raise RuntimeError(f"All warmup requests failed: {w_fail[0]['error']}")
        if args.dump_sample and w_ok and w_ok[0].get("sample_text"):
            sample_path = args.output_json.replace(".json", "_sample.txt")
            with open(sample_path, "w", encoding="utf-8") as sf:
                sf.write(w_ok[0]["sample_text"])
            print(f"[{args.label}] Saved sample text ({len(w_ok[0]['sample_text'])} chars) to {sample_path}", flush=True)

    m_before = parse_prom_metrics(fetch_text(f"{args.url}/metrics"))
    info_before = fetch_text(f"{args.url}/get_server_info")

    mode_desc = f"rps={args.request_rate}" if args.request_rate else f"concurrency={args.concurrency}"
    print(f"[{args.label}] Running timed benchmark: N={args.num_prompts}, {mode_desc} ...", flush=True)
    res = await run_phase(
        args.url, args.model, prompts, args.num_prompts, args.concurrency, args.request_rate,
        args.max_tokens, args.use_min_tokens, start_idx=args.warmup, dump_first=False
    )

    m_after = parse_prom_metrics(fetch_text(f"{args.url}/metrics"))
    info_after = fetch_text(f"{args.url}/get_server_info")

    ok = [r for r in res["results"] if r["ok"]]
    fail = [r for r in res["results"] if not r["ok"]]
    wall_s = res["wall_s"]

    prompt_toks = [r["prompt_tokens"] for r in ok]
    comp_toks = [r["completion_tokens"] for r in ok]
    ttfts = [r["ttft_ms"] for r in ok]
    e2es = [r["e2e_ms"] for r in ok]
    tpots = [r["tpot_ms"] for r in ok]
    itls = [x for r in ok for x in r["itl_ms"]]
    chunks = [r["num_chunks"] for r in ok]

    total_comp = sum(comp_toks)
    total_in = sum(prompt_toks)

    summary = {
        "label": args.label,
        "model": args.model,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "num_prompts": args.num_prompts,
        "warmup": args.warmup,
        "completed_requests": len(ok),
        "failed_requests": len(fail),
        "first_error": fail[0]["error"] if fail else None,
        "wall_time_s": round(wall_s, 4),
        "req_per_s": round(len(ok) / wall_s, 4) if wall_s > 0 else 0.0,
        "output_tok_per_s": round(total_comp / wall_s, 2) if wall_s > 0 else 0.0,
        "input_tok_per_s": round(total_in / wall_s, 2) if wall_s > 0 else 0.0,
        "total_tok_per_s": round((total_in + total_comp) / wall_s, 2) if wall_s > 0 else 0.0,
        "mean_prompt_tokens": round(statistics.mean(prompt_toks), 2) if prompt_toks else 0.0,
        "mean_completion_tokens": round(statistics.mean(comp_toks), 2) if comp_toks else 0.0,
        "min_completion_tokens": min(comp_toks) if comp_toks else 0,
        "max_completion_tokens": max(comp_toks) if comp_toks else 0,
        "mean_sse_chunks": round(statistics.mean(chunks), 2) if chunks else 0.0,
        "ttft_ms": summarize(ttfts),
        "e2e_ms": summarize(e2es),
        "tpot_ms": summarize(tpots),
        "itl_ms": summarize(itls),
        "spec_metrics": extract_spec_deltas(m_before, m_after),
    }
    if info_after:
        try:
            summary["sglang_server_info"] = json.loads(info_after)
        except Exception:
            pass

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[{args.label}] DONE: ok={len(ok)}/{args.num_prompts} wall={wall_s:.2f}s "
        f"out_tok/s={summary['output_tok_per_s']} req/s={summary['req_per_s']} | "
        f"TTFT p50/p90={summary['ttft_ms']['p50']:.1f}/{summary['ttft_ms']['p90']:.1f}ms | "
        f"TPOT p50/p90={summary['tpot_ms']['p50']:.2f}/{summary['tpot_ms']['p90']:.2f}ms | "
        f"E2E p50/p90={summary['e2e_ms']['p50']/1000:.2f}/{summary['e2e_ms']['p90']/1000:.2f}s | "
        f"in={summary['mean_prompt_tokens']} out={summary['mean_completion_tokens']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="/hf-cache/enterprise_10k_prompts.jsonl")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--num-prompts", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--use-min-tokens", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--dump-sample", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
