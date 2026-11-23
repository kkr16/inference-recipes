#!/usr/bin/env python3
"""Generate 128 unique, realistic 10,000-token enterprise escalation prompts."""

import argparse
import json
import random
import uuid
from transformers import AutoTokenizer

COMPANIES = [
    "Vanguard Cloud Systems", "Meridian FinTech Corp", "Apex Logistics Global",
    "Solstice Health Partners", "Stratos Retail Group", "Aether Telecom Inc",
    "Crestview Insurance Co", "Helios Energy Grid", "Northstar Payments Ltd",
    "Orion Aviation Services", "Vertex Media Networks", "Lumina Biotech Labs",
    "Cobalt Mobility Corp", "Summit Hospitality Group", "Pinnacle Banking Corp",
    "Cascade E-Commerce Holdings"
]

CUSTOMERS = [
    "Elena Vance", "Marcus Thorne", "Priya Patel", "David Chen", "Sofia Alvarez",
    "Liam O'Connor", "Amara Okafor", "Hannah Lindqvist", "Kenji Takahashi",
    "Nadia Benali", "Gabriel Silva", "Chloe Dubois", "Tariq Al-Mansoor",
    "Rachel Kim", "Mateusz Kowalski", "Olivia Brooks"
]

AGENTS = [
    "Jordan Hayes (Tier-3 Escalations)", "Maya Singh (Enterprise TAM)",
    "Carlos Mendez (Platform SRE Liaison)", "Sienna Park (Trust & Compliance)",
    "Devon Brooks (Billing & SLA Specialist)", "Anya Ivanova (API Integrations)",
    "Noah Williams (Support Lead)", "Zara Ahmed (AI Routing Ops)"
]

CHANNELS = [
    "Live Chat Widget", "Salesforce Case Sync", "WhatsApp Business API",
    "Apple Messages for Business", "Email Thread", "Voice-to-Text Transcript",
    "Instagram DM Escalation", "Slack Connect Shared Channel"
]

PRODUCTS = [
    "Enterprise Support Platform", "Unified Agent Console", "Care Lite Routing Engine",
    "Social Publishing & Governance", "AI+ Sentiment & Intent Classifier",
    "Conversational IVR Connector", "Omnichannel SLA Breach Predictor",
    "CRM Bi-Directional Webhook Bus", "Live Chat WebSocket Gateway",
    "Automated Quality Management (QM) Suite"
]

ERROR_CODES = [
    "ERR_429_RATE_LIMIT_BURST", "ERR_504_WEBHOOK_DISPATCH_TIMEOUT",
    "ERR_409_CASE_STATE_RACE", "ERR_502_UPSTREAM_TLS_RESET",
    "ERR_422_SCHEMA_PAYLOAD_DRIFT", "ERR_503_SHARD_REBALANCE_PAUSE",
    "ERR_401_OAUTH_TOKEN_SKEW", "ERR_500_DEDUP_LOCK_CONTENTION"
]

REGIONS = ["us-east-1", "eu-west-1", "eu-central-1", "ap-southeast-1", "us-west-2"]

TOPICS = [
    "delayed webhook delivery causing SLA breach alerts across tier-1 VIP queues",
    "duplicate ticket creation during simultaneous WhatsApp and email customer replies",
    "sudden spike in negative sentiment scores after automated bot handoff to live agents",
    "OAuth refresh token race condition interrupting Salesforce bi-directional case sync",
    "incorrect PII redaction masking order confirmation numbers in regulated finance queues",
    "voice transcript diarization dropping agent turns during warm supervisor transfers",
    "rate-limit throttling on outbound proactive notifications during flash sales events",
    "routing rule regression sending German-language escalations to English-only pods",
]

POLICY_CLAUSES = [
    "Enterprise SLA Tier-1 guarantees 99.95% webhook delivery within 5,000 ms p95 and mandatory root-cause disclosure within 24 hours of any Sev-1 incident.",
    "Compliance Rule 4.2 requires zero unmasked payment card digits or government IDs in exported QA transcripts and full audit logging of every supervisor override.",
    "Escalation Policy 7B mandates automatic VIP tagging when customer wait time exceeds 180 seconds or when consecutive sentiment polarity drops below -0.65.",
    "API Fair-Use Section 11 allows burst concurrency up to 2,500 requests per minute per tenant shard before adaptive backoff headers (Retry-After) are enforced.",
    "Retention Mandate 9.1 specifies that cross-channel conversation merges must preserve original channel timestamps, agent assignment history, and CSAT survey attribution.",
    "Agent Coaching Rubric C3 evaluates empathy phrasing, accurate policy citation, proactive workaround offering, and confirmation of customer resolution before closing.",
]

SUFFIX_INSTRUCTION = (
    "\n\n=== FINAL ANALYSIS DIRECTIVE ===\n"
    "Produce an exhaustive, structured 15-section enterprise escalation dossier covering: "
    "(1) Executive Incident Summary, (2) Chronological Multi-Channel Timeline, "
    "(3) Technical Root-Cause Hypotheses Ranked by Evidence, (4) Webhook & API Telemetry Anomaly Breakdown, "
    "(5) Customer Sentiment Trajectory & Churn Risk Score, (6) SLA & Contractual Credit Exposure, "
    "(7) PII / Compliance Audit Findings, (8) Agent Handling & Coaching Scorecard, "
    "(9) Routing & Automation Misconfiguration Review, (10) Immediate Mitigation Playbook, "
    "(11) Long-Term Architectural Hardening Plan, (12) Draft Executive Apology Email to Customer, "
    "(13) Internal Engineering Jira Ticket Specification, (14) QA Calibration Recommendations, and "
    "(15) Preventive Monitoring Alert Thresholds. Write at least 1,500 words with thorough technical detail.\n"
    "## Section 1: Executive Incident Summary\n"
)


def make_turn_block(rng: random.Random, turn_idx: int, ticket_num: int) -> str:
    cust = rng.choice(CUSTOMERS)
    agent = rng.choice(AGENTS)
    ch = rng.choice(CHANNELS)
    prod = rng.choice(PRODUCTS)
    err = rng.choice(ERROR_CODES)
    reg = rng.choice(REGIONS)
    topic = rng.choice(TOPICS)
    clause = rng.choice(POLICY_CLAUSES)
    lat = rng.randint(140, 9800)
    queue_depth = rng.randint(12, 480)
    sent = round(rng.uniform(-0.92, 0.45), 2)
    minute = (turn_idx * 3) % 60
    hour = 8 + (turn_idx * 3) // 60

    template_kind = turn_idx % 4
    if template_kind == 0:
        return (
            f"[{hour:02d}:{minute:02d}:14 UTC | Turn {turn_idx:03d} | Channel: {ch} | Region: {reg}]\n"
            f"Customer ({cust}): Our operations team noticed {topic} on {prod} starting right after the scheduled batch import for ticket batch #{ticket_num + turn_idx}. "
            f"Approximately {queue_depth} active conversations were delayed, and our internal monitor captured {err} with p99 response latency hitting {lat} ms. "
            f"Can your engineering liaison verify whether the tenant partition experienced lock contention or retry storm amplification?\n"
            f"Telemetry Snapshot: sentiment_score={sent}, retry_attempt={rng.randint(1,5)}, payload_bytes={rng.randint(1800, 42000)}, sla_timer_remaining_sec={rng.randint(15, 600)}.\n\n"
        )
    elif template_kind == 1:
        return (
            f"[{hour:02d}:{minute:02d}:42 UTC | Turn {turn_idx:03d} | Internal Agent Note by {agent}]\n"
            f"Reviewed trace_id=tr_{uuid.UUID(int=rng.getrandbits(128)).hex[:16]} in {reg} for {prod}. "
            f"Observed worker pool saturation during burst ingestion ({queue_depth} concurrent threads). "
            f"Error signature {err} correlated with downstream CRM acknowledgment latency spiking to {lat} ms. "
            f"Applicable governance note: {clause} "
            f"Recommended interim mitigation: enable dead-letter replay queue, throttle non-urgent bulk exports by 25%, and pin VIP queue routing to dedicated shard.\n\n"
        )
    elif template_kind == 2:
        return (
            f"[{hour:02d}:{minute:02d}:55 UTC | Turn {turn_idx:03d} | Automated System & Webhook Audit]\n"
            f"Event=DISPATCH_AUDIT | Product={prod} | Error={err} | LatencyMs={lat} | QueueDepth={queue_depth} | Sentiment={sent}\n"
            f"Diagnostic Context: During cross-channel state reconciliation between {ch} and core case store, optimistic concurrency check triggered {rng.randint(2,19)} retries. "
            f"Agent {agent} flagged potential schema mismatch on custom_field_v{rng.randint(2,9)} and requested verification against customer {cust}'s sandbox configuration.\n\n"
        )
    else:
        return (
            f"[{hour:02d}:{minute:02d}:08 UTC | Turn {turn_idx:03d} | Agent Reply ({agent}) -> {cust}]\n"
            f"Thank you for sharing the exact correlation IDs and timestamps. Our SRE team confirmed transient pressure in {reg} affecting {prod} around {topic}. "
            f"We verified that {clause} We have already isolated the noisy backlog partition (depth={queue_depth}, observed latency={lat} ms under {err}) "
            f"and initiated automated replay verification so zero customer messages are lost while sentiment remains tracked ({sent}).\n\n"
        )


def build_exact_prompt(tok, idx: int, target_tokens: int) -> str:
    rng = random.Random(20261119 + idx * 9973)
    case_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
    company = COMPANIES[idx % len(COMPANIES)]
    cust = CUSTOMERS[(idx * 3) % len(CUSTOMERS)]
    prod = PRODUCTS[(idx * 5) % len(PRODUCTS)]
    topic = TOPICS[(idx * 7) % len(TOPICS)]

    header = (
        f"Case-ID: {case_uuid} | Timestamp: 2026-11-19T{10 + (idx % 12):02d}:{(idx * 7) % 60:02d}:{(idx * 13) % 60:02d}Z | "
        f"Enterprise Contact Center Escalation Packet #{idx:04d}\n"
        f"Tenant: {company} | Primary Stakeholder: {cust} | Affected Suite: {prod}\n"
        f"Escalation Summary: Investigation into {topic}.\n\n"
    )

    chunks = [header]
    turn = 1
    while len(chunks) < 120:
        chunks.append(make_turn_block(rng, turn, 100000 + idx * 1000))
        turn += 1
        if turn % 10 == 0:
            # Check token count periodically so we don't overshoot wildly
            cur_text = "".join(chunks) + SUFFIX_INSTRUCTION
            if len(tok.encode(cur_text, add_special_tokens=False)) >= target_tokens + 200:
                break

    # Ensure enough tokens
    while len(tok.encode("".join(chunks) + SUFFIX_INSTRUCTION, add_special_tokens=False)) < target_tokens + 150:
        chunks.append(make_turn_block(rng, turn, 100000 + idx * 1000))
        turn += 1

    body_text = "".join(chunks)
    body_ids = tok.encode(body_text, add_special_tokens=False)
    suffix_ids = tok.encode(SUFFIX_INSTRUCTION, add_special_tokens=False)

    n = target_tokens - len(suffix_ids)
    for _ in range(40):
        prefix = tok.decode(body_ids[:n], skip_special_tokens=False)
        candidate = prefix + "\n" + SUFFIX_INSTRUCTION
        cur_ids = tok.encode(candidate, add_special_tokens=False)
        diff = target_tokens - len(cur_ids)
        if diff == 0:
            return candidate
        if abs(diff) > 1:
            n += diff
        else:
            # Fine tune with 1-token adjustments if BPE boundary shifts by 1
            if diff > 0:
                for pad in [" log", " note", " ref", " ok", "."]:
                    test = prefix + pad + "\n" + SUFFIX_INSTRUCTION
                    if len(tok.encode(test, add_special_tokens=False)) == target_tokens:
                        return test
            n += diff

    # Hard fallback loop if boundary stubborn
    prefix = tok.decode(body_ids[:n], skip_special_tokens=False)
    while len(tok.encode(prefix + "\n" + SUFFIX_INSTRUCTION, add_special_tokens=False)) > target_tokens:
        prefix = prefix[:-1]
    while len(tok.encode(prefix + "\n" + SUFFIX_INSTRUCTION, add_special_tokens=False)) < target_tokens:
        prefix += " x"
        if len(tok.encode(prefix + "\n" + SUFFIX_INSTRUCTION, add_special_tokens=False)) > target_tokens:
            prefix = prefix[:-2] + " "
    candidate = prefix + "\n" + SUFFIX_INSTRUCTION
    assert len(tok.encode(candidate, add_special_tokens=False)) == target_tokens, (
        f"Failed exact length for idx={idx}: {len(tok.encode(candidate, add_special_tokens=False))}"
    )
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--output", default="/hf-cache/enterprise_10k_prompts.jsonl")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--target-tokens", type=int, default=10000)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(args.num_prompts):
            p = build_exact_prompt(tok, i, args.target_tokens)
            n_false = len(tok.encode(p, add_special_tokens=False))
            n_true = len(tok.encode(p, add_special_tokens=True))
            assert n_false == args.target_tokens, f"idx {i}: {n_false} != {args.target_tokens}"
            if i == 0:
                print(f"Sample verification idx=0: add_special_tokens=False -> {n_false}, True -> {n_true}")
            f.write(json.dumps({"id": i, "tokens": n_false, "prompt": p}) + "\n")
    print(f"Successfully wrote {args.num_prompts} exact-{args.target_tokens}-token prompts to {args.output}")


if __name__ == "__main__":
    main()
