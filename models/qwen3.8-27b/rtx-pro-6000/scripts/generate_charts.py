#!/usr/bin/env python3
"""Generate SemiAnalysis InferenceMAX-style Pareto charts for Qwen3.8-27B on RTX PRO 6000.

Zero third-party Python dependencies: emits crisp SVGs and optionally rasterizes
them to 2x HiDPI PNGs using headless Google Chrome / Chromium.
"""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


FONT_SANS = "Roboto, 'Liberation Sans', 'DejaVu Sans', sans-serif"
FONT_MONO = "'Noto Sans Mono', 'Liberation Mono', 'DejaVu Sans Mono', monospace"

# SemiAnalysis-inspired high-contrast palette
COLORS = {
    "tp1_nvfp4_mtp4": "#0D9488",        # Teal (Champion DP=8 x TP=1)
    "tp1_nvfp4_mtp4_mode3": "#0F766E",  # Deep Teal
    "tp2_nvfp4_mtp4_w5": "#2563EB",     # Royal Blue
    "tp4_nvfp4_mtp4_ipc": "#7C3AED",    # Violet
    "tp8_nvfp4_mtp4_w5": "#EA580C",     # Burnt Orange
    "tp8_nvfp4_mtp5_w5": "#DC2626",     # Crimson (Fastest interactivity)
    "tp1_fp8_mtp4": "#D97706",          # Amber
    "tp1_nvfp4_nomtp": "#64748B",       # Slate
    "tp1_fp8_nomtp": "#94A3B8",         # Light Slate
    "tp1_bf16_naive": "#B45309",        # Warm Amber (1-GPU Naive BF16 baseline)
    "tp8_bf16_naive": "#9CA3AF",        # Gray
    "tp8_bf16_mtp4_ipc": "#6B7280",     # Dark Gray
}


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def marker_svg(shape: str, cx: float, cy: float, r: float, fill: str, stroke: str = "#FFFFFF") -> str:
    sw = 1.6
    if shape == "circle":
        return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if shape == "square":
        s = r * 1.7
        return f'<rect x="{cx - s/2:.1f}" y="{cy - s/2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1.5" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if shape == "diamond":
        d = r * 1.35
        pts = f"{cx:.1f},{cy-d:.1f} {cx+d:.1f},{cy:.1f} {cx:.1f},{cy+d:.1f} {cx-d:.1f},{cy:.1f}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if shape == "triangle":
        d = r * 1.35
        pts = f"{cx:.1f},{cy-d:.1f} {cx+d*0.95:.1f},{cy+d*0.8:.1f} {cx-d*0.95:.1f},{cy+d*0.8:.1f}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if shape == "star":
        pts = []
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            rad = r * 1.55 if i % 2 == 0 else r * 0.68
            pts.append(f"{cx + rad*math.cos(ang):.1f},{cy + rad*math.sin(ang):.1f}")
        return f'<polygon points="{" ".join(pts)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def compute_pareto_upper_right(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return non-dominated points maximizing both x and y, sorted by x ascending."""
    pts = sorted(points, key=lambda p: (-p[0], -p[1]))
    frontier = []
    max_y = -1e18
    for x, y in pts:
        if y > max_y:
            frontier.append((x, y))
            max_y = y
    frontier.sort(key=lambda p: p[0])
    return frontier


def compute_pareto_upper_left(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return non-dominated points minimizing x (latency) and maximizing y (throughput)."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    max_y = -1e18
    for x, y in pts:
        if y > max_y:
            frontier.append((x, y))
            max_y = y
    return frontier


def build_interactivity_pareto(metrics: Dict[str, Any]) -> str:
    W, H = 1240, 740
    ML, MR, MT, MB = 92, 42, 118, 92
    PW, PH = W - ML - MR, H - MT - MB

    x_min, x_max = 0.0, 720.0
    y_min, y_max = 0.0, 580.0

    def sx(x: float) -> float:
        return ML + (x - x_min) / (x_max - x_min) * PW

    def sy(y: float) -> float:
        return MT + PH - (y - y_min) / (y_max - y_min) * PH

    # Ordered from baselines -> champions
    plot_specs = [
        ("tp1_bf16_naive",      "TP=1 BF16 Naive (No MTP)",                    "circle",   True,  2.0),
        ("tp1_fp8_nomtp",       "TP=1 FP8 Baseline (No MTP)",                  "circle",   True,  2.0),
        ("tp1_nvfp4_nomtp",     "TP=1 NVFP4 (No MTP)",                         "circle",   True,  2.2),
        ("tp1_fp8_mtp4",        "TP=1 FP8 + MTP-4 (Tuned)",                    "square",   True,  2.3),
        ("tp1_nvfp4_mtp4",      "TP=1 NVFP4 + MTP-4 (Champion · DP=8)",        "circle",   False, 3.0),
        ("tp2_nvfp4_mtp4_w5",   "TP=2 NVFP4 + MTP-4 (PCIE_IPC + Triton)",      "square",   False, 2.8),
        ("tp4_nvfp4_mtp4_ipc",  "TP=4 NVFP4 + MTP-4 (PCIE_IPC + P2P=SYS)",     "diamond",  False, 2.8),
        ("tp8_nvfp4_mtp4_w5",   "TP=8 NVFP4 + MTP-4 (PCIE_IPC + Triton)",      "triangle", False, 2.8),
        ("tp8_nvfp4_mtp5_w5",   "TP=8 NVFP4 + MTP-5 (PCIE_IPC + Triton)",      "star",     False, 2.8),
    ]

    all_nvfp4_pts: List[Tuple[float, float]] = []
    for k in ["tp1_nvfp4_mtp4", "tp1_nvfp4_mtp4_mode3", "tp2_nvfp4_mtp4_w5", "tp4_nvfp4_mtp4_ipc", "tp8_nvfp4_mtp4_w5", "tp8_nvfp4_mtp5_w5"]:
        for p in metrics["series"][k]["points"]:
            all_nvfp4_pts.append((p["interactivity_tok_s_user"], p["out_tok_s_per_gpu"]))
    frontier = compute_pareto_upper_right(all_nvfp4_pts)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        # Subtle top accent bar (SemiAnalysis editorial touch)
        f'<rect x="0" y="0" width="{W}" height="5" fill="#0D9488"/>',
        # Plot background
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="#FAFBFD" stroke="#E2E8F0" stroke-width="1"/>',
    ]

    # Title & subtitle
    lines.append(
        f'<text x="{ML}" y="48" font-family="{FONT_SANS}" font-size="25" font-weight="700" fill="#0F172A">'
        f'Qwen3.8-27B on NVIDIA RTX PRO 6000 Blackwell — Inference Pareto Frontier</text>'
    )
    lines.append(
        f'<text x="{ML}" y="74" font-family="{FONT_SANS}" font-size="14" fill="#475569">'
        f'Output Throughput per GPU vs. Interactivity (1000 / TPOT p50) · 10k ISL / 500 OSL Uncached · vLLM 0.34.0 · GCP G4 (SM120, PCIe Gen5)</text>'
    )
    # "Better ->" badge top-right of plot
    lines.append(
        f'<rect x="{ML + PW - 134}" y="{MT + 12}" width="122" height="28" rx="14" fill="#ECFDF5" stroke="#A7F3D0" stroke-width="1"/>'
        f'<text x="{ML + PW - 73}" y="{MT + 31}" text-anchor="middle" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="#047857">Better  ↗</text>'
    )

    # Gridlines & ticks
    for xv in range(0, 721, 100):
        px = sx(xv)
        lines.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT+PH}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{px:.1f}" y="{MT+PH+24}" text-anchor="middle" font-family="{FONT_MONO}" font-size="12" fill="#475569">{xv}</text>')
    for yv in range(0, 581, 100):
        py = sy(yv)
        lines.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{ML+PW}" y2="{py:.1f}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{ML-12}" y="{py+4:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="12" fill="#475569">{yv}</text>')

    # Axes
    lines.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')

    # Axis labels
    lines.append(
        f'<text x="{ML + PW/2:.1f}" y="{H - 36}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A">'
        f'Interactivity — Output Tokens / sec / User  (1000 / TPOT p50)</text>'
    )
    lines.append(
        f'<text x="28" y="{MT + PH/2:.1f}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A" '
        f'transform="rotate(-90 28 {MT + PH/2:.1f})">Output Throughput per GPU  (tok/s/GPU)</text>'
    )

    # Shaded Pareto frontier envelope
    if frontier:
        poly_pts = [f"{sx(frontier[0][0]):.1f},{sy(0):.1f}"]
        for x, y in frontier:
            poly_pts.append(f"{sx(x):.1f},{sy(y):.1f}")
        poly_pts.append(f"{sx(frontier[-1][0]):.1f},{sy(0):.1f}")
        lines.append(f'<polygon points="{" ".join(poly_pts)}" fill="#0D9488" fill-opacity="0.06"/>')
        path_d = "M " + " L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in frontier)
        lines.append(f'<path d="{path_d}" fill="none" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="7,5" stroke-opacity="0.55"/>')

    # Plot each series
    for key, label, shape, is_dashed, sw in plot_specs:
        pts = metrics["series"][key]["points"]
        # Sort by interactivity ascending so lines draw cleanly left-to-right
        pts_sorted = sorted(pts, key=lambda p: p["interactivity_tok_s_user"])
        color = COLORS[key]
        dash = ' stroke-dasharray="5,4"' if is_dashed else ""
        opacity = "0.78" if is_dashed else "1.0"
        d = "M " + " L ".join(f"{sx(p['interactivity_tok_s_user']):.1f},{sy(p['out_tok_s_per_gpu']):.1f}" for p in pts_sorted)
        lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash} stroke-opacity="{opacity}"/>')
        r = 5.5 if not is_dashed else 4.2
        for p in pts_sorted:
            cx, cy = sx(p["interactivity_tok_s_user"]), sy(p["out_tok_s_per_gpu"])
            lines.append(marker_svg(shape, cx, cy, r, color))

    # Concurrency & hero callout annotations (carefully positioned to avoid overlap)
    annotations = [
        # (x, y, dx, dy, text, color, anchor)
        (32.77, 521.55, 18, -14, "TP=1 c=24: 522 tok/s/GPU (4,172 on 8x)", "#0F766E", "start"),
        (198.93, 149.35, 16, -24, "TP=1 c=1: 199 tok/s/u (3.86s p90)", "#0D9488", "start"),
        (329.71, 105.61, 14, -22, "TP=2 c=1: 330 tok/s/u (2.66s p90)", "#2563EB", "start"),
        (443.26, 64.66,  10, -40, "TP=4 c=1: 443 tok/s/u (2.13s p90)", "#7C3AED", "start"),
        (592.07, 38.83,  -8, -22, "TP=8 MTP-4: 592 tok/s/u (1.76s)", "#EA580C", "end"),
        (669.34, 41.25, -14, -76, "TP=8 MTP-5 + Triton: 669 tok/s/u (1.65s p90!)", "#DC2626", "end"),
        (26.36, 24.49,   22,  -4, "1× BF16 Naive: 24 → 169 tok/s (c=1..16)", "#B45309", "start"),
    ]
    for x, y, dx, dy, txt, col, anchor in annotations:
        px, py = sx(x), sy(y)
        tx, ty = px + dx, py + dy
        lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty+3:.1f}" stroke="{col}" stroke-width="1" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{FONT_SANS}" font-size="11.5" font-weight="600" fill="{col}">{esc(txt)}</text>')

    # Legend box (upper right, below "Better" badge)
    lx, ly = ML + PW - 365, MT + 54
    lh = 22 * (len(plot_specs) + 1) + 18
    lines.append(f'<rect x="{lx}" y="{ly}" width="353" height="{lh}" rx="8" fill="#FFFFFF" fill-opacity="0.94" stroke="#CBD5E1" stroke-width="1"/>')
    # Frontier legend entry first
    fy = ly + 22
    lines.append(f'<line x1="{lx+14}" y1="{fy-4}" x2="{lx+40}" y2="{fy-4}" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="6,4" stroke-opacity="0.65"/>')
    lines.append(f'<text x="{lx+48}" y="{fy}" font-family="{FONT_SANS}" font-size="11.5" font-weight="700" fill="#0F172A">NVFP4 Pareto Frontier (TP=1 → TP=8)</text>')
    for idx, (key, label, shape, is_dashed, sw) in enumerate(reversed(plot_specs)):
        ry = ly + 22 * (idx + 2)
        col = COLORS[key]
        dash = ' stroke-dasharray="4,3"' if is_dashed else ""
        lines.append(f'<line x1="{lx+14}" y1="{ry-4}" x2="{lx+40}" y2="{ry-4}" stroke="{col}" stroke-width="{sw}"{dash}/>')
        lines.append(marker_svg(shape, lx + 27, ry - 4, 4.3, col))
        lines.append(f'<text x="{lx+48}" y="{ry}" font-family="{FONT_SANS}" font-size="11.5" fill="#1E293B">{esc(label)}</text>')

    # Footer / source
    lines.append(
        f'<text x="{ML}" y="{H - 12}" font-family="{FONT_SANS}" font-size="11" fill="#94A3B8">'
        f'Source: inference-recipes · Zero prefix caching · 0 failed requests across all plotted points</text>'
    )
    lines.append(
        f'<text x="{ML + PW}" y="{H - 12}" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="#94A3B8">'
        f'RTX PRO 6000 (96 GB GDDR7)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def build_node_e2e_pareto(metrics: Dict[str, Any]) -> str:
    W, H = 1240, 740
    ML, MR, MT, MB = 96, 42, 118, 92
    PW, PH = W - ML - MR, H - MT - MB

    # Log-scale X axis for E2E p90 latency (1.2s to 75s)
    x_min_log, x_max_log = math.log10(1.2), math.log10(75.0)
    y_min, y_max = 0.0, 4600.0

    def sx(x: float) -> float:
        return ML + (math.log10(max(x, 1.2)) - x_min_log) / (x_max_log - x_min_log) * PW

    def sy(y: float) -> float:
        return MT + PH - (y - y_min) / (y_max - y_min) * PH

    plot_specs = [
        ("tp1_bf16_naive",      "1 × TP=1 BF16 Naive (1-GPU Baseline)",       "diamond",  True,  2.2, "out_tok_s"),
        ("tp8_bf16_naive",      "1 × TP=8 BF16 Naive",                        "circle",   True,  2.0, "node_8gpu_out_tok_s"),
        ("tp8_bf16_mtp4_ipc",   "1 × TP=8 BF16 + MTP-4 (PCIE_IPC + P2P=SYS)", "square",   True,  2.2, "node_8gpu_out_tok_s"),
        ("tp1_fp8_mtp4",        "DP=8 × TP=1 FP8 + MTP-4 (Tuned)",            "square",   True,  2.4, "node_8gpu_out_tok_s"),
        ("tp1_nvfp4_mtp4",      "DP=8 × TP=1 NVFP4 + MTP-4 (Champion)",       "circle",   False, 3.0, "node_8gpu_out_tok_s"),
        ("tp2_nvfp4_mtp4_w5",   "DP=4 × TP=2 NVFP4 + MTP-4 (PCIE_IPC)",       "square",   False, 2.8, "node_8gpu_out_tok_s"),
        ("tp4_nvfp4_mtp4_ipc",  "DP=2 × TP=4 NVFP4 + MTP-4 (PCIE_IPC)",       "diamond",  False, 2.8, "node_8gpu_out_tok_s"),
        ("tp8_nvfp4_mtp4_w5",   "1 × TP=8 NVFP4 + MTP-4 (IPC + Triton)",      "triangle", False, 2.8, "node_8gpu_out_tok_s"),
        ("tp8_nvfp4_mtp5_w5",   "1 × TP=8 NVFP4 + MTP-5 (IPC + Triton)",      "star",     False, 2.8, "node_8gpu_out_tok_s"),
    ]

    all_nvfp4_pts = []
    for k in ["tp1_nvfp4_mtp4", "tp1_nvfp4_mtp4_mode3", "tp2_nvfp4_mtp4_w5", "tp4_nvfp4_mtp4_ipc", "tp8_nvfp4_mtp4_w5", "tp8_nvfp4_mtp5_w5"]:
        for p in metrics["series"][k]["points"]:
            all_nvfp4_pts.append((p["e2e_p90_s"], p["node_8gpu_out_tok_s"]))
    frontier = compute_pareto_upper_left(all_nvfp4_pts)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<rect x="0" y="0" width="{W}" height="5" fill="#2563EB"/>',
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="#FAFBFD" stroke="#E2E8F0" stroke-width="1"/>',
    ]

    # Title & subtitle
    lines.append(
        f'<text x="{ML}" y="48" font-family="{FONT_SANS}" font-size="25" font-weight="700" fill="#0F172A">'
        f'8x RTX PRO 6000 Node Capacity vs. E2E Latency p90 (DP vs. TP Scaling)</text>'
    )
    lines.append(
        f'<text x="{ML}" y="74" font-family="{FONT_SANS}" font-size="14" fill="#475569">'
        f'Total 8-GPU Node Output Throughput vs. E2E p90 Latency (log scale) · Qwen3.8-27B · 10k ISL / 500 OSL Uncached · vLLM 0.34.0</text>'
    )
    # "Better <- up" badge top-left of plot
    lines.append(
        f'<rect x="{ML + 14}" y="{MT + 12}" width="122" height="28" rx="14" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="1"/>'
        f'<text x="{ML + 75}" y="{MT + 31}" text-anchor="middle" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="#1D4ED8">↖  Better</text>'
    )

    # 2.0s SLA vertical threshold
    x_2s = sx(2.0)
    lines.append(f'<rect x="{ML}" y="{MT}" width="{x_2s - ML:.1f}" height="{PH}" fill="#FEF2F2" fill-opacity="0.45"/>')
    lines.append(f'<line x1="{x_2s:.1f}" y1="{MT}" x2="{x_2s:.1f}" y2="{MT+PH}" stroke="#DC2626" stroke-width="1.8" stroke-dasharray="6,4"/>')
    lines.append(f'<text x="{x_2s + 7:.1f}" y="{MT + 26}" font-family="{FONT_SANS}" font-size="12" font-weight="700" fill="#DC2626">2.0s E2E p90 SLA</text>')

    # Gridlines & ticks
    for xv in [1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 65]:
        px = sx(xv)
        lines.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT+PH}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lbl = f"{xv:g}s"
        lines.append(f'<text x="{px:.1f}" y="{MT+PH+24}" text-anchor="middle" font-family="{FONT_MONO}" font-size="12" fill="#475569">{lbl}</text>')
    for yv in range(0, 4601, 500):
        py = sy(yv)
        lines.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{ML+PW}" y2="{py:.1f}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{ML-12}" y="{py+4:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="12" fill="#475569">{yv:,}</text>')

    # Axes
    lines.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')

    # Axis labels
    lines.append(
        f'<text x="{ML + PW/2:.1f}" y="{H - 36}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A">'
        f'End-to-End Request Latency p90  (seconds, log scale — lower is better)</text>'
    )
    lines.append(
        f'<text x="26" y="{MT + PH/2:.1f}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A" '
        f'transform="rotate(-90 26 {MT + PH/2:.1f})">8-GPU Node Output Throughput  (tok/s)</text>'
    )

    # Frontier line
    if frontier:
        path_d = "M " + " L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in frontier)
        lines.append(f'<path d="{path_d}" fill="none" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="7,5" stroke-opacity="0.55"/>')

    # Series
    for key, label, shape, is_dashed, sw, yfield in plot_specs:
        pts = sorted(metrics["series"][key]["points"], key=lambda p: p["e2e_p90_s"])
        color = COLORS[key]
        dash = ' stroke-dasharray="5,4"' if is_dashed else ""
        opacity = "0.85" if is_dashed else "1.0"
        d = "M " + " L ".join(f"{sx(p['e2e_p90_s']):.1f},{sy(p[yfield]):.1f}" for p in pts)
        lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash} stroke-opacity="{opacity}"/>')
        r = 5.5 if not is_dashed else 4.2
        for p in pts:
            lines.append(marker_svg(shape, sx(p["e2e_p90_s"]), sy(p[yfield]), r, color))

    # Hero callouts
    annotations = [
        (31.279, 4172.4, -14, -18, "DP=8 × TP=1 Peak: 4,172 tok/s (c=24/GPU)", "#0F766E", "end"),
        (7.262,  2735.8, -14, -18, "DP=8 Balanced: 2,736 tok/s @ 7.26s p90", "#0D9488", "end"),
        (4.174,  1999.3,  -8, -26, "DP=8 c=2: 1,999 tok/s @ 4.17s", "#0D9488", "end"),
        (2.660,   844.8,  -6, -54, "DP=4 × TP=2: 845 tok/s @ 2.66s", "#2563EB", "end"),
        (2.129,   517.3,  16,  52, "DP=2 × TP=4: 517 tok/s @ 2.13s", "#7C3AED", "start"),
        (1.648,   330.0,   8, -160, "TP=8 MTP-5: 1.65s p90 (352ms under SLA)", "#DC2626", "start"),
        (20.420,   24.5, -18, -82, "1× BF16 Naive c=1: 24 tok/s @ 20.4s", "#B45309", "end"),
        (61.648,  168.6,  -8, -28, "1× BF16 Naive c=16: 169 tok/s @ 61.6s", "#B45309", "end"),
    ]
    for x, y, dx, dy, txt, col, anchor in annotations:
        px, py = sx(x), sy(y)
        tx, ty = px + dx, py + dy
        lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty+3:.1f}" stroke="{col}" stroke-width="1" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{FONT_SANS}" font-size="11.5" font-weight="600" fill="{col}" paint-order="stroke" stroke="#FFFFFF" stroke-width="3">{esc(txt)}</text>')

    # Legend box (middle-right open bay, strictly right of DP=4xTP=2 21.1s & above 1x BF16 Naive)
    lx, ly = ML + PW - 322, MT + PH - 310
    lh = 20.5 * (len(plot_specs) + 1) + 16
    lines.append(f'<rect x="{lx}" y="{ly}" width="312" height="{lh:.0f}" rx="8" fill="#FFFFFF" fill-opacity="0.95" stroke="#CBD5E1" stroke-width="1"/>')
    fy = ly + 21
    lines.append(f'<line x1="{lx+12}" y1="{fy-4}" x2="{lx+34}" y2="{fy-4}" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="6,4" stroke-opacity="0.65"/>')
    lines.append(f'<text x="{lx+42}" y="{fy}" font-family="{FONT_SANS}" font-size="10.8" font-weight="700" fill="#0F172A">NVFP4 8-GPU Pareto Frontier</text>')
    for idx, (key, label, shape, is_dashed, sw, _yf) in enumerate(reversed(plot_specs)):
        ry = ly + 20.5 * (idx + 2)
        col = COLORS[key]
        dash = ' stroke-dasharray="4,3"' if is_dashed else ""
        lines.append(f'<line x1="{lx+12}" y1="{ry-4}" x2="{lx+34}" y2="{ry-4}" stroke="{col}" stroke-width="{sw}"{dash}/>')
        lines.append(marker_svg(shape, lx + 23, ry - 4, 4.0, col))
        lines.append(f'<text x="{lx+42}" y="{ry}" font-family="{FONT_SANS}" font-size="10.5" fill="#1E293B">{esc(label)}</text>')

    # Footer
    lines.append(
        f'<text x="{ML}" y="{H - 12}" font-family="{FONT_SANS}" font-size="11" fill="#94A3B8">'
        f'Source: inference-recipes · DP=8 × TP=1 delivers 3.6×–7.1× higher 8-GPU node throughput than TP=4/TP=8</text>'
    )
    lines.append(
        f'<text x="{ML + PW}" y="{H - 12}" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="#94A3B8">'
        f'8x RTX PRO 6000 (GCP G4)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def svg_to_png(svg_path: Path, png_path: Path, width: int = 1240, height: int = 740) -> bool:
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as tf:
        tf.write(
            f'<!doctype html><html><body style="margin:0;padding:0;background:#fff;overflow:hidden;">'
            f'{svg_path.read_text(encoding="utf-8")}</body></html>'
        )
        html_path = tf.name
    try:
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--force-device-scale-factor=2",
                f"--window-size={width},{height}",
                f"--screenshot={png_path}",
                f"file://{html_path}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    finally:
        Path(html_path).unlink(missing_ok=True)


def build_workload_pareto(
    wl_data: Dict[str, Any],
    *,
    title: str,
    subtitle: str,
    accent: str,
    x_max: float,
    x_step: int,
    y_max: float,
    y_step: int,
    annotations: List[Tuple[float, float, float, float, str, str, str]],
    footer_left: str,
) -> str:
    W, H = 1240, 740
    ML, MR, MT, MB = 92, 42, 118, 92
    PW, PH = W - ML - MR, H - MT - MB

    def sx(x: float) -> float:
        return ML + (x / x_max) * PW

    def sy(y: float) -> float:
        return MT + PH - (y / y_max) * PH

    wl_colors = {
        "tp1_nvfp4_nomtp": "#94A3B8",
        "tp1_nvfp4_mtp4": "#64748B",
        "tp8_nvfp4_base": "#9CA3AF",
        "tp1_nvfp4_dyn_mtp": "#0D9488",
        "tp2_nvfp4_dyn_triton": "#2563EB",
        "tp4_nvfp4_dyn_triton": "#7C3AED",
        "tp8_nvfp4_dyn_triton": "#DC2626",
    }

    plot_specs = [
        ("tp1_nvfp4_nomtp",      "TP=1 NVFP4 Baseline (No MTP)",                        "circle",   True,  2.1),
        ("tp1_nvfp4_mtp4",       "TP=1 NVFP4 + Static MTP-4 (Prior Baseline)",          "square",   True,  2.1),
        ("tp8_nvfp4_base",       "TP=8 NVFP4 + Static MTP-4 (PCIE_IPC + CUMEM=0)",      "triangle", True,  2.1),
        ("tp1_nvfp4_dyn_mtp",    "TP=1 Dyn-MTP 6/3 + Rust (Champion · DP=8)",           "circle",   False, 3.0),
        ("tp2_nvfp4_dyn_triton", "TP=2 PIX Dyn-MTP + LocalArgmax + Triton (DP=4)",      "square",   False, 2.8),
        ("tp4_nvfp4_dyn_triton", "TP=4 NODE Dyn-MTP + LocalArgmax + Triton (DP=2)",     "diamond",  False, 2.8),
        ("tp8_nvfp4_dyn_triton", "TP=8 SYS Dyn-MTP + LocalArgmax + Triton (1×TP=8)",    "star",     False, 2.8),
    ]

    champ_keys = ["tp1_nvfp4_dyn_mtp", "tp2_nvfp4_dyn_triton", "tp4_nvfp4_dyn_triton", "tp8_nvfp4_dyn_triton"]
    all_champ_pts: List[Tuple[float, float]] = []
    for k in champ_keys:
        for p in wl_data["series"][k]["points"]:
            all_champ_pts.append((p["interactivity_tok_s_user"], p["out_tok_s_per_gpu"]))
    frontier = compute_pareto_upper_right(all_champ_pts)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<rect x="0" y="0" width="{W}" height="5" fill="{accent}"/>',
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="#FAFBFD" stroke="#E2E8F0" stroke-width="1"/>',
    ]

    lines.append(
        f'<text x="{ML}" y="48" font-family="{FONT_SANS}" font-size="25" font-weight="700" fill="#0F172A">{esc(title)}</text>'
    )
    lines.append(
        f'<text x="{ML}" y="74" font-family="{FONT_SANS}" font-size="14" fill="#475569">{esc(subtitle)}</text>'
    )
    lines.append(
        f'<rect x="{ML + PW - 134}" y="{MT + 12}" width="122" height="28" rx="14" fill="#ECFDF5" stroke="#A7F3D0" stroke-width="1"/>'
        f'<text x="{ML + PW - 73}" y="{MT + 31}" text-anchor="middle" font-family="{FONT_SANS}" font-size="12.5" font-weight="700" fill="#047857">Better  ↗</text>'
    )

    for xv in range(0, int(x_max) + 1, x_step):
        px = sx(xv)
        lines.append(f'<line x1="{px:.1f}" y1="{MT}" x2="{px:.1f}" y2="{MT+PH}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{px:.1f}" y="{MT+PH+24}" text-anchor="middle" font-family="{FONT_MONO}" font-size="12" fill="#475569">{xv}</text>')
    for yv in range(0, int(y_max) + 1, y_step):
        py = sy(yv)
        lines.append(f'<line x1="{ML}" y1="{py:.1f}" x2="{ML+PW}" y2="{py:.1f}" stroke="#E2E8F0" stroke-width="1" stroke-dasharray="4,4"/>')
        lines.append(f'<text x="{ML-12}" y="{py+4:.1f}" text-anchor="end" font-family="{FONT_MONO}" font-size="12" fill="#475569">{yv:,}</text>')

    lines.append(f'<line x1="{ML}" y1="{MT+PH}" x2="{ML+PW}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+PH}" stroke="#334155" stroke-width="1.5"/>')

    lines.append(
        f'<text x="{ML + PW/2:.1f}" y="{H - 36}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A">'
        f'Interactivity — Output Tokens / sec / User  (1000 / TPOT p50)</text>'
    )
    lines.append(
        f'<text x="26" y="{MT + PH/2:.1f}" text-anchor="middle" font-family="{FONT_SANS}" font-size="14.5" font-weight="600" fill="#0F172A" '
        f'transform="rotate(-90 26 {MT + PH/2:.1f})">Output Throughput per GPU  (tok/s/GPU)</text>'
    )

    if frontier:
        poly_pts = [f"{sx(frontier[0][0]):.1f},{sy(0):.1f}"]
        for x, y in frontier:
            poly_pts.append(f"{sx(x):.1f},{sy(y):.1f}")
        poly_pts.append(f"{sx(frontier[-1][0]):.1f},{sy(0):.1f}")
        lines.append(f'<polygon points="{" ".join(poly_pts)}" fill="{accent}" fill-opacity="0.06"/>')
        path_d = "M " + " L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in frontier)
        lines.append(f'<path d="{path_d}" fill="none" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="7,5" stroke-opacity="0.55"/>')

    for key, label, shape, is_dashed, sw in plot_specs:
        pts = sorted(wl_data["series"][key]["points"], key=lambda p: p["interactivity_tok_s_user"])
        color = wl_colors[key]
        dash = ' stroke-dasharray="5,4"' if is_dashed else ""
        opacity = "0.78" if is_dashed else "1.0"
        d = "M " + " L ".join(f"{sx(p['interactivity_tok_s_user']):.1f},{sy(p['out_tok_s_per_gpu']):.1f}" for p in pts)
        lines.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash} stroke-opacity="{opacity}"/>')
        r = 5.5 if not is_dashed else 4.2
        for p in pts:
            lines.append(marker_svg(shape, sx(p["interactivity_tok_s_user"]), sy(p["out_tok_s_per_gpu"]), r, color))

    for x, y, dx, dy, txt, col, anchor in annotations:
        px, py = sx(x), sy(y)
        tx, ty = px + dx, py + dy
        lines.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{tx:.1f}" y2="{ty+3:.1f}" stroke="{col}" stroke-width="1" stroke-opacity="0.55"/>')
        lines.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-family="{FONT_SANS}" font-size="11.5" font-weight="600" fill="{col}">{esc(txt)}</text>')

    lx, ly = ML + PW - 392, MT + 54
    lh = 22 * (len(plot_specs) + 1) + 18
    lines.append(f'<rect x="{lx}" y="{ly}" width="380" height="{lh}" rx="8" fill="#FFFFFF" fill-opacity="0.95" stroke="#CBD5E1" stroke-width="1"/>')
    fy = ly + 22
    lines.append(f'<line x1="{lx+14}" y1="{fy-4}" x2="{lx+40}" y2="{fy-4}" stroke="#0F172A" stroke-width="2.2" stroke-dasharray="6,4" stroke-opacity="0.65"/>')
    lines.append(f'<text x="{lx+48}" y="{fy}" font-family="{FONT_SANS}" font-size="11.5" font-weight="700" fill="#0F172A">NVFP4 Dynamic MTP Pareto Frontier</text>')
    for idx, (key, label, shape, is_dashed, sw) in enumerate(reversed(plot_specs)):
        ry = ly + 22 * (idx + 2)
        col = wl_colors[key]
        dash = ' stroke-dasharray="4,3"' if is_dashed else ""
        lines.append(f'<line x1="{lx+14}" y1="{ry-4}" x2="{lx+40}" y2="{ry-4}" stroke="{col}" stroke-width="{sw}"{dash}/>')
        lines.append(marker_svg(shape, lx + 27, ry - 4, 4.3, col))
        lines.append(f'<text x="{lx+48}" y="{ry}" font-family="{FONT_SANS}" font-size="11.5" fill="#1E293B">{esc(label)}</text>')

    lines.append(
        f'<text x="{ML}" y="{H - 12}" font-family="{FONT_SANS}" font-size="11" fill="#94A3B8">{esc(footer_left)}</text>'
    )
    lines.append(
        f'<text x="{ML + PW}" y="{H - 12}" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="#94A3B8">'
        f'RTX PRO 6000 (96 GB GDDR7)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    model_dir = Path(__file__).resolve().parents[1]
    results_dir = model_dir / "results"
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))

    charts = [
        ("pareto_interactivity_vs_throughput", build_interactivity_pareto(metrics)),
        ("pareto_node_throughput_vs_e2e_p90", build_node_e2e_pareto(metrics)),
    ]

    wl_metrics_path = results_dir / "metrics_1k8k_8k1k.json"
    if wl_metrics_path.exists():
        wlm = json.loads(wl_metrics_path.read_text(encoding="utf-8"))
        charts.append((
            "pareto_1k8k_interactivity_vs_throughput",
            build_workload_pareto(
                wlm["workloads"]["1k8k"],
                title="Qwen3.8-27B on RTX PRO 6000 — 1k / 8k (Decode-Heavy) Pareto Frontier",
                subtitle="Output Throughput per GPU vs. Interactivity (1000 / TPOT p50) · 1,024 ISL / 8,192 OSL Uncached · vLLM 0.34.0 · Dynamic MTP + Local Argmax + Rust",
                accent="#0D9488",
                x_max=620.0,
                x_step=100,
                y_max=4100.0,
                y_step=500,
                annotations=[
                    (60.94, 3712.39,  24,  -12, "TP=1 Dyn-MTP c=64: 3,712 /GPU (29,699 on 8x!)", "#0F766E", "start"),
                    (86.43, 2626.55,  22,  -14, "TP=1 c=32: 2,627 /GPU @ 86 tok/s/u", "#0D9488", "start"),
                    (121.94, 1840.66, 26,  -22, "TP=2 c=32: 1,841 /GPU @ 122 tok/s/u", "#2563EB", "start"),
                    (157.58, 1190.39, 20,  -18, "TP=4 c=32: 1,190 /GPU @ 158 tok/s/u", "#7C3AED", "start"),
                    (265.46, 494.43,  16,  -38, "TP=8 c=16: 494 /GPU @ 265 tok/s/u", "#DC2626", "start"),
                    (414.08, 102.11,  -8,  -56, "TP=4 c=1: 414 tok/s/u (19.9s E2E)", "#7C3AED", "end"),
                    (544.37, 67.70,  -10, -108, "TP=8 c=1: 544 tok/s/u (15.2s E2E · 8.1× vs MTP-0)", "#DC2626", "end"),
                    (32.83, 2081.96,   8,   28, "No MTP (2,082)", "#64748B", "start"),
                ],
                footer_left="Source: inference-recipes · Exact 1,024/8,192 uncached · Dynamic MTP + Local Argmax + Rust + Triton · 0 failed requests",
            ),
        ))
        charts.append((
            "pareto_8k1k_interactivity_vs_throughput",
            build_workload_pareto(
                wlm["workloads"]["8k1k"],
                title="Qwen3.8-27B on RTX PRO 6000 — 8k / 1k (Prefill-Heavy) Pareto Frontier",
                subtitle="Output Throughput per GPU vs. Interactivity (1000 / TPOT p50) · 8,192 ISL / 1,024 OSL Uncached · vLLM 0.34.0 · Dynamic MTP + Local Argmax + Rust",
                accent="#7C3AED",
                x_max=700.0,
                x_step=100,
                y_max=1200.0,
                y_step=200,
                annotations=[
                    (53.63, 1072.93,  22,  -12, "TP=1 Dyn-MTP c=32: 1,073 /GPU (8,583 on 8x!)", "#0F766E", "start"),
                    (159.36, 518.24,  20,  -26, "TP=1 c=4: 518 /GPU @ 159 tok/s/u", "#0D9488", "start"),
                    (202.84, 183.51,   4,   54, "TP=1 c=1: 203 tok/s/u (566ms TTFT)", "#0D9488", "middle"),
                    (322.58, 132.50,  16,  -88, "TP=2 c=1: 323 tok/s/u (3.79s E2E)", "#2563EB", "start"),
                    (539.37, 102.15, -10,  -48, "TP=4 c=1: 539 tok/s/u (2.49s E2E)", "#7C3AED", "end"),
                    (635.32, 58.45,  -10, -138, "TP=8 c=1: 635 tok/s/u (2.21s E2E · 1.57ms TPOT)", "#DC2626", "end"),
                    (31.43, 777.26,    8,   34, "No MTP (777)", "#64748B", "start"),
                ],
                footer_left="Source: inference-recipes · Exact 8,192/1,024 uncached · DP=8 × TP=1 delivers 6.56× node capacity vs TP=8 · 0 failed requests",
            ),
        ))

    for stem, svg_str in charts:
        svg_path = results_dir / f"{stem}.svg"
        png_path = results_dir / f"{stem}.png"
        svg_path.write_text(svg_str, encoding="utf-8")
        ok = svg_to_png(svg_path, png_path)
        print(f"Wrote {svg_path.name}" + (f" + {png_path.name} (2x HiDPI)" if ok else " (Chrome not found; skipped PNG)"))


if __name__ == "__main__":
    main()
