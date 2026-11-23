#!/usr/bin/env python3
"""Generate a self-contained ReadTheDocs-style GitHub Pages site in docs/ from repo content.

Zero third-party Python dependencies. Reads:
  - models/**/metadata.yaml
  - models/<model>/<accelerator>/README.md
  - models/<model>/<accelerator>/recipes/*/{serve.sh,deploy-gke.yaml,metadata.yaml}
  - models/<model>/<accelerator>/results/*.png (copied into docs/assets/)
  - hardware/*.md
  - bench/README.md

Outputs:
  - docs/index.html
  - docs/assets/...
  - docs/llms.txt
  - docs/sitemap.xml
"""

import glob
import html
import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

SITE_URL = "https://kkr16.github.io/inference-recipes/"
REPO_URL = "https://github.com/kkr16/inference-recipes"


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0 and ":" in line:
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if not val:
                data[key] = {}
                current_section = key
            else:
                current_section = None
                data[key] = _parse_scalar(val)
        elif indent > 0 and current_section and ":" in line:
            key, val = line.strip().split(":", 1)
            data[current_section][key.strip()] = _parse_scalar(val.strip())
    return data


def _parse_scalar(val: str) -> Any:
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"').strip("'") for item in inner.split(",")]
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def inline_md(text: str, asset_prefix: str = "") -> str:
    """Convert inline Markdown (code, bold, italic, links, images) to safe HTML."""
    # Protect inline code spans first
    code_spans: List[str] = []

    def _stash_code(m: re.Match) -> str:
        code_spans.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00CODE{len(code_spans)-1}\x00"

    text = re.sub(r"`([^`]+)`", _stash_code, text)
    text = html.escape(text)

    # Images ![alt](src)
    def _img(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if not src.startswith(("http://", "https://", "data:")) and asset_prefix:
            src = f"{asset_prefix}/{Path(src).name}"
        return f'<figure class="chart-fig"><img src="{src}" alt="{alt}" loading="lazy" onclick="openLightbox(this.src, this.alt)"/><figcaption>{alt}</figcaption></figure>'

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img, text)

    # Links [label](href)
    def _link(m: re.Match) -> str:
        lbl, href = m.group(1), m.group(2)
        if href.startswith(("http://", "https://")):
            return f'<a href="{href}" target="_blank" rel="noopener">{lbl}</a>'
        if href.startswith("recipes/"):
            slug = href.strip("/").split("/")[-1]
            return f'<a href="#recipe-{slugify(slug)}" onclick="navTo(\'recipes\')">{lbl}</a>'
        return f'<a href="{REPO_URL}/blob/main/{href.lstrip("./")}" target="_blank" rel="noopener">{lbl}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _link, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    for idx, span in enumerate(code_spans):
        text = text.replace(f"\x00CODE{idx}\x00", span)
    return text


def md_to_html(md_text: str, asset_prefix: str = "", skip_h1: bool = False) -> str:
    """Clean, dependency-free Markdown-to-HTML converter supporting GFM tables, alerts, code blocks, lists."""
    lines = md_text.splitlines()
    out: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            out.append("<hr/>")
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_body = "\n".join(code_lines)
            if lang == "mermaid":
                out.append(f'<div class="code-wrap"><div class="code-bar"><span>topology diagram</span></div><pre class="mermaid-pre"><code>{html.escape(code_body)}</code></pre></div>')
            else:
                lbl = lang or "text"
                out.append(
                    f'<div class="code-wrap"><div class="code-bar"><span>{html.escape(lbl)}</span>'
                    f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></div>'
                    f'<pre><code class="lang-{html.escape(lbl)}">{html.escape(code_body)}</code></pre></div>'
                )
            continue

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if hm:
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            if lvl == 1 and skip_h1:
                i += 1
                continue
            sid = slugify(title)
            out.append(f'<h{lvl} id="{sid}">{inline_md(title, asset_prefix)}</h{lvl}>')
            i += 1
            continue

        # Blockquote / GitHub Alert
        if stripped.startswith(">"):
            bq_lines = []
            while i < n and lines[i].strip().startswith(">"):
                bq_lines.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            alert_kind = ""
            if bq_lines and re.match(r"^\[!(IMPORTANT|TIP|WARNING|NOTE|CAUTION)\]$", bq_lines[0].strip(), re.I):
                alert_kind = re.match(r"^\[!([A-Z]+)\]$", bq_lines[0].strip(), re.I).group(1).lower()
                bq_lines = bq_lines[1:]
            inner = md_to_html("\n".join(bq_lines), asset_prefix)
            if alert_kind:
                out.append(f'<div class="callout callout-{alert_kind}"><div class="callout-title">{alert_kind.upper()}</div>{inner}</div>')
            else:
                out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # GFM Table
        if "|" in stripped and i + 1 < n and re.match(r"^\|?[\s:\-|]+\|?$", lines[i + 1].strip()) and "-" in lines[i + 1]:
            def split_row(r: str) -> List[str]:
                r = r.strip().strip("|")
                return [c.strip() for c in r.split("|")]

            headers = split_row(lines[i])
            aligns_raw = split_row(lines[i + 1])
            aligns = []
            for a in aligns_raw:
                if a.startswith(":") and a.endswith(":"):
                    aligns.append("center")
                elif a.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(split_row(lines[i]))
                i += 1
            th_html = "".join(
                f'<th style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{inline_md(h, asset_prefix)}</th>'
                for idx, h in enumerate(headers)
            )
            body_rows = []
            for r in rows:
                tds = "".join(
                    f'<td style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{inline_md(c, asset_prefix)}</td>'
                    for idx, c in enumerate(r)
                )
                body_rows.append(f"<tr>{tds}</tr>")
            out.append(f'<div class="table-wrap"><table><thead><tr>{th_html}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>')
            continue

        # Unordered / Ordered list
        if re.match(r"^([-*]|\d+\.)\s+", stripped):
            is_ol = bool(re.match(r"^\d+\.\s+", stripped))
            tag = "ol" if is_ol else "ul"
            items = []
            while i < n:
                cur = lines[i].strip()
                if not cur:
                    break
                m_item = re.match(r"^(?:[-*]|\d+\.)\s+(.*)$", cur)
                if m_item:
                    items.append(m_item.group(1))
                    i += 1
                elif lines[i].startswith("  ") and items:
                    items[-1] += " " + cur
                    i += 1
                else:
                    break
            li_html = "".join(f"<li>{inline_md(it, asset_prefix)}</li>" for it in items)
            out.append(f"<{tag}>{li_html}</{tag}>")
            continue

        # Paragraph
        para = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt or nxt.startswith(("#", ">", "```", "---", "|")) or re.match(r"^([-*]|\d+\.)\s+", nxt):
                break
            para.append(nxt)
            i += 1
        out.append(f'<p>{inline_md(" ".join(para), asset_prefix)}</p>')

    return "\n".join(out)


def collect_repo_data(repo_root: Path, docs_dir: Path) -> Dict[str, Any]:
    assets_dir = docs_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Recipes from metadata.yaml
    recipes = []
    for mp in sorted(glob.glob(str(repo_root / "models" / "**" / "metadata.yaml"), recursive=True)):
        p = Path(mp)
        meta = parse_simple_yaml(p.read_text(encoding="utf-8"))
        rdir = p.parent
        meta["slug"] = rdir.name
        meta["rel_path"] = rdir.relative_to(repo_root).as_posix()
        meta["model_slug"] = rdir.parents[2].name
        meta["hw_slug"] = rdir.parents[1].name
        meta["guide_id"] = f"guide-{slugify(meta['model_slug'] + '-' + meta['hw_slug'])}"
        serve_p = rdir / "serve.sh"
        gke_p = rdir / "deploy-gke.yaml"
        meta["serve_sh"] = serve_p.read_text(encoding="utf-8") if serve_p.exists() else ""
        meta["deploy_gke"] = gke_p.read_text(encoding="utf-8") if gke_p.exists() else ""
        recipes.append(meta)

    # 2. Model/hardware guides
    guides = []
    for rp in sorted(glob.glob(str(repo_root / "models" / "*" / "*" / "README.md"))):
        p = Path(rp)
        hw_dir = p.parent
        hw_slug = hw_dir.name
        model_slug = hw_dir.parent.name
        gid = f"guide-{slugify(model_slug + '-' + hw_slug)}"
        asset_sub = assets_dir / f"{model_slug}-{hw_slug}"
        asset_sub.mkdir(parents=True, exist_ok=True)
        # Copy chart PNGs & SVGs into docs/assets/<model>-<hw>/
        charts = []
        results_dir = hw_dir / "results"
        if results_dir.is_dir():
            for img in sorted(results_dir.glob("*.png")):
                shutil.copy2(img, asset_sub / img.name)
                charts.append(f"assets/{model_slug}-{hw_slug}/{img.name}")
        md = p.read_text(encoding="utf-8")
        first_h1 = ""
        for ln in md.splitlines():
            if ln.startswith("# "):
                first_h1 = ln[2:].strip()
                break
        guides.append({
            "id": gid,
            "model_slug": model_slug,
            "hw_slug": hw_slug,
            "title": first_h1 or f"{model_slug} on {hw_slug}",
            "rel_path": p.relative_to(repo_root).as_posix(),
            "charts": charts,
            "html": md_to_html(md, asset_prefix=f"assets/{model_slug}-{hw_slug}", skip_h1=True),
        })

    # 3. Hardware docs
    hardware_docs = []
    for hp in sorted((repo_root / "hardware").glob("*.md")):
        md = hp.read_text(encoding="utf-8")
        h1 = hp.stem
        for ln in md.splitlines():
            if ln.startswith("# "):
                h1 = ln[2:].strip()
                break
        hardware_docs.append({
            "id": f"hw-{slugify(hp.stem)}",
            "slug": hp.stem,
            "title": h1,
            "html": md_to_html(md, skip_h1=True),
        })

    # 4. Bench README
    bench_md = (repo_root / "bench" / "README.md").read_text(encoding="utf-8") if (repo_root / "bench" / "README.md").exists() else ""
    bench_html = md_to_html(bench_md, skip_h1=True)

    return {
        "recipes": recipes,
        "guides": guides,
        "hardware": hardware_docs,
        "bench_html": bench_html,
    }


def render_html(data: Dict[str, Any]) -> str:
    recipes = data["recipes"]
    guides = data["guides"]
    hardware = data["hardware"]

    today = date.today().isoformat()
    num_recipes = len(recipes)
    num_models = len({r["model_slug"] for r in recipes})
    num_hw = len({r["hw_slug"] for r in recipes})

    # Build sidebar nav items
    guide_nav = "\n".join(
        f'<a class="nav-item" data-page="{g["id"]}" href="#{g["id"]}" onclick="navTo(\'{g["id"]}\');return false;">'
        f'<span class="nav-dot"></span>{html.escape(g["model_slug"])} · {html.escape(g["hw_slug"])}</a>'
        for g in guides
    )
    hw_nav = "\n".join(
        f'<a class="nav-item" data-page="{h["id"]}" href="#{h["id"]}" onclick="navTo(\'{h["id"]}\');return false;">'
        f'<span class="nav-dot hw-dot"></span>{html.escape(h["slug"])}</a>'
        for h in hardware
    )

    # Leaderboard rows
    lb_rows = []
    for r in recipes:
        m = r.get("metrics", {})
        feats = r.get("features", [])
        badges = "".join(f'<span class="pill">{html.escape(str(f))}</span>' for f in feats)
        lb_rows.append(
            f'<tr data-search="{html.escape((str(r.get("model","")) + " " + str(r.get("hardware","")) + " " + str(r.get("topology","")) + " " + str(r.get("quantization","")) + " " + " ".join(map(str, feats))).lower())}">'
            f'<td><strong>{html.escape(str(r.get("model","")))}</strong><div class="sub-cell">{html.escape(str(r.get("checkpoint","")))}</div></td>'
            f'<td><code>{html.escape(str(r.get("hardware","")))}</code><div class="sub-cell">{html.escape(str(r.get("cloud_instance","")))}</div></td>'
            f'<td><code>{html.escape(str(r.get("topology","")))}</code></td>'
            f'<td><span class="quant-badge quant-{slugify(str(r.get("quantization","")))}">{html.escape(str(r.get("quantization","")))}</span> '
            f'{html.escape(str(r.get("engine","")))} {html.escape(str(r.get("engine_version","")))}<div class="pill-row">{badges}</div></td>'
            f'<td class="num"><strong>{html.escape(str(m.get("c1_e2e_p90_s","—")))} s</strong></td>'
            f'<td class="num"><strong>{html.escape(str(m.get("peak_out_tok_s","—")))} tok/s</strong><div class="sub-cell">{html.escape(str(m.get("peak_load","")))}</div></td>'
            f'<td class="actions">'
            f'<a href="#{r["guide_id"]}" onclick="navTo(\'{r["guide_id"]}\');return false;">Guide</a> · '
            f'<a href="#recipe-{slugify(r["slug"])}" onclick="navTo(\'recipes\', \'recipe-{slugify(r["slug"])}\');return false;">Launch</a>'
            f'</td></tr>'
        )

    # Featured Pareto charts on Overview
    featured_charts = []
    for g in guides:
        for ch in g["charts"]:
            lbl = Path(ch).stem.replace("_", " ").title()
            featured_charts.append(
                f'<div class="chart-card"><img src="{ch}" alt="{html.escape(lbl)}" loading="lazy" onclick="openLightbox(this.src, this.alt)"/></div>'
            )

    # Recipe cards page
    recipe_cards = []
    for r in recipes:
        rid = f"recipe-{slugify(r['slug'])}"
        m = r.get("metrics", {})
        serve_block = (
            f'<div class="code-wrap"><div class="code-bar"><span>serve.sh</span>'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></div>'
            f'<pre><code>{html.escape(r["serve_sh"])}</code></pre></div>'
        ) if r["serve_sh"] else ""
        gke_block = (
            f'<div class="code-wrap"><div class="code-bar"><span>deploy-gke.yaml</span>'
            f'<button class="copy-btn" onclick="copyCode(this)">Copy</button></div>'
            f'<pre><code>{html.escape(r["deploy_gke"])}</code></pre></div>'
        ) if r["deploy_gke"] else ""
        recipe_cards.append(f"""
        <article class="recipe-card" id="{rid}">
          <div class="recipe-head">
            <div>
              <h3>{html.escape(r['model'])} — <code>{html.escape(r['slug'])}</code></h3>
              <p class="recipe-meta">
                <code>{html.escape(r['hardware'])}</code> · <code>{html.escape(r['topology'])}</code> ·
                {html.escape(r['engine'])} {html.escape(str(r['engine_version']))} · {html.escape(r['quantization'])} ·
                c=1 E2E p90: <strong>{html.escape(str(m.get('c1_e2e_p90_s','—')))} s</strong> ·
                Peak: <strong>{html.escape(str(m.get('peak_out_tok_s','—')))} tok/s</strong> ({html.escape(str(m.get('peak_load','')))})
              </p>
            </div>
            <a class="gh-link" href="{REPO_URL}/tree/main/{r['rel_path']}" target="_blank" rel="noopener">View on GitHub ↗</a>
          </div>
          <div class="tabs">
            <button class="tab-btn active" onclick="switchTab(this, '{rid}-serve')">serve.sh</button>
            <button class="tab-btn" onclick="switchTab(this, '{rid}-gke')">deploy-gke.yaml</button>
          </div>
          <div class="tab-pane active" id="{rid}-serve">{serve_block}</div>
          <div class="tab-pane" id="{rid}-gke">{gke_block}</div>
        </article>
        """)

    # Guide pages
    guide_sections = []
    for g in guides:
        guide_sections.append(f"""
        <section class="page" id="page-{g['id']}">
          <div class="page-header">
            <div class="breadcrumb">Models / {html.escape(g['model_slug'])} / {html.escape(g['hw_slug'])}</div>
            <h1>{html.escape(g['title'])}</h1>
            <a class="gh-link" href="{REPO_URL}/blob/main/{g['rel_path']}" target="_blank" rel="noopener">Edit / View Source on GitHub ↗</a>
          </div>
          <div class="prose">{g['html']}</div>
        </section>
        """)

    # Hardware pages
    hw_sections = []
    for h in hardware:
        hw_sections.append(f"""
        <section class="page" id="page-{h['id']}">
          <div class="page-header">
            <div class="breadcrumb">Hardware Reference</div>
            <h1>{html.escape(h['title'])}</h1>
            <a class="gh-link" href="{REPO_URL}/blob/main/hardware/{h['slug']}.md" target="_blank" rel="noopener">View Source on GitHub ↗</a>
          </div>
          <div class="prose">{h['html']}</div>
        </section>
        """)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Inference Recipes — Tuned LLM Serving Benchmarks &amp; Pareto Frontiers</title>
<meta name="description" content="Reproducible LLM serving recipes, controlled ablations, and SemiAnalysis-style Pareto benchmarks across GPUs and serving engines."/>
<link rel="canonical" href="{SITE_URL}"/>
<meta property="og:type" content="website"/>
<meta property="og:title" content="Inference Recipes — Tuned LLM Serving Benchmarks"/>
<meta property="og:description" content="Reproducible LLM serving recipes, SemiAnalysis-style Pareto charts, and runnable vLLM/SGLang configs."/>
<meta property="og:url" content="{SITE_URL}"/>
<link rel="icon" href='data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="%230D9488"/><path d="M8 22 L13 15 L18 18 L24 10" stroke="white" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'/>
<style>
:root {{
  --bg: #ffffff;
  --bg-subtle: #f8fafc;
  --bg-sidebar: #f1f5f9;
  --text: #0f172a;
  --text-muted: #475569;
  --border: #e2e8f0;
  --accent: #0d9488;
  --accent-2: #2563eb;
  --accent-soft: #ecfdf5;
  --code-bg: #0f172a;
  --code-text: #e2e8f0;
  --shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
}}
[data-theme="dark"] {{
  --bg: #0b1120;
  --bg-subtle: #111827;
  --bg-sidebar: #0f172a;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --border: #1e293b;
  --accent: #2dd4bf;
  --accent-2: #60a5fa;
  --accent-soft: rgba(45,212,191,0.12);
  --code-bg: #020617;
  --code-text: #e2e8f0;
  --shadow: 0 1px 3px rgba(0,0,0,0.4);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  display: flex;
  min-height: 100vh;
}}
a {{ color: var(--accent-2); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Sidebar */
.sidebar {{
  width: 272px;
  flex-shrink: 0;
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  padding: 20px 16px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
.brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  font-size: 17px;
  color: var(--text);
  letter-spacing: -0.02em;
}}
.brand-badge {{
  width: 28px; height: 28px; border-radius: 7px;
  background: linear-gradient(135deg, #0d9488, #2563eb);
  display: grid; place-items: center; color: #fff; font-size: 14px; font-weight: 800;
}}
.nav-group-title {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  font-weight: 700;
  margin: 4px 8px 6px;
}}
.nav-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 7px;
  color: var(--text);
  font-size: 13.5px;
  font-weight: 500;
  transition: background 0.12s;
}}
.nav-item:hover {{ background: var(--border); text-decoration: none; }}
.nav-item.active {{ background: var(--accent-soft); color: var(--accent); font-weight: 700; }}
.nav-dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }}
.hw-dot {{ background: var(--accent-2); }}
.sidebar-footer {{
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12.5px;
}}
.theme-btn {{
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}}

/* Main content */
.main {{
  flex: 1;
  min-width: 0;
  max-width: 1160px;
  margin: 0 auto;
  padding: 32px 40px 80px;
}}
.page {{ display: none; }}
.page.active {{ display: block; }}

/* Hero */
.hero {{
  border-bottom: 1px solid var(--border);
  padding-bottom: 24px;
  margin-bottom: 28px;
}}
.hero h1 {{
  margin: 0 0 8px;
  font-size: 32px;
  letter-spacing: -0.025em;
  line-height: 1.2;
}}
.hero p {{ margin: 0; color: var(--text-muted); font-size: 15.5px; max-width: 780px; }}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  margin: 22px 0 6px;
}}
.kpi {{
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
  box-shadow: var(--shadow);
}}
.kpi-label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi-val {{ font-size: 24px; font-weight: 800; color: var(--text); margin-top: 4px; letter-spacing: -0.02em; }}
.kpi-sub {{ font-size: 12px; color: var(--accent); font-weight: 600; margin-top: 2px; }}

/* Filter bar & tables */
.filter-bar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin: 20px 0 12px;
  flex-wrap: wrap;
}}
.filter-bar input {{
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--bg);
  color: var(--text);
  font-size: 13.5px;
  width: 300px;
  max-width: 100%;
}}
.table-wrap {{
  overflow-x: auto;
  border: 1px solid var(--border);
  border-radius: 10px;
  margin: 14px 0 24px;
  background: var(--bg);
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13.5px;
}}
th, td {{
  padding: 10px 13px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
th {{
  background: var(--bg-subtle);
  font-weight: 700;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  white-space: nowrap;
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover {{ background: var(--bg-subtle); }}
td.num {{ text-align: right; white-space: nowrap; }}
td.actions {{ white-space: nowrap; font-weight: 600; }}
.sub-cell {{ font-size: 11.5px; color: var(--text-muted); margin-top: 2px; }}
.pill-row {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }}
.pill {{
  font-size: 11px;
  padding: 1px 7px;
  border-radius: 99px;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  color: var(--text-muted);
}}
.quant-badge {{
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  padding: 1px 7px;
  border-radius: 5px;
  margin-right: 4px;
}}
.quant-nvfp4 {{ background: #ccfbf1; color: #0f766e; }}
.quant-fp8 {{ background: #fef3c7; color: #b45309; }}
.quant-bf16 {{ background: #e2e8f0; color: #334155; }}

/* Charts */
.charts-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
  margin: 20px 0 28px;
}}
.chart-card, .chart-fig {{
  margin: 18px 0;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
  box-shadow: var(--shadow);
}}
.chart-card img, .chart-fig img {{
  width: 100%;
  height: auto;
  display: block;
  border-radius: 6px;
  cursor: zoom-in;
}}
.chart-fig figcaption {{
  font-size: 12px;
  color: #64748b;
  margin-top: 8px;
  text-align: center;
}}

/* Prose & Callouts */
.page-header {{
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 24px;
}}
.page-header h1 {{ margin: 4px 0 8px; font-size: 28px; letter-spacing: -0.02em; }}
.breadcrumb {{ font-size: 12px; font-weight: 700; color: var(--accent); text-transform: uppercase; letter-spacing: 0.06em; }}
.gh-link {{ font-size: 13px; font-weight: 600; }}
.prose h2 {{ margin: 30px 0 12px; font-size: 21px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
.prose h3 {{ margin: 22px 0 10px; font-size: 17px; }}
.prose h4 {{ margin: 16px 0 8px; font-size: 15px; }}
.prose code {{
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 0.88em;
  background: var(--bg-subtle);
  border: 1px solid var(--border);
  padding: 1px 5px;
  border-radius: 4px;
}}
.callout {{
  border-left: 4px solid var(--accent);
  background: var(--bg-subtle);
  border-radius: 0 8px 8px 0;
  padding: 12px 16px;
  margin: 16px 0;
}}
.callout-warning {{ border-left-color: #d97706; }}
.callout-tip {{ border-left-color: #0d9488; }}
.callout-important {{ border-left-color: #2563eb; }}
.callout-title {{ font-size: 11.5px; font-weight: 800; letter-spacing: 0.06em; margin-bottom: 4px; color: var(--accent); }}
.callout-warning .callout-title {{ color: #d97706; }}
.callout-important .callout-title {{ color: #2563eb; }}

/* Code blocks & Tabs */
.code-wrap {{
  border-radius: 9px;
  overflow: hidden;
  border: 1px solid var(--border);
  margin: 12px 0 18px;
  background: var(--code-bg);
}}
.code-bar {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  background: rgba(255,255,255,0.06);
  color: #94a3b8;
  font-size: 12px;
  font-family: monospace;
}}
.copy-btn {{
  background: rgba(255,255,255,0.12);
  color: #f8fafc;
  border: none;
  border-radius: 5px;
  padding: 3px 9px;
  font-size: 11.5px;
  cursor: pointer;
}}
.copy-btn:hover {{ background: rgba(255,255,255,0.22); }}
pre {{
  margin: 0;
  padding: 14px 16px;
  overflow-x: auto;
  color: var(--code-text);
  font-size: 12.5px;
  line-height: 1.5;
}}
pre code {{ background: none !important; border: none !important; padding: 0 !important; color: inherit; }}

.recipe-card {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 22px;
  background: var(--bg-subtle);
}}
.recipe-head {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  flex-wrap: wrap;
}}
.recipe-head h3 {{ margin: 0 0 4px; font-size: 17px; }}
.recipe-meta {{ margin: 0 0 12px; font-size: 13px; color: var(--text-muted); }}
.tabs {{ display: flex; gap: 6px; margin-bottom: 4px; }}
.tab-btn {{
  background: var(--bg);
  color: var(--text-muted);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 12px;
  font-size: 12.5px;
  font-weight: 600;
  cursor: pointer;
}}
.tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.tab-pane {{ display: none; }}
.tab-pane.active {{ display: block; }}

/* Lightbox */
.lightbox {{
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(15,23,42,0.88);
  z-index: 999;
  place-items: center;
  padding: 24px;
  cursor: zoom-out;
}}
.lightbox.open {{ display: grid; }}
.lightbox img {{ max-width: 96vw; max-height: 92vh; border-radius: 8px; background: #fff; }}

@media (max-width: 900px) {{
  body {{ flex-direction: column; }}
  .sidebar {{ width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border); }}
  .main {{ padding: 20px 16px 60px; }}
}}
</style>
</head>
<body>

<aside class="sidebar">
  <div class="brand">
    <div class="brand-badge">IR</div>
    <div>inference-recipes</div>
  </div>

  <div>
    <div class="nav-group-title">Overview</div>
    <a class="nav-item active" data-page="overview" href="#overview" onclick="navTo('overview');return false;">Leaderboard &amp; Pareto</a>
    <a class="nav-item" data-page="recipes" href="#recipes" onclick="navTo('recipes');return false;">Runnable Recipes ({num_recipes})</a>
    <a class="nav-item" data-page="methodology" href="#methodology" onclick="navTo('methodology');return false;">Benchmark Harness</a>
  </div>

  <div>
    <div class="nav-group-title">Model Guides</div>
    {guide_nav}
  </div>

  <div>
    <div class="nav-group-title">Hardware Reference</div>
    {hw_nav}
  </div>

  <div class="sidebar-footer">
    <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub Repo ↗</a>
    <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">Dark</button>
  </div>
</aside>

<main class="main">
  <!-- OVERVIEW -->
  <section class="page active" id="page-overview">
    <div class="hero">
      <h1>Inference Recipes</h1>
      <p>Reproducible, tuned LLM serving recipes, controlled single-variable ablations, and SemiAnalysis-style Pareto frontiers across accelerators and serving engines.</p>
      <div class="kpi-grid">
        <div class="kpi">
          <div class="kpi-label">Fastest Single-Stream</div>
          <div class="kpi-val">1.65 s p90</div>
          <div class="kpi-sub">Qwen3.8-27B · TP=8 NVFP4 MTP-5 (10k/500)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Peak 8-GPU Node</div>
          <div class="kpi-val">4,172 tok/s</div>
          <div class="kpi-sub">DP=8 × TP=1 NVFP4 MTP-4 (3.6×–7.1× vs TP)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Published Recipes</div>
          <div class="kpi-val">{num_recipes}</div>
          <div class="kpi-sub">{num_models} model(s) · {num_hw} accelerator(s)</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Last Updated</div>
          <div class="kpi-val">{today}</div>
          <div class="kpi-sub">Apache 2.0 · Zero-error verified</div>
        </div>
      </div>
    </div>

    <div class="filter-bar">
      <h2 style="margin:0;">Recipe Leaderboard</h2>
      <input type="search" id="lbSearch" placeholder="Filter by model, hardware, quant, TP..." oninput="filterTable(this.value)"/>
    </div>
    <div class="table-wrap">
      <table id="lbTable">
        <thead>
          <tr>
            <th>Model</th>
            <th>Hardware</th>
            <th>Topology</th>
            <th>Engine &amp; Stack</th>
            <th style="text-align:right">c=1 E2E p90</th>
            <th style="text-align:right">Peak Throughput</th>
            <th>Links</th>
          </tr>
        </thead>
        <tbody>
          {"".join(lb_rows)}
        </tbody>
      </table>
    </div>

    <h2>Pareto Frontiers</h2>
    <p style="color:var(--text-muted);margin-top:-6px;">Click any chart to zoom. Generated directly from verified <code>results/metrics.json</code> runs.</p>
    <div class="charts-grid">
      {"".join(featured_charts)}
    </div>
  </section>

  <!-- RUNNABLE RECIPES -->
  <section class="page" id="page-recipes">
    <div class="page-header">
      <div class="breadcrumb">Copy &amp; Deploy</div>
      <h1>Runnable Serving Recipes</h1>
      <p style="color:var(--text-muted);margin:0;">Exact <code>serve.sh</code> launch commands and Kubernetes / GKE manifests for every published operating tier.</p>
    </div>
    {"".join(recipe_cards)}
  </section>

  <!-- METHODOLOGY -->
  <section class="page" id="page-methodology">
    <div class="page-header">
      <div class="breadcrumb">Benchmarks</div>
      <h1>Benchmarking Harness &amp; Methodology</h1>
      <a class="gh-link" href="{REPO_URL}/tree/main/bench" target="_blank" rel="noopener">View bench/ on GitHub ↗</a>
    </div>
    <div class="prose">{data['bench_html']}</div>
  </section>

  <!-- MODEL GUIDES -->
  {"".join(guide_sections)}

  <!-- HARDWARE PAGES -->
  {"".join(hw_sections)}
</main>

<div class="lightbox" id="lightbox" onclick="this.classList.remove('open')">
  <img id="lightboxImg" src="" alt=""/>
</div>

<script>
function navTo(pageId, anchorId) {{
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(a => a.classList.remove('active'));
  const target = document.getElementById('page-' + pageId) || document.getElementById('page-overview');
  target.classList.add('active');
  const nav = document.querySelector('.nav-item[data-page="' + pageId + '"]');
  if (nav) nav.classList.add('active');
  if (anchorId) {{
    setTimeout(() => {{
      const el = document.getElementById(anchorId);
      if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}, 30);
  }} else {{
    window.scrollTo({{ top: 0, behavior: 'instant' }});
  }}
  history.replaceState(null, '', '#' + (anchorId || pageId));
}}

function switchTab(btn, paneId) {{
  const card = btn.closest('.recipe-card');
  card.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  card.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(paneId).classList.add('active');
}}

function copyCode(btn) {{
  const code = btn.closest('.code-wrap').querySelector('code').innerText;
  navigator.clipboard.writeText(code).then(() => {{
    const orig = btn.innerText;
    btn.innerText = 'Copied!';
    setTimeout(() => btn.innerText = orig, 1400);
  }});
}}

function filterTable(q) {{
  q = q.trim().toLowerCase();
  document.querySelectorAll('#lbTable tbody tr').forEach(tr => {{
    tr.style.display = !q || tr.dataset.search.includes(q) ? '' : 'none';
  }});
}}

function openLightbox(src, alt) {{
  const lb = document.getElementById('lightbox');
  const img = document.getElementById('lightboxImg');
  img.src = src; img.alt = alt || '';
  lb.classList.add('open');
}}

function toggleTheme() {{
  const html = document.documentElement;
  const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  document.getElementById('themeBtn').innerText = next === 'dark' ? 'Light' : 'Dark';
}}

// Handle initial hash navigation
(function() {{
  const h = location.hash.replace(/^#/, '');
  if (!h) return;
  if (h.startsWith('recipe-')) {{ navTo('recipes', h); }}
  else if (document.getElementById('page-' + h)) {{ navTo(h); }}
}})();
</script>
</body>
</html>
"""


def render_llms_txt(data: Dict[str, Any]) -> str:
    lines = [
        "# inference-recipes",
        "",
        f"> Reproducible, tuned LLM serving recipes and Pareto benchmarks ({SITE_URL})",
        "",
        "## Recipes",
    ]
    for r in data["recipes"]:
        m = r.get("metrics", {})
        lines.append(
            f"- {r['model']} on {r['hardware']} ({r['topology']}, {r['engine']} {r['engine_version']} {r['quantization']}): "
            f"c=1 E2E p90={m.get('c1_e2e_p90_s')}s, Peak={m.get('peak_out_tok_s')} tok/s ({m.get('peak_load')}) "
            f"[{REPO_URL}/tree/main/{r['rel_path']}]"
        )
    lines.append("")
    lines.append("## Guides")
    for g in data["guides"]:
        lines.append(f"- {g['title']}: {REPO_URL}/blob/main/{g['rel_path']}")
    return "\n".join(lines) + "\n"


def render_sitemap() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f'  <url><loc>{SITE_URL}</loc><lastmod>{date.today().isoformat()}</lastmod></url>\n'
        "</urlset>\n"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    data = collect_repo_data(repo_root, docs_dir)
    (docs_dir / "index.html").write_text(render_html(data), encoding="utf-8")
    (docs_dir / "llms.txt").write_text(render_llms_txt(data), encoding="utf-8")
    (docs_dir / "sitemap.xml").write_text(render_sitemap(), encoding="utf-8")
    print(f"Built docs/index.html ({len(data['recipes'])} recipes, {len(data['guides'])} guides, {len(data['hardware'])} hardware docs)")


if __name__ == "__main__":
    main()
