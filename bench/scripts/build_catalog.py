#!/usr/bin/env python3
"""Generate the root README.md benchmark catalog table from models/**/metadata.yaml."""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List

START_MARKER = "<!-- CATALOG:START -->"
END_MARKER = "<!-- CATALOG:END -->"


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Minimal dependency-free parser for flat + 1-level nested recipe metadata.yaml."""
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


def load_recipes(repo_root: Path) -> List[Dict[str, Any]]:
    recipes = []
    for path_str in sorted(glob.glob(str(repo_root / "models" / "**" / "metadata.yaml"), recursive=True)):
        p = Path(path_str)
        meta = parse_simple_yaml(p.read_text(encoding="utf-8"))
        meta["_recipe_rel"] = p.parent.relative_to(repo_root).as_posix()
        # Guide is two levels up from recipes/<recipe_slug>/metadata.yaml -> models/<model>/<hw>/README.md
        hw_readme = p.parent.parent.parent / "README.md"
        if hw_readme.exists():
            meta["_guide_rel"] = hw_readme.relative_to(repo_root).as_posix()
        else:
            meta["_guide_rel"] = meta["_recipe_rel"]
        recipes.append(meta)
    recipes.sort(key=lambda r: (str(r.get("date", "")), str(r.get("model", "")), str(r.get("topology", ""))), reverse=True)
    return recipes


def render_table(recipes: List[Dict[str, Any]]) -> str:
    header = (
        "| Model | Hardware | Topology | Engine & Stack | Workload | `c=1` E2E p90 | Peak Throughput | Date | Recipe |\n"
        "|---|---|---|---|---|---:|---:|---|---|"
    )
    rows = [header]
    for r in recipes:
        metrics = r.get("metrics", {}) if isinstance(r.get("metrics"), dict) else {}
        features = r.get("features", [])
        feat_str = f" ({', '.join(features)})" if features else ""
        stack = f"{r.get('engine', '')} {r.get('engine_version', '')} · {r.get('quantization', '')}{feat_str}".strip()
        c1_p90 = metrics.get("c1_e2e_p90_s", "—")
        c1_str = f"**{c1_p90} s**" if c1_p90 != "—" else "—"
        peak = metrics.get("peak_out_tok_s", "—")
        peak_note = metrics.get("peak_load", "")
        peak_str = f"**{peak} tok/s**" + (f" ({peak_note})" if peak_note else "") if peak != "—" else "—"
        rows.append(
            f"| **{r.get('model', '')}** | `{r.get('hardware', '')}` | `{r.get('topology', '')}` "
            f"| {stack} | {r.get('workload', '')} | {c1_str} | {peak_str} | {r.get('date', '')} "
            f"| [Guide]({r['_guide_rel']}) · [Files]({r['_recipe_rel']}) |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build README catalog table from metadata.yaml files.")
    parser.add_argument("--write", action="store_true", help="Update README.md in-place between catalog markers.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    recipes = load_recipes(repo_root)
    table = render_table(recipes)

    if not args.write:
        print(table)
        return

    readme_path = repo_root / "README.md"
    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}", re.DOTALL)
    replacement = f"{START_MARKER}\n{table}\n{END_MARKER}"
    if not pattern.search(content):
        raise SystemExit(f"Markers {START_MARKER} / {END_MARKER} not found in {readme_path}")
    readme_path.write_text(pattern.sub(replacement, content), encoding="utf-8")
    print(f"Updated {readme_path} with {len(recipes)} recipes.")


if __name__ == "__main__":
    main()
