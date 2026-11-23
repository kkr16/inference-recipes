# Documentation Site (`kkr16.github.io/inference-recipes`)

A self-contained ReadTheDocs-style static site auto-generated from the repository's `models/`, `hardware/`, and `bench/` content.

## Regenerate

```bash
python3 bench/scripts/build_site.py
```

## View Locally

Open `docs/index.html` in any browser (no build tools or external dependencies required).

## Publish with GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → Branch: `main`, Folder: `/docs`.
Live URL: https://kkr16.github.io/inference-recipes/
