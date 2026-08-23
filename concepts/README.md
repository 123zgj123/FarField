# Concept co-occurrence graphs

English · 简体中文

`farfield research` reads `concepts/arxiv_registry.json`. The production slice is `ds-arxiv-concepts-2026`. Those `G_full.json` / `G_le_T.json` files exceed GitHub's 100 MB limit, so a clone ships **`attn-concepts-s1/`** (~8 MB) instead. If the production graph is missing, `load_assets` uses this demo slice — same mission path, smaller neighbourhood.

产品路径默认读 `ds-arxiv-concepts-2026`。大图不进 git。克隆后自动落在演示切片 `attn-concepts-s1/`，循环本身不变。

Shipped per slice: `manifest.json`, `heldout_post_T.json`. Only `attn-concepts-s1` also ships the graph JSON.

```bash
# Replay recorded OAI pages (gitignored concepts/*/raw/)
PYTHONPATH=src python3 scripts/build_concepts.py --replay

# Network only when raw pages are missing
PYTHONPATH=src python3 scripts/build_concepts.py
```

A digest mismatch means a different corpus. Do not swap the graph under the same id.
