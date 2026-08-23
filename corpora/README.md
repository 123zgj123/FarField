# Citation corpora

English · 简体中文

Frozen citation graphs used when sampling concept pairs. Each slice ships `manifest.json`, graph JSON, and `heldout_post_T.json`. Raw OpenAlex pages live in gitignored `*/raw/` and can be replayed with `scripts/build_corpora.py`.

本目录是取样用的引文图。原始 API 页不进 git，用 `scripts/build_corpora.py` 重建。

| id | contents / 内容 |
|---|---|
| `attn-2017` | Attention / Transformer neighbourhood, cut at 2017 |
| `attn-2017-p2` | Same neighbourhood, companion slice |
| `sparse-2017` | sparse-recovery neighbourhood, cut at 2017 |
| `sparse-2017-p2` | Same neighbourhood, companion slice |
| `cacheob-2017` | cache-oblivious neighbourhood, cut at 2017 |
| `cacheob-2017-p2` | Same neighbourhood, companion slice |

Do not swap a graph under the same id if the manifest digest would change. Registry: `registry.json`.
