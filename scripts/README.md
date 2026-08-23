# Operator scripts

English · 简体中文

The product entry is `python3 -m farfield`, not this folder.

产品入口是 `python3 -m farfield`，不在这里。

| file | purpose / 作用 |
|---|---|
| `console.py` | web console → http://localhost:8765/ (`webui/index.html` + `i18n.js`; binds 127.0.0.1) |
| `build_corpora.py` | rebuild `corpora/` citation graphs from OpenAlex pages |
| `build_concepts.py` | rebuild `concepts/` co-occurrence graphs from arXiv OAI pages |

Report JSON, LLM caches, and local measurement scripts are not shipped.
