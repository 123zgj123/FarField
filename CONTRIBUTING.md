# Contributing

English · 简体中文

FarField is a research runtime with a scientific contract. Please keep that contract intact.

## Do

- Open one pull request per change.
- Add or extend tests under `tests/` for any behaviour you touch.
- Keep API keys in `$FARFIELD_LLM_KEY_FILE`. Never put keys in argv, artifacts, or git.
- Update [README.md](README.md) and [README.zh-CN.md](README.zh-CN.md) together when the public surface changes.

## Do not

- Weaken world binding, two-arm fairness, or the WORLD / GENERATED / SYNTHETIC ladder.
- Commit `var/`, LLM caches, report JSON, or `worlds/wishlist.json`.
- Treat Elo / QD as the research ladder.
- Change kernel behaviour in a “cleanup” or rename PR.

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -q
```

---

FarField 带科学合同。请保住合同，不要在整理类 PR 里改内核行为。一次变更一个 PR。密钥只走 `$FARFIELD_LLM_KEY_FILE`。公开说明同时改英文与中文 README。
