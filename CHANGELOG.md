# Changelog

## 2026-09-07 — OpenWorld research integrity fixes

- Explicit world ids now load the catalog manifest, verify the recorded digest,
  and copy the original artifacts into the mission workspace. Missing worlds
  and changed data fail instead of producing a substitute.
- Manifestless inputs remain GENERATED regardless of their names. Generated
  provenance survives derivation, and derived manifests use stable file order.
- Evidence submitted through events passes the same admission checks as direct
  evidence. A matching frozen-world digest is required for WORLD admission.
- Default research no longer invents validator snapshots for arbitrary topics.
  Builtin exploratory summaries describe available tables, graphs, and sequences
  while leaving the research question open. Verification recomputes supported
  observations instead of trusting an attestation flag alone.
- OpenWorld writes `RESEARCH_STATUS.md` and explicitly reports
  `research_complete: false`. General LLM/literature work and research packets
  remain available through `--legacy-pipeline`; both READMEs and CLI help explain
  this distinction.
- Anchor events expose the actual memory key. Tests use this key instead of
  assuming the nearest graph concept is also the memory anchor.
- Live-SWE raw trajectories are documented as optional external data. Their two
  integration tests skip when absent; missing-world checks still run. CI now
  also discovers the contract and historical regression test directories.

Validation on Python 3.11: 1,072 main-suite tests (6 skipped), 30 contract tests,
and 2 historical regression tests; no failures. Ten research-integrity regression
tests cover the fixes. A local Iris smoke run read 150 original rows, preserved
the open prediction question, and reproduced its event-state digest. Live model
research quality was not evaluated in this update.

中文：本次修复重点是恢复真实数据绑定、阻止生成数据晋升为 WORLD、校验事件证据，
并明确默认 OpenWorld 的能力边界。完整文献、提案和实验流程请使用
`--legacy-pipeline`；默认路径提供本地探索性观测及研究状态记录。
