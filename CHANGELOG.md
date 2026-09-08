# Changelog

## 2026-09-08 — ScientificState research workers and policy admission

- Refresh the English and Chinese project introductions, document actual CLI
  workspace creation and Python resumption, and separate system design from
  validation evidence. Remove placeholder demo emphasis from the READMEs.
- CLI research and the local console now call the same `run_research` entry.
  OpenWorld selects actions from `ScientificState`; the existing literature,
  hypothesis, registered-probe, verification, and synthesis workers execute
  those actions. `--legacy-pipeline` is a deprecated compatibility flag for
  that same controller. The historical Python pipeline remains a regression
  oracle, not another product control loop.
- Literature artifacts preserve verified citation identities, dates, and
  source spans. Paper states and transitions remain GENERATED interpretations;
  missing fields remain unknown. Transitions require citation lineage,
  temporal order, and grounded conceptual change. Structural applicability
  and historical-copy checks constrain operator reuse without proving novelty.
- First-hop exploration has trajectory-supported, assumption-reframe, and
  unconstrained speculative lanes. Later hops use canonical parent evidence;
  explicit execution failures can motivate a bounded failure reframe without
  being treated as scientific negatives.
- Experiments register identities and execute actual source against bound
  world bytes. Hypothesis-disagreement and gain/cost estimates guide scheduling
  as declared heuristics, not measured information gain. Replayable belief
  scores use reliability-weighted counts, not calibrated Bayesian inference.
  Scoped verified state requires the relevant reproduction, mechanism,
  identity, and independent heldout checks.
- Python callers can supply explicit `heldout_worlds`; the CLI accepts
  repeatable `--heldout-world` arguments. Each session records
  its service configuration and policy snapshot; `policy_file` and
  `policy_log` connect the existing mission-end research-policy workflow.
  Admission requires an evaluator-issued immutable single-use receipt binding
  the candidate, baseline, failure replay, nearby/heldout/regression cases,
  costs, and strictly positive heldout gain. Missing evaluation rejects an
  update. Research-time evolution only records proposals; it does not install
  capabilities or automatically promote judges.
- Trusted Python applications can supply `policy_suite` and `policy_evaluator`.
  Evaluation occurs after segment completion and log append, then admission
  consumes the evaluator-issued receipt. Missing trusted evaluation rejects.
- Synthesis and optional paper compilation now derive scientific scope and
  exact world identity from the same registered protocol/probe. Saved generated
  fact-sheet overrides cannot clear `cannot_corroborate` or prohibited
  inferences. Explicit identity mismatches and unsupported confirmation are
  rejected; the deterministic prose guard is not a complete semantic audit.
  Missing world provenance is labelled unknown, not published.
- Production artifacts live in `SCIENTIFIC_STATE.json`,
  `SCIENTIFIC_EVENTS.json`, `literature/`, and
  `research_workers/<evidence-id>/candidates/<card-id>/`.
  `RESEARCH_STATUS.md` and `research_complete: false` describe an open study,
  not a successfully completed paper. Both READMEs distinguish these outputs
  from historical bilingual research packets and document the current model
  configuration without including credentials.

Partial live validation: `gpt-5.6-sol` returned `OK` on a connectivity request
(17 tokens). An initial research mission encountered source HTTP 429/timeouts.
After the raw-term retrieval correction, a later check retrieved four real
verified papers and made four extraction calls totaling 2,731 tokens. Subsequent
bounded runs generated a typed speculative hypothesis, performed its literature
check, and invoked real diagnosis/probe-writing workers. A generated script's
measurement identifier differed from registration and was blocked, not counted
as scientific evidence. The controller then produced a failure-reframe branch.
The last segment exhausted its four-call budget and ended with event replay
matching the persisted state. The writer now receives an explicit registered
measurement identifier; that final prompt correction has offline regression
coverage, not a subsequent live result. These checks do not demonstrate completed
research, scientific novelty, live mission parity, or effective policy
self-improvement. See [VALIDATION.md](VALIDATION.md) for tests and limitations.

中文：CLI 与控制台统一由 ScientificState 驱动，接通真实文献与已登记实验 worker。
论文轨迹、三条探索分支和优先级仍是推测或启发；信念分数是可回放记账，非校准贝叶斯。
研究策略只能在任务结束后凭完整留出评估回执准入，不自动晋升评判器。
真实调用已覆盖生成、假设文献检查、诊断/脚本生成和失败重构；指标不一致被拒绝，
没有变成实验结果。最后一个有上限的运行段状态回放一致。运行结束不等于研究成功。

## 2026-09-07 — OpenWorld research integrity fixes

Historical entry: the CLI routing distinction below was superseded by the
single-controller integration on 2026-09-08.

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
