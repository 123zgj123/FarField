# ScientificState runtime validation — 2026-09-08

This is an implementation and validation record, not a claim of scientific
discovery or demonstrated recursive self-improvement.

## What changed

The CLI and console use one `run_research` / OpenWorld controller. Mature
literature, generation, diagnosis, probe execution, host reproduction, and
writing workers are adapters in that loop. `kernel.py`, `events.py`, the
World/Evidence identity implementations, and ClaimSpec infrastructure were not
rewritten. State extensions and admission checks preserve generated provenance.

Paper-state extraction and grounded transition retrieval live in the existing
exploration code. Three proposal lanes share the worker and event contracts.
Later hops consume canonical parent evidence, or record a bounded failure
reframe without calling failure an experimental result. Belief accounting is
deterministic and replayable; experiment priority estimates hypothesis
disagreement, not measured information gain.

Policy changes are routing-policy candidates, not autonomous kernel or judge
modifications. Admission requires disjoint historical-failure, nearby, heldout,
and regression replay partitions, measured costs, and a protected single-use
receipt with strict heldout improvement. The evaluator is trusted application
code; its absence rejects admission. This is not evidence that live RSI works.

## Required regression coverage

| Requirement | Regression files |
|---|---|
| Arbitrary-topic real literature worker dispatch | `test_openworld_workers.py`, `test_research_control.py`, `test_research_literature_queries.py` |
| THEORIZE independent of validator fixtures | `test_research_branches.py`, `test_research_world_compile.py` |
| GENERATED never promoted to WORLD | `test_research_integrity.py`, `test_research_belief.py` |
| Reject reconstruction of historical target | `test_research_trajectory.py` |
| Same operator across distinct research topics | `test_research_trajectory.py` |
| Detect lightly reworded historical copies | `test_research_trajectory.py` |
| Second hop reads actual first-hop evidence | `test_research_branches.py` |
| Failed first hop permits explicit reframe | `test_research_branches.py` |
| Belief updates replay from events | `test_research_belief.py`, `test_research_control.py` |
| Competitors change experiment selection | `test_research_selection.py` |
| Policy patch requires protected heldout admission | `test_research_policy_admission.py`, `test_research_policy.py` |
| Legacy/OpenWorld execution parity | `test_research_worker_parity.py`, `test_research_replication.py` |

The parity tests run actual registered subprocesses against frozen test worlds
with recorded-model responses. They compare identities, preregistration order,
positive/negative results, reproduction, unavailable services, and policy
refusals. They are **not** a live multi-topic mission parity benchmark.

Additional regressions reject mismatched hypothesis IDs, graph predicates used
as scientific falsifiers, wrong world/object bindings, changed diagnosis
handles or metrics, and unsupported claim escalation during paper drafting.
Resume rebuilds derived snapshots from unchanged scientific events.

## Reproduce offline verification

Latest complete run: 1,202 main-suite tests (6 skipped, no failures/errors),
30 contract tests and 2 historical regression tests passed. `git diff --check`
passed. The selected generation, scope, replication and parity paths also
received an independent bounded code review; offline correctness does not
establish live scientific value.

```bash
PYTHONPATH=src FARFIELD_LLM_MODE=replay python3.11 -m unittest discover -s tests -q
PYTHONPATH=src FARFIELD_LLM_MODE=replay python3.11 -m unittest discover -s tests/contracts -q
PYTHONPATH=src FARFIELD_LLM_MODE=replay python3.11 -m unittest discover -s tests/regressions -q
git diff --check
```

## Live-service observations and remaining validation

The exact official model `gpt-5.6-sol` answered a connectivity request. Subsequent
bounded research runs used the same model and the attested Fisher Iris world.
Credentials were process-only, not persisted in source or model caches.

Live retrieval exposed malformed stemmed queries; the corrected queries
retrieved four verified papers. Model outputs subsequently exposed inconsistent
competitor identifiers, missing closest-prior citations, and missing world
compilation context. The interface defects have regression coverage. Missing
citations or grounded historical trajectories still block honestly.

A later live response produced a speculative hypothesis with a concrete
falsifier, named competing explanation, discriminating proposal, and compiled
world/object identity. The controller performed a separate literature check on
that hypothesis. This establishes worker integration, **not** novelty, correct
causal interpretation, or verification of the hypothesis.

The final bounded segment invoked diagnosis and probe-writing workers. It
rejected a generated measurement description that did not match the registered
metric identifier (`probe_execution_blocked`), then generated an explicit
failure-reframe branch. No arm values or WORLD probe evidence were invented.
It used four model calls / 22,549 tokens and stopped at its declared envelope;
the final state digest matched event replay. A subsequent code correction makes
the registered measurement identifier explicit in the writer prompt while
retaining the rejection guard. This last prompt change has offline regression
coverage and has not been retried live.

Open validation requirements remain: diverse real mission parity, a useful
grounded trajectory corpus, independent replication worlds with suitable
scientific scope, and an application-owned heldout policy evaluation benchmark.
Lexical copy checks do not certify novelty; registered names and source checks
do not fully prove arbitrary generated programs implement their prose claims;
the deterministic draft guard is not a complete semantic audit. None of these
limits is replaced by LLM votes or fabricated observations.
