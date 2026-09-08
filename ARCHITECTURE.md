# FarField system design

[English overview](README.md) · [中文概览](README.zh-CN.md) · [Validation](VALIDATION.md)

FarField is an experimental, ScientificState-driven research runtime. Its
trajectory-conditioned and evidence-driven mechanisms are implemented research
infrastructure, not evidence that the system has learned to make discoveries.

## Authority and control

There is one product-level scientific controller:

```text
ScientificState → Frontier → Scheduler → Action → Existing Research Worker
       ↑                                                   ↓
       └── Scientific Event ← Trusted Kernel ← CandidateResult
```

[runtime.py](src/farfield/extras/openworld/runtime.py) connects the CLI and console
to OpenWorld. [workers.py](src/farfield/extras/openworld/workers.py) adapts existing
research services. Workers return candidates; they do not choose the next
research action or launch another mission.

The Trusted Kernel, event sourcing, World/Evidence identity, and ClaimSpec
infrastructure were not rewritten in this integration. State extensions and
admission checks operate within those contracts. Models can propose plans,
interpret papers, and write executable candidates; they cannot mint trusted
observations or authorize their own verifier changes.

| Action | Worker integration | Boundary |
|---|---|---|
| SURVEY | arXiv/livefeed, sources, prior work and literature spans | Missing citations or unavailable sources remain gaps |
| THEORIZE | generate, explore, reframe, diagnose | Structured hypotheses are GENERATED proposals |
| PROBE | diagnose, probeexp, hostexp | A generated plan is not an executed result |
| VERIFY | identity checks, registered execution, reproduction, heldout fixtures | Verification is scoped to actual evidence and objects |
| SYNTHESIZE | brief, packet, paperplan, writeup | Text cannot increase claim strength without the evidence contract |

## Trajectory-conditioned exploration

[explore.py](src/farfield/extras/explore.py) represents a paper using its problem,
research question, gap, assumptions, representation, mechanism, objective,
method, evidence, evaluation regime, limitations, and unresolved questions.
Citation identity and source spans remain attached. Extracted interpretations
are GENERATED; absent information stays unknown.

A historical transition needs citation lineage, temporal order, and changed
components grounded in exact source spans. It records source and target states,
trigger, changed and preserved components, operator, mechanism/assumption/
objective/evaluation deltas, supporting evidence, applicability, and failure
conditions. A paper pair is not automatically a reliable transition.

The finite, extensible operator vocabulary includes:

- assumption removal/inversion, problem reframing, representation/objective changes;
- decomposition, mechanism replacement, adaptive control, feedback introduction;
- formalization, boundary search, scaling, cross-domain transfer;
- verifier shift and failure-driven redesign.

An operator is a proposed scientific transformation. In particular,
`verifier_shift` does not grant permission to replace the Trusted Kernel.

The first-hop frontier provides three candidate lanes:

| Lane | Anchor |
|---|---|
| `trajectory_supported` | Applicable, grounded historical transitions |
| `assumption_reframe` | An assumption, gap, or boundary in the current problem |
| `unconstrained` | Exploratory hypotheses without a required retrieved trajectory |

Missing trajectory support blocks that lane rather than fabricating examples.
Candidates retain source trajectories, reused operators, closest prior work,
inherited and genuinely changed components, claim/mechanism deltas, and an
explanation of why they are not historical reconstruction. Copy checks and the
existing apply-X-to-Y rejection rule constrain proposals; neither proves novelty.

Current retrieval uses structural matching of research-state conditions, not
embedding distance as scientific value. Coverage and matching are limited:
there is no broad learned transition corpus or trained
`P(operator | research_state, gap, evidence)` yet.

## Evidence-driven later hops

A later scientific branch reads its canonical parent's admitted evidence,
unresolved questions, and relevant failure or contradiction state. Hypotheses
undergo a separate literature check and propose a discriminating experiment
against explicit competing explanations.

```text
State → operator → hypothesis → literature / competing explanations
  ↑                                      ↓
  └──────── admitted evidence ← registered probe
```

An execution failure can justify a bounded `failure_reframe` branch. This is
recovery from a failed research attempt, not a scientifically negative result
or an evidence-supported discovery. Further exploration is not an unrestricted
sequence of latent-space jumps.

## Belief and experiment selection

[state.py](src/farfield/extras/openworld/state.py) and
[reducer.py](src/farfield/extras/openworld/reducer.py) record prior confidence,
supporting/contradicting evidence, reliability, competitors, unresolved
assumptions, posterior score, and update provenance.

Replay deterministically derives reliability-weighted Beta-count bookkeeping
from admitted evidence, starting from Beta(2, 2). Generated proposals and
infrastructure failures give no production belief credit. This is not a
calibrated Bayesian model: competing theories are recorded, not jointly inferred,
and repetitions must not be interpreted as independent samples.

Experiment proposals specify hypotheses compared, outcomes under each,
nuisance factors, a minimal discriminating intervention, and positive,
negative, and inconclusive interpretations.

[scheduler.py](src/farfield/extras/openworld/scheduler.py) retains the heuristic
scheduler and adds hypothesis-aware disagreement/resolution estimates alongside
scientific gain, transfer, frontier value, cost, and risk. Changing competing
hypotheses can change the experiment ranking. These are estimates from declared
predictions—not measured information gain or validated utility signals for RL.

## Evidence and promotion

Speculative and verified research are separate views of the scientific state.
Bold proposals are allowed in the speculative frontier; they cannot silently
become long-term verified knowledge.

- WORLD evidence requires attested bytes and actual execution against the bound
  object. A schema match alone does not establish scientific object identity.
- GENERATED/SYNTHETIC results cannot corroborate the target-world claim.
  Constructed checks may reveal scoped coherence or feasibility limitations.
- Registration binds claim, source, world, arms, measure, expected direction,
  and protocol identity before execution. Both arms use the same measurement.
- Probe outcomes distinguish execution blocked, policy refused, scientific
  negative, positive, and inconclusive. A refusal supplies no arm values.
- Exact-instance reproduction checks the same EvidenceID. Independent transfer
  or replication needs suitable explicitly supplied heldout conditions.
- Scoped promotion requires the relevant literature, falsification, mechanism,
  executable evidence, identity, reproduction, and heldout checks.

Synthesis reads authoritative protocol/probe facts and preserves unknowns,
scope gaps, and prohibited inferences. A deterministic guard rejects explicit
unsupported confirmation; it does not completely audit arbitrary scientific
prose. Likewise, matching registered identifiers and source bytes does not
fully prove a generated program implements the stated causal mechanism.

## Research-policy admission

Policy improvement is a separate mission-boundary workflow using
[routing.py](src/farfield/extras/routing.py), not another scientific controller:

```text
Completed segment → recurring failure → candidate routing-policy patch
  → historical failure replay / nearby tasks / heldout / regression / cost
  → evaluator-issued receipt → admit or reject for a later session
```

The current candidate schema is restricted to routing-policy weights and fitting
provenance. It is not a general facility for autonomous source-code, kernel,
judge, or skill replacement.

Trusted Python applications can provide `policy_suite` and `policy_evaluator`
to `run_research`. Admission requires disjoint evaluation partitions, measured
costs, strict heldout improvement, and an immutable single-use receipt binding
the candidate and incumbent. Missing trusted evaluation rejects the update.
The active policy is snapshotted for a session; admission affects later sessions.

Research-time `OpenWorld.evolve()` only records proposals. A model score,
single-task improvement, survival rate, or self-consistency is not an admission
receipt. Tests establish these local guards, not successful live RSI.

## Operational boundaries

- Sessions have a finite horizon and service budget. Python callers can resume
  an existing workspace from events; the CLI currently creates a new directory.
- Arbitrary-topic literature dispatch does not imply arbitrary-topic execution.
  Catalog and acquisition recipes remain limited; missing capabilities fail closed.
- Offline worker parity uses actual subprocesses with recorded model responses.
  It is not a diverse live mission benchmark.
- The latest bounded live validation reached real retrieval and probe construction,
  but a metric mismatch blocked execution; it did not produce verified findings.

See [VALIDATION.md](VALIDATION.md) for reproducible checks and remaining work.
