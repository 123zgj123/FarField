<div align="center">

# FarField

**State-driven research. Evidence-gated conclusions.**

An experimental local research runtime that uses ScientificState to coordinate literature, hypotheses, registered experiments, verification, and synthesis.

[English](README.md) · [简体中文](README.zh-CN.md) · [System design](ARCHITECTURE.md) · [Validation](VALIDATION.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-003a70)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-9a7b2e)](LICENSE)
[![CI](https://github.com/123zgj123/FarField/actions/workflows/tests.yml/badge.svg)](https://github.com/123zgj123/FarField/actions/workflows/tests.yml)
[![Core dependencies](https://img.shields.io/badge/core-Python%20stdlib-5c6b7a)](pyproject.toml)

[Guijia Zhang](https://123zgj123.github.io)

</div>

> **Research software, not a validated autonomous scientist.** The runtime and its admission rules are testable; autonomous discovery, general-purpose experiment execution, and beneficial live policy self-improvement have not been demonstrated. See [validation and limitations](VALIDATION.md).

## What FarField does

Given a research intent, FarField maintains explicit questions, hypotheses, competing explanations, evidence, contradictions, and research debt. OpenWorld selects eligible actions from that state; existing research workers carry them out.

The aim is to explore transformations of scientific reasoning—not to treat distance between ideas as scientific value.

- **Inspect the reasoning state.** Scientific events, evidence identities, belief updates, and unresolved gaps remain replayable.
- **Explore several branches.** Historical transformations, assumption reframing, and unconstrained exploration share one speculative frontier.
- **Test before promoting.** Registered two-arm probes, exact world binding, reproduction, and applicable heldout checks constrain conclusions.
- **Keep failure informative.** Execution failures, policy refusals, scientific negatives, and inconclusive results remain distinct.
- **Evaluate policy changes separately.** A research-policy candidate cannot be installed without trusted heldout evaluation.

FarField is useful for researchers building and auditing automated-research workflows with explicit literature sources and executable, frozen experimental worlds. It is not a one-command route from an arbitrary topic to a verified paper.

## One scientific control loop

```text
ScientificState → Frontier → Scheduler → Action → Existing Research Worker
       ↑                                                   ↓
       └── Scientific Event ← Trusted Kernel ← CandidateResult
```

This is a state-dependent loop, not a fixed sequence of stages. CLI and console both use `run_research`. Workers do not start independent research missions.

| Action | Existing capabilities used |
|---|---|
| SURVEY | Literature retrieval, citation/source verification, prior work, source spans |
| THEORIZE | Hypothesis generation, exploration, reframing, diagnosis |
| PROBE | Diagnosis, registered probe construction, actual execution |
| VERIFY | Exact object and artifact checks, reproduction, explicit heldout worlds |
| SYNTHESIZE | Research briefs, packets, fact sheets, paper planning and drafting |

The Trusted Kernel, event sourcing, provenance, and WORLD/GENERATED separation remain the authority boundary. Models propose claims and scripts; they cannot declare their own outputs to be scientific evidence.

Read the [system design](ARCHITECTURE.md) for trajectory extraction, later-hop conditions, belief accounting, experiment selection, and policy admission.

## Quick start

Requires **Python 3.11+**. The core runtime and offline tests use the standard library. Individual experimental programs may require their own executor environment and dependencies.

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField

# Offline verification: no API key required.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src FARFIELD_LLM_MODE=replay \
  python3.11 -m unittest discover -s tests -q
```

### Run a bounded live research session

Save your provider key in a private local file outside the repository. Do not put it in source code or pass the key itself on the command line. See [.env.example](.env.example) for configuration names; the example below uses explicit environment exports.

```bash
mkdir -p "$HOME/.config/farfield"
chmod 700 "$HOME/.config/farfield"
# Save your provider key privately as ~/.config/farfield/llm.key first.
chmod 600 "$HOME/.config/farfield/llm.key"

export FARFIELD_LLM_KEY_FILE="$HOME/.config/farfield/llm.key"
export FARFIELD_LLM_BASE_URL=https://api.openai.com/v1
export FARFIELD_LLM_MODEL=gpt-5.6-sol

PYTHONPATH=src python3.11 -m farfield research /tmp/farfield-research \
  --topic "Does sepal geometry predict iris species?" \
  --world fisher-iris --horizon 20 --api-calls 30
```

This permits **live, potentially billed model requests** and network literature retrieval. Use a model available to your provider account. Missing credentials, rate limits, absent literature, or unsupported experiment bindings can block progress; no stand-in evidence is created.

The positional path is a **project root**. Each CLI invocation creates a new timestamped workspace under `/tmp/farfield-research/var/missions/`. A bound Iris fixture makes the example concrete; it does not guarantee a novel hypothesis or successful experiment. `--horizon` bounds controller steps and `--api-calls` bounds model calls, not a dollar budget.

For independent validation, add repeatable `--heldout-world <world-id>` arguments using suitable catalog fixtures. A rerun on the same bytes is reproduction, not independent replication.

### Resume an existing research state

The CLI currently creates a new mission directory. To continue an existing state, use the Python API with the **original workspace, topic, and compatible world**, under the same environment configuration:

```python
from pathlib import Path
from farfield.extras.openworld.runtime import run_research

for event in run_research(
    "Does sepal geometry predict iris species?",
    workspace=Path("/path/to/existing/var/missions/<mission-id>"),
    world="fisher-iris",
    horizon=20,
    api_calls=30,
):
    print(event)
```

Run Python with `PYTHONPATH=src`. State is reconstructed from scientific events; unresolved work receives a new bounded session. This is resumability, not an unattended daemon.

### Local console

```bash
PYTHONPATH=src python3.11 scripts/console.py
```

Open `http://localhost:8765/`. The console uses the same research entry point and binds to loopback by default. Use SSH forwarding for remote access.

## What counts as evidence?

**High-entropy proposals; low-risk promotion.**

Hypotheses and paper interpretations remain speculative. Generated plans, imagined scenarios, reviewer opinions, and model self-consistency cannot become WORLD evidence.

A promotable result needs the relevant literature and falsification checks, an executable probe on the exact bound scientific object, admitted measurements, verification, and reproduction or transfer checks required by its scope. A descriptive association does not establish a causal mechanism.

The probe records one of five outcomes:

```text
probe_execution_blocked       no usable experimental result
probe_policy_refused          execution/admission not permitted
probe_scientifically_negative measured evidence against the hypothesis
probe_positive               measured support within the probe's scope
probe_inconclusive           insufficient discrimination
```

A positive probe is not automatically a verified finding. Unsupported claim strengthening, missing world digests, changed protocol bytes, and mismatched EvidenceIDs fail closed. See [evidence and promotion](ARCHITECTURE.md#evidence-and-promotion).

## Research artifacts

Inside each mission workspace:

```text
SCIENTIFIC_STATE.json   questions, theories, evidence, beliefs, frontier
SCIENTIFIC_EVENTS.json replayable scientific event log
RESEARCH_STATUS.md     verified scope and remaining gaps
literature/            source artifacts, paper states, transitions
llm_cache/             recorded model requests and responses
worlds/<world-id>/     bound fixture bytes and manifests
research_workers/<evidence-id>/candidates/<card-id>/
  protocol.json        registration and identities
  experiment.py        executable source
  probe.json           measurements, verdict, provenance, scope
  metrics.json         executable metrics
  research_note.md     synthesis, when scheduled
  PAPER_FACTS.json     synthesis facts, when scheduled
  paper/               optional draft artifacts
```

Artifacts appear only when the corresponding action produces them. A blocked run may have no measurements. `research_complete: false` describes an open study, not a successfully completed paper. Runtime folders and credentials are excluded from the public repository.

To rerun an existing registered protocol:

```bash
PYTHONPATH=src python3.11 -m farfield execute /path/to/candidate-folder --on-slice
```

This exact-instance reproduction does not establish transfer to a new world.

## Current validation and limits

The [validation record](VALIDATION.md) maps the requested behaviors to tests and separates offline regressions from live observations.

- **Offline:** 1,202 main-suite tests, including 6 skips; 30 contract tests and 2 historical regressions pass.
- **Live integration:** bounded Iris sessions retrieved real papers, generated structured hypotheses, and reached diagnosis/probe construction. The final recorded segment blocked a metric mismatch before execution and produced a failure-reframe branch—not WORLD probe evidence or a verified discovery.
- **Trajectory coverage:** extraction and structural retrieval exist, but a broad grounded transition corpus and learned operator-selection distribution do not.
- **Experimental coverage:** catalog and acquisition support are limited. Registered identities do not fully prove that arbitrary generated code implements its prose claim.
- **Estimation:** belief scores are deterministic bookkeeping, not calibrated Bayesian confidence; scheduler priorities are heuristics, not measured information gain.
- **Self-improvement:** policy admission is enforced locally; beneficial live research-policy RSI has not been demonstrated.

Historical far-jump experiments did not establish an advantage over matched random search. Embedding distance is neither a novelty certificate nor a scientific-value signal.

## Repository and compatibility

| Path | Purpose |
|---|---|
| [src/farfield/extras/openworld/](src/farfield/extras/openworld/) | Scientific controller, state, scheduler, worker adapters |
| [src/farfield/](src/farfield/) | Trusted runtime and existing research capabilities |
| [tests/](tests/) | Offline, contract, parity, and regression tests |
| [worlds/](worlds/) | Small attested experimental fixtures |
| [scripts/](scripts/) / [webui/](webui/) | Local console and maintenance tools |
| [corpora/](corpora/) / [concepts/](concepts/) | Literature and historical graph assets |

Historical Python pipeline functions remain regression references. `--legacy-pipeline` is deprecated and routes to the same controller; it does not restore the old stage loop. Older bilingual `RESEARCH_PACKET*.md` outputs are archival formats, not promised outputs of the current command.

Some legacy options still appear in CLI help but are not forwarded to the current controller. In particular, do not rely on `--no-host-execute` or old human-gate options to disable execution. Use offline tests for a no-live-service check; live research may schedule registered execution.

See [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), and `PYTHONPATH=src python3.11 -m farfield --help`.

## Citation

If you use FarField in research, please cite [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE).
