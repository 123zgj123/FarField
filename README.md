<div align="center">

# FarField

**Explore far. Promote only what the same scientific object can bear.**

A local research loop for automated scientific inquiry: far-field search, registered two-arm experiments, and an evidence contract that the language model cannot rewrite.

[English](README.md) · [简体中文](README.zh-CN.md) · [Citation](#citation)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-003a70)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-9a7b2e)](LICENSE)
[![CI](https://github.com/123zgj123/FarField/actions/workflows/tests.yml/badge.svg)](https://github.com/123zgj123/FarField/actions/workflows/tests.yml)
[![Dependencies](https://img.shields.io/badge/stdlib%20only-no%20pip%20deps-5c6b7a)](pyproject.toml)

[Guijia Zhang](https://123zgj123.github.io)

</div>

## Demo

Recorded English console run on DeepSeek-V4-Pro. Topic: *combining diffusion models with agent-tool authorization security*. The mission keeps `diffusion model × quasipolynomial time algorithm`, builds a GENERATED `labeled_traces` world, reports treatment 4 / control 10 as `supports`, and corroborates nothing.

<p align="center">
  <a href="https://123zgj123.github.io/FarField/demo.html">
    <img src="assets/farfield-demo-poster.png" alt="Play FarField console demo (78s)" width="920">
  </a>
</p>

## Abstract

Automated research systems can now draft plausible ideas, experiments, and paper-shaped text. The open problem is not generation. It is **when an AI-produced result should be believed**.

FarField is an inspectable local loop. Given a topic, it searches a concept graph, proposes a falsifiable hypothesis, **registers a two-arm experiment before any number is observed**, and emits a protocol another researcher can rerun. The model may write the claim, the mechanism, and the script. Graph checks, prior-art checks, world binding, and treatment–control arithmetic are ordinary Python — the model does not sit on those gates.

A confirmation is allowed only when **binding, construction, lever, and probe measure the same scientific object**. Schema match is not scientific match. A novel can dress up as any sequence-shaped claim; a generated automaton cannot stand in for a missing I/O trace. Synthetic or constructed data may weaken a hypothesis. They cannot corroborate it.

The finish line is not a conference PDF. It is a research packet.

## Contributions

1. **Far-field search as location, not belief.** Operators move through a concept graph; distance is a search heuristic, never a novelty proof.
2. **Registration precedes observation.** Treatment, control, metric, competing explanation, and expected direction are fixed before execution.
3. **The same-object contract.** A `WORLD` support requires an attested fixture of the claim's object, a declared lever of that world, a probe that measures that object, and a placebo copy that loses the verdict. Incomplete acquisitions and generated stand-ins cannot climb.
4. **Asymmetric evidence.** Negative evidence is cheaper than positive evidence. Ranking, eloquence, and reviewer scores cannot promote a result.

<p align="center">
  <img src="assets/overview.png" alt="FarField loop: topic, search, hypothesis, screening, registered two-arm experiment, research packet" width="920">
</p>

## Why FarField

Many auto-research systems can generate plausible ideas, experiments, or paper-shaped drafts. The harder problem is deciding when an AI-generated result should actually be trusted.

Three failure modes are especially easy to introduce:

- the model's explanation is treated as evidence for its own hypothesis;
- an experiment is changed after the model has already seen the numbers;
- a claim is "confirmed" on data created specifically for that claim.

FarField is built around a stricter rule:

> **Explore broadly; promote conclusions conservatively.**

The search process is allowed to be speculative. The evidence process is not.

FarField therefore separates **hypothesis generation** from **scientific judgment**:

- Far-field search proposes combinations outside the topic's immediate concept neighborhood.
- The model writes a claim, mechanism, prediction, and experiment implementation.
- Executable checks decide whether the hypothesis is admissible.
- The treatment/control design and expected direction are registered before execution.
- Synthetic or generated data may weaken a hypothesis, but cannot confirm it.
- Ranking among surviving ideas is advisory only; it cannot accept, reject, or promote a scientific result.

The finish line is not a conference PDF. It is an inspectable research packet and a protocol that a colleague can run independently.

## Research loop

| Stage | What FarField does |
| --- | --- |
| **Search** | Retrieves related work and samples less-local concept pairs from a concept co-occurrence graph. |
| **Hypothesis** | The model proposes a falsifiable claim, mechanism, prediction, and likely failure condition. |
| **Screening** | Deterministic graph and text checks reject already-known combinations and structurally weak hypotheses. |
| **Prior-art check** | Retrieved literature is checked before spending compute on a claim that may already be known. |
| **Registration** | The competing explanation, treatment arm, control arm, metric, and expected direction are fixed before any result is observed. |
| **Experiment** | Both arms use the same measurement and differ only in the mechanism under test. |
| **Evidence** | Scientific outcomes are stored separately from infrastructure failures such as timeout, missing data, or an incompatible dataset. |
| **Protocol** | Surviving hypotheses are compiled into `RESEARCH_PACKET.md` and `protocol.json` for independent reruns. |

A short run on constructed data is a coherence test, not confirmation.

Confirmation requires an attested frozen dataset and a rerun of the same registered experiment.

## Far-field exploration

FarField does not ask the model to simply "brainstorm novel ideas."

Instead, it separates:

```text
where to search
    ↓
what hypothesis to propose
    ↓
whether the hypothesis survives
```

The search stage samples locations away from the topic's immediate concept neighborhood, then retrieves real concepts near those locations.

Current far-field operators include:

- **directional** — move away from the seed along a controlled direction;
- **interpolate** — move toward a centroid of distant concepts;
- **low-density** — search sparse regions of the concept space;
- **analogy** — apply analogy-style offsets between distant concepts.

The search stage proposes **where to look**, not what to believe.

Each jump records its operator, landing, retrieved concepts, and distance from the seed neighborhood so that the exploration path can be inspected and replayed.

Far-field distance is not treated as proof of novelty. Candidate hypotheses still have to survive graph checks, literature checks, and experiments.

## Hypotheses are proposals, not evidence

The model may write:

- a claim;
- a mechanism;
- a prediction;
- a likely falsifier;
- a discarded approach;
- why that approach failed;
- a reframe;
- a reviewer objection.

These fields help structure the research process, but they are not scientific evidence.

FarField follows a simple rule:

> **Model-generated text can propose a claim, but cannot establish that the claim is true.**

A hypothesis only gains scientific status through external checks and experimental evidence.

## Evidence model

FarField distinguishes where experimental evidence came from.

### Frozen world

A frozen world is an experimental dataset selected independently of the current result, with its provenance and content bound to the run.

A successful experiment on a compatible frozen world may support a hypothesis.

### Generated world

A generated world is constructed for the current registered experiment.

It can expose a broken mechanism or a bad hypothesis, but it cannot independently confirm the claim that caused it to be constructed.

### Synthetic world

A synthetic world is a small constructed dataset used for coherence checks, debugging, or bounded experimentation.

It follows the same asymmetric rule.

```text
synthetic / generated result
        |
        +-- weakens       -> scientific negative evidence
        |
        +-- supports      -> not confirmation
```

This asymmetry is intentional:

> **A system should be allowed to disprove its own idea more easily than it can prove it.**

### Forward-simulatable worlds

A frozen world is real, but bytes alone are not analyzable: a token stream from a novel can dress up as the world of any sequence-shaped claim. Each schema family therefore carries a **runtime-owned dynamics bridge** — wrapping the frozen bytes, never rewriting them:

- **Levers** are the only ways a mechanism can act on this world (edge dropout, token dropout, base mutation, row noise, transition dropout, ...).
- **Observables** are measurements the runtime computes — verifiers the model did not write.
- **Forward simulation** rolls the frozen payload under a lever along an intensity ramp; the trajectory is deterministic given the world digest, lever, and seed, so any auditor can recompute it.

Before a diagnosis is written, the runtime scouts the bound world (one simulation per lever) and hands the measured dose-response reading to the designer. The diagnosis must then bind its mechanism to a declared lever and its predicted quantity to a declared observable — or say `none`, which **unbinds the world**: the probe still runs as a coherence check, but it cannot corroborate. A lever the bridge does not declare is refused outright; the model may choose a handle, not invent one. Named `operations` must be a subset of the schema's capabilities — an empty `pass` no longer lets a missing verb through. After registration the runtime simulates the chosen lever once more; an inert forecast forces a redesign instead of burning a probe. Manifest `domains` are instance identity (Pride keeps only prose / english / narrative); sketch/hash/cache are schema hints that rank, they do not gate, and a foreign object (agent-tool traces vs a novel) is refused. `undirected_named_graph` (Florentine families) shares the undirected-graph bridge. A WORLD `supports` still has to clear a placebo consumption gate: the same script on a structure-destroyed copy must lose its verdict (displacement vs the registered margin, not a |z| test), or it cannot corroborate. `REGISTER_PROTOCOL` is chained before the first `run_probe`. A literature lineage that names a different scientific object unbinds a freeze that lexical matching already attached.

Everything the simulator produces is **scout evidence**: it is chained (`SIMULATE_WORLD`) so an audit can see what the designer saw, it can expose an inert lever or kill a surrogate binding, and it feeds redesigns after an uninformative probe — but no promotion gate ever counts it. `supports` still comes only from the registered two-arm probe and its host and executor-owned reproductions.

Binding, construction, levers, and the probe measure are one predicate: the claim's object. On a catalog miss the mission constructs a GENERATED world of that same family so the registered experiment can simulate; `io` stays incomplete. A constructed world can weaken. It cannot corroborate. The next mission runs freeze recipes already named on the wishlist; new claims may bind the new freeze, finished cards are not reopened. A competing explanation is another declared lever of this world, not an English word list.

### Evidence identity and replication tiers

"The experiment was rerun" is not one thing. FarField assigns every execution an **EvidenceID** — a hash over the claim, the experiment script, the world, and the exact bytes read — and classifies reruns on a replication ladder:

| Tier | Meaning |
| --- | --- |
| `E0` synthetic | invented or generated data; can weaken, cannot confirm |
| `E1` same-instance | the host reran the *same bytes* under the same protocol |
| `E2` scale replication | the same source at a different slice or full size |
| `E3` independent | a different attested dataset |
| `E4` external | another environment submitted the two arm numbers |

Only an `E1` reproduction with a **matching EvidenceID** can confirm a probe result. A run on the full parent dataset is a scale replication: it gets its own EvidenceID inside the same replication group, and it generalizes the claim rather than reproducing it.

Confirmation is **fail-closed**. A missing digest, a missing EvidenceID, or an incomplete protocol is a refusal, not a warning — "not provably the same evidence object" and "a different evidence object" are the same answer.

Claims carry the same discipline: a research line survives rewording, but any substantive edit to the claim is a new **hypothesis revision**, and a new revision inherits the line's context — never its truth status. Evidence for "mechanism A" does not ride along when the claim quietly becomes "mechanism B".

Every epistemic act — freezing a world, registering a hypothesis, registering a protocol, executing it, each confirmation-tier rerun, promoting a claim — is appended to a digest-chained event log. The chain proves the *order* of acts: the data was frozen before the hypothesis saw it, the hypothesis was registered before its evidence, and the protocol was registered before the run that confirmed it. Editing, dropping, or reordering an event breaks every later digest.

The chain is a **gate, not a diary**. Before a claim may be written as `corroborated` or `verified`, the promotion step audits the ledger: the digests must verify, the hypothesis revision must have been registered before its evidence, and the protocol registration must precede the execution that claims to confirm it. No chain, a broken chain, or the wrong order refuses the promotion — the evidence may be real, but unprovable order is not provenance. The research state's own promotion log is audited the same way before every write.

Each ladder rung is strictly harder than the one below it. `corroborated` requires the E1 host reproduction above. `verified` additionally requires **executor-owned replication**: the trusted runtime — never the model — reruns the *registered* script under seeds it derives from the evidence identity itself (`seed_i = H(experiment_digest ‖ world_digest ‖ i)`, unknowable at writing time because the digest covers the script's own bytes), records every execution and the aggregation in the event chain, and computes the statistics. The verdict is granted only when the **entire 95% confidence interval** of the per-seed separations clears the pre-registered margin — a mean that clears while the interval straddles is uninformative by name, and a script that produces identical numbers under every seed is *invariant*, which is indistinguishable from ignoring the seed and therefore cannot confirm. At promotion the ledger is reverse-checked: the seed executions and the aggregate that names them must actually be in the chain. The governing principle: **the model may generate computation; it may never generate the evidence that audits it** — execution identity, raw measurements, replicates, digests, ordering, and promotion proofs are produced by the runtime alone.

## Two-arm experiments

Experiments are registered before execution.

Conceptually, every probe follows the same structure:

```python
control = measure(
    world,
    mechanism_enabled=False,
)

treatment = measure(
    world,
    mechanism_enabled=True,
)
```

Both arms operate on the same data and use the same metric.

The experimental variable is the proposed mechanism itself.

Before the experiment runs, FarField fixes:

- the competing explanation;
- the treatment arm;
- the control arm;
- the measurement;
- the expected direction;
- the compute tier.

The experiment implementation is generated only after those choices are registered.

This prevents the model from redefining success after observing the result.

## Experiment execution

Generated experiment code is validated before execution.

The runtime checks that:

- both treatment and control are present;
- both arms use the same measurement;
- the experiment respects the registered design;
- imports stay within the allowed module set;
- the script does not use forbidden runtime capabilities;
- execution stays within its compute budget;
- outputs contain finite treatment and control measurements.

A runtime failure is not automatically a scientific failure.

For example:

```text
timeout
missing file
invalid script
incompatible dataset
runner failure
```

are recorded separately from:

```text
supports
weakens
uninformative
```

FarField does not convert infrastructure errors into evidence against a hypothesis.

## Prior-art checks

A hypothesis may be interesting but already known.

Before investing further experimental effort, FarField retrieves literature around the candidate concepts and checks whether the claim or its central mechanism is already present in prior work.

If the idea is already established, the correct action is to read the paper rather than rediscover it.

Prior-art closure and experimental falsification are therefore separate outcomes.

## Research state

A run can reuse structured research state from earlier missions.

The state distinguishes information such as:

```text
known
open
contradicted
rejected
promising
```

Speculative hypotheses are not automatically turned into long-term knowledge.

Evidence-supported findings and persistent open questions can influence later search, while failed or rejected directions can be retained as negative memory to avoid repeating the same dead end.

This allows FarField to accumulate research context without treating every previous model output as truth.

## Evidence ladder

Research conclusions are promoted conservatively.

A typical line may move through states such as:

```text
speculative
    ↓
corroborated
    ↓
verified
```

Negative evidence can instead move it toward:

```text
weakened
closed_by_prior
```

Promotion depends on experimental evidence rather than model confidence or writing quality.

A synthetic result alone cannot move a claim into a confirmed state.

## Ranking is advisory

FarField may compare surviving research ideas to help decide what a researcher should inspect first.

This ranking can consider factors such as:

- experimental evidence;
- literature gaps;
- feasibility;
- potential value;
- research interest.

But idea ranking is deliberately separated from scientific judgment.

A high-ranked hypothesis is not automatically true.

A model reviewer, tournament score, or Elo score cannot override:

- graph checks;
- prior-art closure;
- registered experiments;
- experimental arithmetic;
- evidence promotion rules.

## Research policy evolution

FarField can learn which exploration operators have historically produced stronger research trajectories.

The system records trajectories such as:

```text
operator
    ↓
hypothesis
    ↓
screening
    ↓
experiment
    ↓
evidence
    ↓
promotion
```

A candidate routing policy may then change which search operators are preferred in later missions.

However, a new routing policy is not installed simply because it performs well on the data used to create it.

It must also outperform the current policy on held-out missions.

Only evidence-backed outcomes are allowed to steer future search. Synthetic support and subjective idea ranking are not used as routing rewards.

## Verifier evolution

FarField can also propose new text predicates when repeated failure patterns appear.

A new predicate does not immediately gain permission to reject hypotheses.

It first enters a **shadow** state:

```text
candidate verifier
       ↓
shadow judge
       ↓
observed across later missions
       ↓
held-out validation
       ↓
lethal judge
```

A shadow judge records what it would have rejected, but has no authority to kill a hypothesis.

Promotion requires evidence that the judge consistently predicts real failures without rejecting supported hypotheses.

This keeps verifier improvement separate from verifier authority.

## Skills

Successful research procedures can be distilled into reusable skills.

A skill can describe:

- when a procedure is useful;
- how to construct an experiment;
- how treatment and control should differ;
- what measurement to use;
- what failure modes to watch for.

Skills may guide later model calls, but a skill is not scientific evidence.

Distilled guidance cannot:

- bypass experiment registration;
- change the expected direction after execution;
- override a verifier;
- promote a hypothesis;
- rewrite experimental arithmetic.

Executable authority remains in the runtime.

## Local and inspectable

FarField is intentionally local and inspectable.

The core runtime uses:

- Python 3.11+
- standard-library Python for the runtime and tests
- local event and research artifacts
- frozen experimental fixtures
- replayable experiment records
- an optional OpenAI-compatible model backend

The model is used where generation or judgment is useful.

Deterministic code is used where a result should be reproducible.

## Requirements

- Python 3.11 or newer
- No third-party packages for the core runtime, tests, or console
- For live model calls: an OpenAI-compatible API key stored through `$FARFIELD_LLM_KEY_FILE`

Do not commit API keys to git or pass them through command-line arguments.

## Installation

Clone the repository:

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField
```

There is nothing to install with `pip`.

Add `src/` to `PYTHONPATH`.

Run the test suite without an API key:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
python3 -m unittest discover -s tests -q
```

A clone includes a small concept graph:

```text
concepts/attn-concepts-s1/
```

It is roughly 8 MB and is sufficient to run the pipeline.

The larger production concept graph exceeds GitHub's file-size limit. If you have the recorded source pages, it can be reconstructed with:

```bash
python3 scripts/build_concepts.py --replay
```

Without it, FarField automatically falls back to the bundled graph.

## Quick start

Create a local directory for the model key:

```bash
mkdir -p ~/.config/farfield
chmod 700 ~/.config/farfield
```

Write your provider key to:

```text
~/.config/farfield/llm.key
```

and restrict its permissions:

```bash
chmod 600 ~/.config/farfield/llm.key
```

Configure an OpenAI-compatible endpoint:

```bash
export FARFIELD_LLM_KEY_FILE=$HOME/.config/farfield/llm.key
export FARFIELD_LLM_BASE_URL=https://api.deepseek.com/v1
export FARFIELD_LLM_MODEL=deepseek-v4-pro
```

Run a research mission:

```bash
PYTHONPATH=src python3 -m farfield research /tmp/farfield-research \
  --topic "compress genomic sequence collections with succinct data structures"
```

Open the local console:

```bash
PYTHONPATH=src python3 scripts/console.py
```

Then visit:

```text
http://localhost:8765/
```

The console binds to `127.0.0.1` by default. The recorded walkthrough is in [Demo](#demo).

For remote access, prefer SSH port forwarding.

Set `FARFIELD_HOST` only if you intentionally want to expose the service on a LAN.

See `.env.example` for available environment variables.

For the full command list:

```bash
PYTHONPATH=src python3 -m farfield --help
```

Additional commands include:

```text
freeze
execute
run
attest-run
campaign
```

## Freezing experimental worlds

FarField can bind experiments to frozen external data.

For example:

```bash
PYTHONPATH=src python3 -m farfield freeze \
  --id example-world \
  --file /path/to/data.txt \
  --schema text_stream \
  --slice all \
  --title "Example frozen dataset" \
  --source "Original dataset provenance"
```

The frozen artifact records the data, schema, slice rule, and provenance used by later experiments.

The experiment cannot silently replace the dataset after seeing a result.

## Executing a compiled protocol

A compiled research packet contains a registered `protocol.json`.

Run it locally with:

```bash
PYTHONPATH=src python3 -m farfield execute \
  /path/to/candidate-folder
```

FarField executes the registered experiment and records its treatment/control measurements.

By default `execute` rebuilds the attested parent dataset, so the run is recorded as an `E2` scale replication under its own EvidenceID. Pass `--on-slice` to rerun the exact bytes the probe saw — the `E1` same-instance reproduction that can confirm the probe's EvidenceID.

For heavier registered experiments:

```bash
PYTHONPATH=src python3 -m farfield run \
  /path/to/candidate-folder \
  --runner host-heavy
```

External runners can also be used when the registered experiment cannot run on the local host.

## External replication

When an experiment is executed outside the local runtime, FarField can record the resulting treatment and control values without letting the external runner directly declare the scientific verdict.

Prepare:

```json
{
  "treatment": 123.0,
  "control": 157.0
}
```

and record it with:

```bash
PYTHONPATH=src python3 -m farfield attest-run \
  /path/to/candidate-folder \
  --metrics metrics.json \
  --environment "A100 GPU, CUDA 13, PyTorch 3.x" \
  --runner "external replication"
```

The registered experiment definition remains authoritative.

The runner supplies measurements, not the interpretation of those measurements.

## Outputs

A research run produces inspectable local artifacts rather than only a chat transcript.

Typical outputs include:

```text
RESEARCH_PACKET.md
protocol.json
experiment.py
metrics.json
evidence_snapshot/
```

### `RESEARCH_PACKET.md`

Human-readable summary of:

- the research question;
- surviving hypotheses;
- literature evidence;
- experiment design;
- experimental status;
- open falsifiers;
- next steps.

### `protocol.json`

Machine-readable registered experiment protocol.

It records the experiment definition needed for an independent rerun.

### `experiment.py`

The executable treatment/control experiment.

### `metrics.json`

The treatment and control measurements generated by the experiment.

### `evidence_snapshot/`

Frozen literature or experimental evidence used during the run.

Run-specific state is stored under `var/` and is ignored by git.

## Repository layout

```text
src/farfield/       core runtime and research loop
tests/              stdlib unittest suite
worlds/             frozen experimental datasets
corpora/            citation-graph assets
concepts/           concept co-occurrence graphs
scripts/            console and rebuild utilities
examples/           small runtime / ledger examples
webui/              local research console
assets/             README figures
.agents/skills/     optional research and experiment hooks
```

The public repository contains the runnable runtime, tests, small replayable assets, and example fixtures.

Large local corpora, generated mission artifacts, API keys, and private development notes are not included.

## Design principles

FarField follows a small set of rules.

### 1. Model proposals are not results

LLMs can propose hypotheses and code.

They cannot declare that their own hypothesis is scientifically established.

### 2. Register before observing

The experimental comparison and expected direction are fixed before treatment/control values are available.

### 3. Use symmetric controls

Treatment and control should differ only in the mechanism being tested.

### 4. Negative evidence is easier than positive evidence

Synthetic or generated data can expose a broken claim.

They cannot independently confirm it.

### 5. Runtime failure is not scientific evidence

A timeout or implementation bug should not be interpreted as falsification.

### 6. Ranking is not verification

Idea quality and scientific truth are different questions.

### 7. Exploration and write-back use different thresholds

A speculative direction may be worth testing without being worth storing as long-term knowledge, policy, or verifier logic.

## What FarField is not

FarField is not intended to:

- automatically produce a submission-ready conference paper;
- treat LLM confidence or eloquence as experimental evidence;
- accept a hypothesis because an LLM reviewer likes it;
- claim confirmation from a dataset invented for the current hypothesis;
- silently rewrite an experiment after observing its outcome;
- turn every successful trajectory into permanent system behavior;
- replace independent human or external replication.

Its goal is narrower:

> **Turn an open research direction into a falsifiable, evidence-bound experiment that another researcher can inspect and rerun.**

## Status

FarField is an experimental research system.

The repository is intended to make its research loop, evidence rules, and failure modes inspectable rather than to present the system as a finished autonomous scientist.

Current limitations include:

- experiment coverage depends on available frozen worlds;
- many research questions still require external GPU, simulation, or physical experiments;
- far-field distance is a search heuristic, not a guarantee of scientific novelty;
- routing and verifier evolution require enough cross-mission evidence to become meaningful;
- local actor labels are logical roles rather than a complete security boundary.

These limitations are recorded explicitly rather than hidden behind model-generated confidence.

## Citation

If you use FarField in research, please cite the repository using [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
