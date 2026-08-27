<div align="center">

# FarField

**Explore far. Promote only what the same scientific object can bear.**

A local research loop for automated scientific inquiry: far-field search, registered two-arm experiments, and an evidence contract the language model cannot rewrite.

[English](README.md) · [简体中文](README.zh-CN.md) · [Citation](#citation)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-003a70)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-9a7b2e)](LICENSE)
[![CI](https://github.com/123zgj123/FarField/actions/workflows/tests.yml/badge.svg)](https://github.com/123zgj123/FarField/actions/workflows/tests.yml)
[![Dependencies](https://img.shields.io/badge/stdlib%20only-no%20pip%20deps-5c6b7a)](pyproject.toml)

[Guijia Zhang](https://123zgj123.github.io)

</div>

## News

- **2026.08** — Evidence contract: only an attested `WORLD` probe may corroborate or stop far-field spray. A catalog miss is diagnosed and wait-listed; the runtime does not invent a generated stand-in. Per-idea working memory stays on that concept pair.
- **Demo video** — A console walkthrough will be uploaded separately. The poster below is a placeholder.

<p align="center">
  <a href="demo.html">
    <img src="assets/farfield-demo-poster.png" alt="FarField console (demo recording forthcoming)" width="920">
  </a>
</p>

## Start here

| You want to… | Go to |
|---|---|
| Understand the scientific contract | [Why FarField](#why-farfield) and [Evidence contract](#evidence-contract) |
| Install and run a mission | [Install](#install) and [Quick start](#quick-start) |
| Open the local console | [Console](#console) |
| Rerun a compiled protocol | [Execute a protocol](#execute-a-protocol) |
| Cite the software | [Citation](#citation) |

## Abstract

Automated research systems can now draft plausible ideas, experiments, and paper-shaped text. The open problem is not generation. It is **when an AI-produced result should be believed**.

FarField is an inspectable local loop. Given a topic, it searches a concept graph, proposes a falsifiable hypothesis, **registers a two-arm experiment before any number is observed**, and emits a protocol another researcher can rerun. The model may write the claim, the mechanism, and the script. Graph checks, prior-art checks, world binding, and treatment–control arithmetic are ordinary Python — the model does not sit on those gates.

A confirmation is allowed only when **binding, construction, lever, and probe measure the same scientific object**. Schema match is not scientific match. Synthetic or constructed data may weaken a hypothesis. They cannot corroborate it.

The finish line is not a conference PDF. It is a research packet.

## Why FarField

Three failure modes are easy to introduce in auto-research:

1. the model's explanation is treated as evidence for its own hypothesis;
2. an experiment is rewritten after the numbers are seen;
3. a claim is “confirmed” on data created for that claim.

FarField is built around a stricter rule:

> **Explore broadly; promote conclusions conservatively.**

Search may be speculative. Evidence may not. Ranking, Elo, and reviewer scores are recorded colleague opinions. They cannot accept, reject, or promote a scientific result.

<p align="center">
  <img src="assets/overview.png" alt="FarField loop: topic, search, hypothesis, screening, registered two-arm experiment, research packet" width="920">
</p>

## Research loop

| Stage | What happens |
|---|---|
| **Search** | Far-field operators move on a concept co-occurrence graph. Distance is a search heuristic, not a novelty proof. |
| **Hypothesis** | The model writes a claim, mechanism, prediction, and a discarded approach. These fields are proposals, not evidence. |
| **Screening** | Deterministic graph and text checks refuse already-known pairs and structurally weak claims. |
| **Prior art** | Retrieved literature can close a claim (`closed_by_prior`) before compute is spent rediscovering it. |
| **Registration** | Competing explanation, arms, metric, expected direction, and compute tier are fixed before execution. |
| **Experiment** | Both arms call the same `measure`; they differ only by a mechanism flag. |
| **Evidence** | `supports` / `weakens` / `uninformative` are scientific. Timeouts and missing files are not. |
| **Packet** | Surviving ideas compile to `RESEARCH_PACKET.md` and `protocol.json`. |

Far-field spray stops when an attested-world `supports` is already on the books, or when the jump cap is reached. A briefing or a synthetic support does not close exploration.

## Evidence contract

```text
synthetic / generated result
        |
        +-- weakens    →  scientific negative evidence
        |
        +-- supports   →  coherence check, not confirmation
```

| Kind | Meaning |
|---|---|
| `WORLD` | Attested fixture copied into `data/` and actually read by the script. May climb after a matching host EvidenceID. |
| `GENERATED` | Constructed for this registration. May weaken; cannot corroborate, distill, or stop spray. |
| `SYNTHETIC` | Invented coherence check. Same asymmetry. |

`--world auto` matches a claim against `worlds/`. A miss is `WORLD_INCOMPATIBLE`: diagnose, wishlist, skip the probe — never a silent substitute. Forward simulation steps the registered lever until absorbing, plateau, or horizon. Its analysis is idea analysis. It cannot climb the ladder.

Working memory (H) upgrades **the same concept pair**. Distilled skills stay under `candidates/<card_id>/skills/`. Trusted `plugin.py` in the repo is the only shared capability.

Confirmation is fail-closed. A missing digest, a missing EvidenceID, or an incomplete protocol is a refusal. `farfield execute` records `protocol_executed`; that is not a discovery.

## Install

Requires Python 3.11+. The core runtime and tests use the standard library only.

```bash
git clone https://github.com/123zgj123/FarField.git
cd FarField
```

There is nothing to install with `pip`. Add `src/` to `PYTHONPATH`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -q
```

Tests do not need an API key. A clone includes `concepts/attn-concepts-s1/` (~8 MB), enough to run the loop. Larger graphs exceed GitHub's file limit; rebuild from recorded pages with `python3 scripts/build_concepts.py --replay` when you have them.

## Quick start

```bash
mkdir -p ~/.config/farfield
chmod 700 ~/.config/farfield
# write the provider key, then:
chmod 600 ~/.config/farfield/llm.key

export FARFIELD_LLM_KEY_FILE=$HOME/.config/farfield/llm.key
export FARFIELD_LLM_BASE_URL=https://api.deepseek.com/v1
export FARFIELD_LLM_MODEL=deepseek-v4-pro

PYTHONPATH=src python3 -m farfield research /tmp/farfield-research \
  --topic "compress genomic sequence collections with succinct data structures"
```

To continue a scientific line, reuse the same `--state-store` and `--world`. A new mission folder is an audit trail, not a new identity. Do not commit API keys or pass them on the command line. See `.env.example`.

### Console

```bash
PYTHONPATH=src python3 scripts/console.py
```

Then open `http://localhost:8765/`. The console binds to `127.0.0.1` by default. Prefer SSH forwarding for remote access.

### Execute a protocol

```bash
PYTHONPATH=src python3 -m farfield execute /path/to/candidate-folder --on-slice
```

`--on-slice` reruns the probe bytes (E1). Default `execute` rebuilds the attested parent (E2) when the freeze cache exists. Other commands: `freeze`, `run`, `attest-run`, `campaign`. See `python3 -m farfield --help`.

## Outputs

```text
RESEARCH_PACKET.md   human research plan (not a paper)
protocol.json        registered experiment
experiment.py        two-arm script
metrics.json         treatment / control
evidence_snapshot/   frozen retrieval used this run
```

Mission state lives under `var/` and is gitignored.

## Repository

```text
src/farfield/       core runtime
tests/              stdlib unittest
worlds/             attested fixtures
corpora/            citation-graph assets
concepts/           concept co-occurrence graphs
scripts/            console and rebuild
examples/           small ledger examples
webui/              local console
assets/             figures
.agents/skills/     trusted research hooks
```

The public tree is the runtime, tests, small replayable assets, and fixtures. Keys, mission folders, and internal design notes are not included.

## Status

FarField is an experimental research system. The repository makes the loop and its failure modes inspectable; it is not a finished autonomous scientist.

- Coverage depends on frozen worlds in `worlds/`.
- Many questions still need GPU, cluster, or physical experiments (`attest-run` records those numbers; they cannot corroborate).
- Far-field distance is a search heuristic, not scientific novelty.
- Routing and verifier evolution need enough cross-mission evidence to matter.

## Citation

If you use FarField in research, please cite [`CITATION.cff`](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
