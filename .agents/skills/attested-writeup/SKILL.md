---
name: attested-writeup
description: Compile a venue draft from FarField attested.json using an existing paper-writing skill, not a homemade LaTeX engine. Use when compile-artifact, RESEARCH_PACKET, conference draft, or tectonic PDF is requested.
stage: writeup, intern
admitted: true
audience: research
---

# Attested write-up

`farfield compile-artifact` dumps attested fields. It is not a paper compiler. The wheels:

- [jin-s13/ai-research-writing-skill](https://github.com/jin-s13/ai-research-writing-skill) — evidence-backed LaTeX from logs/code/venue template
- [NiuYingchun/latex-paper-skills](https://github.com/NiuYingchun/latex-paper-skills) — `empirical-paper-writer` with planned/placeholder/verified results
- PDF: [Tectonic](https://tectonic-typesetting.github.io/) or `latexmk`, not a FarField pdflatex wrapper
- Polaris Paper Writer is a full lab app (CRDT + server tectonic). Use it if the lab already runs Polaris; do not vendor it. Numbers still have to come from `host_run.json` / `external_run.json` via attest-run.

## Procedure

1. `farfield compile-artifact <card>` → `artifact/attested.json` + `artifact/WRITING.md`.
2. Point the writing skill at that JSON, `RESEARCH_PACKET.md`, and `protocol.md`. Fetch the venue template from the skill's official-source manifest.
3. Results status is `empty` until a host/external run exists. Do not paste 20s probe metrics into a results table. Do not let the writing model invent a verdict.
4. A compiled PDF is a submission shape, not a discovery, and does not climb the ladder.

Sakana AI Scientist / AIDE write papers by searching experiments. That is a different product. Do not import their tree search into FarField.
