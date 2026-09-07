---
name: experiment-audit
description: Audit a candidate experiment.py against fair two-arm, attested field paths, and probe kind honesty. Use before running a probe or when intern-checking a protocol.
stage: intern, probe
admitted: true
audience: research
entry: plugin.py
---

# Experiment audit (FarField)

Adapted from ARIS `/experiment-audit`. The hook is causal; the prompt is colour.

Interns invoke `{"action":"skill","name":"experiment-audit","args":{"path":"experiment.py","schema":"...","claim":"..."}}`.

Checks (same teeth as PluginHost, not a new reviewer):

1. `fair-two-arm-probe` — both arms call one measure
2. `worldfields.probe_reads_attested_fields` — the script reads the freeze's layout
3. kind honesty — GENERATED/SYNTHETIC cannot be labelled WORLD in the protocol

A refusal is an unlock condition. It cannot rewrite expected_direction.
