---
name: external-gpu-run
description: Prepare a SkyPilot task for a FarField two-arm protocol instead of writing a GPU scheduler. Use when farfield run --runner external, GPU/cluster/Slurm jobs, or attest-run after an off-box experiment.
stage: probe, intern
admitted: true
audience: research
entry: plugin.py
---

# External GPU run

FarField does not schedule GPUs. The wheel is [SkyPilot](https://github.com/skypilot-org/skypilot) plus its agent skill (`skypilot-org/skypilot/agent/skills/skypilot`). Polaris SSH lab and AIDE/AI-Scientist tree search are other products — do not copy them into the probe loop (tree search after seeing numbers is p-hacking).

## Procedure

1. `farfield run <card> --runner external` writes `runner_receipt.json` and, via this plugin, `sky_task.yaml` in SkyPilot's schema.
2. Launch with the SkyPilot skill: `sky jobs launch sky_task.yaml` (or `sky launch` while debugging). Fill `accelerators:` there — do not invent a GPU type in FarField.
3. The job must run the same `experiment.py` and emit `{"treatment": <num>, "control": <num>}`. No `verdict` key.
4. `farfield attest-run --metrics metrics.json`. Status is `externally_replicated`, not corroborated. Do not open the sandbox to the network.

`plugin.py` only emits the YAML (and optionally `sky launch` when `FARFIELD_SKY_LAUNCH=1`). It never invents arm numbers.
