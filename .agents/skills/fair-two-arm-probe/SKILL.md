---
name: fair-two-arm-probe
description: Write a two-arm probe where both arms call the same measure and differ only by a mechanism flag. Use when implementing or reviewing FarField experiment.py, treatment vs control, or a diagnosis that must not hard-code a win.
stage: diagnose, probe, intern
admitted: true
audience: research
entry: plugin.py
---

# Fair two-arm probe

Both arms call one `measure(config)`. The only difference is a boolean that turns the proposed mechanism on or off. The sandbox then writes `{"treatment": measure(on), "control": measure(off)}`.

`plugin.py` is the verifier. PluginHost calls `validate_probe` before a script is admitted and `validate_diagnosis` if the two arms are identical. Interns invoke it with `{"action":"skill","name":"fair-two-arm-probe","args":{"path":"experiment.py"}}`.

Refuse, do not repair:

- numeric literals for both arms (`80` vs `120`)
- one arm `pass`, the other counting
- treatment increments unconditionally, control only under a predicate

The expected direction was pre-registered before this script. Do not pick the outcome. Do not measure wall-clock; count work. The counted quantity still has to be the claim's object, not the distant mechanism's bookkeeping.
