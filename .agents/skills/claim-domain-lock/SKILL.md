---
name: claim-domain-lock
description: Keep a FarField hypothesis and its write-up in the claim's own field. Use when generating a far-field card, writing a briefing title, or when a topic such as agent safety must not be painted onto a hardness or indexing claim.
stage: generate, writeup, intern
admitted: true
audience: research
entry: plugin.py
---

# Claim-domain lock

The distant concept supplies a mechanism, not a new field. Claim and mechanism must still be about the researcher's topic.

The write-up locks to the claim, not the topic. Do not retitle a graph-hardness, spanner, or indexing result as LLM tool-use safety because that was the mission topic. If the claim is about hardness, the title stays in hardness.

Schema match is not scientific match. A `text_stream` freeze of Pride must not bind a tool-use distillation claim because both mention tokens. Fixture domain tags must overlap the claim. The probe measure must report a quantity of that same object — not the far-field algorithm's internal cost (min-cut edge ops on an Agent Card / AUTH_REQUIRED claim).

`plugin.py` is the verifier. PluginHost calls `lock_claim` at generation, `validate_diagnosis` / `validate_probe` before evidence, and `lock_writeup` at briefing. Interns invoke it with `{"action":"skill","name":"claim-domain-lock","args":{...}}`.
