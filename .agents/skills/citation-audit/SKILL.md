---
name: citation-audit
description: Drop retrieved papers that never name the claim object, and refuse unverified rows as citations. Use before a briefing bibliography or intern-checking RESEARCH_PACKET related work.
stage: intern, writeup
admitted: true
audience: research
entry: plugin.py
---

# Citation audit (FarField)

Adapted from ARIS `/citation-audit`. The hook calls `papers_on_claim_object`. Status still comes from `verify_papers`.

Interns invoke `{"action":"skill","name":"citation-audit","args":{"claim":"...","topic":"...","works":[...]}}`.

- `verified` + on-object → may cite
- `verified` + off-object → noise, list separately
- `unverified` / `verify_pending` → not a citation
- GENERATED worlds are not papers
- On-object means a topic bigram or two distinctive unigrams. Sharing `world` or `model` alone is noise.
