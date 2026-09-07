---
name: novelty-check
description: Check live literature against the claim object before treating a briefing as related work. Use with research-lit and prior-kill, never with a claim written after the fact.
stage: intern
admitted: true
audience: research
---

# Novelty check (FarField)

Adapted from ARIS `/novelty-check`. Reuse `research-lit` + prior-kill. Do not vendor a second bibliographic engine.

1. Query **topic object phrases**, not the far concept.
2. Admit only `verified` rows (`verify_papers`).
3. Prior-kill needs a contiguous span that carries the claim's distinctive phrases.
4. Off-object hits are noise, not related work (`papers_on_claim_object`).
5. A neighbour paper is a race, not a veto. ABANDON only when a span restates this claim's result.

A miss is an agenda target, not a solved claim. Limitation sentences stay open.
