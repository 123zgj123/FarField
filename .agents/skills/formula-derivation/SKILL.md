---
name: formula-derivation
description: Compile an honest derivation package from a FarField card's derivation, claim, mechanism, and prediction. Use when the human packet needs a formula section, when a theory sketch must stay locked to registered quantities, or when notes are not yet coherent enough to pretend they are a theorem.
stage: writeup, intern
admitted: true
audience: research
---

# Formula derivation (FarField)

Adapted from [ARIS `/formula-derivation`](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep). FarField does not run a theorem prover or invent symbols after the fact.

Compiler: `farfield.extras.packet.render_derivation_note` → `ideas/<card_id>/idea-stage/RESEARCH_BRIEF.md` section 领域知识.

## Goal

Produce exactly one of:

1. a coherent sketch of the **registered** quantity (claim + prediction + derivation field)
2. a blocker: NOT YET COHERENT, because no derivation was registered

Do not emit a fake polished theorem story.

## Freeze the invariant object first

The object is the topic/claim object (tool-call fields, token stream, seat-scoped legal actions), not the far-field node (min-cut, sorting network, balanced tree). If the derivation talks about the far node’s internal cost, refuse and point at `claim-domain-lock`.

## Status labels

- `COHERENT AS STATED` — derivation field exists and names the same quantity as the prediction
- `NOT YET COHERENT` — empty derivation; do not fill it with new algebra

Unverified model text. Never evidence. Never a ladder input.

## Do not

- Add assumptions that are not on the card
- Translate Pride token counts into a post-training loss
- Use GPU / RLHF / reward-model notation the freeze does not store
