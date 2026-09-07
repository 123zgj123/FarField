---
name: claim-domain-lock
description: Keep a FarField hypothesis and its write-up in the claim's own field. Use when generating a far-field card, writing a briefing title, or when a topic such as agent safety must not be painted onto a hardness or indexing claim.
stage: generate, writeup, intern
admitted: true
audience: research
entry: plugin.py
---

# Claim-domain lock

The distant concept supplies a mechanism, not a new field. Claim and mechanism must still be about the researcher's topic. When the topic names a bigram (`chain of thought`, `safety properties`), that bigram must appear as adjacent words — sharing two unigrams (`reasoning` + `trace`) in different clauses is not coverage and is how a wavelet-tree card used to pass a reasoning lock.

The write-up locks to the claim, not the topic. Do not retitle a graph-hardness, spanner, or indexing result as LLM tool-use safety because that was the mission topic. If the claim is about hardness, the title stays in hardness.

Schema match is not scientific match. A `text_stream` freeze of Pride must not bind a tool-use distillation claim because both mention tokens. Fixture domain tags must overlap the claim. The probe measure must report a quantity of that same object — not the far-field algorithm's internal cost (min-cut edge ops on an Agent Card / AUTH_REQUIRED claim).

A matching field name is not enough. `created_tools` on a labeled_traces freeze lives on each trace, not on each step; `step.get("created_tools")` is empty. Diagnosis and probe must use the attested field paths in the schema `load_hint` / layout. A claim that names wall-clock duration on a freeze that does not store it is a missing-field diagnosis (`world_lever: none`), not a two-arm on invented step times. Papers offered as briefing or control baselines must still name the claim's object; a drought-index paper is not a SWE-agent control because both say "index". The freeze brand (Live-SWE-agent, …) is not the claim object and is not a literature ticket. Idea rehearsal (`farfield world-sim`) admits verified papers whose phrases belong to the registered `world_lever` and freeze schema (for example `truncate` on `labeled_traces` may admit a code world model paper). It does not compile onto a parallel operator such as `code_rollout`.

Scout the bound world **before** generating a card. The menu (levers / observables / inert pairs / attested field paths) is the mission prior. A freeze with no `room_to_move` stops spray and wishlists; it does not sample four graph nouns. Generation must compile the distant concept onto a declared lever (`world_lever`, `world_observable`, `far_maps_to_lever`). Object-wrong is `WORLD_INCOMPATIBLE` (wishlist, never Pride). Object-right but uncompiled is `no_handle`: the same pair's `idea_rounds` rewrites the mechanism against the lever table.

`plugin.py` is the verifier. PluginHost calls `lock_claim` at generation, `validate_diagnosis` / `validate_probe` before evidence, and `lock_writeup` at briefing. Interns invoke it with `{"action":"skill","name":"claim-domain-lock","args":{...}}`.
