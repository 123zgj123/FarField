---
name: experiment-plan
description: Compile a claim-driven experiment roadmap from FarField attested fields (two-arm protocol, freeze, competing explanation). Use after a card enters, when writing EXPERIMENT_PLAN.md, or when execute needs a run order that defends C1 rather than a benchmark wishlist.
stage: writeup, intern, diagnose
admitted: true
audience: research
---

# Experiment plan (FarField)

Adapted from [ARIS `/experiment-plan`](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) — claim map, compact blocks, must-run vs nice-to-have, milestone gates. Not vendored. FarField does not run ARIS, Codex MCP, or a GPU queue.

The compiler is `farfield.extras.packet.render_experiment_plan`. Interns do not call another LLM to invent GPU-hours or results.

## What this skill is for

Turn a surviving card into a **claim → evidence → run order** that is executed now.

`idea_kind` is first-class. Compile the plan that matches the kind:

- `question` — observational contrast this freeze can score; no invented effect size
- `probe` — two-arm intervention this freeze can feel
- `acquire` — harvest / import / derive recipe; **do not** register a two-arm on this freeze
- `theory` — derivation locked to registered quantities; cannot climb

An acquire card occupying `ideas/` is a live idea, not a failed probe.

Four things the plan must defend:

1. the method moves the **named object** (not a cosine-neighbour field)
2. the dominant contribution is isolated (competing explanation / lever off)
3. extra complexity is unnecessary
4. a frontier primitive (LLM, RLHF, GPU decoder) is either measured on attested fields or explicitly not claimed

## Constants

- `MAX_PRIMARY_CLAIMS = 2`
- `MAX_CORE_BLOCKS = 5`
- Output: `ideas/<idea-name>/refine-logs/EXPERIMENT_PLAN.md` and `EXPERIMENT_PLAN_EN.md` (compiled, self-contained). The same blocks are inlined into `RESEARCH_BRIEF.md` / `RESEARCH_BRIEF_EN.md` so the research plan is complete without a pointer.

## Workflow (compile, do not generate)

1. Read attested fields: claim, prediction, diagnosis (treatment/control/expected_direction/alternative), bound `world_id` / schema / kind, dead_end, objection.
2. Freeze **C1** and **Anti-C1**. Empty alternative stays empty — do not invent a competing explanation in prose and then treat it as registered.
3. Emit blocks: main two-arm, novelty isolation, object-lock / no-surrogate, host replication. Delete any block that has no attested handle.
4. Milestone order: M0 sanity (field paths) → M1 two-arm → M2 anti-claim → M3 host. supports and weakens both stop. uninformative redesigns the test, not the claim.
5. Separate must-run from nice-to-have. Do not pad baselines.

## FarField teeth (do not weaken)

- Probe arms share one `measure`; PluginHost `fair-two-arm-probe` refuses hard-coded wins.
- Schema match is not scientific match. Pride / SNAP / fasta cannot stand in for tool-call or game traces. Schema inference reads the topic object with the far concept's terms stripped (`without_far_terms`): `feedback edge set` must not vote `undirected_graph` onto an executable-program claim.
- `program_state` is the schema for self-improvement / code-world-model / RSI claims (cells + mutable validators + an update log; levers include `replay_gate` and `freeze_validators`). It is in-mission constructable and not catalog-freezable yet.
- GENERATED construction (when no freeze matches) can weaken. It cannot corroborate. A GENERATED `uninformative` or `object_absent` (0/0 on both arms) is not a result and does not occupy the spray seat — adapt the constructed world or keep exploring.
- One mission, one world, bound at `workspace/world/`. Do not rebind mid-packet. A GENERATED world that never met the object may be adapted in place; an attested freeze may not.
- Query literature on topic object phrases, not the far concept.
- Do not fabricate results. Plan evidence; do not claim evidence.

## Composing

`attested-writeup` consumes this plan plus `protocol.md`. `fair-two-arm-probe` is the verifier for the script. `claim-domain-lock` still wins over a venue-shaped title.
