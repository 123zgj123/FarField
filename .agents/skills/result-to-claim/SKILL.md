---
name: result-to-claim
description: Compile what a FarField probe actually earned for C1. Use after evidence, when writing refine-logs/EXPERIMENT_RESULTS.md, or when a briefing is about to treat GENERATED supports as a discovery.
stage: writeup, intern
admitted: true
audience: research
---

# Result to claim (FarField)

Adapted from ARIS `/result-to-claim`. FarField does not run a second Codex judge. The compiler is `farfield.extras.packet.render_result_to_claim` → `ideas/<card_id>/refine-logs/EXPERIMENT_RESULTS.md`.

## Gate

Read attested fields only: `probe_kind`, `verdict`, `world_id`, claim.

| kind | supports means |
|------|----------------|
| WORLD | direction matched on a freeze; still needs host EvidenceID |
| GENERATED | coherence on a constructed task world; cannot corroborate |
| SYNTHETIC | invented data; can weaken only |

weakens stops the mechanism. uninformative redesigns the test, not the claim. `object_absent` (0/0 on both arms) means the probe never met the object: it is not a C1 result, not a live line, and not a license to treat GENERATED 0/0 as `plan_executable`.

## Do not

- Invent a paper-ready claim from 20s numbers
- Climb the ladder from GENERATED / SYNTHETIC
- Rebind Pride, SNAP, or fasta because the far pair mentioned tokens or graphs
