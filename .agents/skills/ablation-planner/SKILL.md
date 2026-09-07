---
name: ablation-planner
description: Compile must-run isolations from the registered competing explanation and world lever. Use when writing the 消融实验 block of EXPERIMENT_PLAN.md or when a diagnosis left alternative empty.
stage: diagnose, writeup, intern
admitted: true
audience: research
---

# Ablation planner (FarField)

Adapted from ARIS `/ablation-planner`. Not a GPU study list. Compiler: `render_ablation_plan` → `ideas/<card_id>/refine-logs/EXPERIMENT_PLAN.md` (实验块 2).

Must-run is the registered control (mechanism off) plus the card's `alternative`. Empty alternative stays empty — do not mint a competing explanation in prose and treat it as pre-registered.

Nice-to-have is extra seeds on the *same* freeze after host_ok. Pad baselines are not ablations.

`fair-two-arm-probe` still refuses asymmetric arms. `claim-domain-lock` still wins over a venue-shaped isolation story.
