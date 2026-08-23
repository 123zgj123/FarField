---
name: campaign-horizon
description: Keep a multi-mission scientific line as a FarField campaign ledger plus Cursor /loop, instead of vendoring Argus four-role Voyage or LangGraph. Use when rolling back a stage, spanning missions, or running unattended ticks.
stage: intern
admitted: true
audience: research
---

# Long-horizon campaign

Argus (lbx154/Argus) persists a Voyage as files and separates control / execution / records. FarField already has missions + events.jsonl. Do not vendor Argus's four LLM roles or LangGraph checkpointers — those are runtimes, not this evaluation function.

The wheel for waking an agent is Cursor `/loop` (cloud subscription timer or local monitored shell). The wheel for scientific identity is `farfield campaign`:

```
farfield campaign init --id line-a --intent "frozen standing intent"
farfield campaign append --id line-a --mission var/missions/<id>
farfield campaign rollback --id line-a --reason "method search is a dead end"
```

Rollback records no-go. It must not rewrite finished verdicts, EvidenceIDs, or corroborated status. Authorities (`operator|manager|planner|engineer|reviewer`) are recorded names, not four model agents.

Unattended ticks: `/loop 1h append the latest mission folder to campaign line-a if RESEARCH_PACKET.md exists`. Do not let a tick redesign after `weakens`.
