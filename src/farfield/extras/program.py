"""Runtime H for this idea: what the next in-mission refine may propose.

Self-evolution in this codebase is rewriting the same idea against working
H compiled from this mission so far. It is not polishing a briefing, and
it is not a new far-field crossover. The program is also persisted so a
later mission cannot spray over this pair. Remaining scientific work on
an executable plan is `farfield execute`, not another generate.

The program is compiled from attested mission records. It cannot climb
the promotion ladder and is not evidence. The next refine of the *same
pair* in this mission reads it. A new far jump does not.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .explore import is_continue_program, is_live_program, pair_structurally_dead
from .world import is_world_kind, world_attested


def compile_program(
    topic: str,
    *,
    ranked: list[dict[str, Any]] | None = None,
    found: list[dict[str, Any]] | None = None,
    kills: list[dict[str, Any]] | None = None,
    previous: dict[str, Any] | None = None,
    world: Any = None,
) -> dict[str, Any] | None:
    """Build the working generator brief from this mission's books. No LLM.

    The program is the Memory slot of runtime H. The next refine of the
    same pair in this mission reads it. A new far jump reads the
    exploration notebook of blocked routes, not this brief. Once the
    plan is executable, `next_card_must` names execute work, not a
    later research generate.
    """
    found = list(found or [])
    ranked = list(ranked or [])
    kills = list(kills or [])
    if not found and not kills and not previous:
        return None

    by_id = {str(row.get("card_id") or ""): row for row in found}
    lead: dict[str, Any] | None = None

    def _worldish(row: dict[str, Any]) -> bool:
        return world_attested(row)

    def _closed(row: dict[str, Any]) -> bool:
        return bool(row.get("prior_kills") or row.get("verdict") == "weakens")

    def _merged(item: dict[str, Any]) -> dict[str, Any]:
        return {**by_id.get(str(item.get("card_id") or ""), {}), **item}

    def _attested_open(row: dict[str, Any]) -> bool:
        return (
            not _closed(row)
            and _worldish(row)
            and str(row.get("ledger") or "") != "diagnostic"
            and row.get("verdict") in {"supports", "uninformative"}
        )

    world_live = any(_attested_open(_merged(item)) for item in ranked) or any(
        _attested_open(row) for row in found
    )

    for item in ranked:
        rec = {**by_id.get(str(item.get("card_id") or ""), {}), **item}
        if _closed(rec):
            continue
        if (
            str(rec.get("ledger") or "") == "diagnostic"
            and not rec.get("world_incompatible")
        ):
            continue
        if world_live and not _worldish(rec) and not rec.get("world_incompatible"):
            continue
        lead = rec
        break
    if lead is None:
        for row in found:
            if row.get("world_incompatible") and not _closed(row):
                lead = row
                break
            if row.get("has_brief") or row.get("verdict") in (
                "supports",
                "uninformative",
            ):
                lead = row
                break

    kill_reasons = Counter()
    killed_pairs: list[str] = []
    for kill in kills:
        pair = kill.get("pair") or []
        # Text-gate deaths are repaired by idea_rounds on the same pair.
        # Only a graph-dead pair is a combination not to generate again.
        if len(pair) >= 2 and pair_structurally_dead(kill.get("killed_by")):
            killed_pairs.append(f"{pair[0]} × {pair[1]}")
        for reason in kill.get("killed_by") or []:
            kill_reasons[str(reason)] += 1
    dominant = ", ".join(f"{name} ({n})" for name, n in kill_reasons.most_common(3))
    weakened = [
        str(row.get("claim") or "").strip()
        for row in found
        if row.get("verdict") == "weakens" and row.get("claim")
    ]

    avoid: list[str] = []
    if dominant:
        avoid.append(f"gate failures that dominated this mission: {dominant}")
    avoid.extend(killed_pairs[:4])
    avoid.extend(f"weakened mechanism: {claim[:160]}" for claim in weakened[:2])
    if previous and previous.get("do_not_generate"):
        prev_avoid = str(previous["do_not_generate"])
        if prev_avoid not in avoid:
            avoid.append(prev_avoid)

    if lead:
        claim = str(lead.get("claim") or "").strip()
        verdict = lead.get("verdict")
        experiment = str(lead.get("experiment") or "").strip()
        failed = [
            str(item).strip()
            for item in (lead.get("failed_experiments") or [])
            if str(item or "").strip()
        ]
        if lead.get("world_incompatible") and verdict not in {"supports", "weakens"}:
            must = (
                "this pair has no attested freeze; do not invent a dataset "
                "and do not construct a substitute world for verification; "
                "keep evolving THIS idea's memory against the missing object "
                "or freeze a matching public source — other ideas do not "
                "inherit this gap"
            )
        elif verdict == "supports":
            kind = str(lead.get("probe_kind") or "").upper()
            if kind == "GENERATED":
                must = (
                    "the last support used a constructed world; freeze a "
                    "matching attested schema (farfield freeze) then execute "
                    "the same registration; this result cannot occupy the "
                    "continue seat, distill, or corroborate"
                )
            elif not is_world_kind(kind):
                must = (
                    "the last support was SYNTHETIC (invented data); freeze "
                    "an attested fixture, then farfield execute the same "
                    "registration — do not generate a dataset the idea is "
                    "guaranteed to win"
                )
            elif lead.get("host_ok") is False:
                must = (
                    "run farfield execute on the attested parent of the "
                    "same EvidenceID; the 20s slice is a filter, not a "
                    "discovery, and not a reason to generate a new card"
                )
            else:
                if (
                    lead.get("ablation_required")
                    or lead.get("mechanism_identified") is False
                ):
                    must = (
                        "execute 实验块 2 (ablate the competing explanation) "
                        "on THIS freeze via farfield execute; do not generate "
                        "a new pair or change the scientific object"
                    )
                    if experiment:
                        must += f"; registered experiment: {experiment[:160]}"
                else:
                    must = (
                        "execute the registered protocol (M0 then M1, then "
                        "farfield execute). Do not generate a new distant pairing"
                    )
                if lead.get("host_ok") is True:
                    must += (
                        "; host protocol_executed is not a discovery — keep "
                        "the same attested freeze"
                    )
        elif verdict == "uninformative":
            prior = "; ".join(failed[:3]) if failed else experiment
            kind = str(lead.get("probe_kind") or "").upper()
            switch = bool(
                lead.get("must_switch_mechanism")
                or (
                    previous
                    and previous.get("must_switch_mechanism")
                    and list(previous.get("pair") or [])[:2]
                    == list(lead.get("pair") or [])[:2]
                )
            )
            if lead.get("object_absent"):
                must = (
                    "the last probe never met the claim's object (both arms "
                    "0); construct or freeze a fixture that actually stores "
                    "the claimed quantities before rerunning — an eligibility "
                    "check over an empty world is not an experiment"
                )
            elif kind == "GENERATED":
                must = (
                    "the last uninformative ran on a constructed world; it "
                    "does not occupy the continue seat. Freeze a matching "
                    "attested schema or rework the same pair against the "
                    "constructed lever table — do not treat that rehearsal "
                    "as an executable plan"
                )
            elif switch:
                must = (
                    "execute the plan's must-run lever switch on THIS freeze; "
                    "do not generate a new card and do not rewrite the same "
                    "experiment in new words"
                )
            else:
                must = (
                    "execute the sharper must-run items in "
                    "EXPERIMENT_PLAN.md on THIS pair and THIS freeze; "
                    "do not generate a new pair"
                )
            if prior:
                must += f"; do not repeat: {prior}"
        else:
            must = (
                experiment
                or "write a claim that would move this line, not a new spray "
                "across unused distant concepts"
            )
        program = {
            "commit": claim or str((previous or {}).get("commit") or topic),
            "why": str(lead.get("mechanism") or "")[:400]
            or "this mission's evidence did not weaken the line",
            "stakes": str(lead.get("prediction") or "")[:240],
            "next_card_must": must[:600],
            "do_not_generate": "; ".join(avoid)[:600],
            "card_id": str(lead.get("card_id") or ""),
            "pair": list(lead.get("pair") or []),
            "verdict": verdict,
            "probe_kind": str(lead.get("probe_kind") or ""),
            "source": "compiled",
            "prior_kills": bool(lead.get("prior_kills") or lead.get("killed")),
        }
        nodes = [
            str(item) for item in (lead.get("pair_nodes") or []) if item
        ][:2]
        if len(nodes) == 2:
            program["pair_nodes"] = nodes
        if lead.get("object_absent"):
            # A probe that never met its object is not an executable plan;
            # the stored program must not occupy the neighbourhood seat.
            program["object_absent"] = True
        if lead.get("ablation_required"):
            program["ablation_required"] = True
        if (
            program.get("verdict") == "uninformative"
            and is_world_kind(program.get("probe_kind"))
            and (
                lead.get("must_switch_mechanism")
                or (
                    previous
                    and previous.get("must_switch_mechanism")
                    and list(previous.get("pair") or [])[:2] == list(program.get("pair") or [])[:2]
                )
            )
        ):
            program["must_switch_mechanism"] = True
    else:
        program = {
            "commit": str((previous or {}).get("commit") or "")
            or f"Stay inside {topic}; the last spray did not enter research",
            "why": (
                "generation budget was spent on combinations the gates refused;"
                " another distant menu is not progress"
            ),
            "stakes": "",
            "next_card_must": (
                f"Avoid the checks that dominated kills ({dominant or 'text discipline'})"
                " and write a measurable, falsifiable claim in the topic's own field"
            ),
            "do_not_generate": "; ".join(avoid)[:600],
            "card_id": "",
            "pair": list((previous or {}).get("pair") or []),
            "verdict": None,
            "probe_kind": "",
            "source": "compiled",
        }
    if lead:
        pair = list(program.get("pair") or [])
        if len(pair) >= 2:
            label = f"{pair[0]} × {pair[1]}"
            alt = f"{pair[0]} x {pair[1]}"
            program["do_not_generate"] = "; ".join(
                part.strip()
                for part in str(program.get("do_not_generate") or "").split(";")
                if part.strip() and label not in part and alt not in part
            )
    skills: list[dict[str, Any]] = []
    seen_skill = set()
    for row in list((previous or {}).get("skills") or []) + [
        lead.get("distilled_skill") if lead else None
    ]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or name in seen_skill:
            continue
        seen_skill.add(name)
        skills.append(row)
    if skills:
        program["skills"] = skills[:4]
    notes = [
        str(item).strip()
        for item in (
            list((lead or {}).get("design_notes") or [])
            + list((previous or {}).get("design_notes") or [])
        )
        if str(item or "").strip()
    ]
    if notes:
        program["design_notes"] = list(dict.fromkeys(notes))[:6]
    world_id = ""
    if lead:
        world_id = str(lead.get("world_id") or "")
    if not world_id:
        world_id = str(getattr(world, "id", "") or "")
    if not world_id:
        world_id = str((previous or {}).get("world_id") or "")
    if world_id:
        program["world_id"] = world_id
    if lead and isinstance(lead.get("how_found"), dict):
        program["how_found"] = dict(lead["how_found"])
    program["h"] = _compile_h(
        lead=lead,
        found=found,
        kill_reasons=kill_reasons,
        world=world,
        previous=previous,
    )
    return program


def prompt_lines(program: dict[str, Any] | None) -> tuple[str, ...]:
    """Shown to the generator. Empty keeps cached prompts byte-identical."""
    if not program or not str(program.get("commit") or "").strip():
        return ()
    lines = [
        f"Continue this line: {program['commit']}",
        f"Why this line (not a new spray): {program.get('why') or 'it is the committed program'}",
    ]
    if program.get("stakes"):
        lines.append(f"Stakes: {program['stakes']}")
    lines.append(
        "Remaining duty (execute or in-mission refine; not a later "
        "research generate): "
        + str(program.get("next_card_must") or "execute the committed plan")
    )
    if program.get("do_not_generate"):
        lines.append(f"Do not generate: {program['do_not_generate']}")
    h = program.get("h")
    if isinstance(h, dict) and h:
        parts = [
            f"{slot}={h[slot]}"
            for slot in ("memory", "skills", "tools", "verifiers", "routing")
            if str(h.get(slot) or "").strip()
        ]
        if parts:
            lines.append(
                "Runtime H (self-evolution objects, not evidence, cannot climb): "
                + "; ".join(parts)
            )
        honesty = str(h.get("honesty") or "").strip()
        if honesty:
            lines.append(honesty)
    return tuple(lines)


def continue_duty_lines(program: dict[str, Any] | None) -> tuple[str, ...]:
    """Lock this pair against spray. Empty unless a WORLD plan is seated.

    Ablation, a sharper test, or a lever switch are already must-run
    rows. Do not generate a new card to perform them.
    """
    if not is_continue_program(program):
        return ()
    pair = list((program or {}).get("pair") or [])
    if len(pair) < 2:
        return ()
    freeze = str((program or {}).get("world_id") or "").strip()
    claim = str((program or {}).get("commit") or "")
    freeze_line = (
        f"Bound freeze: {freeze}. Stay on that instance; "
        "do not unbind it to hunt a different object."
        if freeze
        else "Keep the attested freeze this pair already measured."
    )
    duty = (
        "The plan is executable. Run refine-logs/EXPERIMENT_PLAN.md "
        "(M0 then M1, then farfield execute). Do not open a new distant "
        "landing. Do not rewrite the claim to hunt a different object."
    )
    if is_live_program(program):
        duty = (
            "The plan is executable. Run the registered protocol and "
            "must-run ablation on this freeze. Do not pick a new distant "
            "concept. Do not reduce the claim to a different scientific "
            "object than the freeze."
        )
    return (
        f"Keep pair {pair[0]} × {pair[1]} exactly. Do not generate a new card.",
        f"Previous claim: {claim}",
        freeze_line,
        duty,
    )


def _compile_h(
    *,
    lead: dict[str, Any] | None,
    found: list[dict[str, Any]],
    kill_reasons: Counter,
    world: Any,
    previous: dict[str, Any] | None,
) -> dict[str, str]:
    """Attested runtime H. Not evidence. Cannot climb.

    DESIGN §5: we do not train weights. In-mission improvement is
    H = {Memory, Skills, Tools, Verifiers, Routing} compiled after each
    landing, including evidence, so the next refine/generate in this
    mission is a better idea. Skills stay trusted plugins.
    """
    world_id = str(getattr(world, "id", "") or "")
    schema = str(getattr(world, "schema", "") or "")
    if lead:
        world_id = world_id or str(lead.get("world_id") or "")
        schema = schema or str(lead.get("world_schema") or "")
    if world_id:
        tools = f"attested world {world_id}" + (f" schema {schema}" if schema else "")
    else:
        tools = "no attested world bound this mission; SYNTHETIC cannot corroborate"
    if any(
        row.get("generated_world")
        or str(row.get("probe_kind") or "").upper() == "GENERATED"
        for row in found
    ):
        tools += "; constructed GENERATED world this iteration (cannot corroborate)"
    if lead and lead.get("host_ok") is True:
        tools += "; host protocol_executed (not a discovery)"
    elif lead and lead.get("host_ok") is False:
        tools += (
            "; host execute blocked: "
            + str(lead.get("host_status") or "unspecified")[:120]
        )
    if lead and isinstance(lead.get("idea_analysis"), dict):
        analysis = lead["idea_analysis"]
        tools += (
            "; idea analysis "
            + str(analysis.get("stop_reason") or "unknown")
            + ": "
            + str(analysis.get("summary") or "")[:200]
        )
    if lead and isinstance(lead.get("idea_world"), dict):
        world_note = lead["idea_world"]
        objects = [
            str(item)
            for item in (world_note.get("objects") or [])
            if str(item or "").strip()
        ]
        if objects:
            tools += "; idea-world objects: " + ", ".join(objects[:4])
        iterate = str(world_note.get("iterate") or "").strip()
        if iterate:
            verifiers_extra_idea = iterate
        else:
            verifiers_extra_idea = ""
    else:
        verifiers_extra_idea = ""
    dominant = ", ".join(f"{name} ({n})" for name, n in kill_reasons.most_common(3))
    verifiers = dominant or "no gate kills recorded"
    if verifiers_extra_idea:
        verifiers += "; idea-world iterate: " + verifiers_extra_idea[:200]
    failed = []
    if lead:
        failed = [
            str(item).strip()
            for item in (lead.get("failed_experiments") or [])
            if str(item or "").strip()
        ]
    if failed:
        verifiers += "; discarded experiments: " + "; ".join(failed[:2])
    probe_notes = []
    for row in found:
        verdict = str(row.get("verdict") or "")
        if verdict not in {"supports", "weakens", "uninformative"}:
            continue
        kind = str(row.get("probe_kind") or "unknown")
        probe_notes.append(f"{verdict}/{kind}")
    if probe_notes:
        verifiers += "; this-mission probes: " + ", ".join(probe_notes[:4])
    if lead:
        pipe = str(lead.get("pipeline_bottleneck") or "")
        if pipe and pipe not in {"informative", "unresolved"}:
            verifiers += f"; pipeline {pipe}"
    if lead and isinstance(lead.get("world_sim_h"), dict):
        sim_mem = str((lead["world_sim_h"] or {}).get("memory") or "").strip()
        if sim_mem:
            tools += "; " + sim_mem[:240]
    if lead:
        structural = [
            str(item.get("summary") or "").strip()
            for item in (lead.get("world_sim_issues") or [])
            if isinstance(item, dict) and item.get("type") == "structural"
        ]
        if structural:
            verifiers += "; world-sim structural: " + "; ".join(structural[:3])
    if lead and lead.get("ablation_required"):
        verifiers += "; ablation_required: competing explanation not isolated"
    notes = [
        str(item).strip()
        for item in (
            list((lead or {}).get("design_notes") or [])
            + list((previous or {}).get("design_notes") or [])
        )
        if str(item or "").strip()
    ]
    if notes:
        verifiers += "; this-mission design notes: " + "; ".join(notes[:3])
    skill_names = []
    for row in list((previous or {}).get("skills") or []) + [
        (lead or {}).get("distilled_skill")
    ]:
        if isinstance(row, dict) and str(row.get("name") or "").strip():
            skill_names.append(str(row["name"]))
    skills = (
        "this pair's distilled procedures: " + ", ".join(skill_names[:4])
        if skill_names
        else "no distilled procedure for this pair yet"
    )
    ops = Counter(str(row.get("operator") or "unknown") for row in found)
    world_ops = Counter(
        str(row.get("operator") or "unknown")
        for row in found
        if is_world_kind(row.get("probe_kind"))
        and row.get("verdict") == "supports"
        and row.get("host_ok") is True
    )
    if world_ops:
        routing = "WORLD-supported operators: " + ", ".join(
            f"{name} ({n})" for name, n in world_ops.most_common(3)
        )
    elif ops:
        routing = "operators that entered: " + ", ".join(
            f"{name} ({n})" for name, n in ops.most_common(4)
        )
    else:
        routing = "no operator earned an entered card"
    memory = ""
    if lead and str(lead.get("claim") or "").strip():
        memory = str(lead.get("claim") or "").strip()[:200]
    elif previous and previous.get("commit"):
        memory = str(previous.get("commit") or "").strip()[:200]
    else:
        memory = "no committed line; rewrite the current idea against this H, do not spray"
    return {
        "memory": memory,
        "skills": skills,
        "tools": tools,
        "verifiers": verifiers,
        "routing": routing,
        "honesty": (
            "H is this idea's runtime state. It upgrades the next refine "
            "of THE SAME idea in this mission (memory, skills, tools, "
            "verifiers). It is not evidence, cannot climb, and must not "
            "be pasted into a different idea's generation. A generated "
            "world or WORLD_SIM rehearsal is diagnostic and cannot "
            "corroborate. An executable plan is executed, not regenerated."
        ),
    }


def target_line(program: dict[str, Any] | None) -> str | None:
    if not program or not str(program.get("commit") or "").strip():
        return None
    must = str(program.get("next_card_must") or "").strip()
    commit = str(program["commit"]).strip()
    if must:
        return f"committed program: {must}"
    return f"committed program: continue {commit}"


def block(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    return (
        "Committed research program (this constrains generation; it is not"
        " evidence and cannot climb the promotion ladder). A new distant"
        " combination that ignores this program is a failed generation, not"
        " exploration:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )
