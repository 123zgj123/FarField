"""The generation program: what the *next generate in this mission* may propose.

Self-evolution in this codebase is rewriting the same idea against working
H compiled from this mission so far. It is not polishing a briefing, and
it is not a new far-field crossover. The program is also persisted so a
later mission can continue the same line, but the first consumer is the
next `generate_card` / `refine_card` call in the current run.

The program is compiled from attested mission records. It cannot climb
the promotion ladder and is not evidence.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .explore import pair_structurally_dead


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

    The program is the Memory slot of runtime H. The next refine or jump
    *in this mission* reads it. Tools / verifiers / routing sit beside it.
    """
    found = list(found or [])
    ranked = list(ranked or [])
    kills = list(kills or [])
    if not found and not kills and not previous:
        return None

    by_id = {str(row.get("card_id") or ""): row for row in found}
    lead: dict[str, Any] | None = None

    def _worldish(row: dict[str, Any]) -> bool:
        return str(row.get("probe_kind") or "").upper() in {"WORLD", "REAL", "FIXTURE"}

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
                    "the last support used a constructed world; freeze "
                    "a matching attested schema or keep evolving this idea — "
                    "do not spray a new pair"
                )
            elif kind not in {"WORLD", "REAL", "FIXTURE"}:
                must = (
                    "the last support was SYNTHETIC (invented data); the next "
                    "card must measure an attested world fixture under data/, "
                    "not invent a dataset that the idea is guaranteed to win"
                )
            elif lead.get("host_ok") is False:
                must = (
                    "the WORLD filter supported the line but the parent was "
                    "not executed; freeze the attested source or run "
                    "farfield execute; do not treat the 20s slice as a discovery"
                )
            else:
                must = (
                    experiment
                    or "deepen this same far concept: change the competing explanation "
                    "or the regime, do not start a new distant pairing"
                )
                if lead.get("host_ok") is True:
                    must += (
                        "; host protocol_executed is not a discovery — the "
                        "next card must change the competing explanation or "
                        "the regime on the same attested world"
                    )
        elif verdict == "uninformative":
            prior = "; ".join(failed[:3]) if failed else experiment
            must = (
                "the last two-arm test was UNINFORMATIVE; switch the "
                "mechanism flag, the metric, the scale, or the competing "
                "explanation — do not rewrite the same experiment in new words"
            ) + (f"; do not repeat: {prior}" if prior else "")
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
            "next_card_must": must[:400],
            "do_not_generate": "; ".join(avoid)[:600],
            "card_id": str(lead.get("card_id") or ""),
            "pair": list(lead.get("pair") or []),
            "verdict": verdict,
            "probe_kind": str(lead.get("probe_kind") or ""),
            "source": "compiled",
        }
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
        "The next hypothesis must: "
        + str(program.get("next_card_must") or "advance the committed line")
    )
    if program.get("do_not_generate"):
        lines.append(f"Do not generate: {program['do_not_generate']}")
    h = program.get("h")
    if isinstance(h, dict) and h:
        parts = [
            f"{slot}={h[slot]}"
            for slot in ("memory", "tools", "verifiers", "routing")
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
    dominant = ", ".join(f"{name} ({n})" for name, n in kill_reasons.most_common(3))
    verifiers = dominant or "no gate kills recorded"
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
    ops = Counter(str(row.get("operator") or "unknown") for row in found)
    world_ops = Counter(
        str(row.get("operator") or "unknown")
        for row in found
        if str(row.get("probe_kind") or "").upper() in {"WORLD", "REAL", "FIXTURE"}
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
        "tools": tools,
        "verifiers": verifiers,
        "routing": routing,
        "honesty": (
            "H is this idea's runtime state. It upgrades the next round of "
            "THE SAME idea (memory, tools, verifiers). It is not evidence, "
            "cannot climb, and must not be pasted into a different idea's "
            "generation. A generated world is diagnostic and cannot corroborate."
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
