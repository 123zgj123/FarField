"""Compile an executable research packet from fields the mission already attested.

This is the finish line, not another generation. The mission-level file is the
complete research plan (Chinese and English). Each live idea also keeps an
isolated folder so execute does not mix two claims in one prompt.

Citations may only name papers retrieved this mission. Object-mismatched rows
are listed as noise, not related work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

from .brief import ResearchBrief
from .generate import GeneratedCard
from .ideakind import ACQUIRE, PROBE, QUESTION, THEORY, idea_kind_of
from .hostexp import PROTOCOL_EXECUTED, HOST_TIMEOUT_SECONDS
from .livefeed import FreshWork
from .packet_copy import empty as _empty
from .packet_copy import lang_of
from .packet_copy import t as _t
from .probeexp import resolve_tier


SYNTHETIC_DISCLAIMER = (
    "The cheap probe ran on a constructed dataset. Its numbers are evidence "
    "about the mechanism's coherence under the pre-registered direction, not "
    "an empirical result on a real benchmark, wet lab, or public corpus."
)

GENERATED_DISCLAIMER = (
    "The cheap probe ran on a world constructed this iteration for the "
    "registered experiment. It can weaken. It cannot corroborate. A matching "
    "farfield freeze is required before WORLD."
)

NOT_A_PAPER = (
    "This packet is not a conference paper, workshop submission, or claim "
    "that the science is done. It is a research plan compiled from attested "
    "fields: idea, theory sketch, pre-registered experiment, and a host "
    "protocol. Run `farfield execute` on this folder. protocol_executed is "
    "not a discovery."
)

PROTOCOL_NOT_DISCOVERY = (
    "Even a host run on the attested parent only shows the pre-registered "
    "direction on that public trace. It does not establish a scientific result."
)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        payload = obj.to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def _cite(work: Any) -> str:
    if hasattr(work, "cite_id"):
        return str(work.cite_id() or "")
    data = work if isinstance(work, dict) else {}
    # FreshWork.to_dict stores the citable id under work_id; a doi:… or
    # openalex:… row rendered from a dict must not print "(no id)".
    return str(
        data.get("arxiv_id")
        or data.get("cite_id")
        or data.get("work_id")
        or data.get("id")
        or ""
    )


def _paper_row(work: Any) -> dict[str, str]:
    if isinstance(work, FreshWork):
        return {
            "cite_id": work.cite_id(),
            "title": work.title,
            "published": work.published,
            "venue": work.venue or "",
        }
    data = _as_dict(work)
    return {
        "cite_id": _cite(work),
        "title": str(data.get("title") or ""),
        "published": str(data.get("published") or ""),
        "venue": str(data.get("venue") or ""),
    }


def _retrieved_ids(works: list[Any]) -> set[str]:
    return {row["cite_id"] for row in (_paper_row(work) for work in works) if row["cite_id"]}


def runnable_baselines(works: list[Any]) -> list[dict[str, str]]:
    """This mission's papers as controls to implement, not related-work prose."""
    rows: list[dict[str, str]] = []
    for work in works or []:
        row = _paper_row(work)
        if not row["cite_id"]:
            continue
        data = work if isinstance(work, dict) else _as_dict(work)
        abstract = ""
        if hasattr(work, "abstract"):
            abstract = str(getattr(work, "abstract", "") or "")
        else:
            abstract = str(data.get("abstract") or "")
        cue = " ".join(abstract.split())[:240]
        rows.append(
            {
                "cite_id": row["cite_id"],
                "title": row["title"],
                "published": row["published"],
                "method_cue": cue,
                "role": (
                    "implement as the control arm on the bound world; "
                    "do not paste this as related work"
                ),
            }
        )
    return rows


@dataclass(frozen=True)
class IdeaProtocol:
    markdown: str
    payload: dict[str, Any]
    readme: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "payload": dict(self.payload),
            "readme": self.readme,
        }


def render_protocol(
    topic: str,
    card: GeneratedCard,
    brief: ResearchBrief,
    works: list[Any],
    *,
    probe: dict[str, Any] | None = None,
    diagnosis: Any = None,
    prior: dict[str, Any] | None = None,
    world: Any = None,
) -> IdeaProtocol:
    """Compile one card's protocol. No LLM call."""
    papers = [_paper_row(work) for work in (works or [])]
    retrieved = _retrieved_ids(works or [])
    read_first = [cid for cid in brief.read_first if cid in retrieved]
    diag = _as_dict(diagnosis)
    prior_data = _as_dict(prior)
    probe_data = dict(probe or {})

    cheap: dict[str, Any] | None = None
    if probe_data:
        kind = str(probe_data.get("kind") or "SYNTHETIC").upper() or "SYNTHETIC"
        cheap = {
            "kind": kind,
            "status": probe_data.get("status"),
            "measure": probe_data.get("measure") or "",
            "treatment": probe_data.get("treatment"),
            "control": probe_data.get("control"),
            "verdict": probe_data.get("verdict"),
            "expected_direction": probe_data.get("expected_direction") or diag.get("expected_direction") or "",
            "alternative": probe_data.get("alternative") or diag.get("alternative") or "",
            "disclaimer": (
                SYNTHETIC_DISCLAIMER
                if kind == "SYNTHETIC"
                else GENERATED_DISCLAIMER
                if kind == "GENERATED"
                else (
                    "This probe read a frozen attested world under data/. "
                    "The bytes were copied before the script ran; they were "
                    "not invented inside experiment.py."
                )
            ),
        }

    success = ""
    if diag.get("expected_direction"):
        success = (
            f"treatment vs control in the pre-registered direction "
            f"{diag['expected_direction']}"
        )
        if diag.get("margin") is not None and diag.get("margin") != "":
            success += f" with margin {diag['margin']}"
            if diag.get("margin_reason"):
                success += f" ({diag['margin_reason']})"
    elif brief.baseline:
        success = f"beats the named baseline: {brief.baseline}"

    next_week = {
        "experiment": str(diag.get("experiment") or "") or (
            brief.first_steps[0] if brief.first_steps else brief.approach
        ),
        "treatment_arm": str(diag.get("treatment_arm") or ""),
        "control_arm": str(diag.get("control_arm") or ""),
        "baseline": brief.baseline,
        "success": success,
        "fail_if": brief.risks,
        "first_steps": list(brief.first_steps),
    }

    world_meta = (
        world.to_dict()
        if world is not None and hasattr(world, "to_dict")
        else None
    )
    tier = resolve_tier(str(diag.get("compute_tier") or ""))
    derivation = str(getattr(card, "derivation", "") or "").strip()
    experimental_design = {
        "hypothesis": card.claim,
        "competing_explanation": str(diag.get("alternative") or ""),
        "independent_variable": (
            "mechanism flag only ("
            + str(diag.get("mechanism_flag") or "mechanism_enabled")
            + "): both arms call the same measure on the same attested bytes"
        ),
        "dependent_variable": (
            (cheap or {}).get("measure")
            or next_week["experiment"]
            or "pre-registered numeric measure"
        ),
        "treatment_arm": next_week["treatment_arm"],
        "control_arm": next_week["control_arm"],
        "expected_direction": str(diag.get("expected_direction") or ""),
        "alternative_direction": str(diag.get("alternative_direction") or ""),
        "margin": diag.get("margin"),
        "margin_reason": str(diag.get("margin_reason") or ""),
        "analysis": (
            "judge_probe arithmetic: relative |treatment-control|/|control| "
            "below the margin is uninformative (failed experiment, not a "
            "failed idea); otherwise the sign against expected_direction is "
            "supports or weakens. The model cannot change the sign afterwards."
        ),
        "filter": (
            "20s sandbox on the constructed world under data/; stdlib whitelist; "
            "no network. GENERATED: discriminator only, cannot corroborate."
            if str((world_meta or {}).get("role") or "") == "generated"
            else (
                "20s sandbox on the frozen slice under data/; stdlib whitelist; "
                "no network. This is a discriminator, not the science."
            )
        ),
        "host": (
            f"farfield execute on the attested parent "
            f"(compute tier {tier.name}, timeout {int(tier.timeout_seconds)}s, "
            "same source_digest). "
            "Missing parent cache is blocked rather than silently using the slice."
        ),
        "instance": (world_meta or {}).get("id") or "",
        "schema": (world_meta or {}).get("schema") or "",
        "source_digest": (world_meta or {}).get("source_digest") or "",
        "slice_rule": (world_meta or {}).get("slice_rule") or None,
        "stopping": (
            "supports and weakens both stop. uninformative redesigns the "
            "experiment, quoting the failed design as text, never the "
            "treatment/control numbers. Do not chase a preferred sign."
        ),
        "baseline_cite_id": str(diag.get("baseline_cite_id") or ""),
        "baseline_method": str(diag.get("baseline_method") or ""),
    }

    payload: dict[str, Any] = {
        "kind": "research_protocol",
        "not_a_paper": True,
        "card_id": card.card_id,
        "idea_id": card.card_id,
        "idea_kind": idea_kind_of(card),
        "lineage_id": str(getattr(card, "lineage_id", "") or ""),
        "origin": str(getattr(card, "origin", "") or ""),
        "title": brief.title,
        "plain_title": brief.plain_title,
        "one_liner": brief.one_liner,
        "why_it_matters": brief.why_it_matters,
        "topic": topic,
        "claim": card.claim,
        "mechanism": card.mechanism,
        "prediction": card.prediction,
        "pair": list(card.pair),
        "gap": brief.gap,
        "idea": brief.idea,
        "approach": brief.approach,
        "baseline": brief.baseline,
        "first_steps": list(brief.first_steps),
        "risks": brief.risks,
        "read_first": read_first,
        "diagnosis": {
            "alternative": str(diag.get("alternative") or ""),
            "experiment": str(diag.get("experiment") or ""),
            "treatment_arm": str(diag.get("treatment_arm") or ""),
            "control_arm": str(diag.get("control_arm") or ""),
            "expected_direction": str(diag.get("expected_direction") or ""),
            "alternative_direction": str(diag.get("alternative_direction") or ""),
            "expected_if_alternative": str(diag.get("expected_if_alternative") or ""),
            "margin": diag.get("margin"),
            "margin_reason": str(diag.get("margin_reason") or ""),
            "baseline_cite_id": str(diag.get("baseline_cite_id") or ""),
            "baseline_method": str(diag.get("baseline_method") or ""),
            "mechanism_flag": str(diag.get("mechanism_flag") or "mechanism_enabled"),
            "compute_tier": tier.name,
        }
        if diag
        else None,
        "cheap_probe": cheap,
        "experiment_attempts": list(probe_data.get("attempts") or []),
        "discarded_experiments": [
            str(item).strip()
            for item in (probe_data.get("discarded") or [])
            if str(item or "").strip()
        ],
        "papers": papers,
        "runnable_baselines": runnable_baselines(works),
        "world": world_meta,
        "experiment_digest": str(probe_data.get("experiment_digest") or ""),
        "world_digest": str(
            probe_data.get("world_digest")
            or (world_meta or {}).get("digest")
            or ""
        ),
        "data_digest": str(
            probe_data.get("data_digest")
            or probe_data.get("world_digest")
            or (world_meta or {}).get("digest")
            or ""
        ),
        "evidence_id": str(probe_data.get("evidence_id") or ""),
        "derivation": derivation,
        "experimental_design": experimental_design,
        "host": {
            "command": "farfield run <this-folder>",
            "runner": tier.name if tier.name in {"host", "host-heavy"} else "host",
            "compute_tier": tier.name,
            "timeout_seconds": tier.timeout_seconds,
            "prefer_parent": True,
            "external": (
                "farfield run --runner external (emits SkyPilot sky_task.yaml); "
                "launch with the SkyPilot skill; then attest-run --metrics"
            ),
            "artifact": (
                "farfield compile-artifact <this-folder> dumps attested.json; "
                "feed jin-s13/ai-research-writing-skill — not a FarField paper"
            ),
        },
        "paper_outline": [
            {"section": "Abstract", "from": "claim"},
            {"section": "Introduction", "from": "gap"},
            {"section": "Theory sketch", "from": "derivation or mechanism", "status": "unverified model text"},
            {"section": "Method", "from": "mechanism + diagnosis arms"},
            {"section": "Experiments", "from": "cheap probe + host protocol"},
            {"section": "Runnable baselines", "from": "this mission's papers"},
            {"section": "Results", "from": "host_run.json after farfield execute", "status": "empty until executed"},
            {"section": "Limitations", "from": "honesty: protocol_executed is not a discovery"},
        ],
        "prior": prior_data or None,
        "next_week": next_week,
        "honesty": [
            NOT_A_PAPER,
            (
                GENERATED_DISCLAIMER
                if str((cheap or {}).get("kind") or "") == "GENERATED"
                else SYNTHETIC_DISCLAIMER
            ),
            PROTOCOL_EXECUTED,
            PROTOCOL_NOT_DISCOVERY,
        ],
    }

    theory = derivation or card.mechanism
    theory_status = (
        "derivation supplied by the generator"
        if derivation
        else "no derivation field; mechanism is shown as the sketch"
    )
    plain_heading = (
        f"{brief.plain_title}（{brief.title}）"
        if brief.plain_title
        else brief.title
    )
    md: list[str] = [
        f"# Research plan: {plain_heading}",
        "",
        NOT_A_PAPER,
        "",
    ]
    if brief.one_liner or brief.why_it_matters:
        md.extend(["**这张卡在说什么**", ""])
        if brief.one_liner:
            md.extend([f"- 在问什么：{brief.one_liner}"])
        if brief.why_it_matters:
            md.extend([f"- 为什么值得答：{brief.why_it_matters}"])
        md.append("")
    md.extend(
        [
            f"*Topic.* {topic}",
            "",
            f"*Combination.* {card.pair[0]} × {card.pair[1]}",
            "",
            "## 1. Idea",
            "",
            card.claim,
            "",
        ]
    )
    if brief.gap:
        md.extend([f"*Gap.* {brief.gap}", ""])
    if brief.idea:
        md.extend([brief.idea, ""])
    md.extend(
        [
            "## 2. Theory sketch (unverified model text, not evidence)",
            "",
            theory,
            "",
            f"Status: {theory_status}. This sketch is model text. It does not climb the ladder.",
            "",
            "## 3. Why the mechanism might hold",
            "",
            card.mechanism,
            "",
        ]
    )
    if getattr(card, "dead_end", ""):
        md.extend(
            [
                "### Discarded approach (unverified model text, not evidence)",
                "",
                f"- Failed approach: {card.dead_end}",
                f"- Why it failed: {card.why_failed}",
                f"- Reframe: {card.reframe}",
                f"- Strongest objection: {card.objection}",
                "",
            ]
        )
    md.extend(
        [
            "## 4. Experimental design (pre-registered)",
            "",
            f"- Hypothesis: {experimental_design['hypothesis']}",
            f"- Competing explanation: {experimental_design['competing_explanation'] or 'none registered'}",
            f"- Independent variable: {experimental_design['independent_variable']}",
            f"- Dependent variable: {experimental_design['dependent_variable']}",
            f"- Treatment arm: {experimental_design['treatment_arm'] or 'n/a'}",
            f"- Control arm: {experimental_design['control_arm'] or 'n/a'}",
            f"- Expected direction: {experimental_design['expected_direction'] or 'n/a'}",
            f"- If the alternative is true: {experimental_design['alternative_direction'] or 'n/a'} — {diag.get('expected_if_alternative') or 'n/a'}",
            f"- Margin: {experimental_design['margin']} ({experimental_design['margin_reason'] or 'n/a'})",
            f"- Instance: {experimental_design['instance'] or 'unbound'} ({experimental_design['schema'] or 'no schema'})",
            f"- Parent digest: `{experimental_design['source_digest'] or 'n/a'}`",
            f"- Slice rule: {experimental_design['slice_rule'] or 'n/a'}",
            f"- Analysis: {experimental_design['analysis']}",
            f"- Stopping: {experimental_design['stopping']}",
            f"- Filter: {experimental_design['filter']}",
            f"- Host: {experimental_design['host']}",
            "",
            str(diag.get("alternative") or "No competing explanation was registered."),
            "",
            f"## 5. Cheap probe already run ({cheap.get('kind') if cheap else 'not run'})",
            "",
        ]
    )
    if cheap and cheap.get("status") == "ran":
        md.extend(
            [
                f"- Measure: {cheap.get('measure') or 'n/a'}",
                f"- Kind: {cheap.get('kind')}",
                f"- Treatment: {cheap.get('treatment')}",
                f"- Control: {cheap.get('control')}",
                f"- Pre-registered direction: {cheap.get('expected_direction') or 'n/a'}",
                f"- Verdict by arithmetic: **{cheap.get('verdict')}**",
                "",
                str(cheap.get("disclaimer") or SYNTHETIC_DISCLAIMER),
                "",
            ]
        )
    elif cheap:
        md.extend(
            [
                f"The probe did not produce two arms ({cheap.get('status')}:"
                f" {probe_data.get('error') or 'no detail'}).",
                "",
            ]
        )
    else:
        md.extend(["No cheap probe was run.", ""])

    attempts = [
        row
        for row in (probe_data.get("attempts") or [])
        if isinstance(row, dict)
    ]
    discarded = [
        str(item).strip()
        for item in (probe_data.get("discarded") or [])
        if str(item or "").strip()
    ]
    if len(attempts) > 1 or discarded:
        md.extend(
            [
                "## Failed experiment branches (process data, not a new claim)",
                "",
                "The claim did not change. These are earlier tests that could "
                "not discriminate, or scripts that failed before two arms existed.",
                "",
            ]
        )
        for row in attempts:
            bottleneck = row.get("bottleneck") or "informative"
            verdict = row.get("verdict") or row.get("status") or "n/a"
            experiment = str(row.get("experiment") or "").strip() or "(no design)"
            md.append(
                f"- Attempt {int(row.get('attempt') or 0) + 1}: {bottleneck} "
                f"→ {verdict}. {experiment}"
            )
        md.append("")

    md.extend(
        [
            "## 6. Host protocol (the real experiment)",
            "",
            next_week["experiment"],
            "",
            "Run on the attested parent, not the 20s slice:",
            "",
            "```",
            "PYTHONPATH=src python3 -m farfield execute <this-folder>",
            "```",
            "",
            f"Timeout {int(HOST_TIMEOUT_SECONDS)}s, stdlib whitelist, no network, same source_digest. "
            "protocol_executed is not a discovery.",
            "",
        ]
    )
    if next_week["treatment_arm"] or next_week["control_arm"]:
        md.extend(
            [
                f"- Treatment arm: {next_week['treatment_arm'] or 'n/a'}",
                f"- Control arm: {next_week['control_arm'] or 'n/a'}",
            ]
        )
    steps = [
        f"{i}. {step}"
        for i, step in enumerate(next_week["first_steps"], start=1)
    ] or ["1. (none registered)"]
    md.extend(
        [
            f"- Baseline: {next_week['baseline'] or 'n/a'}",
            f"- Success: {next_week['success'] or 'n/a'}",
            f"- Fail if: {next_week['fail_if'] or 'n/a'}",
            "",
            "### First steps",
            "",
            *steps,
            "",
            "## 7. Runnable baselines from this mission (implement, do not summarise)",
            "",
        ]
    )
    baselines = payload["runnable_baselines"]
    if baselines:
        for row in baselines:
            cue = f" — {row['method_cue']}" if row.get("method_cue") else ""
            md.append(
                f"- [{row['cite_id']}] {row['title']}{cue}"
            )
            md.append(f"  Role: {row['role']}")
        named = str(diag.get("baseline_cite_id") or "")
        method = str(diag.get("baseline_method") or "")
        if named:
            md.extend(["", f"Named control: `{named}` — {method or 'see method_cue'}", ""])
        else:
            md.append("")
    else:
        md.extend(["- (none retrieved this mission)", ""])

    md.extend(
        [
            "## 8. Paper outline (fill results after host execute; this is not a paper)",
            "",
            "- Abstract ← claim",
            "- Introduction ← gap",
            "- Theory sketch ← derivation / mechanism (unverified)",
            "- Method ← diagnosis arms",
            "- Experiments ← cheap probe (filter) + `farfield execute` (parent)",
            "- Results ← `host_run.json` only",
            "- Limitations ← protocol_executed is not a discovery",
            "",
            "## Literature retrieved this mission only",
            "",
        ]
    )
    if papers:
        for row in papers:
            venue = f", {row['venue']}" if row["venue"] else ""
            mark = " — read first" if row["cite_id"] in read_first else ""
            md.append(
                f"- [{row['cite_id']}] {row['title']} ({row['published']}{venue}){mark}"
            )
    else:
        md.append("- (none retrieved)")
    md.append("")
    if read_first:
        md.extend(["Read first: " + ", ".join(read_first), ""])
    if prior_data:
        md.extend(
            [
                "## Prior warning",
                "",
                f"Closest supporting span: {prior_data.get('title')} "
                f"[{prior_data.get('cite_id')}], support "
                f"{prior_data.get('support', prior_data.get('coverage'))}"
                + (
                    f" — {prior_data['span']}"
                    if prior_data.get("span")
                    else ""
                )
                + ".",
                "",
            ]
        )
    md.extend(
        [
            "## What later work would show",
            "",
            card.prediction,
            "",
            "## Honesty",
            "",
            f"- {NOT_A_PAPER}",
            f"- {GENERATED_DISCLAIMER if str((cheap or {}).get('kind') or '') == 'GENERATED' else SYNTHETIC_DISCLAIMER}",
            f"- {PROTOCOL_EXECUTED}",
            f"- {PROTOCOL_NOT_DISCOVERY}",
            "- `supports` on SYNTHETIC or GENERATED data is a coherence check. `corroborated` / `verified` require an attested WORLD fixture. Neither word means the science is done.",
            "",
        ]
    )

    readme_intro = (
        [f"- 在问什么：{brief.one_liner}"] if brief.one_liner else []
    ) + (
        [f"- 为什么值得答：{brief.why_it_matters}"] if brief.why_it_matters else []
    )
    readme = "\n".join(
        [
            f"# {plain_heading}",
            "",
            *(readme_intro + [""] if readme_intro else []),
            "This folder is a research plan a colleague — or `farfield execute` — can pick up. It is not a paper.",
            "",
            "| File | What it is |",
            "| --- | --- |",
            "| `protocol.md` | Research plan: idea, theory sketch, design, host protocol |",
            "| `protocol.json` | Same fields, machine-readable |",
            "| `note.md` / `note.tex` | Compiled briefing from attested fields |",
            "| `experiment.py` | Cheap 20s filter (not the science) |",
            "| `metrics.json` | Arithmetic on the in-loop world or SYNTHETIC data |",
            "| `host_run.json` | Host execution of this protocol, if run |",
            "| `papers.json` | Only papers retrieved this mission (runnable baselines) |",
            "| `brief.json` | Model briefing after domain lock |",
            "",
            "Do not cite `metrics.json` as an empirical result. Run `farfield execute` on this folder. protocol_executed is not a discovery.",
            "",
            f"Claim: {card.claim}",
            "",
        ]
    )
    return IdeaProtocol(markdown="\n".join(md), payload=payload, readme=readme)


def _bucket(row: dict[str, Any]) -> str:
    if row.get("prior_kills"):
        return "closed_by_prior"
    verdict = row.get("verdict")
    if verdict == "weakens":
        return "weakened"
    if verdict == "supports":
        return "do_next"
    if str(row.get("experiment") or "").strip() or row.get("has_brief"):
        return "do_next"
    constructed = bool(row.get("generated_world")) or str(
        row.get("probe_kind") or ""
    ).upper() == "GENERATED"
    if constructed:
        return "do_next"
    return "still_open"


def _heading(rec: dict[str, Any], *, lang: str = "zh") -> str:
    title = str(rec.get("title") or "").strip()
    plain = str(rec.get("plain_title") or "").strip()
    if plain and title and plain != title:
        if lang_of(lang) == "en":
            return f"{plain} ({title})"
        return f"{plain}（{title}）"
    if plain:
        return plain
    if title:
        return title
    claim = " ".join(str(rec.get("claim") or "").split())
    if len(claim) > 88:
        return claim[:85].rstrip() + "…"
    if claim:
        return claim
    return idea_folder_name(rec)


def _pair_text(rec: dict[str, Any]) -> str:
    pair = rec.get("pair") or []
    if len(pair) >= 2:
        return f"{pair[0]} × {pair[1]}"
    return ""


_SLUG_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def idea_folder_name(
    rec: dict[str, Any],
    *,
    taken: set[str] | None = None,
) -> str:
    """Human folder name from the claim object, not `gen_<digest>`.

    The machine `card_id` stays under `candidates/` for execute. Idea
    packets a researcher opens are named by far concept + lever.
    """
    cached = str(rec.get("idea_name") or "").strip()
    if cached and not cached.startswith("gen_"):
        slug = re.sub(r"[^a-z0-9]+", "-", cached.lower()).strip("-")
        if slug:
            return _unique_slug(slug, taken)
    pair = rec.get("pair") or []
    far = str(pair[1] if len(pair) >= 2 else "")
    flag = str(rec.get("mechanism_flag") or rec.get("world_lever") or "")
    kind = idea_kind_of(rec)
    title = str(rec.get("plain_title") or rec.get("title") or "")
    claim = str(rec.get("claim") or "")
    raw = " ".join(
        part
        for part in ((kind if kind != PROBE else ""), far, flag)
        if str(part).strip()
    )
    if not raw.strip():
        raw = title or claim or "idea"
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", raw.lower())
        if token
        and not token.startswith("gen")
        and not (len(token) > 1 and token in _SLUG_STOP)
    ]
    if not tokens:
        tokens = ["idea"]
    slug = "-".join(tokens[:8])[:56].strip("-") or "idea"
    return _unique_slug(slug, taken)


def _unique_slug(slug: str, taken: set[str] | None) -> str:
    if taken is None:
        return slug
    base = slug
    n = 2
    while slug in taken:
        slug = f"{base}-{n}"[:56].strip("-")
        n += 1
    taken.add(slug)
    return slug


def packet_worthy(rec: dict[str, Any]) -> bool:
    """False when the idea must not occupy an `ideas/` folder.

    `weakens` has stopped. A no_handle draft that never ran is not a
    second scientific line. `uninformative` stays: the plan is
    executable; remaining work is `farfield execute` on this pair.
    Acquire and theory occupy `ideas/` without a two-arm.
    """
    if rec.get("prior_kills") or rec.get("world_incompatible"):
        return False
    if rec.get("verdict") == "weakens":
        return False
    experiment = str(rec.get("experiment") or "").lower()
    if rec.get("verdict") in (None, "") and "not runnable" in experiment:
        return False
    return True


def live_idea_records(
    *,
    ranked: list[dict[str, Any]] | None = None,
    found: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    found_rows = list(found or [])
    by_id = {str(row.get("card_id") or ""): row for row in found_rows}
    buckets: dict[str, list[dict[str, Any]]] = {
        "do_next": [],
        "closed_by_prior": [],
        "weakened": [],
        "still_open": [],
    }
    ordered = list(ranked or [])
    if not ordered:
        ordered = [
            {"rank": i, "card_id": row.get("card_id"), "why": []}
            for i, row in enumerate(found_rows, start=1)
        ]
    for item in ordered:
        record = by_id.get(str(item.get("card_id") or ""), {})
        merged = {**record, **{k: v for k, v in item.items() if v is not None}}
        buckets[_bucket(merged)].append(merged)
    live = buckets["do_next"] + buckets["still_open"]
    return live, buckets


def live_packet_records(
    *,
    ranked: list[dict[str, Any]] | None = None,
    found: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    live, buckets = live_idea_records(ranked=ranked, found=found)
    kept: list[dict[str, Any]] = []
    stopped = list(buckets.get("weakened") or [])
    for rec in live:
        if packet_worthy(rec):
            kept.append(rec)
        else:
            stopped.append(rec)
    buckets["weakened"] = stopped
    return kept, buckets


def _kind(rec: dict[str, Any]) -> str:
    return str(rec.get("probe_kind") or "SYNTHETIC").upper()


def _is_world_sim(rec: dict[str, Any]) -> bool:
    if _kind(rec) == "WORLD_SIM":
        return True
    return str(rec.get("results_status") or "") == "imagined"


def _week_steps(rec: dict[str, Any], *, lang: str = "zh") -> list[str]:
    steps = [str(item).strip() for item in (rec.get("first_steps") or []) if str(item or "").strip()]
    if steps:
        return steps
    kind = idea_kind_of(rec)
    built: list[str] = []
    if kind == ACQUIRE:
        built.append(_t(lang, "step_acquire"))
        if rec.get("prediction"):
            built.append(f"{_t(lang, 'step_prediction')}{rec['prediction']}")
        built.append(_t(lang, "step_metrics"))
        return built
    if kind == THEORY:
        built.append(_t(lang, "step_theory"))
        built.append(_t(lang, "step_metrics"))
        return built
    if kind == QUESTION:
        built.append(_t(lang, "step_question"))
    if rec.get("experiment"):
        built.append(str(rec["experiment"]))
    treatment = str(rec.get("treatment_arm") or "").strip()
    control = str(rec.get("control_arm") or "").strip()
    if treatment:
        built.append(f"{_t(lang, 'step_treatment')}{treatment}")
    if control:
        built.append(f"{_t(lang, 'step_control')}{control}")
    if rec.get("generated_world") or _kind(rec) == "GENERATED":
        built.append(_t(lang, "step_generated"))
    elif rec.get("verdict") == "supports" and _kind(rec) in {"WORLD", "REAL", "FIXTURE"}:
        built.append(_t(lang, "step_execute"))
    if rec.get("prediction"):
        built.append(f"{_t(lang, 'step_success')}{rec['prediction']}")
    if rec.get("verdict") == "uninformative":
        built.append(_t(lang, "step_uninformative"))
    if rec.get("verdict") == "weakens":
        built.append(_t(lang, "step_weakens"))
    if not built and rec.get("claim"):
        built.append(f"{_t(lang, 'step_register')}{rec['claim']}")
        if rec.get("prediction"):
            built.append(f"{_t(lang, 'step_prediction')}{rec['prediction']}")
    built.append(_t(lang, "step_metrics"))
    return built


def _honesty_line(rec: dict[str, Any], *, lang: str = "zh") -> str:
    kind = _kind(rec)
    verdict = rec.get("verdict")
    sep = "; " if lang_of(lang) == "en" else "；"
    stop = "." if lang_of(lang) == "en" else "。"
    if rec.get("prior_kills"):
        return _t(lang, "honesty_kills")
    if verdict == "weakens":
        return _t(lang, "honesty_weakens")
    idea = idea_kind_of(rec)
    if idea == ACQUIRE:
        return _t(lang, "honesty_acquire") + stop
    if idea == THEORY:
        return _t(lang, "honesty_theory") + stop
    bits: list[str] = []
    if idea == QUESTION:
        bits.append(_t(lang, "honesty_question"))
    if rec.get("generated_world") or kind == "GENERATED":
        bits.append(_t(lang, "honesty_generated"))
    elif rec.get("world_id"):
        bits.append(f"{_t(lang, 'honesty_bound')} `{rec.get('world_id')}`")
    if verdict == "supports":
        if kind in {"WORLD", "REAL", "FIXTURE"}:
            bits.append(_t(lang, "honesty_world_supports"))
        else:
            bits.append(f"{kind} {_t(lang, 'honesty_kind_supports')}")
    elif verdict == "uninformative":
        bits.append(_t(lang, "honesty_uninformative"))
    elif rec.get("experiment"):
        bits.append(_t(lang, "honesty_registered"))
    return sep.join(bits) + stop if bits else _t(lang, "honesty_default")


def _derivation_text(rec: dict[str, Any]) -> str:
    return str(rec.get("derivation") or "").strip()


def render_derivation_note(rec: dict[str, Any]) -> str:
    """Honest derivation package from registered fields. Does not invent math."""
    derivation = _derivation_text(rec)
    claim = str(rec.get("claim") or "").strip()
    mechanism = str(rec.get("mechanism") or "").strip()
    prediction = str(rec.get("prediction") or "").strip()
    lines = [
        "# Derivation package",
        "",
        "Unverified model text plus registered quantities. Not a theorem. "
        "Not evidence. Do not add symbols that are not already on the card.",
        "",
        "## Target",
        "",
        claim or "（本场未登记主张）",
        "",
        "## Registered quantities",
        "",
        prediction or "（预测句未点名可测量）",
        "",
        "## Mechanism sketch",
        "",
        mechanism or "（未登记机制）",
        "",
        "## Derivation as stated",
        "",
    ]
    if derivation:
        lines.extend([derivation, ""])
        lines.extend(
            ["**Status.** COHERENT AS STATED — only as a sketch, not a proof.", ""]
        )
    else:
        lines.extend(
            [
                "本场没有 `derivation` 字段。不要补一套未登记的公式。",
                "",
                "**Status.** NOT YET COHERENT — freeze the quantity on the "
                "claim/prediction before deriving.",
                "",
            ]
        )
    return "\n".join(lines)


def _evidence_projection_text(rec: dict[str, Any]) -> str:
    """Scientific fact sentences. Never lets a number rename the claim."""
    stored = str(rec.get("evidence_projection") or "").strip()
    if stored:
        return stored
    raw_spec = rec.get("claim_spec")
    if not isinstance(raw_spec, dict):
        return ""
    try:
        from .claimspec import claim_spec_from_payload, project_evidence_to_claim

        spec = claim_spec_from_payload(raw_spec)
    except Exception:
        return ""
    projection = project_evidence_to_claim(
        spec,
        arithmetic_verdict=str(rec.get("verdict") or ""),
        observed_effect={
            key: rec[key]
            for key in ("delta", "treatment", "control", "separation")
            if key in rec
        },
        metric=str(rec.get("world_observable") or rec.get("metric") or ""),
    )
    return projection.scientific_text()


def render_result_to_claim(rec: dict[str, Any]) -> str:
    """What the probe earned. GENERATED supports is not a discovery."""
    kind = _kind(rec)
    verdict = str(rec.get("verdict") or "").strip() or "（尚未跑探针）"
    claim = str(rec.get("claim") or "").strip() or "（未登记主张）"
    world_id = str(rec.get("world_id") or "").strip() or "（未绑定）"
    if _is_world_sim(rec):
        return "\n".join(
            [
                "# Result to claim",
                "",
                "WORLD_SIM imagined numbers are withheld. They cannot occupy this readout.",
                "",
                f"- claim: {claim}",
                "- probe_kind: `WORLD_SIM`",
                "- results_status: `imagined`",
                "",
                "## What C1 earned",
                "",
                "Nothing on the evidence ladder. World rehearsal is diagnostic.",
                "",
                "## What C1 did not earn",
                "",
                "- A conference result",
                "- A climb on SYNTHETIC, GENERATED, or WORLD_SIM",
                "",
                "## Next",
                "",
                "- Read `world-sim/` as diagnostic rehearsal only. Do not rewrite expected_direction.",
                "",
            ]
        )
    projection = _evidence_projection_text(rec)
    scoped = str(rec.get("scoped_verdict") or "").strip()
    lines = [
        "# Result to claim",
        "",
        "Compiled from attested fields. Not a second LLM judge. "
        "Numbers do not rename the ladder.",
        "",
        f"- claim: {claim}",
        f"- probe_kind: `{kind}`",
        f"- verdict: `{verdict}`",
        *( [f"- scoped_verdict: `{scoped}`"] if scoped else [] ),
        f"- world: `{world_id}`",
        "",
        "## What C1 earned",
        "",
    ]
    if projection:
        lines.extend([projection, ""])
    if verdict == "supports" and kind == "WORLD":
        lines.append(
            "Two-arm arithmetic on an attested freeze matched the "
            "pre-registered direction. Pending host replication of the "
            "same EvidenceID before corroboration."
        )
    elif verdict == "supports":
        lines.append(
            f"`supports` on `{kind}` is a coherence check. It can "
            "encourage a freeze of this schema. It cannot corroborate, "
            "distill, or stop spray."
        )
    elif verdict == "weakens":
        lines.append(
            "The registered direction failed. Stop this mechanism. "
            "Do not redesign to hunt a support."
        )
    elif verdict == "uninformative":
        lines.append(
            "Arms did not separate. Sharpen the test on the same object. "
            "Do not change the claim."
        )
    else:
        lines.append("No scientific verdict yet. Plan evidence; do not claim it.")
    lines.extend(
        [
            "",
            "## What C1 did not earn",
            "",
            "- A conference result",
            "- A climb on SYNTHETIC, GENERATED, or WORLD_SIM",
            "- Permission to rebind a cousin freeze (Pride / SNAP / fasta)",
            "",
            "## Next",
            "",
        ]
    )
    if kind in {"GENERATED", "SYNTHETIC"}:
        lines.append(
            "- Freeze a matching schema (`farfield freeze`) before WORLD."
        )
    elif verdict == "supports":
        lines.append("- `farfield execute` the same protocol on the attested parent.")
    else:
        lines.append(
            "- Stay on this task world. Read `refine-logs/EXPERIMENT_PLAN.md`. "
            "Follow-up runs listed as 必须运行 are not promotion."
        )
    lines.append("")
    return "\n".join(lines)


def render_ablation_plan(rec: dict[str, Any]) -> str:
    """Must-run isolations from the registered competing explanation."""
    alternative = str(rec.get("alternative") or "").strip()
    lever = str(rec.get("world_lever") or "").strip()
    kind = _kind(rec)
    lines = [
        "# Ablation plan",
        "",
        "Compiled from attested alternative / lever. Not a GPU queue. "
        "Do not invent a new competing explanation here and treat it as registered.",
        "",
        "## Must-run",
        "",
        "1. Mechanism off (control arm already registered).",
        "",
    ]
    if alternative:
        lines.append(f"2. Competing explanation: {alternative}")
        lines.append("")
    else:
        lines.append(
            "2. No alternative was registered this mission. The next "
            "diagnosis of this pair must name one before claiming isolation."
        )
        lines.append("")
    if lever:
        lines.append(f"3. World lever already named: `{lever}`.")
        lines.append("")
    lines.extend(
        [
            "## Nice-to-have (do not pad)",
            "",
            "- Extra seeds on the *same* freeze after host_ok.",
            "- Slice robustness of the bound parent, not a new dataset.",
            "",
            "## Forbidden",
            "",
            "- Changing expected_direction after seeing numbers",
            f"- Treating `{kind}` supports as ablation success on a real benchmark",
            "- Running the far-field node's internal cost as the measure",
            "",
        ]
    )
    return "\n".join(lines)


def render_kill_argument(rec: dict[str, Any]) -> str:
    """Strongest registered objection. Not a new review LLM."""
    objection = str(rec.get("objection") or "").strip()
    falsifier = str(rec.get("falsifier") or "").strip()
    dead = str(rec.get("dead_end") or "").strip()
    lines = [
        "# Kill argument",
        "",
        "The strongest no already on the card. Do not open a second "
        "reviewer model and treat its memo as evidence.",
        "",
        "## Attack (registered)",
        "",
        objection or "（本场未登记 objection）",
        "",
        "## Falsifier",
        "",
        falsifier or "（未登记）",
        "",
        "## Discarded approach",
        "",
        dead or "（未登记 dead_end）",
        "",
        "## Still unresolved if empty",
        "",
        "An empty attack is not a pass. The next refine of this pair "
        "must write an objection that is not a restatement of the claim.",
        "",
    ]
    return "\n".join(lines)


def render_analyze_results(rec: dict[str, Any]) -> str:
    """Readout of attested numbers. Observation vs interpretation."""
    kind = _kind(rec)
    treatment = rec.get("treatment")
    control = rec.get("control")
    verdict = rec.get("verdict")
    lines = [
        "# Analyze results",
        "",
        f"probe_kind: `{kind}`",
        "",
        "## Observation",
        "",
    ]
    if _is_world_sim(rec):
        lines.append("- WORLD_SIM imagined arm numbers are withheld from this readout.")
    elif treatment not in (None, "") and control not in (None, ""):
        lines.append(
            f"- treatment={treatment} control={control} verdict=`{verdict}`"
        )
    else:
        lines.append("- No arm numbers attested on this card yet.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    if kind in {"GENERATED", "SYNTHETIC"}:
        lines.append(
            "These numbers speak only to coherence under the pre-registered "
            "direction. They are not a benchmark result."
        )
    elif _is_world_sim(rec):
        lines.append("No interpretation: WORLD_SIM is diagnostic rehearsal, not a verdict.")
    elif verdict == "supports":
        lines.append(
            "Direction matched on the bound freeze. Host replication is still required."
        )
    elif verdict == "weakens":
        lines.append("Direction failed. The mechanism is not earning its keep.")
    else:
        lines.append("No interpretation beyond the attested verdict.")
    lines.append("")
    return "\n".join(lines)


def _pipe(text: str) -> str:
    return str(text or "").replace("|", "/")


def _structural_holes(rec: dict[str, Any]) -> list[str]:
    holes: list[str] = []
    for item in rec.get("world_sim_issues") or []:
        if not isinstance(item, dict) or item.get("type") != "structural":
            continue
        summary = str(item.get("summary") or "").strip()
        if summary:
            holes.append(summary)
    return holes


def _experiment_plan_body(
    rec: dict[str, Any],
    *,
    lang: str = "zh",
) -> list[str]:
    """Claim map through risks. Shared by the standalone plan and the research plan."""
    claim = str(rec.get("claim") or "").strip() or _empty(lang)
    prediction = str(rec.get("prediction") or "").strip()
    alternative = str(rec.get("alternative") or "").strip()
    experiment = str(rec.get("experiment") or "").strip()
    treatment = str(rec.get("treatment_arm") or "").strip()
    control = str(rec.get("control_arm") or "").strip()
    expected = str(rec.get("expected_direction") or "").strip()
    world_id = str(rec.get("world_id") or "").strip() or _empty(lang)
    schema = str(rec.get("world_schema") or "").strip()
    lever = str(rec.get("world_lever") or "").strip()
    kind = _kind(rec)
    why = str(rec.get("why_it_matters") or prediction or _t(lang, "why_default"))
    block1 = _t(lang, "block1_short")
    block2 = _t(lang, "block2_short")
    lines = [
        f"## {_t(lang, 'claim_map')}",
        "",
        _t(lang, "claim_map_header"),
        _t(lang, "claim_map_sep"),
        f"| C1: {_pipe(claim)} | {_pipe(why)} | "
        f"{_pipe(prediction or _t(lang, 'evidence_default'))} | {block1}, {block2} |",
        "",
    ]
    if alternative:
        lines.append(
            f"| Anti-C1 | {_t(lang, 'anti_exclude')}「{_pipe(alternative)}」 | "
            f"{_t(lang, 'anti_evidence')} | {block2} |"
        )
        lines.append("")
    if lever:
        control_sys = (
            f"{_t(lang, 'control_other_lever')}（{_t(lang, 'current_lever')}=`{lever}`）"
            if lang_of(lang) == "zh"
            else f"{_t(lang, 'control_other_lever')} ({_t(lang, 'current_lever')}=`{lever}`)"
        )
    else:
        control_sys = (
            f"{_t(lang, 'control_other_lever')}（{_t(lang, 'no_far_word')}）"
            if lang_of(lang) == "zh"
            else f"{_t(lang, 'control_other_lever')} ({_t(lang, 'no_far_word')})"
        )
    host_priority = (
        _t(lang, "must_run")
        if kind == "WORLD" and rec.get("verdict") == "supports"
        else _t(lang, "optional")
    )
    lines.extend(
        [
            f"## {_t(lang, 'blocks')}",
            "",
            f"### {_t(lang, 'block1')}",
            "",
            f"- **{_t(lang, 'claim_tested')}**: C1 — {claim}",
            f"- **{_t(lang, 'dataset')}**: `{world_id}` / `{schema or kind}`",
            f"- **{_t(lang, 'compared')}**: {_t(lang, 'treatment')} "
            f"`{treatment or _t(lang, 'mech_on')}` {_t(lang, 'vs')} "
            f"{_t(lang, 'control')} `{control or _t(lang, 'mech_off')}`",
            f"- **{_t(lang, 'metrics')}**: {prediction or _t(lang, 'named_field')}",
            f"- **{_t(lang, 'setup')}**: {experiment or _t(lang, 'same_measure')}",
            f"- **{_t(lang, 'success')}**: {_t(lang, 'success_sign')} "
            f"`{expected or 'expected_direction'}` {_t(lang, 'is_supports')}",
            f"- **{_t(lang, 'failure')}**: {_t(lang, 'block1_failure')}",
            f"- **{_t(lang, 'priority')}**: {_t(lang, 'must_run')}",
            "",
            f"### {_t(lang, 'block2')}",
            "",
            f"- **{_t(lang, 'claim_tested')}**: {_t(lang, 'c1_isolation')} — "
            f"{alternative or _t(lang, 'block2_no_alt')}",
            f"- **{_t(lang, 'compared')}**: {control_sys}",
            f"- **{_t(lang, 'success')}**: {_t(lang, 'block2_success')}",
            f"- **{_t(lang, 'failure')}**: {_t(lang, 'block2_failure')}",
            f"- **{_t(lang, 'priority')}**: {_t(lang, 'must_run')}",
            "",
            f"### {_t(lang, 'block3')}",
            "",
            f"- **{_t(lang, 'claim_tested')}**: {_t(lang, 'block3_claim')}",
            f"- **{_t(lang, 'setup')}**: {_t(lang, 'field_contract')}",
            f"- **{_t(lang, 'success')}**: {_t(lang, 'block3_success')}",
            f"- **{_t(lang, 'priority')}**: {_t(lang, 'must_run')}",
            "",
            f"### {_t(lang, 'block4')}",
            "",
            f"- **{_t(lang, 'claim_tested')}**: {_t(lang, 'block4_claim')}",
            f"- **{_t(lang, 'setup')}**: {_t(lang, 'block4_setup')}",
            (
                f"- **{_t(lang, 'success')}**: host_ok; only WORLD can corroborate "
                f"(kind=`{kind}`)."
                if lang_of(lang) == "en"
                else f"- **{_t(lang, 'success')}**: host_ok；仅 WORLD 可 corroborate（当前 kind=`{kind}`）。"
            ),
            f"- **{_t(lang, 'priority')}**: {host_priority}",
            "",
            f"## {_t(lang, 'run_order')}",
            "",
            _t(lang, "run_header"),
            _t(lang, "run_sep"),
            _t(lang, "m0"),
            _t(lang, "m1"),
            _t(lang, "m2"),
            _t(lang, "m3"),
            "",
            f"## {_t(lang, 'budget')}",
            "",
            f"- {_t(lang, 'budget_host')}",
            f"- {_t(lang, 'budget_gpu')}",
            f"- {_t(lang, 'budget_no_invent')}",
            "",
            f"## {_t(lang, 'risks')}",
            "",
        ]
    )
    if rec.get("world_incompatible"):
        lines.append(f"- **{_t(lang, 'risk')}**: {_t(lang, 'risk_incompatible')}")
    if kind in {"GENERATED", "SYNTHETIC"}:
        lines.append(
            f"- **{_t(lang, 'risk')}**: kind=`{kind}`. {_t(lang, 'risk_generated')}"
        )
    if rec.get("verdict") == "uninformative":
        lines.append(f"- **{_t(lang, 'risk')}**: {_t(lang, 'risk_uninformative')}")
    for summary in _structural_holes(rec)[:4]:
        if lang_of(lang) == "en":
            lines.append(
                f"- **{_t(lang, 'risk')}** (world-sim structural hole): {summary} → "
                f"**{_t(lang, 'mitigation')}**: fix the plan structure, then execute; "
                "imagined numbers cannot enter RESULTS."
            )
        else:
            lines.append(
                f"- **{_t(lang, 'risk')}** (world-sim 结构洞): {summary} → "
                f"**{_t(lang, 'mitigation')}**: 改计划结构后执行；想象数字不能进 RESULTS。"
            )
    if rec.get("objection"):
        lines.append(f"- **{_t(lang, 'risk')}**: {rec['objection']}")
    if not any(
        rec.get(key) for key in ("world_incompatible", "verdict", "objection")
    ) and kind not in {"GENERATED", "SYNTHETIC"}:
        lines.append(f"- **{_t(lang, 'risk')}**: {_t(lang, 'risk_direction')}")
    if not lines[-1].startswith("- "):
        lines.append(f"- **{_t(lang, 'risk')}**: {_t(lang, 'risk_direction')}")
    return lines


def render_experiment_plan(
    topic: str,
    rec: dict[str, Any],
    *,
    workspace: str | None = None,
    lang: str = "zh",
) -> str:
    """Claim → evidence → run order. Compiled, not a new generation.

    Self-contained: claim, mechanism, arms, blocks, budget, risks.
    Section names follow the ARIS experiment-plan template.
    """
    del workspace
    claim = str(rec.get("claim") or "").strip() or _empty(lang)
    mechanism = str(rec.get("mechanism") or "").strip()
    prediction = str(rec.get("prediction") or "").strip()
    lines = [
        f"# {_t(lang, 'plan_title')}",
        "",
        f"**{_t(lang, 'problem')}**: {topic.strip() or _empty(lang)}",
        f"**{_t(lang, 'method_thesis')}**: {claim}",
        f"**{_t(lang, 'idea')}**: `{idea_folder_name(rec)}`",
        f"**{_t(lang, 'idea_kind_label')}**: `{idea_kind_of(rec)}`",
        "",
        _t(lang, "plan_preamble"),
        "",
    ]
    idea = idea_kind_of(rec)
    if idea == ACQUIRE:
        lines.extend([_t(lang, "acquire_plan"), ""])
    elif idea == THEORY:
        lines.extend([_t(lang, "theory_plan"), ""])
    elif idea == QUESTION:
        lines.extend([_t(lang, "question_plan"), ""])
    if mechanism:
        lines.extend([f"**{_t(lang, 'mechanism')}**", "", mechanism, ""])
    if prediction:
        lines.extend([f"**{_t(lang, 'prediction')}**", "", prediction, ""])
    if rec.get("treatment_arm") or rec.get("control_arm"):
        lines.extend(
            [
                f"- **{_t(lang, 'treatment')}**: {rec.get('treatment_arm') or _empty(lang)}",
                f"- **{_t(lang, 'control')}**: {rec.get('control_arm') or _empty(lang)}",
            ]
        )
        if rec.get("expected_direction"):
            lines.append(
                f"- **{_t(lang, 'expected')}**: {rec['expected_direction']}"
            )
        lines.append("")
    lines.extend(_experiment_plan_body(rec, lang=lang))
    lines.append("")
    return "\n".join(lines)


def _works_of(rec: dict[str, Any]) -> list[Any]:
    raw = rec.get("works")
    if not raw:
        raw = rec.get("papers_list") or rec.get("retrieved_works") or []
    if isinstance(raw, list):
        return list(raw)
    return []


def _abstract_of(work: Any) -> str:
    if hasattr(work, "abstract"):
        return str(getattr(work, "abstract", "") or "")
    data = _as_dict(work)
    return str(data.get("abstract") or "")


def _split_papers(rec: dict[str, Any], topic: str) -> tuple[list[Any], list[Any]]:
    from .worldfields import papers_on_claim_object

    works = _works_of(rec)
    claim = str(rec.get("claim") or "").strip()
    if not claim:
        # No claim identity → nothing may enter related work.
        return [], list(works)
    aligned = papers_on_claim_object(
        works, claim=claim, topic=topic, minimum="strong"
    )
    aligned_ids = {id(item) for item in aligned}
    other = [item for item in works if id(item) not in aligned_ids]
    return list(aligned), other


def _paper_bullets(works: list[Any], *, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    rows = works if limit is None else works[:limit]
    for work in rows:
        row = _paper_row(work)
        cite = row["cite_id"] or "(no id)"
        title = row["title"] or "(untitled)"
        when = row["published"]
        venue = f", {row['venue']}" if row["venue"] else ""
        stamp = f" ({when}{venue})" if when or venue else ""
        abstract = " ".join(_abstract_of(work).split())
        lines.append(f"- **[{cite}]** {title}{stamp}")
        if abstract:
            lines.append(f"  {abstract[:600]}")
    return lines


def idea_packet_dir(workspace: str | Path, rec: dict[str, Any] | str) -> str:
    name = rec if isinstance(rec, str) else idea_folder_name(rec)
    return str(Path(workspace) / "ideas" / name)


def render_idea_human_packet(
    topic: str,
    rec: dict[str, Any],
    *,
    sibling_ids: list[str] | None = None,
    workspace: str | None = None,
    lang: str = "zh",
    role: str = "folder",
) -> str:
    """Complete research plan for one idea. Experiment design is inlined."""
    del sibling_ids
    name = idea_folder_name(rec)
    aligned, other = _split_papers(rec, topic)
    heading = _heading(rec, lang=lang)
    why = str(rec.get("why_it_matters") or "").strip()
    one = str(rec.get("one_liner") or "").strip()
    gap = str(rec.get("gap") or rec.get("wiki_gap") or "").strip()
    pair = _pair_text(rec)
    derivation = _derivation_text(rec)
    lines: list[str] = []
    if role != "index":
        lines.extend(
            [
                f"# {_t(lang, 'brief_title')}：{heading}"
                if lang_of(lang) == "zh"
                else f"# {_t(lang, 'brief_title')}: {heading}",
                "",
                _t(lang, "brief_preamble"),
                "",
                _t(lang, "not_a_paper"),
                "",
                f"{_t(lang, 'one_idea')}",
                "",
                f"**{_t(lang, 'idea')}**: `{name}`",
                "",
                f"**{_t(lang, 'honesty')}.** {_honesty_line(rec, lang=lang)}",
                "",
                f"**{_t(lang, 'idea_kind_label')}.** `{idea_kind_of(rec)}` — "
                f"{_t(lang, f'kind_{idea_kind_of(rec)}')}",
                "",
            ]
        )
        lines.append(
            _t(lang, "lang_twin_en") if lang_of(lang) == "zh" else _t(lang, "lang_twin_zh")
        )
        lines.append("")
    lines.extend([f"## {_t(lang, 'problem')}", "", str(topic).strip(), ""])
    if one:
        lines.extend([one, ""])
    lines.extend([f"## {_t(lang, 'claim_mech')}", ""])
    if rec.get("claim"):
        lines.extend([f"**{_t(lang, 'claim')}.**", "", str(rec["claim"]), ""])
    if rec.get("mechanism"):
        lines.extend(
            [f"**{_t(lang, 'mechanism')}.**", "", str(rec["mechanism"]), ""]
        )
    if pair:
        lines.extend([f"**{_t(lang, 'pair')}.** {pair}. {_t(lang, 'pair_note')}", ""])
    if rec.get("prediction"):
        lines.extend(
            [f"**{_t(lang, 'prediction')}.**", "", str(rec["prediction"]), ""]
        )
    lines.extend([f"## {_t(lang, 'background')}", ""])
    if why:
        lines.extend([f"- **{_t(lang, 'why_now')}**: {why}", ""])
    if gap:
        lines.extend([f"- **{_t(lang, 'gap')}**: {gap}", ""])
    lines.extend([f"- **{_t(lang, 'key_papers')}**:", ""])
    if aligned:
        lines.extend(_paper_bullets(aligned))
        lines.append("")
    else:
        lines.extend([f"  {_t(lang, 'no_aligned_papers')}", ""])
    if other:
        lines.extend([f"- **{_t(lang, 'off_object_papers')}**:", ""])
        lines.extend(_paper_bullets(other))
        lines.append("")
    if rec.get("dead_end") or rec.get("why_failed"):
        lines.extend(
            [
                f"- **{_t(lang, 'tried_failed')}**:",
                "",
                f"  - {_t(lang, 'dead_path')}：{rec.get('dead_end') or _empty(lang)}"
                if lang_of(lang) == "zh"
                else f"  - {_t(lang, 'dead_path')}: {rec.get('dead_end') or _empty(lang)}",
                f"  - {_t(lang, 'why_failed')}：{rec.get('why_failed') or _empty(lang)}"
                if lang_of(lang) == "zh"
                else f"  - {_t(lang, 'why_failed')}: {rec.get('why_failed') or _empty(lang)}",
                f"  - {_t(lang, 'reframe')}：{rec.get('reframe') or _empty(lang)}"
                if lang_of(lang) == "zh"
                else f"  - {_t(lang, 'reframe')}: {rec.get('reframe') or _empty(lang)}",
                "",
            ]
        )
    lines.extend([f"## {_t(lang, 'world_constraints')}", ""])
    if rec.get("world_id"):
        extra = f" / {rec['world_schema']}" if rec.get("world_schema") else ""
        lever = f" / {_t(lang, 'lever')}=`{rec['world_lever']}`" if rec.get("world_lever") else ""
        lines.append(
            f"- **{_t(lang, 'world')}**: `{rec.get('world_id')}`{extra}{lever} "
            f"({_t(lang, 'kind')}=`{_kind(rec)}`)"
        )
    elif rec.get("generated_world") or _kind(rec) == "GENERATED":
        lines.append(f"- **{_t(lang, 'world')}**: {_t(lang, 'world_generated')}")
    else:
        lines.append(f"- **{_t(lang, 'world')}**: {_t(lang, 'world_unbound')}")
    lines.extend(
        [
            f"- {_t(lang, 'constraint_sandbox')}",
            f"- {_t(lang, 'constraint_world')}",
            f"- {_t(lang, 'constraint_uninformative')}",
            f"- {_t(lang, 'constraint_direction')}",
            "",
        ]
    )
    lines.extend([f"## {_t(lang, 'experiment_design')}", ""])
    idea = idea_kind_of(rec)
    if idea == ACQUIRE:
        lines.extend(
            [
                _t(lang, "acquire_plan"),
                "",
                f"- {_t(lang, 'harvest_must_run')}",
                f"- {_t(lang, 'no_probe_on_freeze')}",
                "",
            ]
        )
    elif idea == THEORY:
        lines.extend([_t(lang, "theory_plan"), ""])
    elif idea == QUESTION:
        lines.extend([_t(lang, "question_plan"), ""])
    if rec.get("experiment"):
        lines.extend(
            [
                f"- [x] {_t(lang, 'registered_two_arm')}",
                "",
                f"  - {_t(lang, 'treatment')}：{rec.get('treatment_arm') or 'n/a'}"
                if lang_of(lang) == "zh"
                else f"  - {_t(lang, 'treatment')}: {rec.get('treatment_arm') or 'n/a'}",
                f"  - {_t(lang, 'control')}：{rec.get('control_arm') or 'n/a'}"
                if lang_of(lang) == "zh"
                else f"  - {_t(lang, 'control')}: {rec.get('control_arm') or 'n/a'}",
            ]
        )
        if rec.get("expected_direction"):
            sep = "：" if lang_of(lang) == "zh" else ": "
            lines.append(f"  - {_t(lang, 'expected')}{sep}{rec['expected_direction']}")
        lines.append("")
    else:
        lines.extend([_t(lang, "no_arms"), ""])
    lines.extend(_experiment_plan_body(rec, lang=lang))
    lines.append("")
    lines.extend([f"## {_t(lang, 'domain')}", ""])
    if derivation:
        lines.extend([_t(lang, "domain_note"), "", derivation, ""])
    else:
        lines.extend([_t(lang, "domain_empty"), ""])
    if rec.get("alternative"):
        lines.extend([f"**{_t(lang, 'competing')}.**", "", str(rec["alternative"]), ""])
    else:
        lines.extend([f"**{_t(lang, 'competing')}.** {_t(lang, 'competing_empty')}", ""])
    if rec.get("objection"):
        lines.extend([f"**{_t(lang, 'kill')}.**", "", str(rec["objection"]), ""])
    lines.extend([f"## {_t(lang, 'nongoals')}", ""])
    lines.extend(
        [
            f"- {_t(lang, 'nongoal_surrogate')}",
            f"- {_t(lang, 'nongoal_sibling')}",
            f"- {_t(lang, 'nongoal_direction')}",
            f"- {_t(lang, 'nongoal_generated')}",
            f"- {_t(lang, 'nongoal_worldsim')}",
            "",
        ]
    )
    if rec.get("dead_end"):
        lines.extend([f"- {_t(lang, 'dead_path')}: {rec['dead_end']}", ""])
    lines.extend([f"## {_t(lang, 'results')}", ""])
    analysis = rec.get("idea_analysis")
    if isinstance(analysis, dict) and str(analysis.get("summary") or "").strip():
        lines.extend(
            [f"**{_t(lang, 'analysis')}.**", "", str(analysis["summary"]), ""]
        )
        stop = str(analysis.get("stop_reason") or "").strip()
        if stop:
            lines.append(f"- {_t(lang, 'stop_reason')}: `{stop}`")
            lines.append("")
    if rec.get("verdict"):
        if _is_world_sim(rec):
            lines.extend([f"**{_t(lang, 'world_sim_note')}**", ""])
        else:
            lines.extend(
                [
                    f"**{_t(lang, 'cheap_probe')}** `{rec['verdict']}` ({_kind(rec)}).",
                    "",
                ]
            )
        if rec.get("verdict") == "uninformative":
            lines.extend([_t(lang, "uninformative_note"), ""])
        if rec.get("verdict") == "weakens":
            lines.extend([_t(lang, "weakens_note"), ""])
    else:
        lines.extend([_t(lang, "no_probe"), ""])
    if rec.get("failed_experiments"):
        failed = [
            str(item).strip()
            for item in rec.get("failed_experiments") or []
            if str(item or "").strip()
        ][:4]
        if failed:
            joiner = "；" if lang_of(lang) == "zh" else "; "
            lines.extend([f"{_t(lang, 'discarded')}: {joiner.join(failed)}", ""])
    lines.extend([f"## {_t(lang, 'next')}", ""])
    world = rec.get("idea_world") if isinstance(rec.get("idea_world"), dict) else {}
    objects = [str(item).strip() for item in (world.get("objects") or []) if str(item or "").strip()]
    missing = [str(item).strip() for item in (world.get("missing") or []) if str(item or "").strip()]
    if objects:
        lines.append(f"- **{_t(lang, 'must_measure')}**: " + "; ".join(objects[:8]))
    if missing:
        lines.append(f"- **{_t(lang, 'missing_now')}**: " + "; ".join(missing[:8]))
    if str(world.get("iterate") or "").strip():
        lines.append(f"- **{_t(lang, 'iterate_same')}**: {world['iterate']}")
    if str(world.get("do_not_repeat") or "").strip():
        lines.append(f"- **{_t(lang, 'do_not_repeat')}**: {world['do_not_repeat']}")
    if objects or missing or world.get("iterate") or world.get("do_not_repeat"):
        lines.append("")
    lines.extend([f"**{_t(lang, 'execute_steps')}.**", ""])
    for i, step in enumerate(_week_steps(rec, lang=lang), start=1):
        lines.append(f"{i}. {step}")
    lines.extend(
        [
            "",
            f"## {_t(lang, 'honesty')}",
            "",
            f"- {_t(lang, 'not_a_paper')}",
            f"- {_t(lang, 'honesty_summary')}",
            "",
        ]
    )
    return "\n".join(lines)


def render_idea_agent_packet(
    topic: str,
    rec: dict[str, Any],
    *,
    sibling_ids: list[str] | None = None,
    workspace: str | None = None,
) -> str:
    """Execute-order for this idea. The plan is already compiled."""
    card_id = str(rec.get("card_id") or "untitled")
    name = idea_folder_name(rec)
    aligned, other = _split_papers(rec, topic)
    pair = _pair_text(rec)
    lines = [
        f"# AGENT_PACKET `{name}`",
        "",
        "audience: execute",
        "not_a_paper: true",
        "isolation: this idea only",
        "",
        "## Scope",
        "",
        f"- idea: `{name}`",
        f"- idea_kind: `{idea_kind_of(rec)}`",
        f"- pair: {pair or 'n/a'}",
        f"- topic: {topic}",
        f"- protocol_dir: `candidates/{card_id}/`",
        "",
        "## Do not",
        "",
        "- Do not open sibling folders under `ideas/`.",
        "- Do not paste this H, claim, or papers into another pair.",
        "- Do not invent citations or wall-clock fields the freeze does not store.",
        "- Do not treat SYNTHETIC / GENERATED / WORLD_SIM numbers as corroborated.",
        "- Do not paste `ideas/<name>/world-sim/` imagined metrics into RESULTS.",
        "- Do not chase a support after weakens.",
        "- Do not cite off-object papers as controls.",
        "- Do not start a new `farfield research` to refine this idea.",
        "",
    ]
    lines.extend(
        [
            "## Execute (this card)",
            "",
            (
                "This card is `acquire`. Do not run a two-arm on this freeze. "
                "Read `refine-logs/EXPERIMENT_PLAN.md`. Harvest / import / derive "
                "a world with the named object_properties, then `farfield freeze`. "
                "Do not treat the current freeze as able to score freeze_validators "
                "or replay_gate."
                if idea_kind_of(rec) == ACQUIRE
                else "This card is `theory`. Do not run a two-arm. Lock the derivation "
                "to registered quantities. It cannot climb."
                if idea_kind_of(rec) == THEORY
                else "Read `refine-logs/EXPERIMENT_PLAN.md`. Run M0 then M1. "
                "Then `farfield execute` the protocol in `candidates/"
                f"{card_id}/`. "
                "Read `refine-logs/EXPERIMENT_RESULTS.md` before writing a discovery sentence. "
                "`world-sim/` is diagnostic rehearsal already finished this mission. "
                "Do not open a sibling idea. Do not rebind a different world family."
            ),
            "",
            "## Claim (unverified model text)",
            "",
            str(rec.get("claim") or ""),
            "",
            "## Mechanism (unverified model text)",
            "",
            str(rec.get("mechanism") or ""),
            "",
            "## Run order",
            "",
        ]
    )
    for i, step in enumerate(_week_steps(rec), start=1):
        lines.append(f"{i}. {step}")
    lines.append("")
    if rec.get("experiment"):
        lines.extend(
            [
                "## Registered experiment",
                "",
                f"- experiment: {rec.get('experiment')}",
                f"- treatment: {rec.get('treatment_arm') or 'n/a'}",
                f"- control: {rec.get('control_arm') or 'n/a'}",
                f"- expected_direction: {rec.get('expected_direction') or 'n/a'}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Registered experiment",
                "",
                "none. Register a two-arm on an attested freeze before probing.",
                "Both arms must share one measure. The claim-named field must be read on the attested object.",
                "",
            ]
        )
    lines.extend(["## Papers you may use as object-aligned controls", ""])
    if aligned:
        for work in aligned:
            row = _paper_row(work)
            lines.append(f"- [{row['cite_id']}] {row['title']}")
        lines.append("")
    else:
        lines.extend(["- none. Re-harvest with topic object phrases.", ""])
    if other:
        lines.extend(["## Papers you must not use as related work", ""])
        for work in other:
            row = _paper_row(work)
            lines.append(f"- [{row['cite_id'] or '?'}] {row['title']}")
        lines.append("")
    world_line = rec.get("world_id") or (
        "GENERATED" if rec.get("generated_world") or _kind(rec) == "GENERATED" else "none"
    )
    lines.extend(
        [
            "## World",
            "",
            f"- bound: `{world_line}`",
            f"- schema: `{rec.get('world_schema') or ''}`",
            f"- probe_kind: `{_kind(rec)}`",
            f"- verdict: `{rec.get('verdict') or 'none'}`",
            "",
            "## Files",
            "",
        ]
    )
    if workspace:
        lines.extend(
            [
                f"- 研究简报: `{idea_packet_dir(workspace, rec)}/idea-stage/RESEARCH_BRIEF.md`",
                f"- 实验计划: `{idea_packet_dir(workspace, rec)}/refine-logs/EXPERIMENT_PLAN.md`",
                f"- protocol: `{workspace}/candidates/{card_id}/protocol.md`",
                f"- papers: `{workspace}/candidates/{card_id}/papers.json`",
                "",
            ]
        )
    else:
        name = idea_folder_name(rec)
        lines.extend(
            [
                f"- 研究简报: `ideas/{name}/idea-stage/RESEARCH_BRIEF.md`",
                f"- 实验计划: `ideas/{name}/refine-logs/EXPERIMENT_PLAN.md`",
                f"- protocol: `candidates/{card_id}/protocol.md`",
                "",
            ]
        )
    lines.extend(
        [
            "## Success / fail",
            "",
            f"- success: {rec.get('prediction') or 'pre-registered direction on an attested WORLD'}",
            f"- fail_if: {rec.get('risks') or rec.get('objection') or 'n/a'}",
            "",
        ]
    )
    return "\n".join(lines)


def render_idea_report(
    topic: str,
    *,
    found: list[dict[str, Any]] | None = None,
    ranked: list[dict[str, Any]] | None = None,
    kills: list[dict[str, Any]] | None = None,
    wiki: list[dict[str, Any]] | None = None,
) -> str:
    """Mission-level idea landscape. Not a paper and not Dream scoring."""
    from .value import idea_usefulness
    from .wiki import gap_lines

    live, buckets = live_idea_records(ranked=ranked, found=found)
    rows = sorted(
        list(found or []),
        key=lambda row: (-int(row.get("idea_usefulness") or idea_usefulness(row)), str(row.get("card_id") or "")),
    )
    lines = [
        f"# Idea 景观：{topic}",
        "",
        "本场探索索引。给人接着打磨同一对象，不是下一场换题 generate，也不是四维打分门。",
        "`idea_usefulness` 只是过程读数：越高越值得在同一 pair 上 iterate。",
        "",
        "## 活线",
        "",
    ]
    if live:
        for rec in live:
            pair = rec.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else "(pair missing)"
            score = int(rec.get("idea_usefulness") or idea_usefulness(rec))
            lines.append(f"- `{combo}` — usefulness {score}")
            if rec.get("claim"):
                lines.append(f"  主张：{rec['claim']}")
            if rec.get("verdict"):
                lines.append(f"  探针：{rec.get('verdict')} / {rec.get('probe_kind') or 'none'}")
            world = rec.get("idea_world") if isinstance(rec.get("idea_world"), dict) else {}
            if world.get("iterate"):
                lines.append(f"  下一步：{world['iterate']}")
            folder = rec.get("idea_name")
            if folder:
                lines.append(f"  文件夹：`ideas/{folder}/`")
            lines.append("")
    else:
        lines.extend(["本场没有留下可继续打磨的进场课题。", ""])

    lines.extend(["## 关掉的路", ""])
    closed = list(buckets.get("weakened") or []) + list(buckets.get("closed_by_prior") or [])
    if closed or kills:
        for rec in closed:
            pair = rec.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else (rec.get("claim") or "draft")
            why = rec.get("why_failed") or rec.get("verdict") or "stopped"
            lines.append(f"- {combo} — {why}")
        for row in list(kills or [])[:8]:
            pair = row.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else "(pair missing)"
            reasons = ", ".join(str(item) for item in (row.get("killed_by") or [])[:4]) or "unspecified"
            extra = str(row.get("why") or row.get("why_failed") or "").strip()
            line = f"- {combo} — `{reasons}`"
            if extra:
                line += f"：{extra}"
            lines.append(line)
        lines.append("")
    else:
        lines.extend(["没有记下的失败路径。", ""])

    lines.extend(["## 文献缺口", ""])
    gaps = list(gap_lines(wiki or []))
    if gaps:
        for gap in gaps[:8]:
            lines.append(f"- {gap}")
        lines.append("")
    else:
        lines.extend(["本场 wiki 还没有 limitation 句子。", ""])

    if rows:
        lines.extend(["## 本场全部着陆（按 usefulness）", ""])
        for rec in rows[:12]:
            pair = rec.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else str(rec.get("card_id") or "")
            score = int(rec.get("idea_usefulness") or idea_usefulness(rec))
            lines.append(
                f"- {score} `{combo}` {rec.get('verdict') or rec.get('outcome') or 'draft'}"
            )
        lines.append("")
    return "\n".join(lines)


def compile_live_idea(
    workspace: Path,
    topic: str,
    rec: dict[str, Any],
    *,
    sibling_ids: list[str] | None = None,
) -> Path:
    """Write one idea's brief, plan, and execute order. Compiler owns the files."""
    from .workspace import write_idea_packets

    card_id = str(rec.get("card_id") or "untitled")
    name = str(rec.get("idea_name") or "") or idea_folder_name(rec)
    rec = {**rec, "idea_name": name}
    ws = str(workspace)
    return write_idea_packets(
        workspace,
        card_id=card_id,
        folder_name=name,
        human_md=render_idea_human_packet(
            topic, rec, sibling_ids=sibling_ids, workspace=ws, lang="zh"
        ),
        human_md_en=render_idea_human_packet(
            topic, rec, sibling_ids=sibling_ids, workspace=ws, lang="en"
        ),
        agent_md=render_idea_agent_packet(
            topic, rec, sibling_ids=sibling_ids, workspace=ws
        ),
        experiment_md=render_experiment_plan(topic, rec, workspace=ws, lang="zh"),
        experiment_md_en=render_experiment_plan(topic, rec, workspace=ws, lang="en"),
        derivation_md=render_derivation_note(rec),
        result_md=render_result_to_claim(rec),
        ablation_md=render_ablation_plan(rec),
        kill_md=render_kill_argument(rec),
        analysis_md=render_analyze_results(rec),
    )


def render_mission_packet(
    topic: str,
    *,
    ranked: list[dict[str, Any]] | None = None,
    found: list[dict[str, Any]] | None = None,
    workspace: str | None = None,
    program: dict[str, Any] | None = None,
    funnel: dict[str, Any] | None = None,
    kills: list[dict[str, Any]] | None = None,
    questions: dict[str, Any] | None = None,
    spent: dict[str, Any] | None = None,
    knowledge_until: str | None = None,
    cards_generated: int | None = None,
    entered: int | None = None,
    lang: str = "zh",
) -> str:
    """Complete research plan for the mission. Execute still isolates by AGENT_PACKET.

    FarField internals (H, funnel, next_card_must) stay in summary.json.
    """
    del program, funnel, questions, cards_generated, entered
    live, buckets = live_idea_records(ranked=ranked, found=found)
    colon = "：" if lang_of(lang) == "zh" else ": "
    lines = [
        f"# {_t(lang, 'packet_title')}{colon}{topic}",
        "",
        _t(lang, "packet_preamble"),
        "",
        _t(lang, "not_a_paper"),
        "",
        _t(lang, "landscape_note"),
        "",
        _t(lang, "packet_twin_en") if lang_of(lang) == "zh" else _t(lang, "packet_twin_zh"),
        "",
    ]
    if workspace:
        ws_label = "工作区" if lang_of(lang) == "zh" else "Workspace"
        lines.extend([f"{ws_label}: `{workspace}`", ""])

    lines.extend([f"## {_t(lang, 'recommend')}", ""])
    if live:
        taken: set[str] = set()
        for rec in live:
            kind = _kind(rec)
            tag = ""
            if rec.get("generated_world") or kind == "GENERATED":
                tag = _t(lang, "generated_tag")
            elif rec.get("world_incompatible"):
                tag = _t(lang, "incompatible_tag")
            lines.append(f"### {_heading(rec, lang=lang)}{tag}")
            lines.append("")
            if packet_worthy(rec):
                name = idea_folder_name(rec, taken=taken)
                rec["idea_name"] = name
                folder = (
                    idea_packet_dir(workspace, rec)
                    if workspace
                    else f"ideas/{name}"
                )
                body = render_idea_human_packet(
                    topic, rec, workspace=workspace, lang=lang, role="index"
                )
                lines.append(body.rstrip())
                lines.append("")
                card_id = str(rec.get("card_id") or "")
                ident = f" (`{card_id}`)" if card_id else ""
                lines.append(f"**{_t(lang, 'folder_copy')}.** `{folder}/`{ident}")
                lines.append("")
            elif rec.get("mechanism") or rec.get("claim"):
                if rec.get("claim"):
                    lines.extend([f"**{_t(lang, 'claim')}.**", "", str(rec["claim"]), ""])
                if rec.get("mechanism"):
                    lines.extend(
                        [
                            f"**{_t(lang, 'mechanism_draft')}.**",
                            "",
                            str(rec["mechanism"]),
                            "",
                        ]
                    )
                if rec.get("prediction"):
                    lines.extend(
                        [f"**{_t(lang, 'prediction')}.**", "", str(rec["prediction"]), ""]
                    )
                lines.append(f"- {_t(lang, 'no_ideas_folder')}")
                lines.append("")
            else:
                lines.append("")
    else:
        lines.extend([_t(lang, "no_live"), ""])

    lines.append(f"## {_t(lang, 'read_first')}")
    lines.append("")
    if buckets["closed_by_prior"]:
        for rec in buckets["closed_by_prior"]:
            lines.append(f"- {rec.get('claim') or _heading(rec, lang=lang)}")
            if rec.get("card_id"):
                lines.append(f"  {_t(lang, 'open_papers')}")
        lines.append("")
    else:
        lines.extend([_t(lang, "no_prior_close"), ""])

    lines.append(f"## {_t(lang, 'do_not_invest')}")
    lines.append("")
    if buckets["weakened"]:
        for rec in buckets["weakened"]:
            lines.append(f"- {rec.get('claim') or _heading(rec, lang=lang)}")
            if rec.get("verdict") == "weakens":
                lines.append(f"  {_t(lang, 'weakened_line')}")
            else:
                lines.append(f"  {_t(lang, 'unrunnable_line')}")
        lines.append("")
    else:
        lines.extend([_t(lang, "no_weakened"), ""])
    if kills:
        lines.append(_t(lang, "gate_kills"))
        for row in kills[:8]:
            pair = row.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else "(pair missing)"
            reasons = ", ".join(str(item) for item in (row.get("killed_by") or [])[:4]) or "unspecified"
            lines.append(f"- {combo} — `{reasons}`")
        lines.append("")

    if knowledge_until or spent:
        lines.extend([f"## {_t(lang, 'scope')}", ""])
        if knowledge_until:
            if lang_of(lang) == "en":
                lines.append(f"- {_t(lang, 'novelty_until')}{knowledge_until}{_t(lang, 'before_lit')}")
            else:
                lines.append(f"- {_t(lang, 'novelty_until')}{knowledge_until}{_t(lang, 'before_lit')}")
        if spent:
            calls = spent.get("api_calls")
            tokens = spent.get("tokens")
            if calls is not None:
                extra = f", ~{tokens} tokens" if tokens is not None else ""
                if lang_of(lang) == "zh":
                    extra = f"，约 {tokens} tokens" if tokens is not None else ""
                    lines.append(f"- {_t(lang, 'model_calls')}{calls} {_t(lang, 'times')}{extra}。")
                else:
                    lines.append(f"- {_t(lang, 'model_calls')}{calls}{extra}.")
        lines.append("")

    lines.extend(
        [
            f"## {_t(lang, 'honesty')}",
            "",
            f"- {_t(lang, 'honesty_not_paper')}",
            f"- {_t(lang, 'honesty_ladder')}",
            f"- {_t(lang, 'honesty_generated_world')}",
            f"- {_t(lang, 'honesty_execute')}",
            f"- {_t(lang, 'honesty_metrics')}",
            f"- {_t(lang, 'honesty_summary')}",
            "",
            f"## {_t(lang, 'glossary')}",
            "",
            f"- {_t(lang, 'gloss_arms')}",
            f"- {_t(lang, 'gloss_verdict')}",
            f"- {_t(lang, 'gloss_kind')}",
            f"- {_t(lang, 'gloss_protocol')}",
            "",
        ]
    )
    return "\n".join(lines)


def rec_from_candidate_folder(folder: Path, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    folder = Path(folder)
    brief: dict[str, Any] = {}
    brief_path = folder / "brief.json"
    if brief_path.is_file():
        loaded = json.loads(brief_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            brief = loaded
    papers: list[Any] = []
    papers_path = folder / "papers.json"
    if papers_path.is_file():
        loaded = json.loads(papers_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            papers = loaded
    protocol: dict[str, Any] = {}
    protocol_path = folder / "protocol.json"
    if protocol_path.is_file():
        loaded = json.loads(protocol_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            protocol = loaded
    rec: dict[str, Any] = {
        "card_id": str(brief.get("card_id") or folder.name),
        "title": str(brief.get("title") or ""),
        "plain_title": str(brief.get("plain_title") or ""),
        "one_liner": str(brief.get("one_liner") or ""),
        "why_it_matters": str(brief.get("why_it_matters") or ""),
        "gap": str(brief.get("gap") or ""),
        "idea": str(brief.get("idea") or ""),
        "approach": str(brief.get("approach") or ""),
        "first_steps": list(brief.get("first_steps") or []),
        "baseline": str(brief.get("baseline") or ""),
        "risks": str(brief.get("risks") or ""),
        "has_brief": True,
        "works": papers,
        "claim": str(protocol.get("claim") or brief.get("idea") or ""),
        "mechanism": str(protocol.get("mechanism") or brief.get("approach") or ""),
        "prediction": str(protocol.get("prediction") or ""),
        "pair": list(protocol.get("pair") or []),
        "dead_end": str(protocol.get("dead_end") or ""),
        "why_failed": str(protocol.get("why_failed") or ""),
        "reframe": str(protocol.get("reframe") or ""),
        "objection": str(protocol.get("objection") or ""),
        "derivation": str(protocol.get("derivation") or ""),
        "alternative": str((protocol.get("diagnosis") or {}).get("alternative") or ""),
        "experiment": str((protocol.get("diagnosis") or {}).get("experiment") or ""),
        "treatment_arm": str((protocol.get("diagnosis") or {}).get("treatment_arm") or ""),
        "control_arm": str((protocol.get("diagnosis") or {}).get("control_arm") or ""),
        "expected_direction": str(
            (protocol.get("diagnosis") or {}).get("expected_direction") or ""
        ),
        "world_id": str((protocol.get("world") or {}).get("id") or ""),
        "world_schema": str((protocol.get("world") or {}).get("schema") or ""),
        "world_lever": str(
            protocol.get("world_lever")
            or (protocol.get("diagnosis") or {}).get("world_lever")
            or ""
        ),
        "mechanism_flag": str(
            (protocol.get("diagnosis") or {}).get("mechanism_flag") or ""
        ),
        "probe_kind": str((protocol.get("cheap_probe") or {}).get("kind") or "SYNTHETIC"),
        "verdict": (protocol.get("cheap_probe") or {}).get("verdict"),
    }
    if extra:
        rec.update({k: v for k, v in extra.items() if v not in (None, "", [], {})})
    return rec


def recompile_workspace_packets(workspace: Path, *, topic: str = "") -> list[Path]:
    """Rewrite isolated idea packets from candidate trees. No LLM."""
    dest = Path(workspace)
    topic_text = str(topic or "").strip()
    topic_path = dest / "topic.txt"
    if not topic_text and topic_path.is_file():
        topic_text = topic_path.read_text(encoding="utf-8").strip()
    extras: dict[str, dict[str, Any]] = {}
    log_path = dest / "research.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("stage") != "verdict":
                continue
            card_id = str(event.get("card_id") or "")
            if not card_id:
                continue
            extras[card_id] = {
                "dead_end": event.get("dead_end"),
                "why_failed": event.get("why_failed"),
                "reframe": event.get("reframe"),
                "objection": event.get("objection"),
                "derivation": event.get("derivation"),
                "claim": event.get("claim"),
                "mechanism": event.get("mechanism"),
                "prediction": event.get("prediction"),
                "pair": event.get("pair"),
            }
    candidates = dest / "candidates"
    recs: list[dict[str, Any]] = []
    if candidates.is_dir():
        for folder in sorted(candidates.iterdir()):
            if not folder.is_dir() or not (folder / "brief.json").is_file():
                continue
            recs.append(
                rec_from_candidate_folder(folder, extra=extras.get(folder.name))
            )
    sibling_ids = [str(row.get("card_id") or "") for row in recs]
    written: list[Path] = []
    from .workspace import prune_idea_packets, write_summary

    live, _buckets = live_packet_records(found=recs)
    taken: set[str] = set()
    for rec in live:
        card_id = str(rec.get("card_id") or "")
        if not card_id:
            continue
        name = idea_folder_name(rec, taken=taken)
        rec["idea_name"] = name
        written.append(
            compile_live_idea(
                dest, topic_text, rec, sibling_ids=sibling_ids
            )
        )
    prune_idea_packets(dest, keep={path.name for path in written})
    index = render_mission_packet(
        topic_text,
        found=recs,
        workspace=str(dest),
        lang="zh",
    )
    index_en = render_mission_packet(
        topic_text,
        found=recs,
        workspace=str(dest),
        lang="en",
    )
    summary: dict[str, Any] = {}
    summary_path = dest / "summary.json"
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            summary = loaded
    summary["idea_packets"] = [str(path) for path in written]
    write_summary(
        dest, topic_text, summary, packet_md=index, packet_md_en=index_en
    )
    return written
