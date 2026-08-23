"""Compile an executable research packet from fields the mission already attested.

This is the finish line, not another generation. Sakana's AI Scientist ships a
PDF that looks like a paper; Google Co-Scientist ships a protocol a lab can
run next week. FarField takes the second shape: a pre-registered experiment
pack whose numbers, paper ids, and arms are copied from the evidence pipeline.
The mission packet is a human research plan. Runtime H stays in summary.json.

The cheap probe is labelled SYNTHETIC. Ladder words (`supports`, `verified`)
are not rewritten as scientific completion. Citations may only name papers
retrieved this mission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .brief import ResearchBrief
from .generate import GeneratedCard
from .hostexp import PROTOCOL_EXECUTED, HOST_TIMEOUT_SECONDS
from .probeexp import resolve_tier
from .livefeed import FreshWork


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
    return str(data.get("arxiv_id") or data.get("cite_id") or data.get("id") or "")


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
        "title": brief.title,
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
    md: list[str] = [
        f"# Research plan: {brief.title}",
        "",
        NOT_A_PAPER,
        "",
        f"*Topic.* {topic}",
        "",
        f"*Combination.* {card.pair[0]} × {card.pair[1]}",
        "",
        "## 1. Idea",
        "",
        card.claim,
        "",
    ]
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

    readme = "\n".join(
        [
            f"# {brief.title}",
            "",
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


def _heading(rec: dict[str, Any]) -> str:
    title = str(rec.get("title") or "").strip()
    if title:
        return title
    claim = " ".join(str(rec.get("claim") or "").split())
    if len(claim) > 88:
        return claim[:85].rstrip() + "…"
    return claim or str(rec.get("card_id") or "untitled")


def _pair_text(rec: dict[str, Any]) -> str:
    pair = rec.get("pair") or []
    if len(pair) >= 2:
        return f"{pair[0]} × {pair[1]}"
    return ""


def _kind(rec: dict[str, Any]) -> str:
    return str(rec.get("probe_kind") or "SYNTHETIC").upper()


def _week_steps(rec: dict[str, Any]) -> list[str]:
    steps = [str(item).strip() for item in (rec.get("first_steps") or []) if str(item or "").strip()]
    if steps:
        return steps
    built: list[str] = []
    if rec.get("experiment"):
        built.append(str(rec["experiment"]))
    treatment = str(rec.get("treatment_arm") or "").strip()
    control = str(rec.get("control_arm") or "").strip()
    if treatment:
        built.append(f"处理臂：{treatment}")
    if control:
        built.append(f"对照臂：{control}")
    if rec.get("generated_world") or _kind(rec) == "GENERATED":
        built.append(
            "便宜探针跑在本轮构造世界上，不能当发现。"
            "下周在匹配 schema 的 freeze 或真实数据上按同一预登记重跑。"
        )
    elif rec.get("verdict") == "supports" and _kind(rec) in {"WORLD", "REAL", "FIXTURE"}:
        built.append("在 attested 母本上按同一 EvidenceID 跑宿主协议。")
    if rec.get("prediction"):
        built.append(f"成功标准：{rec['prediction']}")
    if rec.get("verdict") == "uninformative":
        built.append("便宜探针没分开两臂。不要改主张。换机制开关、度量或竞争解释后再测。")
    if rec.get("verdict") == "weakens":
        built.append("停止投入。弱化后改实验追 support 是 p-hacking。")
    if not built and rec.get("claim"):
        built.append(f"预登记一个只差机制开关的两臂实验，用来检验：{rec['claim']}")
        if rec.get("prediction"):
            built.append(f"预测：{rec['prediction']}")
    built.append("不要把 metrics.json 或 20 秒探针写进论文结果。")
    return built


def _honesty_line(rec: dict[str, Any]) -> str:
    kind = _kind(rec)
    verdict = rec.get("verdict")
    if rec.get("prior_kills"):
        return "主张已见于本场检索。先读那篇，不要再做同一主张。"
    if verdict == "weakens":
        return "两臂按预登记方向削弱了机制。"
    bits: list[str] = []
    if rec.get("generated_world") or kind == "GENERATED":
        bits.append("世界是本轮构造的 GENERATED，可以削弱，不能佐证")
    elif rec.get("world_id"):
        bits.append(f"绑定世界 `{rec.get('world_id')}`")
    if verdict == "supports":
        if kind in {"WORLD", "REAL", "FIXTURE"}:
            bits.append("WORLD 探针支持，待宿主按同一 EvidenceID 确认")
        else:
            bits.append(f"{kind} 探针支持（自洽，不是发现）")
    elif verdict == "uninformative":
        bits.append("两臂没分开，实验设计需要更锐，主张仍开着")
    elif rec.get("experiment"):
        bits.append("已预登记实验")
    return "；".join(bits) + "。" if bits else "下面的字段均来自本场 attested 记录。"


def _plan_body(rec: dict[str, Any], *, workspace: str | None) -> list[str]:
    lines: list[str] = []
    pair = _pair_text(rec)
    if pair:
        lines.append(f"**概念对：** {pair}")
        lines.append("")
    if rec.get("claim"):
        lines.extend(["**主张**", "", str(rec["claim"]), ""])
    if rec.get("mechanism"):
        lines.extend(["**机制（未验证的理论草稿）**", "", str(rec["mechanism"]), ""])
    if rec.get("prediction"):
        lines.extend(["**若机制为真，应看到**", "", str(rec["prediction"]), ""])
    if rec.get("alternative"):
        lines.extend(["**竞争解释**", "", str(rec["alternative"]), ""])
    if rec.get("experiment"):
        lines.extend(["**预登记实验**", "", str(rec["experiment"]), ""])
    if rec.get("treatment_arm") or rec.get("control_arm"):
        lines.append(
            f"- 处理臂：{rec.get('treatment_arm') or 'n/a'}"
        )
        lines.append(
            f"- 对照臂：{rec.get('control_arm') or 'n/a'}"
        )
        if rec.get("expected_direction"):
            lines.append(f"- 预登记方向：{rec['expected_direction']}")
        lines.append("")
    if rec.get("generated_world") or _kind(rec) == "GENERATED":
        schema = rec.get("world_schema") or "symbolic_trace"
        lines.extend(
            [
                f"**世界：** GENERATED `{schema}`。不是 freeze。不能佐证。",
                "",
            ]
        )
    elif rec.get("world_id"):
        extra = f" / {rec['world_schema']}" if rec.get("world_schema") else ""
        lines.extend([f"**世界：** `{rec.get('world_id')}`{extra}", ""])
    if rec.get("verdict"):
        lines.append(f"**便宜探针：** `{rec['verdict']}`（{_kind(rec)}）")
        lines.append("")
    failed = [
        str(item).strip()
        for item in (rec.get("failed_experiments") or [])
        if str(item or "").strip()
    ]
    if failed:
        lines.append("**已放弃的设计（不要重复）：** " + "；".join(failed[:4]))
        lines.append("")
    lines.append(_honesty_line(rec))
    lines.append("")
    lines.append("**下周步骤**")
    lines.append("")
    for i, step in enumerate(_week_steps(rec), start=1):
        lines.append(f"{i}. {step}")
    lines.append("")
    card_id = str(rec.get("card_id") or "")
    if card_id and rec.get("has_brief") and workspace:
        lines.append(f"分卡 runbook：`candidates/{card_id}/protocol.md`")
        lines.append("")
    elif card_id and rec.get("has_brief"):
        lines.append(f"分卡 runbook：`{card_id}/protocol.md`")
        lines.append("")
    return lines


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
) -> str:
    """Human research plan compiled from attested fields. Not a paper.

    FarField internals (H, funnel, next_card_must) stay in summary.json.
    """
    del program, funnel, questions, cards_generated, entered
    found_rows = list(found or [])
    by_id = {str(row.get("card_id") or ""): row for row in found_rows}
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
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
        buckets[_bucket(merged)].append((item, merged))

    live = buckets["do_next"] + buckets["still_open"]
    lines = [
        f"# 研究方案：{topic}",
        "",
        "给人下周开工。只编译本场已经 attested 的字段，不再调用模型。这不是论文。",
        "",
    ]
    if workspace:
        lines.extend([f"工作区：`{workspace}`", ""])

    lines.extend(["## 推荐课题", ""])
    if live:
        rec = live[0][1]
        lines.append(f"### {_heading(rec)}")
        lines.append("")
        lines.extend(_plan_body(rec, workspace=workspace))
        extras = live[1:]
        if extras:
            lines.extend(["## 其它可做的线", ""])
            for item, row in extras:
                lines.append(f"### {_heading(row)}")
                lines.append("")
                lines.extend(_plan_body(row, workspace=workspace))
    else:
        lines.extend(
            [
                "本场没有留下可执行的进场课题。下面「不要走的路」记录了门控或证据关掉的草稿，避免重复同一失败。",
                "",
            ]
        )

    lines.append("## 先读文献（主张已见于本次检索）")
    lines.append("")
    if buckets["closed_by_prior"]:
        for _item, rec in buckets["closed_by_prior"]:
            lines.append(f"- {rec.get('claim') or _heading(rec)}")
            if rec.get("card_id"):
                lines.append(f"  打开本场 papers.json，不要继续这条主张。")
        lines.append("")
    else:
        lines.extend(["没有因先行文献关闭的卡。", ""])

    lines.append("## 不要继续投入")
    lines.append("")
    if buckets["weakened"]:
        for _item, rec in buckets["weakened"]:
            lines.append(f"- {rec.get('claim') or _heading(rec)}")
            lines.append("  机制被削弱。不要改实验追赢。")
        lines.append("")
    else:
        lines.extend(["没有被探针削弱的卡。", ""])
    if kills:
        lines.append("门控未过、不要按原样再生成的草稿：")
        for row in kills[:8]:
            pair = row.get("pair") or []
            combo = f"{pair[0]} × {pair[1]}" if len(pair) >= 2 else "(pair missing)"
            reasons = ", ".join(str(item) for item in (row.get("killed_by") or [])[:4]) or "unspecified"
            lines.append(f"- {combo} — `{reasons}`")
        lines.append("")

    if knowledge_until or spent:
        lines.extend(["## 检索范围", ""])
        if knowledge_until:
            lines.append(f"- 新颖性截止：{knowledge_until} 之前的文献。")
        if spent:
            calls = spent.get("api_calls")
            tokens = spent.get("tokens")
            if calls is not None:
                lines.append(
                    f"- 本场模型调用：{calls} 次"
                    + (f"，约 {tokens} tokens" if tokens is not None else "")
                    + "。"
                )
        lines.append("")

    lines.extend(
        [
            "## 诚实性",
            "",
            "- 这不是会议论文或 workshop 投稿。",
            "- `supports` / `corroborated` / `verified` 量的是预登记方向在 attested 世界上是否成立，不是科学完成。",
            "- 构造数据或 GENERATED 世界上的两臂算术可以削弱，不能佐证。",
            "- `farfield execute` 记为 protocol_executed，不是发现已成立。",
            "- 不要把 `metrics.json` 写进论文结果。",
            "- 漏斗、门杀账本、运行时 H 在 `summary.json`。需要改方案细节时再调用 FarField，不要把本文件当下一跳提示词。",
            "",
        ]
    )
    return "\n".join(lines)
