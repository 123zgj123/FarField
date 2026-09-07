"""Idea rehearsal: forward-simulate a research path, not a freeze brand.

FarField already had two jobs under the word "world": attested fixture
bytes, and idea analysis. This module is the third layer — rehearsal.
The three-tier best/median/worst *role* is a design reference; the
product is not that foreign module (no session server, no branded CLI).

- Input is this idea's brief + experiment plan (framing + plan).
- Phase 1 defines best / median / worst with scientific invariants.
- Phase 2 elaborates imagined experiments keyed to plan_id.
- Writing dump is kind=WORLD_SIM; imagined numbers cannot climb.
- Review splits structural vs data-dependent issues across tiers.
- Assessment writes only this idea's H. It does not rewrite
  expected_direction and does not call execute.

The model never authors fixture dynamics here. Literature phrases are
looked up from the declared dynworld lever and freeze schema. Generation
still welds onto those lever names; rehearsal may not invent a handle.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dynworld import literature_phrases
from .livefeed import FreshWork
from .prior import content_tokens

KIND_WORLD_SIM = "WORLD_SIM"
RESULTS_STATUS_IMAGINED = "imagined"
TIERS = ("best", "median", "worst")
WORLD_SIM_FOLDER = "world-sim"

_CLAIM_HEADING = re.compile(r"^\*\*主张[。.]?\*\*\s*$", re.M)
_BLOCK_HEADING = re.compile(r"^###\s+实验块\s+(\d+)", re.M)
_PROBE_DIR = re.compile(r"(?:^|/)candidates/[^/]+/probe(?:/|$)")


class WorldSimError(ValueError):
    """Rehearsal refused. Fixture bytes and protocol.json are unchanged."""


@dataclass(frozen=True)
class WhatIfExperiment:
    name: str
    claim: str
    description: str
    estimated_gpu_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WhatIfExperiment":
        return cls(
            name=str(data.get("name") or ""),
            claim=str(data.get("claim") or ""),
            description=str(data.get("description") or ""),
            estimated_gpu_hours=float(data.get("estimated_gpu_hours") or 0.0),
        )


@dataclass
class WorldSimInput:
    idea_dir: Path
    what_if: list[WhatIfExperiment] = field(default_factory=list)
    review_mode: str = "simple"
    smart_skip: bool = False
    sim_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_dir": str(self.idea_dir),
            "what_if": [item.to_dict() for item in self.what_if],
            "review_mode": self.review_mode,
            "smart_skip": self.smart_skip,
            "sim_id": self.sim_id,
        }


@dataclass
class TierDefinition:
    name: str
    meaning: str
    key_assumptions: list[str]
    number_ranges: dict[str, list[float]]
    narrative_direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "meaning": self.meaning,
            "key_assumptions": list(self.key_assumptions),
            "number_ranges": {
                key: [float(pair[0]), float(pair[1])]
                for key, pair in self.number_ranges.items()
            },
            "narrative_direction": self.narrative_direction,
        }


@dataclass
class ImaginedExperiment:
    name: str
    plan_id: str
    description: str
    expected_results: dict[str, Any]
    result_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan_id": self.plan_id,
            "description": self.description,
            "expected_results": dict(self.expected_results),
            "result_rationale": self.result_rationale,
            "kind": KIND_WORLD_SIM,
            "results_status": RESULTS_STATUS_IMAGINED,
        }


@dataclass
class WorldSimContext:
    idea_dir: Path
    brief_text: str
    plan_text: str
    claim: str
    mechanism: str
    plan_ids: tuple[str, ...]
    wiki: list[FreshWork]
    what_if: list[WhatIfExperiment]
    levers: tuple[str, ...]
    schema: str
    scout: dict[str, Any]
    context_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "idea_dir": str(self.idea_dir),
            "claim": self.claim,
            "mechanism": self.mechanism,
            "plan_ids": list(self.plan_ids),
            "wiki": [work.cite_id() for work in self.wiki],
            "what_if": [item.to_dict() for item in self.what_if],
            "levers": list(self.levers),
            "schema": self.schema,
            "scout": dict(self.scout),
            "context_hash": self.context_hash,
        }


def is_world_sim_kind(kind: Any) -> bool:
    return str(kind or "").upper() == KIND_WORLD_SIM


def refuse_world_sim_as_evidence(payload: dict[str, Any] | None) -> str | None:
    """Named refusal when imagined numbers try to enter the ladder."""
    row = payload or {}
    if is_world_sim_kind(row.get("kind") or row.get("probe_kind") or row.get("results_status")):
        return "WORLD_SIM imagined numbers cannot enter judge_probe, RESULTS, or corroboration"
    if str(row.get("results_status") or "") == RESULTS_STATUS_IMAGINED:
        return "WORLD_SIM imagined numbers cannot enter judge_probe, RESULTS, or corroboration"
    return None


def paper_rehearses_idea(
    title: str,
    abstract: str,
    claim: str,
    mechanism: str = "",
    *,
    world_lever: str = "",
    competing_lever: str = "",
    schema: str = "",
) -> bool:
    """True when a verified paper names literature of this idea's declared lever.

    Drought-index / obesity papers still miss. A code-world-model paper
    may enter a labeled_traces truncate idea even if it never says the
    freeze brand. A graph rewire idea does not get that paper.
    """
    blob = f"{title} {abstract}".lower()
    if not blob.strip():
        return False
    phrases = literature_phrases(
        levers=tuple(
            item
            for item in (world_lever, competing_lever)
            if str(item).strip() and str(item).strip() != "none"
        ),
        schema=schema,
        claim=claim,
        mechanism=mechanism,
    )
    if not phrases:
        idea = content_tokens(f"{claim} {mechanism}")
        return bool(idea and idea & content_tokens(blob))
    if not any(phrase.lower() in blob for phrase in phrases):
        return False
    idea = content_tokens(f"{claim} {mechanism}")
    if not idea:
        return True
    family_tokens = content_tokens(" ".join(phrases))
    if idea & family_tokens:
        return True
    return bool(idea & content_tokens(blob))


def papers_for_rehearsal(
    works: list[Any] | tuple[Any, ...] | None,
    *,
    claim: str,
    mechanism: str = "",
    world_lever: str = "",
    competing_lever: str = "",
    schema: str = "",
) -> list[Any]:
    """Admit rows that rehearse the registered lever / schema, not a freeze brand."""
    kept: list[Any] = []
    for work in works or []:
        title, abstract = _title_abstract(work)
        if paper_rehearses_idea(
            title,
            abstract,
            claim,
            mechanism,
            world_lever=world_lever,
            competing_lever=competing_lever,
            schema=schema,
        ):
            kept.append(work)
    return kept


def world_sim_dir(idea_dir: Path, sim_id: str) -> Path:
    """Isolation: never under probe/ and never inside a fixture bind."""
    dest = Path(idea_dir) / WORLD_SIM_FOLDER / str(sim_id)
    _assert_not_probe(dest)
    _assert_not_fixture_bind(Path(idea_dir))
    return dest


def _declared_levers(
    scout: dict[str, Any] | None,
    world_lever: str,
) -> tuple[str, ...]:
    names: list[str] = []
    raw = (scout or {}).get("levers")
    if isinstance(raw, dict):
        names.extend(
            str(key)
            for key in raw
            if str(key).strip() and str(key).strip() != "none"
        )
    elif isinstance(raw, (list, tuple)):
        names.extend(
            str(key)
            for key in raw
            if str(key).strip() and str(key).strip() != "none"
        )
    registered = str(world_lever or "").strip()
    if registered and registered != "none" and registered not in names:
        names.insert(0, registered)
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)


def collect_context(
    idea_dir: Path,
    *,
    what_if: list[WhatIfExperiment] | tuple[WhatIfExperiment, ...] = (),
    wiki: list[FreshWork] | tuple[FreshWork, ...] = (),
    scout: dict[str, Any] | None = None,
    claim: str = "",
    mechanism: str = "",
    world_lever: str = "",
    competing_lever: str = "",
    schema: str = "",
) -> WorldSimContext:
    """Hard world-sim precondition: brief + plan must already exist."""
    root = Path(idea_dir)
    _assert_not_fixture_bind(root)
    brief_path = root / "idea-stage" / "RESEARCH_BRIEF.md"
    plan_path = root / "refine-logs" / "EXPERIMENT_PLAN.md"
    if not brief_path.is_file() or not plan_path.is_file():
        raise WorldSimError(
            "world_sim needs idea-stage/RESEARCH_BRIEF.md and "
            "refine-logs/EXPERIMENT_PLAN.md (framing + plan)"
        )
    brief_text = brief_path.read_text(encoding="utf-8")
    plan_text = plan_path.read_text(encoding="utf-8")
    claim_text = str(claim or "").strip() or _claim_from_brief(brief_text)
    mechanism_text = str(mechanism or "").strip() or _mechanism_from_brief(brief_text)
    schema_text = str(schema or "").strip() or str((scout or {}).get("schema") or "")
    lever_text = str(world_lever or "").strip()
    plan_ids = _plan_ids(plan_text)
    extras = list(what_if)
    wiki_kept = papers_for_rehearsal(
        list(wiki),
        claim=claim_text,
        mechanism=mechanism_text,
        world_lever=lever_text,
        competing_lever=competing_lever,
        schema=schema_text,
    )
    digest = context_hash(
        brief_text,
        plan_text,
        wiki=wiki_kept,
        what_if=extras,
    )
    return WorldSimContext(
        idea_dir=root,
        brief_text=brief_text,
        plan_text=plan_text,
        claim=claim_text,
        mechanism=mechanism_text,
        plan_ids=plan_ids,
        wiki=list(wiki_kept),
        what_if=extras,
        levers=_declared_levers(scout, lever_text),
        schema=schema_text,
        scout=dict(scout or {}),
        context_hash=digest,
    )


def context_hash(
    brief_text: str,
    plan_text: str,
    *,
    wiki: list[FreshWork] | tuple[FreshWork, ...] = (),
    what_if: list[WhatIfExperiment] | tuple[WhatIfExperiment, ...] = (),
) -> str:
    cites = sorted(work.cite_id() for work in wiki if work.cite_id())
    extras = [item.to_dict() for item in what_if]
    payload = json.dumps(
        {
            "brief": brief_text,
            "plan": plan_text,
            "wiki": cites,
            "what_if": extras,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def should_skip_world_sim(idea_dir: Path, digest: str) -> dict[str, Any]:
    """Reuse the last report when framing+plan+wiki did not move."""
    root = Path(idea_dir) / WORLD_SIM_FOLDER
    if not root.is_dir():
        return {"skip": False, "reason": "no_prior_world_sim", "last_world_dir": ""}
    latest = ""
    latest_hash = ""
    for meta in sorted(root.glob("*/meta.json")):
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        latest = str(meta.parent)
        latest_hash = str(payload.get("context_hash") or "")
    if latest_hash and latest_hash == digest:
        return {
            "skip": True,
            "reason": "context_unchanged",
            "last_world_dir": latest,
        }
    return {
        "skip": False,
        "reason": "context_changed" if latest else "no_prior_world_sim",
        "last_world_dir": latest,
    }


def validate_tiers(payload: dict[str, Any]) -> dict[str, Any]:
    """Phase 1 teeth: invariants, three tiers, tiers ordered on every metric.

    The teeth are scientific, not typographic. Every Phase 1 answer in
    the cwm-iclr2027 mission carried a rich `scientific_invariants`
    *object* (arms, endpoints, paired design) and a `key_assumptions`
    object; the old list-only reader saw an empty list and threw the
    whole rehearsal away four times out of four. Facts are accepted as
    a list of strings or as an object of named facts. Ordering is
    checked per metric in whichever direction the tiers move: a "gap"
    metric where best is smallest is as valid as a "reduction" metric
    where best is largest — what is refused is a metric that is not
    monotone across best → median → worst.
    """
    if not isinstance(payload, dict):
        raise WorldSimError("Phase 1 must return a JSON object")
    invariants = _fact_list(payload.get("scientific_invariants"))
    if not invariants:
        raise WorldSimError("scientific_invariants must name at least one fixed protocol fact")
    tiers_raw = payload.get("tiers")
    if not isinstance(tiers_raw, dict):
        raise WorldSimError("tiers must be an object with best, median, worst")
    parsed: dict[str, TierDefinition] = {}
    for name in TIERS:
        row = tiers_raw.get(name)
        if not isinstance(row, dict):
            raise WorldSimError(f"missing tier {name}")
        ranges = row.get("number_ranges") or {}
        if not isinstance(ranges, dict):
            raise WorldSimError(f"tier {name} needs number_ranges")
        numeric: dict[str, list[float]] = {}
        for key, value in ranges.items():
            pair = _as_range(value)
            if pair is not None:
                numeric[str(key)] = pair
        if not numeric:
            raise WorldSimError(f"tier {name} needs number_ranges")
        parsed[name] = TierDefinition(
            name=name,
            meaning=str(row.get("meaning") or "").strip(),
            key_assumptions=_fact_list(row.get("key_assumptions")),
            number_ranges=numeric,
            narrative_direction=str(row.get("narrative_direction") or "").strip(),
        )
        if not parsed[name].meaning:
            raise WorldSimError(f"tier {name} needs meaning")
    metrics = set(parsed["best"].number_ranges) & set(parsed["median"].number_ranges) & set(
        parsed["worst"].number_ranges
    )
    if not metrics:
        raise WorldSimError("tiers must share at least one metric")
    for metric in sorted(metrics):
        best = parsed["best"].number_ranges[metric]
        median = parsed["median"].number_ranges[metric]
        worst = parsed["worst"].number_ranges[metric]
        descending = (
            _lo(best) >= _lo(median) >= _lo(worst)
            and _hi(best) >= _hi(median) >= _hi(worst)
        )
        ascending = (
            _lo(best) <= _lo(median) <= _lo(worst)
            and _hi(best) <= _hi(median) <= _hi(worst)
        )
        if not (descending or ascending):
            raise WorldSimError(
                f"{metric} must move monotonically across best → median → worst"
            )
    return {
        "scientific_invariants": invariants,
        "shared_research_constraints": _fact_list(
            payload.get("shared_research_constraints")
        ),
        "tier_varying_factors": _fact_list(payload.get("tier_varying_factors")),
        "tiers": {name: parsed[name].to_dict() for name in TIERS},
    }


def validate_elaboration(
    payload: dict[str, Any],
    *,
    tier: str,
    plan_ids: tuple[str, ...],
    ranges: dict[str, list[float]],
) -> list[ImaginedExperiment]:
    """Phase 2 teeth: plan_id, rationale, numbers inside Phase 1 ranges."""
    rows = payload.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise WorldSimError(f"tier {tier} needs a non-empty experiments list")
    allowed = set(plan_ids) | {"what_if"}
    out: list[ImaginedExperiment] = []
    for item in rows:
        if not isinstance(item, dict):
            raise WorldSimError(f"tier {tier} experiment must be an object")
        plan_id = str(item.get("plan_id") or "").strip()
        if plan_id not in allowed:
            raise WorldSimError(
                f"imagined experiment plan_id {plan_id!r} is not in the plan "
                f"({', '.join(plan_ids) or 'none'}) or what_if"
            )
        rationale = str(item.get("result_rationale") or "").strip()
        if len(rationale.split()) < 6:
            raise WorldSimError("result_rationale needs at least six words")
        expected = item.get("expected_results") or {}
        if not isinstance(expected, dict) or not expected:
            raise WorldSimError("expected_results must be a non-empty object")
        for key, value in expected.items():
            if key in ranges:
                number = float(value)
                lo, hi = ranges[key][0], ranges[key][1]
                if number < min(lo, hi) - 1e-9 or number > max(lo, hi) + 1e-9:
                    raise WorldSimError(
                        f"{key}={number} is outside Phase 1 range [{lo}, {hi}]"
                    )
        out.append(
            ImaginedExperiment(
                name=str(item.get("name") or plan_id or "experiment"),
                plan_id=plan_id,
                description=str(item.get("description") or ""),
                expected_results={str(k): v for k, v in expected.items()},
                result_rationale=rationale,
            )
        )
    return out


def persist_world_sim(
    dest: Path,
    *,
    context: WorldSimContext,
    tiers: dict[str, Any],
    elaborations: dict[str, list[ImaginedExperiment]],
    review_mode: str,
) -> dict[str, Any]:
    """Write isolated reports. Never touches protocol.json, probe/, or fixture bind."""
    _assert_not_probe(dest)
    _assert_not_fixture_bind(dest)
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": KIND_WORLD_SIM,
        "not_a_discovery": True,
        "cannot_corroborate": True,
        "cannot_rewrite_expected_direction": True,
        "cannot_block_execute": True,
        "results_status": RESULTS_STATUS_IMAGINED,
        "context_hash": context.context_hash,
        "claim": context.claim,
        "review_mode": review_mode,
        "levers": list(context.levers),
        "schema": context.schema,
        "plan_ids": list(context.plan_ids),
        "what_if": [item.to_dict() for item in context.what_if],
        "wiki": [work.cite_id() for work in context.wiki],
        "tiers": tiers,
    }
    (dest / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in TIERS:
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        experiments = [item.to_dict() for item in elaborations.get(name) or ()]
        (folder / "imagined_experiments.json").write_text(
            json.dumps(
                {
                    "kind": KIND_WORLD_SIM,
                    "tier": name,
                    "results_status": RESULTS_STATUS_IMAGINED,
                    "experiments": experiments,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (folder / "paper.md").write_text(
            _tier_paper(context, name, tiers, experiments),
            encoding="utf-8",
        )
    return meta


def compile_world_sim_artifact(dest: Path) -> dict[str, Any]:
    """Interchange pack for an existing writing skill. kind=WORLD_SIM."""
    _assert_not_probe(dest)
    meta_path = dest / "meta.json"
    if not meta_path.is_file():
        raise WorldSimError(f"missing meta.json in {dest}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not is_world_sim_kind(meta.get("kind")):
        raise WorldSimError("world-sim artifact meta.kind must be WORLD_SIM")
    dump = {
        "kind": KIND_WORLD_SIM,
        "results_status": RESULTS_STATUS_IMAGINED,
        "not_a_paper": True,
        "not_a_discovery": True,
        "honesty": (
            "Imagined best/median/worst outcomes for rehearsal. "
            "Do not paste these numbers into EXPERIMENT_RESULTS.md "
            "or judge_probe. A writing skill may treat each tier as "
            "an 'if this future' scene."
        ),
        "wheel": "jin-s13/ai-research-writing-skill",
        "claim": meta.get("claim") or "",
        "context_hash": meta.get("context_hash") or "",
        "tiers": {},
    }
    for name in TIERS:
        experiments = dest / name / "imagined_experiments.json"
        paper = dest / name / "paper.md"
        dump["tiers"][name] = {
            "paper": paper.read_text(encoding="utf-8") if paper.is_file() else "",
            "experiments": (
                json.loads(experiments.read_text(encoding="utf-8"))
                if experiments.is_file()
                else {}
            ),
        }
    (dest / "simulated.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "WRITING.md").write_text(
        "Feed simulated.json to an existing writing skill. "
        "kind=WORLD_SIM. Imagined results_status. Not a discovery.\n",
        encoding="utf-8",
    )
    dump["files"] = ["simulated.json", "WRITING.md", "meta.json"]
    return dump


def classify_review_issues(
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-tier: same issue on all three = structural."""
    buckets: dict[str, dict[str, Any]] = {}
    for tier, report in reports.items():
        for issue in report.get("issues") or ():
            if not isinstance(issue, dict):
                continue
            summary = str(issue.get("summary") or "").strip()
            if not summary:
                continue
            key = " ".join(sorted(content_tokens(summary)))
            row = buckets.setdefault(
                key,
                {
                    "summary": summary,
                    "tiers": [],
                    "type": str(issue.get("type") or "data_dependent"),
                },
            )
            if tier not in row["tiers"]:
                row["tiers"].append(tier)
    out = []
    for row in buckets.values():
        if set(row["tiers"]) >= set(TIERS):
            row["type"] = "structural"
        out.append(row)
    return out


def write_assessment(
    dest: Path,
    *,
    issues: list[dict[str, Any]],
    recommendation: str,
    h: dict[str, str] | None = None,
) -> dict[str, str]:
    """Persist diagnostic actions. Does not touch protocol.json."""
    _assert_not_probe(dest)
    structural = [row for row in issues if row.get("type") == "structural"]
    dependent = [row for row in issues if row.get("type") != "structural"]
    lines = [
        "# World simulation assessment",
        "",
        "Diagnostic only. Does not climb. Does not rewrite expected_direction.",
        "Does not block `farfield execute`.",
        "",
        f"**Recommendation.** {recommendation.strip() or 'proceed with registered experiments'}",
        "",
        "## Structural (all three tiers)",
        "",
    ]
    if structural:
        for row in structural:
            lines.append(f"- {row['summary']}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Data-dependent (cross-tier)", ""])
    if dependent:
        for row in dependent:
            lines.append(
                f"- {row['summary']} (tiers: {', '.join(row.get('tiers') or ())})"
            )
    else:
        lines.append("- (none)")
    lines.append("")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "ASSESSMENT.md").write_text("\n".join(lines), encoding="utf-8")
    memory = []
    if structural:
        memory.append("structural: " + "; ".join(row["summary"] for row in structural[:3]))
    if dependent:
        memory.append(
            "data-dependent: " + "; ".join(row["summary"] for row in dependent[:3])
        )
    updated = dict(h or {})
    if memory:
        prev = str(updated.get("memory") or "").strip()
        note = "world-sim: " + " | ".join(memory)
        updated["memory"] = f"{prev}; {note}" if prev else note
    verifiers = str(updated.get("verifiers") or "")
    extra = "world-sim assessment diagnostic only; cannot climb"
    updated["verifiers"] = f"{verifiers}; {extra}" if verifiers else extra
    (dest / "h.json").write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return updated


def simple_review(
    client: Any,
    *,
    paper: str,
    wiki: list[FreshWork],
    purpose: str,
) -> dict[str, Any]:
    """One-pass reviewer: method + novelty + experiments. Scores do not select."""
    cites = [work.cite_id() for work in wiki if work.cite_id()]
    prompt = (
        "Review this world-simulation paper. Answer with one JSON object.\n"
        "Papers you may name: " + ", ".join(cites) + "\n\n"
        f"{paper[:6000]}\n\n"
        '{"novelty":1,"novelty_note":"<=40 words and name a paper id if any",'
        '"issues":[{"summary":"...","type":"structural|data_dependent"}]}'
    )
    completion = client.complete(
        prompt, purpose=purpose, system="You are a reviewer of a world-simulation paper."
    )
    completion.assert_usable()
    payload = _parse_json(completion.text)
    note = str(payload.get("novelty_note") or "").strip()
    if cites and not any(cite in note for cite in cites):
        raise WorldSimError("novelty_note must name a retrieved paper id")
    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        raise WorldSimError("issues must be a list")
    return {
        "novelty": int(payload.get("novelty") or 0),
        "novelty_note": note,
        "issues": [
            {
                "summary": str(item.get("summary") or ""),
                "type": str(item.get("type") or "data_dependent"),
            }
            for item in issues
            if isinstance(item, dict)
        ],
    }


def full_review(
    client: Any,
    *,
    paper: str,
    wiki: list[FreshWork],
    purpose: str,
) -> dict[str, Any]:
    """Two sequential reviews (method/experiments, then novelty). No roundtable."""
    method = simple_review(
        client, paper="[method+experiments]\n" + paper, wiki=wiki, purpose=purpose + ":method"
    )
    novelty = simple_review(
        client, paper="[novelty+knowledge]\n" + paper, wiki=wiki, purpose=purpose + ":novelty"
    )
    issues = list(method.get("issues") or []) + list(novelty.get("issues") or [])
    return {
        "novelty": novelty.get("novelty") or method.get("novelty"),
        "novelty_note": novelty.get("novelty_note") or method.get("novelty_note"),
        "issues": issues,
        "passes": {"method": method, "novelty": novelty},
    }


PHASE1_SYSTEM = (
    "You define three-tier world-simulation scenarios for one research idea. "
    "Scientific invariants stay fixed across tiers. Only effect size, "
    "assumption truth, and failure modes may vary. Imagined experiments "
    "stay on the declared dynworld levers; do not invent a parallel handle. "
    "Answer with one JSON object."
)

PHASE2_SYSTEM = (
    "You elaborate one world-simulation tier's imagined experiments. Stay inside "
    "the given number_ranges. Every experiment needs plan_id and "
    "result_rationale. Answer with one JSON object."
)


def define_tiers(client: Any, context: WorldSimContext) -> dict[str, Any]:
    prompt = (
        f"Claim: {context.claim}\n"
        f"Mechanism: {context.mechanism}\n"
        f"Declared dynworld levers: {', '.join(context.levers) or '(none)'}\n"
        f"Schema: {context.schema or '(none)'}\n"
        f"Plan ids: {', '.join(context.plan_ids)}\n"
        f"Wiki: {', '.join(work.cite_id() for work in context.wiki)}\n"
        f"What-if (appended, plan file unchanged): "
        f"{json.dumps([item.to_dict() for item in context.what_if], ensure_ascii=False)}\n"
        "Brief:\n"
        f"{context.brief_text[:4000]}\n"
        "Plan:\n"
        f"{context.plan_text[:4000]}\n"
        "Return JSON with scientific_invariants (list of strings: the "
        "protocol facts fixed across tiers — arms, endpoint, paired design, "
        "schema), shared_research_constraints (list of strings), "
        "tier_varying_factors (list of strings), and tiers.best/median/worst "
        "each having meaning (string), key_assumptions (list of strings), "
        "number_ranges (object: metric name → [low, high] numbers only; no "
        "booleans), narrative_direction (string). Every metric in "
        "number_ranges must move monotonically from best to median to worst."
    )
    completion = client.complete(
        prompt, purpose="world-sim:phase1", system=PHASE1_SYSTEM
    )
    completion.assert_usable()
    return validate_tiers(_parse_json(completion.text))


def elaborate_tier(
    client: Any,
    context: WorldSimContext,
    *,
    tier: str,
    definition: dict[str, Any],
) -> list[ImaginedExperiment]:
    plan_ids = context.plan_ids
    if context.what_if:
        plan_ids = tuple(list(plan_ids) + ["what_if"])
    prompt = (
        f"Tier: {tier}\n"
        f"Definition: {json.dumps(definition, ensure_ascii=False)}\n"
        f"Plan ids: {', '.join(plan_ids)}\n"
        f"Claim: {context.claim}\n"
        "Return JSON {\"experiments\":[{\"name\",\"plan_id\",\"description\","
        "\"expected_results\",\"result_rationale\"}]}"
    )
    completion = client.complete(
        prompt, purpose=f"world-sim:phase2:{tier}", system=PHASE2_SYSTEM
    )
    completion.assert_usable()
    return validate_elaboration(
        _parse_json(completion.text),
        tier=tier,
        plan_ids=plan_ids,
        ranges=definition.get("number_ranges") or {},
    )


def run_world_sim(
    idea_dir: Path,
    *,
    client: Any | None = None,
    what_if: list[WhatIfExperiment] | tuple[WhatIfExperiment, ...] = (),
    wiki: list[FreshWork] | tuple[FreshWork, ...] = (),
    review_mode: str = "simple",
    smart_skip: bool = False,
    claim: str = "",
    mechanism: str = "",
    scout: dict[str, Any] | None = None,
    world_lever: str = "",
    competing_lever: str = "",
    schema: str = "",
    canned_tiers: dict[str, Any] | None = None,
    canned_elaborations: dict[str, list[dict[str, Any]]] | None = None,
    canned_reviews: dict[str, dict[str, Any]] | None = None,
    h: dict[str, str] | None = None,
    sim_id: str = "",
) -> dict[str, Any]:
    """Orchestrate rehearsal. LLM optional when canned payloads are supplied."""
    extras = [item if isinstance(item, WhatIfExperiment) else WhatIfExperiment.from_dict(item) for item in what_if]
    context = collect_context(
        idea_dir,
        what_if=extras,
        wiki=list(wiki),
        scout=scout,
        claim=claim,
        mechanism=mechanism,
        world_lever=world_lever,
        competing_lever=competing_lever,
        schema=schema,
    )
    skip = should_skip_world_sim(context.idea_dir, context.context_hash)
    if smart_skip and skip.get("skip"):
        return {"ok": True, "skipped": True, **skip, "kind": KIND_WORLD_SIM}
    ident = str(sim_id or uuid.uuid4().hex[:12])
    dest = world_sim_dir(context.idea_dir, ident)
    if canned_tiers is not None:
        tiers = validate_tiers(canned_tiers)
    else:
        if client is None:
            raise WorldSimError("run_world_sim needs an LLM client or canned_tiers")
        tiers = define_tiers(client, context)
    elaborations: dict[str, list[ImaginedExperiment]] = {}
    for name in TIERS:
        definition = tiers["tiers"][name]
        if canned_elaborations and name in canned_elaborations:
            elaborations[name] = validate_elaboration(
                {"experiments": canned_elaborations[name]},
                tier=name,
                plan_ids=tuple(list(context.plan_ids) + (["what_if"] if extras else [])),
                ranges=definition.get("number_ranges") or {},
            )
        else:
            if client is None:
                raise WorldSimError("run_world_sim needs an LLM client or canned_elaborations")
            elaborations[name] = elaborate_tier(
                client, context, tier=name, definition=definition
            )
    persist_world_sim(
        dest,
        context=context,
        tiers=tiers,
        elaborations=elaborations,
        review_mode=review_mode,
    )
    dump = compile_world_sim_artifact(dest)
    reviews: dict[str, dict[str, Any]] = {}
    reviewer = full_review if review_mode == "full" else simple_review
    for name in TIERS:
        paper = (dest / name / "paper.md").read_text(encoding="utf-8")
        if canned_reviews and name in canned_reviews:
            reviews[name] = canned_reviews[name]
        else:
            if client is None:
                reviews[name] = {"novelty": 0, "novelty_note": "", "issues": []}
            else:
                reviews[name] = reviewer(
                    client,
                    paper=paper,
                    wiki=context.wiki,
                    purpose=f"world-sim:review:{name}",
                )
        (dest / name / "review.json").write_text(
            json.dumps(reviews[name], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    issues = classify_review_issues(reviews)
    recommendation = (
        "fix framing/plan before GPU: structural issues appeared on every tier"
        if any(row.get("type") == "structural" for row in issues)
        else "keep the claim; add evidence if only best holds"
    )
    updated_h = write_assessment(
        dest, issues=issues, recommendation=recommendation, h=h
    )
    return {
        "ok": True,
        "skipped": False,
        "kind": KIND_WORLD_SIM,
        "sim_id": ident,
        "dest": str(dest),
        "artifact": dump,
        "issues": issues,
        "h": updated_h,
        "cannot_corroborate": True,
        "cannot_rewrite_expected_direction": True,
        "cannot_block_execute": True,
    }


WORLD_SIM_RESIM_CAP = 1


def maybe_world_sim_after_diagnosis(
    idea_dir: Path,
    *,
    enabled: bool,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Product rehearsal after brief+plan exist. Never blocks execute.

    Structural issues are returned so this mission can rewrite the plan
    once (`WORLD_SIM_RESIM_CAP`). Imagined numbers cannot climb.
    """
    if not enabled:
        return None
    try:
        return run_world_sim(idea_dir, **kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "kind": KIND_WORLD_SIM,
            "error": str(exc),
            "cannot_block_execute": True,
        }


def _assert_not_probe(path: Path) -> None:
    text = str(path).replace("\\", "/")
    if _PROBE_DIR.search(text):
        raise WorldSimError("world-sim reports must not land under candidates/<card>/probe/")


def _assert_not_fixture_bind(path: Path) -> None:
    """Fixture bind is workspace/world or candidates/<card>/world, not rehearsal."""
    root = Path(path)
    if (root / "data").is_dir() and (root / "world.json").is_file():
        raise WorldSimError(
            "world-sim cannot run on a fixture bind directory "
            "(workspace/world or candidates/<card>/world)"
        )
    text = str(root).replace("\\", "/")
    if re.search(r"(?:^|/)world/data(?:/|$)", text):
        raise WorldSimError(
            "world-sim reports must not land under a fixture data/ bind"
        )


def _claim_from_brief(text: str) -> str:
    match = _CLAIM_HEADING.search(text)
    if not match:
        return ""
    rest = text[match.end():].lstrip()
    return rest.split("\n\n", 1)[0].replace("\n", " ").strip()


def _mechanism_from_brief(text: str) -> str:
    marker = "**机制草稿"
    idx = text.find(marker)
    if idx < 0:
        return ""
    rest = text[idx:].split("\n\n", 2)
    if len(rest) < 2:
        return ""
    return rest[1].replace("\n", " ").strip()


def _plan_ids(text: str) -> tuple[str, ...]:
    found = [f"block-{num}" for num in _BLOCK_HEADING.findall(text)]
    return tuple(found)


def _title_abstract(work: Any) -> tuple[str, str]:
    if hasattr(work, "title"):
        return str(getattr(work, "title", "") or ""), str(getattr(work, "abstract", "") or "")
    if isinstance(work, dict):
        return str(work.get("title") or ""), str(work.get("abstract") or "")
    return "", ""


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _fact_list(value: Any) -> list[str]:
    """Named facts as strings: a list, an object of facts, or one string.

    `{"paired_design": "...", "arms": {...}}` becomes
    `["paired_design: ...", "arms: {...}"]`. Nested lists flatten one
    level; nested objects are serialised so nothing the model asserted
    is silently dropped.
    """
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, (dict, list)):
                out.extend(_fact_list(item))
            else:
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            name = str(key).strip()
            if isinstance(item, (dict, list)):
                body = json.dumps(item, ensure_ascii=False, sort_keys=True)
            else:
                body = str(item).strip()
            if not body:
                continue
            out.append(f"{name}: {body}" if name else body)
        return out
    return []


_RANGE_KEYS = (("min", "max"), ("low", "high"), ("lo", "hi"))


def _as_range(value: Any) -> list[float] | None:
    """`[low, high]`, `{"min","max"}`, `{"low","high"}` → [low, high]; else None.

    A prose note or a boolean flag inside number_ranges is not a metric
    and is skipped rather than used to refuse the whole tier.
    """
    if isinstance(value, dict):
        for lo_key, hi_key in _RANGE_KEYS:
            if lo_key in value and hi_key in value:
                value = [value[lo_key], value[hi_key]]
                break
        else:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    if any(isinstance(item, bool) for item in value[:2]):
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def _lo(pair: list[float]) -> float:
    return min(float(pair[0]), float(pair[1]))


def _hi(pair: list[float]) -> float:
    return max(float(pair[0]), float(pair[1]))


def _parse_json(text: str) -> dict[str, Any]:
    blob = (text or "").strip()
    if blob.startswith("```"):
        blob = blob.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise WorldSimError(f"expected JSON object, got {blob[:80]!r}") from exc
    if not isinstance(payload, dict):
        raise WorldSimError("top-level JSON must be an object")
    return payload


def _tier_paper(
    context: WorldSimContext,
    tier: str,
    tiers: dict[str, Any],
    experiments: list[dict[str, Any]],
) -> str:
    definition = (tiers.get("tiers") or {}).get(tier) or {}
    lines = [
        f"# World-simulation paper ({tier})",
        "",
        f"kind: {KIND_WORLD_SIM}",
        f"results_status: {RESULTS_STATUS_IMAGINED}",
        "not_a_discovery: true",
        "",
        "## Claim",
        "",
        context.claim or "(missing)",
        "",
        "## If this future",
        "",
        str(definition.get("meaning") or ""),
        "",
        str(definition.get("narrative_direction") or ""),
        "",
        "## Imagined experiments",
        "",
    ]
    for item in experiments:
        lines.append(f"### {item.get('name')} (`{item.get('plan_id')}`)")
        lines.append("")
        lines.append(str(item.get("description") or ""))
        lines.append("")
        lines.append(f"Expected: `{item.get('expected_results')}`")
        lines.append("")
        lines.append(str(item.get("result_rationale") or ""))
        lines.append("")
    lines.extend(
        [
            "## Honesty",
            "",
            "These numbers are imagined. They cannot corroborate, enter "
            "EXPERIMENT_RESULTS.md, rewrite expected_direction, or block execute.",
            "",
        ]
    )
    return "\n".join(lines)
