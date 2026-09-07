"""Paper-level research proposal and experiment design for one idea.

What the compiled packet cannot do. `RESEARCH_BRIEF.md` and
`EXPERIMENT_PLAN.md` are compiled from attested fields and refuse to
invent prose; that keeps them honest and keeps them thin. A submission
needs the other kind of document as well — background, related work,
the motivation for the mechanism, a theory sketch with its assumptions
written out, method details, threats to validity — and that document
has to be *written*, by a model, from the facts the mission attested.

This module is that writing pass, kept on a leash:

- Facts in, prose out. The model receives `PAPER_FACTS.json`: claim,
  mechanism, prediction, the bound world (id, provenance, n), every
  probe attempt with its arm numbers and verdict, the adversarial
  review's `must_change`, world-sim structural holes, verified papers.
- Citations are audited. Only the ids in the fact sheet may be cited;
  anything else is rewritten to `[unverified]` and listed in `AUDIT.md`.
- Numbers are audited. Any figure in the draft that is not in the fact
  sheet is listed as a draft number, not a result.
- One reviewer round by default. The same model reads the draft as an
  ICLR reviewer and returns `must_fix`; the author pass revises once.
  That loop is what turns a claim into a paper-shaped argument — it is
  the part ARIS-style pipelines do and FarField did not.
- The header of every output says DRAFT and names what is attested.
  Nothing here climbs the ladder or rewrites `expected_direction`.

Outputs under `<idea_dir>/paper/`: `RESEARCH_PROPOSAL.md`,
`EXPERIMENT_DESIGN.md`, `REVIEW.md`, `PAPER_FACTS.json`, `AUDIT.md`,
`paper.json` (completion digests).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .llm import Completion, LLMClient

DEFAULT_ROUNDS = 1
_PROTOCOL_DIR = re.compile(r"protocol_dir:\s*`?candidates/([A-Za-z0-9_]+)/?`?")


def card_id_for(idea_dir: Path) -> str:
    """The candidate this idea folder executes, read off AGENT_PACKET.md."""
    text = _text(Path(idea_dir) / "AGENT_PACKET.md")
    match = _PROTOCOL_DIR.search(text)
    if match:
        return match.group(1)
    raise PaperError(f"{idea_dir} names no candidates/<card_id> in AGENT_PACKET.md")

SYSTEM = (
    "You are the first author of a machine-learning research group preparing an "
    "ICLR submission. You write from a fact sheet. You never invent a citation, a "
    "dataset, or a number: everything numeric or bibliographic in your text must "
    "come from the fact sheet, and anything you propose (a threshold, a budget) "
    "must be labelled as pre-registered proposal, not result. You separate what "
    "the evidence has established from what the paper hypothesizes. You answer "
    "with one JSON object and nothing else."
)

AUTHOR_TEMPLATE = """Write the paper-level research proposal and the experiment design for ONE idea.

Language for prose: {lang_name}. Keep technical terms, metric names, field names, and citations in English inside the {lang_name} prose.

## Fact sheet (the only source of facts)
{facts}

## What the proposal must contain (all sections, in this order, Markdown)
1. Title (a real paper title, not the claim sentence) and a 150-250 word Abstract.
2. Background and problem: what a code world model of executable program state is; why an agent harness that rewrites its own cells and validators (recursive self-improvement) makes validator reliability the central risk; define the setting formally (cells, validators, updates, acceptance, false acceptance, divergence) using the schema fields in the fact sheet.
3. Related work: group the verified papers with relevance "strong" into themes; for each say precisely what it establishes and what it leaves open. Papers with relevance "weak" were retrieved and verified but are tangential: do not cite them in related work; at most, one sentence may list them as retrieved-but-tangential. Cite ONLY ids from the fact sheet, in the form [arxiv_id] or [doi:...]. Never cite by author name or year without the id. You may mention well-known systems by name without a citation ONLY inside a sentence that says they were not retrieved in this mission.
4. Motivation and gap: why the obvious approach fails, which assumption is dropped, and what a positive result would change for practitioners building self-improving harnesses.
5. Method: the mechanism as an algorithm (pseudocode-level detail); the quantities it manipulates; a theory sketch — state the assumptions explicitly, give the argument for the predicted direction, and state what would falsify it. Mark each step "attested" if the fact sheet shows it was run, else "proposed".
6. Scientific role of this idea (read `idea_kind` / `scientific_role` on the fact sheet):
   - question → research question / motivating analysis. Do not rewrite it as a two-arm H1.
   - acquire → methods / infrastructure / world construction. Success is an attested world, not an effect.
   - theory → model / explanation / derived predictions. Unchecked predictions are not WORLD facts.
   - probe → hypotheses H1..Hk aligned with evidence_grid cells; a hypothesis per required cell.
   Never disguise acquire or question as a pseudo two-arm H1 to fit a template.
7. What the evidence so far actually shows: read the probe attempts in the fact sheet honestly (identical arms, saturated worlds, undefined denominators are findings about testability, not noise). Numbers under `exploratory_analyses` are post hoc descriptives on the attested world: you may use them as MOTIVATION, always labelled "exploratory", never as a result; each hypothesis they motivate must name the pre-registered two-arm test that would establish it.
8. Limitations and threats to validity: world provenance (derived vs harvested vs published), single-seed harvests, fixed vs mutable validators, construct validity of false_accept_fraction, generalization beyond the harness family.
9. Attested vs draft: a short table of which statements are attested by the fact sheet and which are this draft's proposals.

## What the experiment design must contain (Markdown)
- Core claims table: C1..Ck (one per hypothesis), why it matters, minimum convincing evidence, linked blocks.
- Worlds: for each world used, id, schema, provenance, n, and whether it exists or must be acquired (say by which of derive / import / harvest, from which source, and why that source has the property the claim needs — e.g. mutable validators for a false-acceptance claim).
- Blocks B1..Bn: claim tested, world, treatment arm, control arm(s) (including an iid / uniform control and a dose-response arm where the mechanism is a rate), exact metric definition with numerator and denominator and how abstentions are handled, matching protocol (what is held equal and with what tolerance, calibrated on data disjoint from evaluation), seeds and paired resampling unit, success criterion with a pre-registered margin and interval, failure interpretation, priority (MUST-RUN / SHOULD / NICE).
- Ablations that isolate the mechanism from generic regularization and data-volume effects.
- Run order with decision gates and kill criteria (what result stops the line).
- Compute: host stdlib 600s / host-heavy 3600s tiers; anything larger is an external run whose numbers cannot corroborate.
- Risks and the mitigation for each.

Return one JSON object:
{{
  "title": "...",
  "abstract": "...",
  "proposal_markdown": "... full Markdown of sections 1-9 ...",
  "experiment_design_markdown": "... full Markdown ...",
  "hypotheses": [{{"id": "H1", "statement": "...", "world": "...", "evidence": "supports|weakens|uninformative|none"}}],
  "citations_used": ["2608.17956v1", "doi:..."],
  "limitations": ["..."],
  "open_questions": ["..."]
}}
"""

REVIEWER_SYSTEM = (
    "You are a rigorous ICLR area chair reading a research proposal draft and its "
    "experiment design. You judge whether the argument is paper-shaped: a real "
    "problem, a mechanism with a reason to work, hypotheses that the proposed "
    "experiments can actually falsify, honest treatment of the evidence so far, "
    "and no claim beyond the fact sheet. You answer with one JSON object."
)

REVIEWER_TEMPLATE = """Review this draft against the fact sheet. Be concrete: name the section and the sentence.

## Fact sheet
{facts}

## Draft proposal
{proposal}

## Draft experiment design
{design}

Return one JSON object:
{{
  "score": 1-10,
  "summary": "three sentences",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "fabrication_or_overclaim": ["any statement not supported by the fact sheet; any citation by author/year without a fact-sheet id; any exploratory number presented as a result; any hypothesis without a named world"],
  "must_fix": ["ordered, specific, actionable"],
  "missing_experiments": ["experiments without which the core claim cannot be defended"]
}}
"""

REVISION_TEMPLATE = """Revise the draft below so that every item in `must_fix` is addressed and every `fabrication_or_overclaim` item is removed or rewritten as a hypothesis. Keep everything that the reviewer did not object to. Same rules: facts only from the fact sheet; cite only fact-sheet ids; label proposals as proposals.

## Fact sheet
{facts}

## Reviewer verdict
{review}

## Current proposal
{proposal}

## Current experiment design
{design}

Return one JSON object with the same schema as the original author answer plus:
  "changes_made": ["what changed, one line each"]
"""

_CITE = re.compile(r"\[(?:arXiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)\]|\[(doi:[^\]\s]+)\]", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d{2,}|\d{3,})(?![\w.])")
# A citation written as prose — "Zhang et al. (2026)", "张等人", "(Smith 2025)",
# "as shown by X and Y" — has no id to audit. It is flagged, not trusted.
_PROSE_CITE = re.compile(
    r"([A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-]+)?\s+et\s+al\.?\s*\(?\s*(?:19|20)\d\d[a-z]?\)?)|"
    r"(\([A-Z][A-Za-z\-]+(?:\s+(?:and|&|,)\s+[A-Z][A-Za-z\-]+)*,?\s+(?:19|20)\d\d[a-z]?\))|"
    r"([\u4e00-\u9fff]{1,4}等人\s*(?:\(|（)?\s*(?:19|20)\d\d)"
)


class PaperError(ValueError):
    """The paper stage refused; nothing was written."""


def gather_facts(workspace: Path, idea_dir: Path, card_id: str, *, catalog_root: Path | None = None) -> dict[str, Any]:
    """Everything the writer may treat as true, read from attested files."""
    workspace = Path(workspace)
    idea_dir = Path(idea_dir)
    cand = workspace / "candidates" / card_id
    protocol = _json(cand / "protocol.json")
    probe = _json(cand / "probe.json")
    metrics = _json(cand / "metrics.json")
    host = _json(cand / "host_run.json")
    world_manifest = _json(workspace / "world" / "world.json")
    world_payload_summary = _world_summary(workspace / "world" / "data" / "world.json")
    facts: dict[str, Any] = {
        "topic": str(protocol.get("topic") or _text(workspace / "topic.txt")).strip(),
        "card_id": card_id,
        "idea_kind": protocol.get("idea_kind") or "probe",
        "idea_id": protocol.get("idea_id") or card_id,
        "lineage_id": protocol.get("lineage_id") or "",
        "origin": protocol.get("origin") or "",
        "scientific_role": protocol.get("idea_kind") or "probe",
        "claim": protocol.get("claim", ""),
        "mechanism": protocol.get("mechanism", ""),
        "prediction": protocol.get("prediction", ""),
        "pair": protocol.get("pair", []),
        "gap": protocol.get("gap", ""),
        "idea": protocol.get("idea", ""),
        "approach": protocol.get("approach", ""),
        "baseline": protocol.get("baseline", ""),
        "risks": protocol.get("risks", []),
        "diagnosis": protocol.get("diagnosis", {}),
        "experiment_attempts": protocol.get("experiment_attempts", []),
        "discarded_experiments": protocol.get("discarded_experiments", []),
        "probe": {
            k: probe.get(k)
            for k in (
                "kind", "verdict", "treatment", "control", "measure", "expected_direction",
                "alternative", "mechanism_identified", "ablation_required", "separation", "margin",
            )
        },
        "probe_attempts": probe.get("attempts") or [],
        "metrics": metrics,
        "host_run": {k: host.get(k) for k in ("status", "verdict", "treatment", "control", "kind") if k in host},
        "world": {
            "id": world_manifest.get("id"),
            "schema": world_manifest.get("schema"),
            "role": world_manifest.get("role"),
            "provenance": world_manifest.get("provenance", ""),
            "title": world_manifest.get("title"),
            "source": world_manifest.get("source"),
            "digest": world_manifest.get("digest"),
            **world_payload_summary,
        },
        "verified_papers": _papers(cand, protocol),
        "adversarial_review": _review_for(workspace / "mission.ndjson", card_id),
        "world_sim_structural_holes": _world_sim_holes(idea_dir / "world-sim"),
        "experiment_results_md": _text(idea_dir / "refine-logs" / "EXPERIMENT_RESULTS.md")[:4000],
        "available_program_state_worlds": _catalog_worlds(catalog_root, schema=str(world_manifest.get("schema") or "")),
        "ladder_status": _promotion_for(workspace / "mission.ndjson", card_id),
        # Post hoc analyses on the bound world: motivation only. The writer
        # must label every number from here as exploratory.
        "exploratory_analyses": _exploratory(catalog_root, str(world_manifest.get("id") or "")),
        # Which cells C1 needs, which are filled (typed), which need a world.
        "evidence_grid": _json(idea_dir / "refine-logs" / "EVIDENCE_GRID.json"),
    }
    return facts


def _exploratory(catalog_root: Path | None, world_id: str) -> list[dict[str, Any]]:
    if not world_id:
        return []
    try:
        from .exploratory import load_records
        from .freeze import default_catalog
    except Exception:
        return []
    root = Path(catalog_root) if catalog_root is not None else default_catalog()
    rows = []
    for record in load_records(root, world_id):
        rows.append(
            {
                "label": record.get("label"),
                "script_digest": record.get("script_digest"),
                "world_digest": record.get("world_digest"),
                "status": "exploratory — post hoc, not pre-registered, cannot corroborate",
                "results": record.get("results"),
            }
        )
    return rows


def compile_paper(
    workspace: Path,
    idea_dir: Path,
    card_id: str,
    *,
    client: LLMClient,
    rounds: int = DEFAULT_ROUNDS,
    lang: str = "zh",
    catalog_root: Path | None = None,
) -> dict[str, Any]:
    """Write `<idea_dir>/paper/`. Returns a record with paths, score, audit."""
    idea_dir = Path(idea_dir)
    facts = gather_facts(workspace, idea_dir, card_id, catalog_root=catalog_root)
    if not str(facts.get("claim") or "").strip():
        raise PaperError(f"no attested claim for {card_id}; nothing to write from")
    facts_text = json.dumps(facts, ensure_ascii=False, indent=1)
    lang_name = {"zh": "Simplified Chinese", "en": "English"}.get(lang, "Simplified Chinese")
    completions: list[Completion] = []

    author = client.complete(
        AUTHOR_TEMPLATE.format(lang_name=lang_name, facts=facts_text),
        purpose=f"paper_author:{card_id}",
        system=SYSTEM,
    )
    author.assert_usable()
    completions.append(author)
    draft = _parse(author, "author")
    reviews: list[dict[str, Any]] = []
    for round_index in range(max(0, int(rounds))):
        review_completion = client.complete(
            REVIEWER_TEMPLATE.format(
                facts=facts_text,
                proposal=draft.get("proposal_markdown", ""),
                design=draft.get("experiment_design_markdown", ""),
            ),
            purpose=f"paper_review:{card_id}:{round_index}",
            system=REVIEWER_SYSTEM,
        )
        review_completion.assert_usable()
        completions.append(review_completion)
        review = _parse(review_completion, "reviewer")
        reviews.append(review)
        if not review.get("must_fix") and not review.get("fabrication_or_overclaim"):
            break
        revision = client.complete(
            REVISION_TEMPLATE.format(
                facts=facts_text,
                review=json.dumps(review, ensure_ascii=False, indent=1),
                proposal=draft.get("proposal_markdown", ""),
                design=draft.get("experiment_design_markdown", ""),
            ),
            purpose=f"paper_revise:{card_id}:{round_index}",
            system=SYSTEM,
        )
        revision.assert_usable()
        completions.append(revision)
        revised = _parse(revision, "revision")
        if revised.get("proposal_markdown") and revised.get("experiment_design_markdown"):
            draft = revised
    return _write(idea_dir, card_id, facts, draft, reviews, completions, lang=lang)


def _write(
    idea_dir: Path,
    card_id: str,
    facts: dict[str, Any],
    draft: dict[str, Any],
    reviews: list[dict[str, Any]],
    completions: list[Completion],
    *,
    lang: str,
) -> dict[str, Any]:
    out = idea_dir / "paper"
    out.mkdir(parents=True, exist_ok=True)
    allowed = {str(p["id"]) for p in facts.get("verified_papers", []) if p.get("id")}
    allowed |= {a.split("v")[0] for a in allowed}
    proposal, cite_audit = _audit_citations(str(draft.get("proposal_markdown") or ""), allowed)
    design, cite_audit2 = _audit_citations(str(draft.get("experiment_design_markdown") or ""), allowed)
    numbers = _audit_numbers(proposal + "\n" + design, facts)
    prose_cites = _audit_prose_citations(proposal + "\n" + design)
    unworlded = [
        str(h.get("id") or "?")
        for h in (draft.get("hypotheses") or [])
        if isinstance(h, dict) and not str(h.get("world") or "").strip()
    ]
    world = facts.get("world") or {}
    header = _header(card_id, facts, world, reviews, lang, draft)
    (out / "RESEARCH_PROPOSAL.md").write_text(header + proposal.rstrip() + "\n", encoding="utf-8")
    (out / "EXPERIMENT_DESIGN.md").write_text(header + design.rstrip() + "\n", encoding="utf-8")
    (out / "PAPER_FACTS.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    review_md = ["# Reviewer rounds (same model as ICLR AC; opinion, not evidence)", ""]
    for index, review in enumerate(reviews):
        review_md.append(f"## Round {index}: score {review.get('score')}")
        review_md.append("")
        review_md.append(str(review.get("summary") or ""))
        for key in ("strengths", "weaknesses", "fabrication_or_overclaim", "must_fix", "missing_experiments"):
            items = review.get(key) or []
            if items:
                review_md.append("")
                review_md.append(f"### {key}")
                review_md.extend(f"- {item}" for item in items)
        review_md.append("")
    if draft.get("changes_made"):
        review_md.append("## Changes made in revision")
        review_md.extend(f"- {item}" for item in draft["changes_made"])
        review_md.append("")
    (out / "REVIEW.md").write_text("\n".join(review_md), encoding="utf-8")
    audit = {
        "unverified_citations_rewritten": sorted(set(cite_audit + cite_audit2)),
        "prose_citations_without_id": prose_cites,
        "hypotheses_without_a_world": unworlded,
        "draft_numbers_not_in_facts": numbers,
        "citations_used_declared": draft.get("citations_used", []),
        "allowed_citation_ids": sorted(allowed),
    }
    audit_md = [
        "# Audit",
        "",
        "Citations are allowed only from the fact sheet's verified papers. Numbers that do not appear in the fact sheet are draft proposals (thresholds, budgets), never results.",
        "",
        f"- unverified citations rewritten to [unverified]: {len(audit['unverified_citations_rewritten'])}",
    ]
    audit_md.extend(f"  - {item}" for item in audit["unverified_citations_rewritten"])
    audit_md.append(
        f"- prose citations without an id (author-year, et al., 等人) — unauditable, treat as unverified: {len(prose_cites)}"
    )
    audit_md.extend(f"  - {item}" for item in prose_cites[:40])
    audit_md.append(f"- hypotheses that name no world to test them on: {len(unworlded)}")
    audit_md.extend(f"  - {item}" for item in unworlded)
    audit_md.append(f"- numbers in the draft that are not in the fact sheet: {len(numbers)}")
    audit_md.extend(f"  - {item}" for item in numbers[:80])
    (out / "AUDIT.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    record = {
        "card_id": card_id,
        "title": draft.get("title", ""),
        "paper_dir": str(out),
        "score": reviews[-1].get("score") if reviews else None,
        "rounds": len(reviews),
        "hypotheses": draft.get("hypotheses", []),
        "limitations": draft.get("limitations", []),
        "open_questions": draft.get("open_questions", []),
        "audit": {k: v for k, v in audit.items() if k != "allowed_citation_ids"},
        "completions": [
            {"purpose": c.artifact_uri, "digest": c.digest, "model": c.model, "tokens": c.total_tokens}
            for c in completions
        ],
        "written_at": date.today().isoformat(),
        "status": "draft",
    }
    (out / "paper.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return record


def _header(card_id: str, facts: dict[str, Any], world: dict[str, Any], reviews: list[dict[str, Any]], lang: str, draft: dict[str, Any]) -> str:
    probe = facts.get("probe") or {}
    status = (facts.get("ladder_status") or {}).get("status") or "speculative"
    title = str(draft.get("title") or "").strip()
    if lang == "zh":
        lines = [
            f"# {title}" if title else f"# 研究方案草稿 `{card_id}`",
            "",
            "> **DRAFT — 由模型从本场 attested 事实表写成，不是结果，不是发现。** 梯子状态不因本文改变。",
            f"> 事实基础：主张 `{card_id}`；世界 `{world.get('id')}`（schema `{world.get('schema')}`，provenance `{world.get('provenance') or 'published'}`）；"
            f"探针 kind `{probe.get('kind')}`，判决 `{probe.get('verdict')}`；梯子 `{status}`。",
            f"> 引用只允许事实表内已验证论文；`paper/AUDIT.md` 列出被改写的引用与未见于事实表的数字。评审轮次 {len(reviews)}，末轮分 {reviews[-1].get('score') if reviews else '—'}/10（同模型意见，不是证据）。",
            "",
        ]
    else:
        lines = [
            f"# {title}" if title else f"# Research proposal draft `{card_id}`",
            "",
            "> **DRAFT — written by a model from this mission's attested fact sheet. Not a result, not a discovery.** The ladder does not move because of this file.",
            f"> Facts: claim `{card_id}`; world `{world.get('id')}` (schema `{world.get('schema')}`, provenance `{world.get('provenance') or 'published'}`); "
            f"probe kind `{probe.get('kind')}`, verdict `{probe.get('verdict')}`; ladder `{status}`.",
            f"> Citations only from verified papers in the fact sheet; see `paper/AUDIT.md`. Reviewer rounds {len(reviews)}, last score {reviews[-1].get('score') if reviews else '—'}/10 (same-model opinion, not evidence).",
            "",
        ]
    return "\n".join(lines) + "\n"


def _audit_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    bad: list[str] = []

    def fix(match: re.Match[str]) -> str:
        ident = match.group(1) or match.group(2) or ""
        base = ident.split("v")[0] if not ident.startswith("doi:") else ident
        if ident in allowed or base in allowed:
            return match.group(0)
        bad.append(ident)
        return f"[unverified: {ident}]"

    return _CITE.sub(fix, text), bad


def _audit_prose_citations(text: str) -> list[str]:
    found: list[str] = []
    for match in _PROSE_CITE.finditer(text):
        token = " ".join((match.group(0) or "").split())
        if token and token not in found:
            found.append(token)
    return found


def _audit_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    fact_blob = json.dumps(facts, ensure_ascii=False)
    fact_numbers = set(_NUMBER.findall(fact_blob))
    seen: list[str] = []
    for match in _NUMBER.finditer(text):
        token = match.group(1)
        if token in fact_numbers or token in seen:
            continue
        if re.fullmatch(r"(19|20)\d\d", token) or re.fullmatch(r"\d{4}\.\d{4,5}", token):
            continue  # a year or an arXiv id, not a measurement
        seen.append(token)
    return seen


def _parse(completion: Completion, role: str) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise PaperError(f"{role} pass did not return JSON: {text[:120]!r}")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PaperError(f"{role} pass returned unparsable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PaperError(f"{role} pass returned {type(payload).__name__}, not an object")
    return payload


def _papers(cand: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Verified rows with abstracts (`papers.json`), else the protocol's cite ids."""
    rows: list[dict[str, Any]] = []
    try:
        raw = json.loads((cand / "papers.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = []
    source = raw if isinstance(raw, list) and raw else (protocol.get("papers") or [])
    for p in source:
        if not isinstance(p, dict):
            continue
        ident = str(p.get("arxiv_id") or p.get("work_id") or p.get("cite_id") or p.get("id") or "").strip()
        if not ident:
            continue
        rows.append(
            {
                "id": ident,
                "title": p.get("title"),
                "published": p.get("published"),
                "venue": p.get("venue"),
                "relevance": p.get("relevance") or _relevance_of(p, protocol),
                "abstract": (p.get("abstract") or "")[:900],
            }
        )
    return rows


def _relevance_of(paper: dict[str, Any], protocol: dict[str, Any]) -> str:
    try:
        from .worldfields import paper_relevance
    except Exception:
        return "unknown"
    return paper_relevance(
        str(paper.get("title") or ""),
        str(paper.get("abstract") or ""),
        str(protocol.get("claim") or ""),
        str(protocol.get("topic") or ""),
    )


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _text(path: Path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _world_summary(path: Path) -> dict[str, Any]:
    payload = _json(path)
    if not payload:
        return {}
    out: dict[str, Any] = {"n": payload.get("n")}
    updates = payload.get("updates")
    if isinstance(updates, list) and updates:
        out["updates"] = len(updates)
        out["accepted"] = sum(1 for u in updates if isinstance(u, dict) and u.get("accepted"))
        out["divergent"] = sum(1 for u in updates if isinstance(u, dict) and u.get("divergent"))
        out["validator_mutations"] = sum(
            1 for u in updates if isinstance(u, dict) and u.get("validator_writes")
        )
        out["cells"] = len(payload.get("cells") or [])
        out["validators"] = len(payload.get("validators") or [])
        out["invariant"] = payload.get("invariant", "")
    return out


def _review_for(path: Path, card_id: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in _events(path):
        if row.get("stage") == "idea_review" and row.get("card_id") == card_id:
            latest = {
                k: row.get(k)
                for k in ("novelty", "clarity", "feasibility", "importance", "critique", "must_change", "value_vector")
            }
    return latest


def _promotion_for(path: Path, card_id: str) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for row in _events(path):
        if row.get("stage") == "promotion" and row.get("card_id") == card_id:
            latest = {k: row.get(k) for k in ("status", "reason", "previous")}
    return latest


def _events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except OSError:
        pass
    return rows


def _world_sim_holes(root: Path) -> list[str]:
    holes: list[str] = []
    root = Path(root)
    if not root.is_dir():
        return holes
    for assessment in sorted(root.glob("*/ASSESSMENT.md")):
        text = _text(assessment)
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "* ")) and len(stripped) > 40:
                holes.append(stripped[2:].strip())
    return holes[:12]


def _catalog_worlds(catalog_root: Path | None, *, schema: str) -> list[dict[str, Any]]:
    try:
        from .freeze import default_catalog
        from .world import load_fixture
    except Exception:
        return []
    root = Path(catalog_root) if catalog_root is not None else default_catalog()
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for directory in sorted(root.iterdir()):
        if not (directory / "manifest.json").is_file():
            continue
        try:
            fixture = load_fixture(directory)
        except Exception:
            continue
        if schema and fixture.schema != schema:
            continue
        if fixture.role in {"test", "fixture", "synthetic"}:
            continue
        rows.append(
            {
                "id": fixture.id,
                "schema": fixture.schema,
                "provenance": fixture.provenance or "published",
                "title": fixture.title,
                **_world_summary(directory / "world.json"),
            }
        )
    return rows
