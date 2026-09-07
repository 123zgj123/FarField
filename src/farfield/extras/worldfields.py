"""Attested field paths: the script must read what the freeze actually has.

Schema match is not scientific match, and a matching field *name* is
not enough either. The Live-SWE labeled_traces freeze stores
`created_tools` on each trace; a probe that looks the same name up on
each step always sees empty and can still emit two numbers. That is
not a measurement of the claimed field.

These checks never climb the ladder. They refuse a diagnosis or probe
before it is admitted, the same kind of tooth as a fair two-arm.
Empty schema or empty claim is a no-op so existing unit tests keep
their bytes.
"""

from __future__ import annotations

import re
from typing import Any

from .schemas import attested_layout

# Phrases a claim may use that this freeze does not store. Naming one
# without also naming an attested stand-in (action_chars, returncode,
# created_tools) is a missing-field diagnosis, not a two-arm.
_UNATTESTED: dict[str, tuple[str, ...]] = {
    "labeled_traces": (
        "job processing time",
        "duration tertile",
        "wall-clock",
        "command duration",
        "processing times",
        "token span from command",
    ),
}

_CLOCK_STANDIN = (
    "action_chars",
    "output_chars",
)

_CREATED = re.compile(r"\bcreated_tools?\b", re.I)
_STEP_CREATED = re.compile(
    r"\bsteps?\s*(?:\[|\.get\()\s*['\"]created_tools?",
    re.I,
)
_TRACE_CREATED = re.compile(
    r"\b(?:trace|traces|raw|row|item|payload|record)\s*"
    r"(?:\[|\.get\()\s*['\"]created_tools?",
    re.I,
)
_DERIVED_CREATED = re.compile(
    r"['\"]created_tool_count['\"]|\bderived\b.{0,40}created_tool",
    re.I | re.S,
)
_TOOL_NAME = re.compile(
    r"created_tools.{0,160}(?:\[['\"]name['\"]\]|\.get\(\s*['\"]name)",
    re.I | re.S,
)
_TOOL_COUNT = re.compile(
    r"len\s*\(\s*.{0,80}created_tools|created_tool_count",
    re.I | re.S,
)


def layout_prompt_block(schema: str) -> str:
    """Paragraph for diagnose/probe prompts. Empty when the schema has none."""
    text = attested_layout(schema)
    if not text:
        return ""
    return "Attested field paths on this freeze: " + text + "\n"


# Top-level JSON keys that identify a schema's instance. A probe bound to
# one of these families must read at least one of its own keys; reading
# another family's keys instead is a script written for a world that is
# not in data/. Text and fasta worlds are files, not keyed objects, and
# are not listed.
_LAYOUT_KEYS: dict[str, frozenset[str]] = {
    "undirected_graph": frozenset({"nodes", "edges"}),
    "labeled_traces": frozenset({"traces"}),
    "program_state": frozenset({"cells", "validators", "updates"}),
    "symbolic_trace": frozenset({"states", "transitions"}),
    "numeric_table": frozenset({"rows", "columns"}),
}

_KEY_READ = re.compile(
    r"(?:\[\s*['\"](?P<sub>[A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]"
    r"|\.get\(\s*['\"](?P<get>[A-Za-z_][A-Za-z0-9_]*)['\"])"
)


def _keys_read(source: str) -> set[str]:
    found: set[str] = set()
    for match in _KEY_READ.finditer(str(source or "")):
        found.add(match.group("sub") or match.group("get") or "")
    found.discard("")
    return found


def probe_reads_bound_layout(source: str, schema: str) -> str | None:
    """Refuse a script whose top-level reads belong to another schema.

    The Darwin–Gödel dropout probe read `states` and `transitions` off a
    world whose bound family was program_state; both arms returned 0/0
    and the run was booked as `object_absent`. That is not a finding
    about the object — it is a script for a different fixture. Empty
    schema, an unlisted family, or a script that reads none of the
    listed keys of any family is a no-op here.
    """
    family = str(schema or "").strip()
    own = _LAYOUT_KEYS.get(family)
    if not own:
        return None
    read = _keys_read(source)
    if read & own:
        return None
    foreign_hits: dict[str, set[str]] = {}
    for other, keys in _LAYOUT_KEYS.items():
        if other == family:
            continue
        hit = (read & keys) - own
        if hit:
            foreign_hits[other] = hit
    if not foreign_hits:
        return None
    detail = "; ".join(
        f"{sorted(keys)} belong to {other}" for other, keys in sorted(foreign_hits.items())
    )
    layout = attested_layout(family)
    return (
        f"the bound world is {family}: read payload{sorted(own)} — the script "
        f"reads {detail}, so both arms would count on a world that is not "
        f"in data/. {layout}".strip()
    )


def _blob(*parts: str) -> str:
    return " ".join(str(part or "") for part in parts)


def missing_attested_field(
    claim: str,
    mechanism: str = "",
    schema: str = "",
    world_lever: str = "",
) -> str | None:
    """Refuse when the claim names a quantity this freeze does not store.

    `world_lever: none` is the honest skip: the prediction has no handle
    on this freeze. That skip is not a two-arm on invented step times.
    """
    if str(world_lever or "").strip().lower() == "none":
        return None
    schema = str(schema or "").strip()
    phrases = _UNATTESTED.get(schema)
    if not phrases:
        return None
    text = _blob(claim, mechanism).lower()
    if not text.strip():
        return None
    hit = next((phrase for phrase in phrases if phrase in text), "")
    if not hit:
        return None
    if any(name in text for name in _CLOCK_STANDIN):
        return None
    layout = attested_layout(schema)
    detail = layout or "the bound freeze does not store that field"
    return (
        f"the claim names {hit!r}, which this {schema} freeze does not "
        f"attest; {detail} Name action_chars or output_chars as the clock, "
        "or set world_lever to none. Do not invent a duration on each step."
    )


def diagnosis_names_claimed_fields(
    experiment: str = "",
    treatment: str = "",
    control: str = "",
    claim: str = "",
    mechanism: str = "",
    schema: str = "",
) -> str | None:
    """Refuse a two-arm that never names the field the claim is about."""
    schema = str(schema or "").strip()
    if schema != "labeled_traces":
        return None
    text = _blob(claim, mechanism)
    body = _blob(experiment, treatment, control)
    if not text.strip() or not body.strip():
        return None
    if _CREATED.search(text) and not _CREATED.search(body):
        return (
            "the claim names created_tools; the two-arm design never "
            "names that field. Treatment and control must load "
            "trace['created_tools'] (or derived['created_tool_count']), "
            "not a different quantity that happens to live on the same freeze."
        )
    return None


def probe_reads_attested_fields(
    source: str,
    claim: str = "",
    mechanism: str = "",
    schema: str = "",
) -> str | None:
    """Refuse a script that looks a claimed field up on the wrong object."""
    schema = str(schema or "").strip()
    if schema != "labeled_traces":
        return None
    text = _blob(claim, mechanism)
    body = str(source or "")
    if not text.strip() or not body.strip():
        return None
    if not _CREATED.search(text):
        return None
    reads_trace = bool(_TRACE_CREATED.search(body) or _DERIVED_CREATED.search(body))
    if reads_trace:
        if _DERIVED_CREATED.search(body) or _TOOL_COUNT.search(body) or _TOOL_NAME.search(
            body
        ):
            return None
        return (
            "created_tools on this freeze is a list of records with a "
            "name field; project item['name'] (or count via len / "
            "derived['created_tool_count']). Tokenizing step['action'] "
            "is not a measurement of created_tools."
        )
    if _CREATED.search(body) and _STEP_CREATED.search(body):
        return (
            "created_tools lives on each trace in this freeze, not on "
            "steps; step.get('created_tools') is always empty. Read "
            "trace['created_tools'] or derived['created_tool_count']."
        )
    if _CREATED.search(body) and not _TRACE_CREATED.search(body):
        return (
            "the claim names created_tools; load it from each trace "
            "(payload['traces'][i]['created_tools']), not from a step dict"
        )
    if not _CREATED.search(body):
        return (
            "the claim names created_tools; the script never reads "
            "trace['created_tools'] or derived['created_tool_count'] "
            "from the bound freeze"
        )
    return None


_PAPER_PAD = frozenset(
    {
        "index",
        "ratio",
        "model",
        "count",
        "value",
        "result",
        "study",
        "method",
        "based",
        "using",
        "paper",
        "system",
        "approach",
        "claim",
        "topic",
        "effect",
        "test",
        "data",
        "field",
        "work",
        "mechanism",
        "prediction",
    }
)


def paper_on_claim_object(
    title: str,
    abstract: str,
    claim: str,
    topic: str = "",
) -> bool:
    """True when the paper still names the claim's object, not only a far word.

    Live retrieve is allowed to be noisy. Briefing and diagnosis baselines
    may not keep a drought-index or obesity paper next to a SWE-agent
    claim just because both say "index" — or "world". A topic bigram is
    enough; anything less needs two distinctive unigrams. One shared
    generic word was the live hole that let 1975 obesity prevalence sit
    in a code-world-model bibliography.
    """
    from .domain import folded_terms, topic_object_phrases

    blob = f"{title} {abstract}".lower()
    if not blob.strip():
        return False
    for gram in topic_object_phrases(topic or claim):
        if " " in gram and gram in blob:
            return True
    surface = folded_terms(blob)
    object_terms = (
        folded_terms(claim) | folded_terms(topic)
    ) - _PAPER_PAD
    object_terms = frozenset(word for word in object_terms if len(word) >= 5)
    if not object_terms:
        return True
    hits = surface & object_terms
    if len(object_terms) == 1:
        return bool(hits)
    return len(hits) >= 2


def paper_relevance(
    title: str,
    abstract: str,
    claim: str,
    topic: str = "",
) -> str:
    """`strong`, `weak`, or `none`: how much of the claim's object the paper names.

    Verified is not relevant. In the cwm-iclr2027 bibliography VisCAD,
    FuncRoom and a healthcare-trajectory pipeline sat next to four code
    world model papers because each named one generic topic bigram
    ("executable program") or two loose unigrams. Strong needs the paper
    to name the object twice over: two distinct topic phrases, or one
    topic phrase plus a claim phrase, or a topic phrase of three words or
    more. One topic phrase alone, or unigrams alone, is weak — retrieved,
    verified, and tangential. Weak rows stay in the record with that
    label; they do not enter related work or baselines while strong rows
    exist.
    """
    from .domain import topic_object_phrases

    blob = f"{title} {abstract}".lower()
    if not blob.strip():
        return "none"
    phrases = [g for g in topic_object_phrases(topic or claim) if " " in g]
    topic_hits = [g for g in phrases if g in blob]
    # A paper that names the object in its *title* is about it; one that
    # mentions "executable programs" once in an abstract about CAD is not.
    title_hits = [g for g in phrases if g in str(title or "").lower()]
    claim_hits = (
        [g for g in topic_object_phrases(claim) if " " in g and g in blob and g not in topic_hits]
        if topic and claim
        else []
    )
    if (
        title_hits
        or len(topic_hits) >= 2
        or (topic_hits and claim_hits)
        or any(len(g.split()) >= 3 for g in topic_hits)
    ):
        return "strong"
    if topic_hits or paper_on_claim_object(title, abstract, claim, topic):
        return "weak"
    return "none"


def _work_text(work: Any) -> tuple[str, str]:
    if hasattr(work, "title"):
        return str(getattr(work, "title", "") or ""), str(getattr(work, "abstract", "") or "")
    if isinstance(work, dict):
        return str(work.get("title") or ""), str(work.get("abstract") or "")
    return "", ""


def papers_on_claim_object(
    works: list[Any] | tuple[Any, ...] | None,
    *,
    claim: str,
    topic: str = "",
    minimum: str = "weak",
) -> list[Any]:
    """Rows that name the claim's object, strongest first.

    Identity is the claim, not the topic. An empty claim is a contract
    violation — callers may not recover by substituting the mission topic.

    `minimum="strong"` returns only strong rows. Thin retrieve stays thin.
    """
    from .claimspec import ContractViolation, LiteratureSet

    if not str(claim or "").strip():
        raise ContractViolation("papers_on_claim_object requires a non-empty claim")
    strong: list[Any] = []
    weak: list[Any] = []
    for work in works or []:
        title, abstract = _work_text(work)
        tier = paper_relevance(title, abstract, claim, topic)
        if tier == "strong":
            strong.append(work)
        elif tier == "weak":
            weak.append(work)
    literature = LiteratureSet(
        claim_object_strong=tuple(strong),
        claim_object_weak=tuple(weak),
    )
    if minimum == "strong":
        return list(literature.related_work())
    return list(literature.claim_object_strong + literature.claim_object_weak)
