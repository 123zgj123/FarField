"""Literature-grounded world lineage: search already retrieved, then construct.

A researcher does not invent a playground for an idea. They read how the
*claim's object* developed, name a published instance, and only then
build a world the idea can be simulated in. This module is that step.

What it is allowed to do:

- Read papers this mission already retrieved (`survey_around` / wiki).
  The LLM extracts a development path and a named instance. It does not
  open the network. Fetching a dataset after seeing the claim is a
  larger invented world — that remains `farfield freeze` on the host.
- Ground the *constructed* world (GENERATED) so it is about that
  instance, not a stub hashed from the claim prose.
- Enrich `var/world_wishlist.json` with a freeze recipe the operator can
  ingest later. A later freeze binds *new* claims; it does not reopen
  this mission's verdict.

What it is forbidden to do:

- Invent a citation that was not retrieved.
- Let the far-field mechanism vote for a different object family than
  the claim's `WorldRequirement`.
- Climb. A lineage-grounded construction is still GENERATED. Scout
  simulation on it can inform a redesign; it cannot corroborate.

If the papers are missing, the parse fails, or the object family
drifts, this module returns None and the existing construct/stub path
runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .schemas import FREEZABLE_SCHEMAS, OBJECT_SCHEMA, SCHEMAS

SYSTEM = (
    "You reconstruct how a scientific object developed, from retrieved"
    " papers only. Answer with one JSON object and nothing else."
)

TEMPLATE = """A research hypothesis needs an experimental world. Do not invent a playground. Read the retrieved papers and name how THIS CLAIM'S object developed, and which published instance the experiment should run on.

Researcher's topic: {topic}
Claim: {claim}
Proposed mechanism: {mechanism}
Prediction: {prediction}
Required object family (the claim's object — the mechanism does not vote): {object_type}

Retrieved papers (the only sources you may cite):
{papers}

Answer with JSON:
{{"lineage": "<=80 words: the development path of the CLAIM's object in these papers — successive constructions, what each added, what is still open",
 "named_instance": "<=30 words: the published instance / protocol / dataset / automaton this experiment should be about",
 "object_type": "{object_type}",
 "schema": "one of {schemas}, or empty if none of them is this object",
 "cite_ids": ["<cite_id from the list above>", "..."],
 "why_this_object": "<=40 words: why named_instance is the claim's scientific object, not the distant mechanism's internal cost",
 "freeze_source": "<=40 words: the public artifact a host should freeze later (dataset name, protocol, URL named in a paper), or empty",
 "freeze_url": "a URL that appears on one of the papers above, or empty",
 "freeze_schema": "one of {freezable}, or empty"}}

cite_ids must be ids from the list. Do not invent a citation. Do not change object_type. Do not describe a world that would make the idea win — describe the object the idea is about.
"""


@dataclass(frozen=True)
class WorldPath:
    """A literature-grounded brief for constructing or freezing a world."""

    card_id: str
    lineage: str
    named_instance: str
    object_type: str
    schema: str
    cite_ids: tuple[str, ...]
    why_this_object: str
    freeze_source: str = ""
    freeze_url: str = ""
    freeze_schema: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "card_id": self.card_id,
            "lineage": self.lineage,
            "named_instance": self.named_instance,
            "object_type": self.object_type,
            "schema": self.schema,
            "cite_ids": list(self.cite_ids),
            "why_this_object": self.why_this_object,
        }
        if self.freeze_source:
            payload["freeze_source"] = self.freeze_source
        if self.freeze_url:
            payload["freeze_url"] = self.freeze_url
        if self.freeze_schema:
            payload["freeze_schema"] = self.freeze_schema
        return payload

    def wishlist_fields(self) -> dict[str, Any]:
        """Extra keys to merge into an unmatched WorldRequirement."""
        extra: dict[str, Any] = {}
        if self.named_instance:
            extra["named_instance"] = self.named_instance
        if self.lineage:
            extra["lineage"] = self.lineage[:400]
        if self.cite_ids:
            extra["lineage_cite_ids"] = list(self.cite_ids)
        if self.freeze_source:
            extra["freeze_source"] = self.freeze_source
        if self.freeze_url:
            extra["freeze_url"] = self.freeze_url
        if self.freeze_schema:
            extra["schema"] = extra.get("schema") or self.freeze_schema
            extra["freeze_schema"] = self.freeze_schema
        return extra


def _paper_rows(works: list[Any] | None) -> list[tuple[str, str, str, str]]:
    """(cite_id, title, abstract, url) for retrieved works."""
    rows: list[tuple[str, str, str, str]] = []
    for work in works or []:
        cite = title = abstract = url = ""
        if hasattr(work, "cite_id"):
            cite = str(work.cite_id() or "")
            title = str(getattr(work, "title", "") or "")
            abstract = str(getattr(work, "abstract", "") or "")
            url = str(getattr(work, "url", "") or "") or str(
                getattr(work, "href", lambda: "")() or ""
            )
        elif isinstance(work, dict):
            cite = str(
                work.get("arxiv_id") or work.get("cite_id") or work.get("id") or ""
            )
            title = str(work.get("title") or "")
            abstract = str(work.get("abstract") or "")
            url = str(work.get("url") or "")
        if not cite:
            continue
        rows.append((cite, title, abstract, url))
    return rows


def _parse_object(text: str) -> dict[str, Any] | None:
    blob = str(text or "").strip()
    if blob.startswith("```"):
        blob = blob.split("\n", 1)[-1].rsplit("```", 1)[0]
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        import json

        payload = json.loads(blob[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _http_url(text: str) -> str:
    raw = str(text or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return raw
    return ""


def path_from_payload(
    card_id: str,
    payload: dict[str, Any],
    *,
    allowed_ids: set[str],
    allowed_urls: set[str],
    required_object: str = "",
) -> WorldPath | None:
    """Validate a model brief. Any invented citation or object drift is None."""
    lineage = str(payload.get("lineage") or "").strip()
    named = str(payload.get("named_instance") or "").strip()
    why = str(payload.get("why_this_object") or "").strip()
    if not lineage or not named or not why:
        return None
    object_type = str(payload.get("object_type") or required_object or "").strip()
    if required_object and object_type != required_object:
        return None
    if not object_type:
        return None
    raw_ids = payload.get("cite_ids") or []
    if not isinstance(raw_ids, (list, tuple)):
        return None
    cite_ids = tuple(str(item).strip() for item in raw_ids if str(item).strip())
    if not cite_ids or any(item not in allowed_ids for item in cite_ids):
        return None
    schema = str(payload.get("schema") or "").strip()
    if not schema:
        schema = OBJECT_SCHEMA.get(object_type, "")
    if schema and schema not in SCHEMAS:
        return None
    freeze_schema = str(payload.get("freeze_schema") or "").strip()
    if freeze_schema and freeze_schema not in FREEZABLE_SCHEMAS:
        freeze_schema = ""
    freeze_url = _http_url(payload.get("freeze_url") or "")
    if freeze_url and freeze_url not in allowed_urls:
        # A URL the papers did not publish is an invented retrieval.
        freeze_url = ""
    return WorldPath(
        card_id=card_id,
        lineage=lineage,
        named_instance=named,
        object_type=object_type,
        schema=schema,
        cite_ids=cite_ids,
        why_this_object=why,
        freeze_source=str(payload.get("freeze_source") or "").strip(),
        freeze_url=freeze_url,
        freeze_schema=freeze_schema,
    )


def write_world_path(
    client: Any,
    card: Any,
    topic: str,
    works: list[Any] | None,
    *,
    requirement: Any = None,
) -> WorldPath | None:
    """Extract a lineage from retrieved papers, or None (never blocks)."""
    rows = _paper_rows(works)
    if client is None or not rows or not hasattr(client, "complete"):
        return None
    object_type = ""
    if requirement is not None:
        if hasattr(requirement, "object_type"):
            object_type = str(requirement.object_type or "")
        elif isinstance(requirement, dict):
            object_type = str(requirement.get("object_type") or "")
    paper_lines = []
    allowed_ids: set[str] = set()
    allowed_urls: set[str] = set()
    for cite, title, abstract, url in rows:
        allowed_ids.add(cite)
        if url:
            allowed_urls.add(url)
        cue = " ".join(abstract.split())[:180]
        href = f" {url}" if url else ""
        paper_lines.append(
            f"- [{cite}] {title}{href}" + (f" — {cue}" if cue else "")
        )
    prompt = TEMPLATE.format(
        topic=topic,
        claim=getattr(card, "claim", ""),
        mechanism=getattr(card, "mechanism", ""),
        prediction=getattr(card, "prediction", ""),
        object_type=object_type or "the claim's object",
        papers="\n".join(paper_lines[:12]),
        schemas=", ".join(sorted(SCHEMAS)),
        freezable=", ".join(FREEZABLE_SCHEMAS),
    )
    try:
        completion = client.complete(
            prompt,
            purpose=f"world_path:{getattr(card, 'card_id', 'card')}",
            system=SYSTEM,
        )
        completion.assert_usable()
        payload = _parse_object(str(completion.text or ""))
    except Exception:
        return None
    if payload is None:
        return None
    return path_from_payload(
        str(getattr(card, "card_id", "") or ""),
        payload,
        allowed_ids=allowed_ids,
        allowed_urls=allowed_urls,
        required_object=object_type,
    )
