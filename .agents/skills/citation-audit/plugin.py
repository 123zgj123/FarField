"""Keep briefing citations on the claim object. Cannot invent papers."""

from __future__ import annotations

import json
from typing import Any

NAME = "citation-audit"
HOOKS = ("execute",)


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    works = args.get("works") or []
    if not isinstance(works, list):
        return json.dumps({"ok": False, "error": "works must be a list"}, ensure_ascii=False)
    from farfield.extras.worldfields import papers_on_claim_object

    kept = papers_on_claim_object(
        works,
        claim=str(args.get("claim") or ""),
        topic=str(args.get("topic") or ""),
    )
    kept_ids = {_cite(item) for item in kept}
    noise = [item for item in works if _cite(item) not in kept_ids]
    return json.dumps(
        {
            "ok": True,
            "on_object": [_row(item) for item in kept],
            "noise": [_row(item) for item in noise],
            "workspace": str(workspace or ""),
        },
        ensure_ascii=False,
    )


def _cite(work: Any) -> str:
    if hasattr(work, "cite_id"):
        return str(work.cite_id() or "")
    if isinstance(work, dict):
        return str(work.get("arxiv_id") or work.get("cite_id") or work.get("id") or "")
    return str(work)


def _row(work: Any) -> dict[str, str]:
    if isinstance(work, dict):
        return {
            "cite_id": _cite(work),
            "title": str(work.get("title") or ""),
        }
    return {
        "cite_id": _cite(work),
        "title": str(getattr(work, "title", "") or ""),
    }
