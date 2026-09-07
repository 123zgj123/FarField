"""Host literature harvest. Retrieve is not admit.

The mission already runs this pipeline at `fresh` and `survey`. This
hook is the intern tooth: the same `admit_works` gate, never a prompt
that tells the model to believe a title.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from farfield.extras.knowledge import admit_works, harvest_recent
from farfield.extras.livefeed import CompositeFeed, FreshWork
from farfield.extras.state import record_wiki

NAME = "research-lit"
HOOKS = ("execute",)
DEFAULT_CORPUS = "ds-arxiv-concepts-2026"


def execute(args: dict[str, Any] | None = None, workspace: Any = None, **_: Any) -> str:
    args = args or {}
    skip_verify = bool(args.get("skip_verify"))
    canned = args.get("works")
    if isinstance(canned, list) and canned:
        rows = [
            item if isinstance(item, FreshWork) else FreshWork.from_dict(item)
            for item in canned
            if isinstance(item, (FreshWork, dict))
        ]
        harvest = admit_works(rows, verify=not skip_verify)
    else:
        concepts = _concepts(args)
        if not concepts:
            return json.dumps(
                {"ok": False, "error": "need topic, concepts, or works"},
                ensure_ascii=False,
            )
        feed = args.get("feed") or CompositeFeed()
        harvest = harvest_recent(
            feed,
            concepts,
            max_results=int(args.get("max_results") or 6),
        )
    added = 0
    state = str(args.get("state_store") or "").strip()
    if state and harvest.works:
        anchor = str(args.get("anchor") or concepts_fallback(args) or "literature")
        added = record_wiki(
            Path(state),
            str(args.get("corpus") or DEFAULT_CORPUS),
            anchor,
            harvest.works,
            seen_at=str(args.get("seen_at") or date.today().isoformat()),
        )
    payload = harvest.to_dict()
    payload["ok"] = True
    payload["added"] = added
    payload["workspace"] = str(workspace or "")
    return json.dumps(payload, ensure_ascii=False)


def _concepts(args: dict[str, Any]) -> tuple[str, ...]:
    raw = args.get("concepts") or args.get("query") or args.get("topic")
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return ()


def concepts_fallback(args: dict[str, Any]) -> str:
    concepts = _concepts(args)
    return concepts[0] if concepts else ""
