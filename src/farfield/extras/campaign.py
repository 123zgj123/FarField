"""A campaign is a sequence of missions against a stable intent.

Argus persists a Voyage as files; Cursor `/loop` wakes an agent. This
module is only the scientific identity (JSON + digest), not a second
runtime: no LangGraph checkpointer, no four LLM roles. Product loop stays
`run_mission`. Rollback records no-go and must not rewrite finished
verdicts. Unattended ticks belong to `/loop` plus skill `campaign-horizon`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUTHORITIES = ("operator", "manager", "planner", "engineer", "reviewer")
STAGE_KINDS = ("open", "mission", "pause", "rollback", "close")


class CampaignError(ValueError):
    """Campaign ledger refused the write."""


def campaigns_dir(root: Path) -> Path:
    return Path(root) / "var" / "campaigns"


def campaign_path(root: Path, campaign_id: str) -> Path:
    ident = _require_id(campaign_id)
    return campaigns_dir(root) / ident / "campaign.json"


def _require_id(campaign_id: str) -> str:
    ident = str(campaign_id or "").strip()
    if not ident or any(ch in ident for ch in "/\\") or ident.startswith("."):
        raise CampaignError(f"bad campaign id {campaign_id!r}")
    return ident


def _digest(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _authority(name: str) -> str:
    text = str(name or "operator").strip().lower() or "operator"
    if text not in AUTHORITIES:
        raise CampaignError(f"unknown authority {name!r}; use {AUTHORITIES}")
    return text


def load_campaign(root: Path, campaign_id: str) -> dict[str, Any]:
    path = campaign_path(root, campaign_id)
    if not path.is_file():
        raise CampaignError(f"campaign {campaign_id!r} is not in var/campaigns/")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError("campaign.json must be an object")
    recorded = str(payload.get("digest") or "")
    if recorded and recorded != _digest(payload):
        raise CampaignError("campaign digest mismatch; refuse a hand edit")
    return payload


def _write(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    ident = _require_id(str(payload.get("id") or ""))
    dest = campaign_path(root, ident)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload["digest"] = _digest(payload)
    dest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def init_campaign(
    root: Path,
    campaign_id: str,
    *,
    intent: str,
    authority: str = "operator",
) -> dict[str, Any]:
    ident = _require_id(campaign_id)
    path = campaign_path(root, ident)
    if path.is_file():
        raise CampaignError(f"campaign {ident!r} already exists")
    heading = str(intent or "").strip()
    if not heading:
        raise CampaignError("campaign intent is empty")
    payload = {
        "id": ident,
        "intent": heading,
        "status": "open",
        "honesty": (
            "a rollback records that a stage is no-go; it does not rewrite "
            "finished mission verdicts, EvidenceIDs, or corroborated status"
        ),
        "missions": [],
        "stages": [
            {
                "n": 1,
                "kind": "open",
                "authority": _authority(authority),
                "at": _now(),
                "note": "intent frozen; operational objective may later pause or roll",
            }
        ],
    }
    return _write(root, payload)


def append_mission(
    root: Path,
    campaign_id: str,
    mission: Path,
    *,
    authority: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    payload = load_campaign(root, campaign_id)
    if payload.get("status") == "close":
        raise CampaignError("closed campaign cannot append a mission")
    folder = Path(mission)
    if not folder.exists():
        raise CampaignError(f"mission path missing: {folder}")
    entry = str(folder)
    if entry in payload["missions"]:
        raise CampaignError(f"mission {entry} is already on this campaign")
    payload["missions"].append(entry)
    payload["stages"].append(
        {
            "n": len(payload["stages"]) + 1,
            "kind": "mission",
            "authority": _authority(authority),
            "at": _now(),
            "mission": entry,
            "note": str(note or "").strip(),
        }
    )
    payload["status"] = "open"
    return _write(root, payload)


def rollback_stage(
    root: Path,
    campaign_id: str,
    *,
    reason: str,
    authority: str = "operator",
) -> dict[str, Any]:
    payload = load_campaign(root, campaign_id)
    if payload.get("status") == "close":
        raise CampaignError("closed campaign cannot roll back")
    why = str(reason or "").strip()
    if len(why) < 8:
        raise CampaignError("rollback needs a recorded reason (>= 8 chars)")
    last = payload["stages"][-1]
    payload["stages"].append(
        {
            "n": len(payload["stages"]) + 1,
            "kind": "rollback",
            "authority": _authority(authority),
            "at": _now(),
            "from_stage": last.get("n"),
            "from_kind": last.get("kind"),
            "reason": why,
            "note": (
                "scientific records in the listed missions stay as written; "
                "this stage is no-go for the operational objective"
            ),
        }
    )
    payload["status"] = "open"
    return _write(root, payload)


def close_campaign(
    root: Path, campaign_id: str, *, authority: str = "operator", note: str = ""
) -> dict[str, Any]:
    payload = load_campaign(root, campaign_id)
    payload["stages"].append(
        {
            "n": len(payload["stages"]) + 1,
            "kind": "close",
            "authority": _authority(authority),
            "at": _now(),
            "note": str(note or "").strip(),
        }
    )
    payload["status"] = "close"
    return _write(root, payload)
