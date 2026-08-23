"""Research routing: learn across missions which operators earn promotions,
and change how the next mission allocates its slots — never how it judges.

(The kernel's `farfield.policy` is the Wave-D install gate; this module is
the *research* policy, so it is named for the one thing it may touch:
routing.)

This is the meta loop the P3 result asks for. Do not mutate ideas by
score — inspect *trajectories*: every mission appends, per card, which
operator produced it and what the executable pipeline then did to it
(killed, briefed, probe verdict, promotion). From that log a candidate
routing policy is proposed deterministically. The commit rule is the
whole point:

    a candidate policy is installed only if it scores strictly higher
    than the incumbent on missions it was NOT fitted to.

Otherwise the update is recorded as rejected and the incumbent stands.
The score is promotion precision — of the cards the policy's weights
prefer, how many ended corroborated or verified — because promotions are
the only outcome in this system that required evidence. WORLD-supports
that have not yet climbed still count at half weight. Elo / value_score
does not enter the score: it is a colleague opinion, not a routing reward.
SYNTHETIC supports never steer search.

An installed policy reorders the operator cycle (far-field and reframe
alike). It cannot touch the gates, the probe arithmetic, the promotion
ladder, or the snapshot discipline: those are how evidence is made, and
evidence is not up for optimisation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LOG_CAP = 200
SMOOTHING = 1.0
SCORE_TOLERANCE = 1e-9
PROMOTED = ("corroborated", "verified")

# Schema 2 began when cards started carrying `probe_kind` and the W2
# observations. Earlier records stay in the log as history, but they
# cannot vote in any settlement: a support that cannot prove it ran on
# an attested world is not evidence for a routing (or judge) change.
LOG_SCHEMA = 2

# The only probe kinds that can attest a real-world run. A missing or
# unknown kind earns nothing — absence of bookkeeping is not a WORLD.
WORLD_KINDS = frozenset({"WORLD", "REAL", "FIXTURE"})


class PolicyError(Exception):
    """The log or policy file refused to load: digest mismatch."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def load_log(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    missions = payload.get("missions")
    if not isinstance(missions, list) or payload.get("digest") != _digest(missions):
        raise PolicyError(
            f"{path} does not match its own digest; refusing a trajectory log"
            " that was edited outside this module"
        )
    return missions


def settlement_missions(missions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The missions allowed to vote: current-schema records only.

    Legacy records predate `probe_kind`, so their 21 supports cannot be
    told apart from SYNTHETIC coherence checks. They remain readable
    history and age out at LOG_CAP, but no meta loop fits or judges on
    them.
    """
    return [
        mission
        for mission in missions
        if int(mission.get("schema") or 0) >= LOG_SCHEMA
    ]


def append_mission(path: Path, record: dict[str, Any]) -> int:
    """Bank one mission's trajectory; returns the log's new length."""
    from .filelock import file_lock

    path = Path(path)
    with file_lock(path):
        missions = load_log(path)
        missions.append({**record, "schema": LOG_SCHEMA})
        missions = missions[-LOG_CAP:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"missions": missions, "digest": _digest(missions)},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return len(missions)


def operator_report(missions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per operator: how many cards, and how far they got."""
    report: dict[str, dict[str, Any]] = {}
    for mission in missions:
        for card in mission.get("cards") or []:
            op = str(card.get("operator") or "unknown")
            row = report.setdefault(
                op,
                {
                    "cards": 0,
                    "survived": 0,
                    "briefed": 0,
                    "supported": 0,
                    "promoted": 0,
                    "world_supported": 0,
                },
            )
            row["cards"] += 1
            row["survived"] += 0 if card.get("killed") else 1
            row["briefed"] += 1 if card.get("briefed") else 0
            row["supported"] += 1 if card.get("verdict") == "supports" else 0
            row["promoted"] += 1 if card.get("promoted") in PROMOTED else 0
            kind = str(card.get("probe_kind") or "").upper()
            if card.get("promoted") in PROMOTED and kind in WORLD_KINDS:
                row["world_supported"] += 1
            elif card.get("host_ok") is True and card.get("verdict") == "supports" and kind in WORLD_KINDS:
                row["world_supported"] += 1
    for row in report.values():
        row["promotion_rate"] = (
            row["promoted"] / row["cards"] if row["cards"] else 0.0
        )
    return report


def propose_policy(missions: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic candidate: Laplace-smoothed promotion rate per operator."""
    report = operator_report(missions)
    weights = {
        op: (row["promoted"] + row.get("world_supported", 0) + SMOOTHING)
        / (row["cards"] + 2 * SMOOTHING)
        for op, row in sorted(report.items())
    }
    return {"weights": weights, "fitted_on": len(missions)}


def _card_credit(card: dict[str, Any]) -> float:
    """Routing reward: host-confirmed WORLD promotions. Never Elo.

    Probe-only WORLD supports are pending confirmation and do not steer
    search. Infrastructure failures have ledger=diagnostic and earn zero.
    """
    if str(card.get("ledger") or "") == "diagnostic":
        return 0.0
    if card.get("promoted") in PROMOTED:
        return 1.0
    kind = str(card.get("probe_kind") or "").upper()
    if (
        card.get("verdict") == "supports"
        and kind in WORLD_KINDS
        and card.get("host_ok") is True
    ):
        return 1.0
    return 0.0


def score_policy(
    policy: dict[str, Any] | None, missions: list[dict[str, Any]]
) -> float:
    """Weighted promotion precision on the given missions.

    With no policy, every operator weighs the same — the uniform incumbent
    a candidate must beat. SYNTHETIC Elo cannot steer search.
    """
    weights = (policy or {}).get("weights") or {}
    gained = 0.0
    spent = 0.0
    for mission in missions:
        for card in mission.get("cards") or []:
            weight = float(weights.get(str(card.get("operator")), 1.0)) if weights else 1.0
            spent += weight
            gained += weight * _card_credit(card)
    return gained / spent if spent else 0.0


def load_policy(path: Path) -> dict[str, Any] | None:
    if not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    body = payload.get("policy")
    if not isinstance(body, dict) or payload.get("digest") != _digest(body):
        raise PolicyError(f"{path} does not match its own digest")
    return body


def commit_if_better(
    policy_path: Path,
    candidate: dict[str, Any],
    heldout: list[dict[str, Any]],
) -> dict[str, Any]:
    """Install the candidate only if it beats the incumbent on held-out
    missions it was not fitted to. The decision is returned either way."""
    from .filelock import file_lock

    path = Path(policy_path)
    with file_lock(path):
        incumbent = load_policy(path)
        candidate_score = score_policy(candidate, heldout)
        incumbent_score = score_policy(incumbent, heldout)
        # Strictly better means measurably better: a live commit once slipped
        # through on the last float ulp when the candidate was arithmetically
        # identical to the uniform incumbent (0.2*x/0.2*y vs x/y).
        committed = bool(heldout) and candidate_score > incumbent_score + SCORE_TOLERANCE
        if committed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"policy": candidate, "digest": _digest(candidate)},
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    return {
        "committed": committed,
        "candidate_score": round(candidate_score, 6),
        "incumbent_score": round(incumbent_score, 6),
        "heldout_missions": len(heldout),
        "reason": (
            "candidate beat the incumbent on held-out missions"
            if committed
            else "candidate did not beat the incumbent on held-out missions;"
            " the incumbent stands"
        ),
    }


def preferred_operators(
    policy: dict[str, Any] | None, operators: tuple[str, ...]
) -> tuple[str, ...]:
    """Order an operator cycle by installed policy weight, ties by name.

    With no installed policy the declared order stands — routing changes
    are earned on held-out missions, never assumed. Works for the
    far-field cycle and the reframe cycle alike.
    """
    weights = (policy or {}).get("weights") or {}
    if not weights:
        return operators
    return tuple(
        sorted(operators, key=lambda op: (-float(weights.get(op, 0.0)), op))
    )


MIN_MISSIONS = 6


def maybe_update(
    log_path: Path,
    policy_path: Path,
    *,
    min_missions: int = MIN_MISSIONS,
) -> dict[str, Any] | None:
    """The whole meta loop, safe to run at the end of every mission.

    Fits a candidate on the even-indexed missions, judges it on the
    odd-indexed ones it never saw, and installs it only if it wins.
    Returns the decision, or None when the log is still too short for the
    held-out half to mean anything. Only current-schema missions vote:
    pre-`probe_kind` history is read but never fitted or judged on.
    """
    missions = settlement_missions(load_log(log_path))
    if len(missions) < min_missions:
        return None
    candidate = propose_policy(missions[0::2])
    decision = commit_if_better(policy_path, candidate, missions[1::2])
    decision["candidate"] = candidate
    return decision
