"""INV-13 research dossier: validation and markdown rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import (
    WAVE_A_PLACEHOLDERS,
    AttestedSpan,
    BlockedRecord,
    BudgetEnvelope,
    ConjectureCard,
)


ALLOWED_KINDS = {"research", "survey"}
FAKE_DISTANCE_TOKENS = {"0", "0.0", "0.00"}


@dataclass
class Dossier:
    seed: str
    campaign_id: str
    envelope: dict[str, float]
    objective: dict[str, Any]
    claims: list[dict[str, Any]]
    conjectures: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    experiments: list[dict[str, Any]]
    open_falsifiers: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    state: dict[str, Any]
    kind: str = "research"
    blocked: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if self.kind not in ALLOWED_KINDS:
            raise ValueError(f"dossier.kind must be research or survey, not {self.kind}")
        if not self.seed.strip():
            raise ValueError("dossier.seed must be the verbatim human seed")
        if not self.campaign_id.strip():
            raise ValueError("dossier.campaign_id must be non-empty")
        BudgetEnvelope.from_dict(self.envelope)
        required_objective = {"text", "id", "digest"}
        missing_obj = required_objective - set(self.objective)
        if missing_obj:
            raise ValueError(f"dossier.objective missing {sorted(missing_obj)}")
        if not self.conjectures:
            raise ValueError("dossier must contain at least one conjecture")
        for claim in self.claims:
            if "conjecture" in claim and claim.get("kind") == "conjecture":
                raise ValueError("claims must not contain conjecture records")
            if not claim.get("evidence_ids"):
                raise ValueError("claims require bound evidence IDs")
        for card in self.conjectures:
            _reject_fake_distance(card.get("graph_distance"))
            if card.get("dv") not in {None, WAVE_A_PLACEHOLDERS["dv"]} and not isinstance(
                card.get("dv"), dict
            ):
                if card.get("dv") in FAKE_DISTANCE_TOKENS:
                    raise ValueError("dv placeholder must be 'not_computed', not a fake zero")
            source = str(card.get("source_id") or "").strip()
            if not source:
                raise ValueError("conjecture must cite a literature source_id")
            if card.get("span_end", -1) < card.get("span_start", 0):
                raise ValueError("conjecture span offsets are invalid")
        for item in self.evidence:
            dv = item.get("dv", WAVE_A_PLACEHOLDERS["dv"])
            # `pre_judgment` covers two different states: no judge ran at all, and
            # a judge ran but moved nothing. Only the first forbids a DV. Reading
            # them as one makes an honestly measured DV of 0.0 indistinguishable
            # from a forged zero, and the campaign cannot settle at all.
            if not self.state.get("judgment_ran") and self.state.get(
                "pre_judgment", True
            ):
                if dv in FAKE_DISTANCE_TOKENS or dv == 0 or dv == 0.0:
                    raise ValueError("evidence.dv placeholder must be 'not_computed', not 0")
                if dv != WAVE_A_PLACEHOLDERS["dv"] and not isinstance(dv, dict):
                    raise ValueError("Wave A evidence.dv must be 'not_computed'")
            elif isinstance(dv, (int, float)) and not isinstance(dv, bool):
                if not 0.0 <= float(dv) <= 1.0:
                    raise ValueError("realized evidence.dv must be in [0, 1]")
            elif dv != WAVE_A_PLACEHOLDERS["dv"] and not isinstance(dv, dict):
                raise ValueError("evidence.dv must be realized, 'not_computed', or a dict")
            if not item.get("source_id"):
                raise ValueError("evidence requires a source ID")
            if not item.get("trust_class"):
                raise ValueError("evidence requires a trust_class")
        for record in self.blocked:
            BlockedRecord(
                missing_capability=record["missing_capability"],
                attempted=record["attempted"],
                unlock_condition=record["unlock_condition"],
            ).validate()
        required_state = {
            "m_digest",
            "p_digest",
            "f_digest",
            "k_digest",
            "rating",
            "taint",
            "grounding",
            "pre_judgment",
            "graph_snapshot_id",
            "neighborhood",
        }
        missing_state = required_state - set(self.state)
        if missing_state:
            raise ValueError(f"dossier.state missing {sorted(missing_state)}")
        if self.state.get("neighborhood") not in {"defined", "undefined"}:
            raise ValueError("state.neighborhood must be defined or undefined")
        if self.state.get("graph_distance") in FAKE_DISTANCE_TOKENS:
            raise ValueError("state graph_distance must not be a fake zero")
        for key in ("m_digest", "p_digest", "f_digest", "k_digest"):
            if self.state.get(key) in FAKE_DISTANCE_TOKENS:
                raise ValueError(f"state.{key} must be 'none', not a fake zero")
        fraction = blocked_fraction(self.experiments, self.blocked)
        if fraction > 0.50 and self.kind != "survey":
            raise ValueError("blocked fraction > 0.50 requires kind=survey")
        if fraction <= 0.50 and self.kind == "survey" and _has_executed_experiment(self.experiments):
            raise ValueError("executed experiments with blocked fraction <= 0.50 are research")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "kind": self.kind,
            "seed": self.seed,
            "campaign_id": self.campaign_id,
            "envelope": dict(self.envelope),
            "objective": dict(self.objective),
            "claims": list(self.claims),
            "conjectures": list(self.conjectures),
            "evidence": list(self.evidence),
            "experiments": list(self.experiments),
            "open_falsifiers": list(self.open_falsifiers),
            "failures": list(self.failures),
            "blocked": list(self.blocked),
            "state": dict(self.state),
        }

    def to_markdown(self) -> str:
        self.validate()
        banner = ""
        if self.kind == "survey":
            banner = (
                "> This is not completed research. The campaign is a literature "
                "survey because experiments could not run (INV-14).\n\n"
            )
        claims = _bullets(self.claims)
        conjectures = _bullets(self.conjectures)
        evidence = _bullets(self.evidence)
        experiments = _bullets(self.experiments)
        falsifiers = _bullets(self.open_falsifiers)
        failures = _bullets(self.failures) or "- none"
        blocked = _bullets(self.blocked) or "- none"
        state = _bullets([self.state])
        envelope = ", ".join(f"{k}={v}" for k, v in sorted(self.envelope.items()))
        return (
            f"{banner}# Farfield dossier (`{self.kind}`)\n\n"
            f"## seed\n\n{self.seed}\n\n"
            f"campaign_id: `{self.campaign_id}`\n\n"
            f"envelope: {envelope}\n\n"
            f"## objective\n\n{_inline(self.objective)}\n\n"
            f"## claims\n\n{claims}\n\n"
            f"## conjectures\n\n{conjectures}\n\n"
            f"## evidence\n\n{evidence}\n\n"
            f"## experiments\n\n{experiments}\n\n"
            f"## open_falsifiers\n\n{falsifiers}\n\n"
            f"## failures\n\n{failures}\n\n"
            f"## blocked\n\n{blocked}\n\n"
            f"## state\n\n{state}\n"
        )


def blocked_fraction(
    experiments: list[dict[str, Any]], blocked: list[dict[str, Any]]
) -> float:
    planned = 0
    blocked_runs = 0
    for item in experiments:
        planned += 1
        if item.get("status") == "blocked":
            blocked_runs += 1
    if blocked and planned == 0:
        planned = len(blocked)
        blocked_runs = len(blocked)
    if planned == 0:
        return 1.0 if blocked else 0.0
    return blocked_runs / planned


def decide_kind(experiments: list[dict[str, Any]], blocked: list[dict[str, Any]]) -> str:
    if blocked_fraction(experiments, blocked) > 0.50:
        return "survey"
    return "research"


def wave_a_state(
    *,
    graph_snapshot_id: str = "none",
    neighborhood: str = "undefined",
    grounding: str | None = None,
    pre_judgment: bool = True,
) -> dict[str, Any]:
    return {
        "m_digest": WAVE_A_PLACEHOLDERS["mpf_digest"],
        "p_digest": WAVE_A_PLACEHOLDERS["mpf_digest"],
        "f_digest": WAVE_A_PLACEHOLDERS["mpf_digest"],
        "k_digest": WAVE_A_PLACEHOLDERS["mpf_digest"],
        "rating": WAVE_A_PLACEHOLDERS["rating"],
        "taint": WAVE_A_PLACEHOLDERS["taint"],
        "grounding": grounding or WAVE_A_PLACEHOLDERS["grounding"],
        "pre_judgment": bool(pre_judgment),
        # Absent or false means no judge ran, so no DV may exist.
        "judgment_ran": False,
        "graph_distance": WAVE_A_PLACEHOLDERS["graph_distance"],
        "graph_snapshot_id": graph_snapshot_id,
        "neighborhood": neighborhood,
    }


def conjecture_from_span(seed: str, span: AttestedSpan, card: ConjectureCard) -> dict[str, Any]:
    card.validate()
    span.validate()
    if card.source_id != span.source_id:
        raise ValueError("conjecture source_id must match the attested span")
    if card.span_start != span.start or card.span_end != span.end:
        raise ValueError("conjecture must cite the attested span offsets")
    payload = card.to_dict()
    payload["dv"] = WAVE_A_PLACEHOLDERS["dv"]
    payload["seed_excerpt"] = seed
    payload["span_text"] = span.text
    return payload


def _has_executed_experiment(experiments: list[dict[str, Any]]) -> bool:
    return any(item.get("status") in {"done", "failed", "ran"} for item in experiments)


def _reject_fake_distance(value: Any) -> None:
    if value in FAKE_DISTANCE_TOKENS or value == 0 or value == 0.0:
        raise ValueError("graph_distance placeholder must be 'unknown', not 0.0")


def _inline(payload: dict[str, Any]) -> str:
    return ", ".join(f"{key}={payload[key]}" for key in payload)


def _bullets(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- none"
    lines = []
    for row in rows:
        body = "; ".join(f"{key}={value}" for key, value in row.items())
        lines.append(f"- {body}")
    return "\n".join(lines)
