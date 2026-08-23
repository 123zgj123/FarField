"""Research State: the single long-term memory of the production path.

One store per (corpus, anchor concept), remembering four different kinds
of fact — and keeping them apart, because they answer different questions:

- **hypotheses**: claims that *entered research* (passed the gates), each
  with a status on the evidence ladder
  `speculative → corroborated → verified`, with `weakened` and
  `closed_by_prior` as the down moves. A card the gates refused never
  appears here: being ineligible is not a scientific outcome.
- **rejected**: the failure memory — combinations the executable judges
  killed, distilled to capsules and fed back into generation prompts as
  directions that already failed. This absorbs the old failure store;
  there is one memory now, not two.
- **program**: the one committed generation brief for the next
  mission — why this line, what the next card must do, what not to
  propose. This is self-evolution of generation, not a polish of a card.
- **unknowns**: questions the evidence failed to settle, written
  explicitly so the next mission can target them instead of jumping at
  random.
- **contradictions**: a probe moving against a previously corroborated
  claim is recorded, never silently overwritten.

Promotion is never a model vote. `supports` / `weakens` come from a
pre-registered two-arm probe (see `diagnose.judge_probe`); `closed_by_prior`
comes from a claim already stated in a paper this mission retrieved. Only
`corroborated` and `verified` claims ride in future prompts as known
ground — a speculative card cannot contaminate long-term memory, which is
what lets generation stay bold.

File discipline: one JSON file, digest over its own content, capped
lists, refuses to load if edited by hand.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import chain
from .evidence import (
    LEDGER_DIAGNOSTIC,
    MIN_HEAVY_REPLICATES,
    SCIENTIFIC_INVALID,
    SCIENTIFIC_REFUTES,
    SCIENTIFIC_SUPPORTS,
    SCIENTIFIC_UNINFORMATIVE,
    classify,
    heavy_confirmation_gaps,
    host_confirmation_gaps,
    host_confirms,
    hypothesis_revision_id,
)
from .livefeed import FreshWork
from .wiki import as_works as wiki_as_works, compile_entry as compile_wiki_entry, upsert as upsert_wiki
from .world import KIND_SYNTHETIC, world_attested

LIST_CAP = 12
HYPOTHESIS_CAP = 40

RANK = {
    "closed_by_prior": -1,
    "weakened": 0,
    "speculative": 1,
    "corroborated": 2,
    "verified": 3,
}

VALIDITY_ACTIVE = "active"
VALIDITY_INVALIDATED = "invalidated"
VALIDITY_STALE = "stale"
VALIDITY_SUPERSEDED = "superseded"
# Evidence inherited across a line merge whose claim revision cannot be
# proven to match the survivor's: kept as history, banned from promotion.
VALIDITY_LEGACY = "legacy_unresolved"


class StateError(Exception):
    """The store refused to load: corrupted content or a digest mismatch."""


def _digest(topics: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(topics, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def hypothesis_id(anchor: str, pair: tuple[str, str] | list[str]) -> str:
    """Stable across missions: (anchor domain, far concept) is the line.

    The near side of a card is always drawn from inside the anchor's
    neighbourhood, so which synonym the model picked ("succinct data" vs
    "succinct data structure") is phrasing, not identity — a live batch
    watched exactly that split one research line in two and break the
    promotion ladder. The far concept is the actual bet; the anchor is a
    deterministic property of the mission, so no fuzzy matching needed.
    """
    far = str(pair[-1]).strip().lower()
    normal = f"{str(anchor).strip().lower()}::{far}"
    return hashlib.sha256(normal.encode()).hexdigest()[:16]


def load(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = payload.get("topics")
    if not isinstance(topics, dict) or payload.get("digest") != _digest(topics):
        raise StateError(
            f"{path} does not match its own digest; refusing to feed missions"
            " a research state that was edited outside this module"
        )
    return topics


def _save(path: Path, topics: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"topics": topics, "digest": _digest(topics)},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@contextmanager
def _mutating(path: Path) -> Iterator[dict[str, Any]]:
    """Load-modify-save under an exclusive lock so parallel missions do not drop buckets."""
    from .filelock import file_lock

    path = Path(path)
    with file_lock(path):
        topics = load(path)
        yield topics
        _save(path, topics)


def topic_key(corpus_id: str, anchor: str) -> str:
    return f"{corpus_id}::{anchor}"


def _bucket(topics: dict[str, Any], key: str) -> dict[str, Any]:
    bucket = topics.setdefault(
        key,
        {
            "hypotheses": {},
            "unknowns": [],
            "contradictions": [],
            "rejected": [],
            "elites": [],
            "program": None,
            "wiki": [],
        },
    )
    for name in ("unknowns", "contradictions", "rejected", "elites", "wiki"):
        bucket.setdefault(name, [])
    bucket.setdefault("hypotheses", {})
    bucket.setdefault("program", None)
    return bucket


def summary(
    topics: dict[str, Any], corpus_id: str, anchor: str, *, cap: int = LIST_CAP
) -> dict[str, Any]:
    """What the next mission in this neighbourhood should treat as ground."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    hypotheses = bucket.get("hypotheses") or {}
    known = [
        {"claim": entry.get("claim"), "status": entry.get("status")}
        for entry in hypotheses.values()
        if entry.get("status") in ("corroborated", "verified")
        and str(entry.get("validity") or VALIDITY_ACTIVE) == VALIDITY_ACTIVE
    ]
    return {
        "known": known[:cap],
        "unknowns": list(bucket.get("unknowns") or [])[:cap],
        "contradictions": list(bucket.get("contradictions") or [])[:cap],
        "rejected": list(bucket.get("rejected") or [])[-cap:],
        "elites": list(bucket.get("elites") or [])[:cap],
        "hypotheses": len(hypotheses),
        "program": bucket.get("program") if isinstance(bucket.get("program"), dict) else None,
        "wiki": list(bucket.get("wiki") or [])[:cap],
    }


def probe_history(
    topics: dict[str, Any], corpus_id: str, anchor: str, pair: tuple[str, str] | list[str]
) -> dict[str, Any] | None:
    """This hypothesis line's last probe design and verdict, or None."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    entry = (bucket.get("hypotheses") or {}).get(hypothesis_id(anchor, pair)) or {}
    last = entry.get("last_probe")
    return dict(last) if isinstance(last, dict) else None


def line_status(
    topics: dict[str, Any], corpus_id: str, anchor: str, pair: tuple[str, str] | list[str]
) -> str | None:
    """This hypothesis line's current ladder status, or None if unseen."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    entry = (bucket.get("hypotheses") or {}).get(hypothesis_id(anchor, pair)) or {}
    status = entry.get("status")
    return status if status in RANK else None


def rejections_for(
    topics: dict[str, Any], corpus_id: str, anchor: str
) -> list[dict[str, Any]]:
    """The failure capsules for this neighbourhood, prompt-ready.

    Same shape the generator's failures block has always eaten:
    `{"context": "a x b", "mechanism": "which checks killed it"}`.
    """
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    return [
        capsule
        for capsule in bucket.get("rejected") or []
        if isinstance(capsule, dict)
    ]


def record_rejections(
    path: Path,
    corpus_id: str,
    anchor: str,
    kills: list[dict[str, Any]],
) -> int:
    """Bank this mission's gate kills as failure capsules; returns how many
    were new. The same dead combination reported twice is one lesson."""
    if not kills:
        return 0
    with _mutating(path) as topics:
        bucket = _bucket(topics, topic_key(corpus_id, anchor))
        known = {
            capsule.get("context")
            for capsule in bucket["rejected"]
            if isinstance(capsule, dict)
        }
        added = 0
        for kill in kills:
            pair = list(kill.get("pair") or ["", ""])
            context = f"{pair[0]} x {pair[1]}"
            if context in known:
                continue
            bucket["rejected"].append(
                {
                    "context": context,
                    "mechanism": ", ".join(kill.get("killed_by") or [])
                    or "text discipline",
                }
            )
            known.add(context)
            added += 1
        bucket["rejected"] = bucket["rejected"][-LIST_CAP:]
    return added


def elites_for(
    topics: dict[str, Any], corpus_id: str, anchor: str
) -> list[dict[str, Any]]:
    """QD elites persisted for this neighbourhood, crossover-ready."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    return [
        dict(row)
        for row in bucket.get("elites") or []
        if isinstance(row, dict) and row.get("pair_nodes")
    ]


def record_elites(
    path: Path,
    corpus_id: str,
    anchor: str,
    elites: list[dict[str, Any]],
) -> int:
    """Replace the neighbourhood's elite set. Only surviving, scored cards."""
    with _mutating(path) as topics:
        bucket = _bucket(topics, topic_key(corpus_id, anchor))
        kept = []
        seen: set[tuple[str, str]] = set()
        for row in elites:
            pair_nodes = tuple(row.get("pair_nodes") or ())
            if len(pair_nodes) != 2 or pair_nodes in seen:
                continue
            seen.add(pair_nodes)
            kept.append(
                {
                    "card_id": str(row.get("card_id") or ""),
                    "pair": list(row.get("pair") or []),
                    "pair_nodes": list(pair_nodes),
                    "claim": str(row.get("claim") or ""),
                    "operator": str(row.get("operator") or ""),
                    "alienness": float(row.get("alienness") or 0),
                    "score": float(row.get("score") or 0),
                }
            )
            if len(kept) >= LIST_CAP:
                break
        bucket["elites"] = kept
    return len(kept)


def program_for(
    topics: dict[str, Any], corpus_id: str, anchor: str
) -> dict[str, Any] | None:
    """The committed generation program for this neighbourhood, or None."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    program = bucket.get("program")
    if isinstance(program, dict) and str(program.get("commit") or "").strip():
        return dict(program)
    return None


def record_program(
    path: Path,
    corpus_id: str,
    anchor: str,
    program: dict[str, Any] | None,
) -> None:
    """Replace the neighbourhood's generation program. One committed line."""
    if not program or not str(program.get("commit") or "").strip():
        return
    with _mutating(path) as topics:
        bucket = _bucket(topics, topic_key(corpus_id, anchor))
        payload = {
            "commit": str(program.get("commit") or "")[:400],
            "why": str(program.get("why") or "")[:400],
            "stakes": str(program.get("stakes") or "")[:240],
            "next_card_must": str(program.get("next_card_must") or "")[:400],
            "do_not_generate": str(program.get("do_not_generate") or "")[:600],
            "card_id": str(program.get("card_id") or ""),
            "pair": list(program.get("pair") or [])[:2],
            "verdict": program.get("verdict"),
            "source": str(program.get("source") or "compiled"),
        }
        raw_h = program.get("h")
        if isinstance(raw_h, dict) and raw_h:
            payload["h"] = {
                str(key)[:40]: str(value)[:400]
                for key, value in raw_h.items()
                if str(value or "").strip()
            }
        bucket["program"] = payload


def wiki_for(
    topics: dict[str, Any], corpus_id: str, anchor: str
) -> list[dict[str, Any]]:
    """Papers this neighbourhood has actually retrieved, oldest-recent last."""
    bucket = topics.get(topic_key(corpus_id, anchor)) or {}
    return [
        dict(row)
        for row in bucket.get("wiki") or []
        if isinstance(row, dict) and (row.get("cite_id") or row.get("title"))
    ]


def wiki_works(
    topics: dict[str, Any], corpus_id: str, anchor: str
) -> list[Any]:
    """Wiki rows reconstructed as FreshWork for prior-support checks."""
    return wiki_as_works(wiki_for(topics, corpus_id, anchor))


def record_wiki(
    path: Path,
    corpus_id: str,
    anchor: str,
    works: list[Any],
    *,
    pair: tuple[str, str] | list[str] | None = None,
    seen_at: str = "",
) -> int:
    """Bank this survey into the neighbourhood wiki. Returns how many were new."""
    if not works:
        return 0
    added = 0
    with _mutating(path) as topics:
        bucket = _bucket(topics, topic_key(corpus_id, anchor))
        incoming: list[dict[str, Any]] = []
        for work in works:
            if isinstance(work, FreshWork):
                incoming.append(compile_wiki_entry(work, pair=pair, seen_at=seen_at))
                continue
            if isinstance(work, dict) and (work.get("title") or work.get("cite_id")):
                payload = dict(work)
                if not payload.get("arxiv_id") and not payload.get("work_id"):
                    payload["work_id"] = str(payload.get("cite_id") or "")
                incoming.append(
                    compile_wiki_entry(
                        FreshWork.from_dict(payload), pair=pair, seen_at=seen_at
                    )
                )
        bucket["wiki"], added = upsert_wiki(list(bucket.get("wiki") or []), incoming)
    return added


def prompt_lines(state_summary: dict[str, Any]) -> tuple[str, ...]:
    """Ground for the generator: known findings and named gaps, never
    speculation. A knowledge gap is a target; a speculative claim is not."""
    lines: list[str] = []
    for entry in state_summary.get("known") or []:
        lines.append(f"Established in earlier missions ({entry['status']}): {entry['claim']}")
    for unknown in state_summary.get("unknowns") or []:
        lines.append(f"Open question from earlier missions: {unknown}")
    for conflict in state_summary.get("contradictions") or []:
        lines.append(f"Unresolved contradiction: {conflict}")
    for elite in state_summary.get("elites") or []:
        pair = elite.get("pair") or ["", ""]
        lines.append(
            f"Elite combination from earlier missions"
            f" ({pair[0]} × {pair[-1] if pair else ''}):"
            f" {str(elite.get('claim') or '')[:120]}"
        )
    return tuple(lines)


def next_status(previous: str | None, outcome: dict[str, Any]) -> tuple[str, str]:
    """The promotion rule, as arithmetic over recorded facts.

    The 20s probe is a filter. A WORLD `supports` without a matching host
    run stays `speculative` (promising / pending confirmation). Only
    `host_confirms` — host_ok, protocol complete, EvidenceID match,
    identified mechanism — may write `corroborated`. Infrastructure
    failure is not a scientific refutation and it invalidates an earlier
    corroboration that depended on the broken artifact. Uninformative
    does not keep a probe-only corroboration; a host-confirmed line stays
    until a later scientific refute or an invalidation.
    """
    prev = previous if previous in RANK else None
    labels = classify(outcome)
    if outcome.get("prior_kills"):
        return "closed_by_prior", (
            "a retrieved paper contains a supporting span that restates the claim"
        )
    if labels["ledger"] == LEDGER_DIAGNOSTIC or labels["scientific"] == SCIENTIFIC_INVALID:
        reason = labels["diagnostic"] or "infrastructure failure"
        if prev in ("corroborated", "verified"):
            return "speculative", (
                f"invalidated: {reason} is not scientific evidence and "
                "cannot keep an earlier corroboration"
            )
        return prev or "speculative", (
            f"{reason}: infrastructure failure is not a scientific outcome"
        )
    scientific = labels["scientific"]
    if scientific == SCIENTIFIC_SUPPORTS:
        if not world_attested(outcome):
            if prev in ("corroborated", "verified") and prev_is_host_ground(outcome, prev):
                return prev, (
                    "earlier host corroboration stands; this probe was "
                    + str(outcome.get("probe_kind") or KIND_SYNTHETIC)
                )
            if prev in ("corroborated", "verified"):
                return "speculative", (
                    "earlier corroboration was not host-confirmed; "
                    "SYNTHETIC support cannot keep it"
                )
            return "speculative", (
                "SYNTHETIC support is a coherence check on invented data, "
                "not corroboration"
            )
        if not host_confirms(outcome):
            if not labels["identified"]:
                return "speculative", (
                    "probe support is unidentified: competing explanations "
                    "were not ablated"
                )
            if outcome.get("host_ok") is True:
                gaps = host_confirmation_gaps(outcome)
                return "speculative", (
                    "host ran but confirmation is fail-closed: "
                    + "; ".join(gaps[:3])
                )
            return "speculative", (
                "probe support is pending host confirmation; "
                "the 20s filter is not corroborated"
            )
        if prev == "verified":
            return "verified", "already verified; the new host run agrees"
        if prev == "corroborated":
            # Verified is the rung above corroborated, so its gate must be
            # strictly harder: a seed-replicated heavy run with its own
            # EvidenceID, on the same world, whose whole confidence
            # interval clears the pre-registered margin. Fail-closed —
            # a bare `supports` flag without the statistics is a refusal.
            heavy_gaps = heavy_confirmation_gaps(outcome)
            if not heavy_gaps:
                return "verified", (
                    "supported on an attested world in two missions and the"
                    " seed-replicated heavy run's confidence interval"
                    " cleared the pre-registered margin"
                )
            if outcome.get("heavy_confirmed") or isinstance(
                outcome.get("heavy"), dict
            ):
                return "corroborated", (
                    "verified is fail-closed: " + "; ".join(heavy_gaps[:3])
                )
            return "corroborated", (
                "supported again; verified awaits the heavier"
                " confirmation run"
            )
        return "corroborated", (
            "the host reran the same attested experiment and the "
            "pre-registered direction still held"
        )
    if scientific == SCIENTIFIC_REFUTES:
        return "weakened", "the probe moved against the pre-registered direction"
    if scientific == SCIENTIFIC_UNINFORMATIVE:
        if prev in ("corroborated", "verified") and outcome.get("host_ok") is True:
            return prev, "earlier host corroboration stands; this probe was uninformative"
        if prev in ("corroborated", "verified"):
            return prev, "earlier corroboration stands; this probe was uninformative"
        return "speculative", "no discriminating evidence yet"
    return prev or "speculative", "no discriminating evidence yet"


def prev_is_host_ground(outcome: dict[str, Any], prev: str) -> bool:
    """SYNTHETIC follow-ups must not launder a missing host confirmation.

    Fail-closed, proof-required: the earlier corroboration stands only
    when `apply_outcomes` resolved a grounding proof from the research
    ledger — the PROMOTE event that granted the previous status, carrying
    a host EvidenceID and a world digest. No proof in the ledger means
    the previous rank is an assertion, and an assertion cannot be kept
    alive by a SYNTHETIC probe. (An earlier revision of this function
    returned True unconditionally; that escape hatch is gone.)
    """
    del prev
    proof = outcome.get("prev_grounding_proof")
    return bool(
        isinstance(proof, dict)
        and str(proof.get("host_evidence_id") or "")
        and str(proof.get("world_digest") or "")
    )


def apply_outcomes(
    path: Path,
    corpus_id: str,
    anchor: str,
    outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold one mission's outcomes into the store; return promotion events.

    Each outcome describes a hypothesis that entered research: card_id,
    pair, claim, prior_kills, verdict, alternative (the competing
    explanation the probe tried to separate). Gate kills never arrive
    here — they go through `record_rejections`.

    The event chain is a gate here, not a diary. Before anything is
    written, the store's own promotion chain must verify — a tampered
    log refuses the whole write. And before an outcome may land as
    `corroborated` or `verified`, its candidate chain must verify *and*
    prove the order: the protocol was registered before the execution
    that now claims to confirm it. A missing chain is a missing proof.
    """
    if not outcomes:
        return []
    events: list[dict[str, Any]] = []
    chain_path = Path(path).with_suffix(".chain.jsonl")
    with _mutating(path) as topics:
        own_audit = chain.verify_chain(chain_path)
        if not own_audit["ok"]:
            raise StateError(
                f"{chain_path} failed its audit ({own_audit['error']}); "
                "refusing to write promotions on top of a tampered "
                "promotion log"
            )
        # One pass over the (already verified) research ledger yields the
        # two facts promotion consumes: which (line, revision) pairs were
        # registered as hypotheses before any evidence, and — per line —
        # the last PROMOTE that carried host-ground provenance, which is
        # the only thing that lets a SYNTHETIC follow-up keep an earlier
        # corroboration alive.
        registered_hypotheses: set[tuple[str, str]] = set()
        grounded_promotes: dict[str, dict[str, Any]] = {}
        for ledger_event in chain.read_events(chain_path):
            kind = str(ledger_event.get("kind") or "")
            payload = ledger_event.get("payload") or {}
            if kind == chain.REGISTER_HYPOTHESIS:
                registered_hypotheses.add(
                    (
                        str(payload.get("line_id") or ""),
                        str(payload.get("revision_id") or ""),
                    )
                )
            elif kind == chain.PROMOTE:
                if str(payload.get("status") or "") in (
                    "corroborated",
                    "verified",
                ) and str(payload.get("host_evidence_id") or ""):
                    grounded_promotes[str(payload.get("line_id") or "")] = payload
        key = topic_key(corpus_id, anchor)
        bucket = _bucket(topics, key)
        hypotheses = bucket["hypotheses"]
        for outcome in outcomes:
            pair = list(outcome.get("pair") or ["", ""])
            hid = hypothesis_id(anchor, pair)
            entry = hypotheses.get(hid) or {
                "pair": pair,
                "claim": str(outcome.get("claim") or ""),
                "status": None,
                "evidence": [],
                "missions": 0,
            }
            previous = entry.get("status")
            # Claim revision identity: the research line survives a rewrite,
            # its truth status does not. Evidence for "mechanism A" must not
            # ride along when the claim quietly becomes "mechanism B".
            claim_text = str(outcome.get("claim") or "").strip()
            revision_reset = False
            if claim_text:
                incoming_revision = hypothesis_revision_id(hid, claim_text)
                stored_revision = str(entry.get("revision_id") or "")
                if not stored_revision and str(entry.get("claim") or "").strip():
                    stored_revision = hypothesis_revision_id(
                        hid, str(entry["claim"])
                    )
                if stored_revision and stored_revision != incoming_revision:
                    revision_reset = True
                    previous = None
                    entry["status"] = None
                    entry["uninformative_streak"] = 0
                    entry.pop("must_switch_mechanism", None)
                    entry.pop("last_probe", None)
                    for item in entry.get("evidence") or []:
                        item["validity"] = VALIDITY_SUPERSEDED
                    entry["superseded_revisions"] = (
                        int(entry.get("superseded_revisions") or 0) + 1
                    )
                entry["revision_id"] = incoming_revision
            # A SYNTHETIC follow-up may keep an earlier corroboration only
            # against a grounding proof resolved from the ledger, and only
            # for the same claim revision — earlier revisions ground nothing.
            proof = grounded_promotes.get(hid)
            if (
                proof is not None
                and not revision_reset
                and str(proof.get("revision_id") or "")
                == str(entry.get("revision_id") or "")
            ):
                outcome = {
                    **outcome,
                    "prev_grounding_proof": {
                        "host_evidence_id": str(proof.get("host_evidence_id") or ""),
                        "world_digest": str(proof.get("world_digest") or ""),
                        "status": str(proof.get("status") or ""),
                    },
                }
            status, reason = next_status(previous, outcome)
            if status in ("corroborated", "verified") and host_confirms(outcome):
                # The confirming host run must be able to show its history:
                # an intact chain in which the protocol registration
                # precedes the execution, plus a hypothesis registered in
                # the research ledger before this promotion. No chain,
                # broken chain, wrong order, or an unregistered hypothesis
                # refuses the promotion — the evidence may be real, but
                # unprovable order is not provenance.
                audit_gaps = list(
                    chain.audit_promotion_gaps(
                        Path(str(outcome.get("chain_path") or "")),
                        experiment_digest=str(outcome.get("experiment_digest") or ""),
                        evidence_id=str(outcome.get("host_evidence_id") or ""),
                    )
                )
                if (
                    hid,
                    str(entry.get("revision_id") or ""),
                ) not in registered_hypotheses:
                    audit_gaps.append(
                        "this hypothesis revision was never registered in"
                        " the research ledger before its evidence"
                    )
                if audit_gaps:
                    status = "speculative"
                    reason = (
                        "the event chain refused this promotion: "
                        + "; ".join(audit_gaps[:3])
                    )
            if status == "verified" and previous != "verified":
                # Verified may not rest on a dict that says a heavy run
                # occurred: the candidate ledger must contain the acts —
                # N seed executions of the registered experiment and the
                # aggregation that names them. A heavy confirmation the
                # ledger cannot prove leaves the line corroborated. A line
                # that is already verified keeps its rank on a plain E1
                # reconfirmation; its heavy proof lives in the chain of the
                # mission that earned it.
                heavy = outcome.get("heavy") or {}
                heavy_ledger_gaps = chain.audit_heavy_gaps(
                    Path(str(outcome.get("chain_path") or "")),
                    evidence_id=str(
                        heavy.get("evidence_id") if isinstance(heavy, dict) else ""
                    )
                    or "",
                    experiment_digest=str(outcome.get("experiment_digest") or ""),
                    min_seeds=MIN_HEAVY_REPLICATES,
                )
                if heavy_ledger_gaps:
                    status = "corroborated"
                    reason = (
                        "verified is ledger-gated: "
                        + "; ".join(heavy_ledger_gaps[:3])
                    )
            if revision_reset:
                reason = (
                    "new hypothesis revision: earlier evidence is context, "
                    "not inherited truth status — " + reason
                )
            labels = classify(outcome)
            entry["status"] = status
            entry["claim"] = str(outcome.get("claim") or entry.get("claim") or "")
            entry["missions"] = int(entry.get("missions") or 0) + 1
            entry["validity"] = VALIDITY_ACTIVE
            if (
                labels["ledger"] == LEDGER_DIAGNOSTIC
                and previous in ("corroborated", "verified")
                and status == "speculative"
            ):
                entry["validity"] = VALIDITY_INVALIDATED
            scientific = labels["scientific"]
            if outcome.get("prior_kills"):
                pass
            elif scientific == SCIENTIFIC_UNINFORMATIVE:
                entry["uninformative_streak"] = int(
                    entry.get("uninformative_streak") or 0
                ) + 1
            elif scientific in (SCIENTIFIC_SUPPORTS, SCIENTIFIC_REFUTES):
                entry["uninformative_streak"] = 0
            if int(entry.get("uninformative_streak") or 0) >= 2:
                entry["must_switch_mechanism"] = True
            verdict = outcome.get("verdict")
            if verdict:
                evidence_item = {
                    "kind": "host" if outcome.get("host_ok") is True else "probe",
                    "direction": {"supports": "+", "weakens": "-"}.get(verdict, "0"),
                    "note": str(outcome.get("evidence_note") or ""),
                    "evidence_id": str(outcome.get("evidence_id") or ""),
                    "ledger": labels["ledger"],
                    "scientific": scientific,
                    "validity": (
                        VALIDITY_INVALIDATED
                        if labels["ledger"] == LEDGER_DIAGNOSTIC
                        else VALIDITY_ACTIVE
                    ),
                }
                entry["evidence"] = (entry.get("evidence") or []) + [evidence_item]
                entry["evidence"] = entry["evidence"][-LIST_CAP:]
                if entry.get("validity") == VALIDITY_INVALIDATED:
                    for item in entry["evidence"]:
                        item["validity"] = VALIDITY_INVALIDATED
                experiment = str(outcome.get("experiment") or "").strip()
                if experiment:
                    # The next mission's diagnosis quotes this back: an
                    # uninformative design must be sharpened, never repeated.
                    discarded = [
                        str(item).strip()
                        for item in (outcome.get("failed_experiments") or [])
                        if str(item or "").strip()
                    ]
                    entry["last_probe"] = {
                        "experiment": experiment,
                        "verdict": verdict,
                        "discarded": discarded[:8],
                        "ledger": labels["ledger"],
                        "scientific": scientific,
                        "evidence_id": str(outcome.get("evidence_id") or ""),
                        "must_switch_mechanism": bool(
                            entry.get("must_switch_mechanism")
                        ),
                    }
            alternative = str(outcome.get("alternative") or "").strip()
            if verdict == "weakens" and previous in ("corroborated", "verified"):
                line = (
                    f"{entry['claim'][:100]} — corroborated earlier, weakened by a"
                    " later probe"
                )
                if line not in bucket["contradictions"]:
                    bucket["contradictions"].append(line)
            elif status == "speculative" and alternative:
                line = f"undecided: {entry['claim'][:90]} — vs — {alternative[:90]}"
                if line not in bucket["unknowns"]:
                    bucket["unknowns"].append(line)
            vector = outcome.get("value_vector")
            if verdict == "supports" and isinstance(vector, dict) and vector:
                peak = max((int(v or 0) for v in vector.values()), default=0)
                if peak <= 2:
                    # The probe held but the reviewer saw little value: the
                    # tension is a named open question, not a silent success —
                    # it gives an "all supports, zero leftovers" topic a target
                    # for its next rerun.
                    line = (
                        "mechanism held in a probe, but its advantage over known"
                        f" methods is unproven: {entry['claim'][:90]}"
                    )
                    if line not in bucket["unknowns"]:
                        bucket["unknowns"].append(line)
            hypotheses[hid] = entry
            events.append(
                {
                    "card_id": outcome.get("card_id"),
                    "pair": pair,
                    "previous": previous,
                    "status": status,
                    "reason": reason,
                    "line_id": hid,
                    "revision_id": str(entry.get("revision_id") or ""),
                    "revision_reset": revision_reset,
                }
            )
        for name in ("unknowns", "contradictions", "rejected"):
            bucket[name] = bucket[name][-LIST_CAP:]
        for event, outcome in zip(events, outcomes):
            chain.append_event(
                chain_path,
                chain.PROMOTE,
                {
                    "topic": key,
                    "card_id": str(event.get("card_id") or ""),
                    "line_id": event["line_id"],
                    "revision_id": event["revision_id"],
                    "revision_reset": event["revision_reset"],
                    "previous": event.get("previous"),
                    "status": event["status"],
                    "reason": event["reason"],
                    "evidence_id": str(outcome.get("evidence_id") or ""),
                    "host_evidence_id": str(
                        outcome.get("host_evidence_id") or ""
                    ),
                    "world_digest": str(outcome.get("world_digest") or ""),
                    "data_digest": str(outcome.get("data_digest") or ""),
                },
            )
        if len(hypotheses) > HYPOTHESIS_CAP:
            # Drop the lowest-ranked, oldest lines first; verified never drops.
            keep = sorted(
                hypotheses.items(),
                key=lambda item: (RANK.get(item[1].get("status"), 0), item[1].get("missions", 0)),
                reverse=True,
            )[:HYPOTHESIS_CAP]
            bucket["hypotheses"] = dict(keep)
    return events


def _entry_revision(entry: dict[str, Any], hid: str) -> str:
    """The revision this entry's evidence belongs to, or '' when unknowable."""
    stored = str(entry.get("revision_id") or "")
    if stored:
        return stored
    claim = str(entry.get("claim") or "").strip()
    return hypothesis_revision_id(hid, claim) if claim else ""


def migrate_identities(path: Path) -> int:
    """One-time rekey of an existing store to (anchor, far concept) identity.

    Research *lines* may merge; hypothesis *revisions* never silently do.
    When two entries collapse onto one key, the survivor is the
    highest-ranked (then most-travelled) entry, and it keeps only its own
    truth status. The other entry's evidence rides along as context, but
    unless its claim revision provably matches the survivor's, every item
    is quarantined as `legacy_unresolved` — losing old evidence is
    acceptable, inheriting an unearned truth status is not. Returns how
    many entries were merged away. Idempotent.
    """
    if not Path(path).is_file():
        return 0
    merged = 0
    with _mutating(path) as topics:
        for key, bucket in topics.items():
            anchor = key.split("::", 1)[1] if "::" in key else key
            hypotheses = bucket.get("hypotheses") or {}
            rekeyed: dict[str, dict[str, Any]] = {}
            for entry in hypotheses.values():
                hid = hypothesis_id(anchor, entry.get("pair") or ["", ""])
                existing = rekeyed.get(hid)
                if existing is None:
                    rekeyed[hid] = entry
                    continue
                merged += 1
                ordered = sorted(
                    (existing, entry),
                    key=lambda e: (RANK.get(e.get("status"), 0), e.get("missions", 0)),
                    reverse=True,
                )
                base, other = ordered
                base_revision = _entry_revision(base, hid)
                other_revision = _entry_revision(other, hid)
                same_revision = bool(base_revision) and base_revision == other_revision
                inherited = [dict(item) for item in (other.get("evidence") or [])]
                if not same_revision:
                    for item in inherited:
                        item["validity"] = VALIDITY_LEGACY
                base["evidence"] = (
                    (base.get("evidence") or []) + inherited
                )[-LIST_CAP:]
                base["missions"] = int(base.get("missions") or 0) + int(
                    other.get("missions") or 0
                )
                if same_revision and not base.get("last_probe") and other.get(
                    "last_probe"
                ):
                    base["last_probe"] = other["last_probe"]
                rekeyed[hid] = base
            bucket["hypotheses"] = rekeyed
    return merged
