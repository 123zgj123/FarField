"""Evidence objects, scientific vs infrastructure ledgers, promotion teeth.

The 20s probe is a filter. It may say a line is promising. It may not
write `corroborated` into the research state. Only a host run of the
*same* experiment on the *same* attested bytes — same EvidenceID — can
climb. Infrastructure failure is a diagnostic, never a scientific
refutation, and it cannot keep an earlier corroboration alive.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .world import KIND_SYNTHETIC, world_attested

LEDGER_SCIENTIFIC = "scientific"
LEDGER_DIAGNOSTIC = "diagnostic"

# --- Replication taxonomy --------------------------------------------------
# "The host reran it" is not one thing. Same bytes, more bytes, other bytes,
# and other environments answer different scientific questions, so they must
# not share an evidence identity. Only E1 may confirm the probe's claim; E2+
# are generalization evidence recorded under their own EvidenceID.
TIER_SYNTHETIC = "E0_synthetic"          # invented / generated for this claim
TIER_SAME_INSTANCE = "E1_same_instance"  # same bytes, same protocol: reproduction
TIER_SCALE = "E2_scale_replication"      # same source, different slice: generalization
TIER_INDEPENDENT = "E3_independent"      # a different attested dataset
TIER_EXTERNAL = "E4_external"            # another environment submitted the numbers

# The confirmation tier (corroborated → verified) must report at least
# this many per-seed replicates; fewer cannot carry a confidence interval
# worth the name.
MIN_HEAVY_REPLICATES = 3

SCIENTIFIC_SUPPORTS = "supports"
SCIENTIFIC_REFUTES = "refutes"
SCIENTIFIC_UNINFORMATIVE = "uninformative"
SCIENTIFIC_INVALID = "invalid"

DIAGNOSTIC_KINDS = frozenset(
    {
        "implementation_error",
        "sandbox_rejection",
        "world_error",
        "infra_timeout",
        "artifact_missing",
        "protocol_violation",
        "world_incompatible",
    }
)

def _mentions_lever(text: str, lever: str) -> bool:
    blob = str(text or "").lower()
    name = str(lever or "").strip().lower()
    if not blob or not name or name == "none":
        return False
    return name in blob or name.replace("_", " ") in blob


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def evidence_id(
    *,
    claim_id: str = "",
    experiment_digest: str = "",
    world_digest: str = "",
    data_digest: str = "",
    protocol_digest: str = "",
    environment_digest: str = "",
) -> str:
    """Immutable identity of one experiment instance.

    Change the data, the protocol, or the environment and this is a new
    evidence object — a scale replication is never "the same run again".
    """
    payload = {
        "claim_id": str(claim_id or ""),
        "experiment_digest": str(experiment_digest or ""),
        "world_digest": str(world_digest or ""),
        "data_digest": str(data_digest or ""),
        "protocol_digest": str(protocol_digest or ""),
        "environment_digest": str(environment_digest or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hypothesis_revision_id(line_id: str, claim: str) -> str:
    """Identity of one claim revision inside a research line.

    A research line ("anchor × far concept") survives rewording; the
    claim revision does not. Any substantive edit to the claim text is a
    new revision, and a new revision inherits the line's context but
    never its truth status — evidence for "mechanism A" must not ride
    along when the claim quietly becomes "mechanism B".
    """
    tokens = re.findall(r"[a-z0-9]+", str(claim or "").lower())
    normal = f"{str(line_id or '').strip()}::{' '.join(tokens)}"
    return hashlib.sha256(normal.encode("utf-8")).hexdigest()[:16]


def replication_group_id(
    *,
    claim_id: str = "",
    experiment_digest: str = "",
    source_digest: str = "",
) -> str:
    """One claim family across E1/E2/E3 runs of the same registered design.

    Distinct EvidenceIDs (different bytes, different environments) that
    test the same registered experiment on the same mother source belong
    to one replication group — related evidence, never the same evidence.
    """
    payload = {
        "claim_id": str(claim_id or ""),
        "experiment_digest": str(experiment_digest or ""),
        "source_digest": str(source_digest or ""),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def diagnostic_kind(outcome: dict[str, Any] | None) -> str | None:
    """Name the infrastructure failure, or None if this is a scientific result."""
    row = outcome or {}
    if row.get("world_incompatible") or row.get("scientific") == "world_incompatible":
        return "world_incompatible"
    if row.get("artifact_mismatch") or row.get("artifact_missing"):
        return "artifact_missing"
    host_error = str(row.get("host_error") or row.get("error") or "").lower()
    status = str(row.get("probe_status") or row.get("status") or "").lower()
    if row.get("host_ok") is False:
        if "timeout" in host_error or status == "timeout":
            return "infra_timeout"
        if "missing" in host_error or "experiment.py" in host_error:
            return "artifact_missing"
        if "digest" in host_error or "mismatch" in host_error:
            return "protocol_violation"
        if "parent cache" in host_error or "world" in host_error:
            return "world_error"
        return "protocol_violation"
    if status in {"timeout", "timed_out"}:
        return "infra_timeout"
    if status in {"refused", "sandbox"}:
        return "sandbox_rejection"
    if status in {"crashed", "error"}:
        return "implementation_error"
    refused = str(row.get("refused") or "")
    if refused == "probe":
        return "sandbox_rejection"
    if refused == "world":
        return "world_incompatible"
    kind = str(row.get("diagnostic") or "").strip().lower()
    if kind in DIAGNOSTIC_KINDS:
        return kind
    if str(row.get("ledger") or "") == LEDGER_DIAGNOSTIC:
        return kind or "implementation_error"
    return None


def scientific_verdict(outcome: dict[str, Any] | None) -> str:
    """Map a recorded outcome onto SUPPORTS / REFUTES / UNINFORMATIVE / INVALID."""
    row = outcome or {}
    if diagnostic_kind(row):
        return SCIENTIFIC_INVALID
    explicit = str(row.get("scientific") or "").strip().lower()
    if explicit in {
        SCIENTIFIC_SUPPORTS,
        SCIENTIFIC_REFUTES,
        SCIENTIFIC_UNINFORMATIVE,
        SCIENTIFIC_INVALID,
    }:
        return explicit
    verdict = str(row.get("verdict") or "").strip().lower()
    if verdict in {"supports", "support"}:
        return SCIENTIFIC_SUPPORTS
    if verdict in {"weakens", "refutes", "refute"}:
        return SCIENTIFIC_REFUTES
    if verdict == "uninformative":
        return SCIENTIFIC_UNINFORMATIVE
    return SCIENTIFIC_INVALID if verdict else SCIENTIFIC_UNINFORMATIVE


def ledger_of(outcome: dict[str, Any] | None) -> str:
    if diagnostic_kind(outcome):
        return LEDGER_DIAGNOSTIC
    if scientific_verdict(outcome) == SCIENTIFIC_INVALID:
        return LEDGER_DIAGNOSTIC
    if (outcome or {}).get("verdict") or (outcome or {}).get("prior_kills"):
        return LEDGER_SCIENTIFIC
    return LEDGER_DIAGNOSTIC


def mechanism_identified(outcome: dict[str, Any] | None) -> bool:
    row = outcome or {}
    if row.get("mechanism_identified") is False:
        return False
    if row.get("ablation_required") and not row.get("ablation_ran"):
        return False
    return True


def ablation_required(
    alternative: str,
    experiment: str,
    *,
    levers: tuple[str, ...] | list[str] = (),
    world_lever: str = "",
) -> bool:
    """True when the alternative is a handle on this object that the experiment did not shut.

    Competing explanations are other declared levers, not an English
    word list. A prose alternative that names no lever of this world is
    also required: the mechanism is then not identified on this object.
    """
    alt = str(alternative or "").strip()
    if not alt:
        return False
    vocab = tuple(
        str(item) for item in levers if str(item or "").strip() and str(item) != "none"
    )
    if not vocab:
        return False
    named = [
        lever
        for lever in vocab
        if lever != world_lever and _mentions_lever(alt, lever)
    ]
    if named:
        return not any(_mentions_lever(experiment, lever) for lever in named)
    return True


def host_confirmation_gaps(outcome: dict[str, Any] | None) -> tuple[str, ...]:
    """Every reason this outcome cannot confirm, by name.

    The contract is fail-closed: a missing EvidenceID, digest, or
    protocol flag is a refusal, not a benefit of the doubt. "Not provably
    the same evidence object" and "a different evidence object" are the
    same answer.
    """
    row = outcome or {}
    gaps: list[str] = []
    if not world_attested(row):
        gaps.append("probe did not run on an attested WORLD")
    if scientific_verdict(row) != SCIENTIFIC_SUPPORTS:
        gaps.append("scientific verdict is not supports")
    diag = diagnostic_kind(row)
    if diag:
        gaps.append(f"diagnostic outcome: {diag}")
    if row.get("host_ok") is not True:
        gaps.append("host did not run the protocol (host_ok is not True)")
    if row.get("protocol_complete") is not True:
        gaps.append("protocol_complete is not True")
    probe_eid = str(row.get("evidence_id") or "")
    host_eid = str(row.get("host_evidence_id") or "")
    if not probe_eid:
        gaps.append("probe evidence_id missing")
    if not host_eid:
        gaps.append("host evidence_id missing")
    if probe_eid and host_eid and probe_eid != host_eid:
        gaps.append("EvidenceID mismatch: the host ran a different evidence object")
    for key in ("world_digest", "experiment_digest", "data_digest"):
        probe_val = str(row.get(key) or "")
        host_val = str(row.get(f"host_{key}") or "")
        if not probe_val:
            gaps.append(f"probe {key} missing")
        if not host_val:
            gaps.append(f"host {key} missing")
        if probe_val and host_val and probe_val != host_val:
            gaps.append(f"{key} mismatch: not the same evidence object")
    replication = str(row.get("host_replication") or "")
    if replication and replication != TIER_SAME_INSTANCE:
        gaps.append(
            f"host run is {replication}, not a same-instance reproduction"
        )
    if not mechanism_identified(row):
        gaps.append("mechanism not identified: competing explanation not ablated")
    if world_attested(row) and scientific_verdict(row) == SCIENTIFIC_SUPPORTS:
        if row.get("world_has_dynamics") and row.get("world_consumed") is not True:
            gaps.append(
                "placebo control: the experiment did not consume the bound world"
            )
    return tuple(gaps)


def host_confirms(outcome: dict[str, Any] | None) -> bool:
    """True only when the host reran the *same* evidence object and it held.

    Fail-closed: same-instance reproduction (E1) with complete, matching
    provenance. A scale replication, an external run, or any missing
    field cannot confirm — see `host_confirmation_gaps` for the reasons.
    """
    return not host_confirmation_gaps(outcome)


def heavy_confirmation_gaps(outcome: dict[str, Any] | None) -> tuple[str, ...]:
    """Every reason the heavy run cannot lift `corroborated` to `verified`.

    The rung above `host_confirms` must be strictly harder, never looser:
    everything the corroboration gate demanded, plus executor-owned
    replication of the *registered* script — same experiment digest the
    probe ran, seeds derived by the runtime, on the same world as the
    corroborating evidence — whose entire 95% confidence interval clears
    the pre-registered margin. The interval is rechecked here from the
    recorded numbers, and the trust boundary is checked by construction:
    the model may generate computation, never the evidence that audits it.
    """
    row = outcome or {}
    heavy = row.get("heavy")
    if not isinstance(heavy, dict) or not heavy:
        return ("no heavy confirmation run recorded",)
    gaps: list[str] = []
    verdict = str(heavy.get("verdict") or "")
    if verdict != SCIENTIFIC_SUPPORTS:
        gaps.append(f"heavy verdict is {verdict or 'missing'}, not supports")
    if str(heavy.get("runner") or "") != "executor":
        gaps.append(
            "replicates were not produced by the trusted executor; "
            "self-reported replication is not replication"
        )
    if heavy.get("replicated") is not True:
        gaps.append("heavy run did not produce per-seed replicates")
    if heavy.get("invariant"):
        gaps.append(
            "the experiment was invariant under the registered seeds; "
            "stability under variation was not demonstrated"
        )
    seeds = heavy.get("seeds") or []
    if len(set(seeds)) < MIN_HEAVY_REPLICATES:
        gaps.append(
            f"only {len(set(seeds))} distinct registered seeds; at least "
            f"{MIN_HEAVY_REPLICATES} are required"
        )
    count = int(heavy.get("n") or 0)
    if count < MIN_HEAVY_REPLICATES:
        gaps.append(
            f"only {count} seed replicates; at least "
            f"{MIN_HEAVY_REPLICATES} are required"
        )
    try:
        margin = float(heavy.get("margin"))
        ci_low = float(heavy.get("ci_low"))
        ci_high = float(heavy.get("ci_high"))
    except (TypeError, ValueError):
        gaps.append("heavy confidence interval is not recorded as numbers")
    else:
        expected_lower = str(heavy.get("expected_direction") or "") == "treatment_lower"
        cleared = ci_high <= -margin if expected_lower else ci_low >= margin
        if not cleared:
            gaps.append(
                "the 95% interval of the heavy separations does not clear "
                "the pre-registered margin"
            )
    if not str(heavy.get("evidence_id") or ""):
        gaps.append("heavy run has no EvidenceID")
    heavy_exp = str(heavy.get("experiment_digest") or "")
    probe_exp = str(row.get("experiment_digest") or "")
    if not heavy_exp:
        gaps.append("heavy run has no experiment digest")
    elif probe_exp and heavy_exp != probe_exp:
        gaps.append(
            "heavy run replayed a different experiment than the registered"
            " protocol"
        )
    heavy_world = str(heavy.get("world_digest") or "")
    if not heavy_world:
        gaps.append("heavy run has no world digest")
    elif heavy_world != str(row.get("world_digest") or ""):
        gaps.append(
            "heavy run measured a different world than the corroborating evidence"
        )
    return tuple(gaps)


def heavy_confirms(outcome: dict[str, Any] | None) -> bool:
    """True only when the seed-replicated heavy run clears every gap."""
    return not heavy_confirmation_gaps(outcome)


def replication_tier(outcome: dict[str, Any] | None) -> str:
    """Which rung of the replication taxonomy this outcome's host run is."""
    row = outcome or {}
    if str(row.get("kind") or "").upper() == "EXTERNAL" or (
        str(row.get("status") or "") == "externally_replicated"
    ):
        return TIER_EXTERNAL
    if not world_attested(row):
        return TIER_SYNTHETIC
    explicit = str(row.get("host_replication") or row.get("replication") or "")
    if explicit:
        return explicit
    probe_data = str(row.get("data_digest") or "")
    host_data = str(row.get("host_data_digest") or "")
    if probe_data and host_data:
        if probe_data == host_data:
            return TIER_SAME_INSTANCE
        if str(row.get("world_digest") or "") == str(
            row.get("host_world_digest") or ""
        ):
            return TIER_SCALE
        return TIER_INDEPENDENT
    return TIER_SYNTHETIC if not row.get("host_ok") else TIER_SCALE


def classify(outcome: dict[str, Any] | None) -> dict[str, Any]:
    """Attach ledger labels without mutating the caller's dict."""
    row = dict(outcome or {})
    diag = diagnostic_kind(row)
    scientific = scientific_verdict(row)
    ledger = LEDGER_DIAGNOSTIC if diag else (
        LEDGER_SCIENTIFIC if scientific != SCIENTIFIC_INVALID else LEDGER_DIAGNOSTIC
    )
    return {
        "ledger": ledger,
        "scientific": scientific,
        "diagnostic": diag,
        "host_confirms": host_confirms(row),
        "confirmation_gaps": host_confirmation_gaps(row),
        "replication": replication_tier(row),
        "identified": mechanism_identified(row),
        "probe_kind": str(row.get("probe_kind") or KIND_SYNTHETIC).upper(),
    }


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def campaign_funnel(
    attempts: list[dict[str, Any]] | None,
    found: list[dict[str, Any]] | None,
    promotions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Layered campaign counts. Do not compare models on corroborated alone."""
    attempts = list(attempts or [])
    found = list(found or [])
    promotions = list(promotions or [])
    by_id = {str(row.get("card_id") or ""): row for row in found}
    generated = len(attempts)
    text_pass = [row for row in attempts if not row.get("killed")]
    world_valid: list[dict[str, Any]] = []
    compiled: list[dict[str, Any]] = []
    probe_ran: list[dict[str, Any]] = []
    probe_informative: list[dict[str, Any]] = []
    probe_supports: list[dict[str, Any]] = []
    host_ran: list[dict[str, Any]] = []
    host_informative: list[dict[str, Any]] = []
    corroborated = [
        row
        for row in promotions
        if row.get("status") in {"corroborated", "verified"}
    ]
    for row in text_pass:
        rec = by_id.get(str(row.get("card_id") or ""), {})
        if rec.get("world_incompatible"):
            continue
        world_valid.append(row)
        if rec.get("probe_ran") or rec.get("verdict") or rec.get("experiment_bottleneck") == "implementation":
            compiled.append(row)
        if rec.get("probe_ran"):
            probe_ran.append(row)
            if rec.get("verdict") in {"supports", "weakens", "uninformative"}:
                probe_informative.append(row)
            if rec.get("verdict") == "supports":
                probe_supports.append(row)
        if rec.get("host_ok") is True:
            host_ran.append(row)
            if rec.get("verdict") in {"supports", "weakens", "uninformative"}:
                host_informative.append(row)
    return {
        "generated": generated,
        "text_pass": len(text_pass),
        "world_valid": len(world_valid),
        "compiled": len(compiled),
        "probe_executed": len(probe_ran),
        "probe_informative": len(probe_informative),
        "probe_supports": len(probe_supports),
        "host_executed": len(host_ran),
        "host_informative": len(host_informative),
        "corroborated": len(corroborated),
        "rates": {
            "P(text_pass|generated)": _rate(len(text_pass), generated),
            "P(world_valid|text_pass)": _rate(len(world_valid), len(text_pass)),
            "P(compile|world_valid)": _rate(len(compiled), len(world_valid)),
            "P(probe_informative|probe_executed)": _rate(
                len(probe_informative), len(probe_ran)
            ),
            "P(host_success|probe_supports)": _rate(len(host_ran), len(probe_supports)),
            "P(corroborated|host_executed)": _rate(len(corroborated), len(host_ran)),
        },
        "honesty": (
            "Do not rank models by corroborated count until workspace "
            "isolation, WORLD binding, and the host confirmation contract hold."
        ),
    }
