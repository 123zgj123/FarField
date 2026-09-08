"""Research routing changes operator allocation, never scientific judgment.

Historical promotion-precision helpers remain available for offline
comparisons. Production admission requires an immutable evaluator receipt:
recurring scientific failures, disjoint completed mission partitions,
protected baseline/candidate replays, matched measured costs, improved
heldout hypothesis discrimination, and no regression. A missing receipt
records rejection and preserves incumbent bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .world import WORLD_KINDS

LOG_CAP = 200
SMOOTHING = 1.0
SCORE_TOLERANCE = 1e-9
PROMOTED = ("corroborated", "verified")

# Schema 2 began when cards started carrying `probe_kind` and the W2
# observations. Earlier records stay in the log as history, but they
# cannot vote in any settlement: a support that cannot prove it ran on
# an attested world is not evidence for a routing (or judge) change.
LOG_SCHEMA = 2

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
    """Low-level offline comparator, retained for historical experiments.

    This promotion-precision comparator is NOT production policy admission.
    Production callers must use ``maybe_update`` with an evaluator receipt.
    """
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

PARTITIONS = ("training", "nearby", "heldout", "regression")


@dataclass(frozen=True)
class ResearchPolicyEvaluation:
    """Opaque handle to one evaluator-issued, immutable, single-use receipt.

    Issuance and consumption must happen in the same application process.
    Disk receipts are audit artifacts, not bearer credentials that a model
    can manufacture or revive after a restart.
    """

    receipt_id: str
    digest: str
    path: str


_ISSUED_EVALUATIONS: dict[str, bytes] = {}


def _policy_bytes_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


def _finite_cost(raw: Any) -> dict[str, float]:
    if not isinstance(raw, Mapping) or set(raw) != {"tokens", "wall_seconds"}:
        raise PolicyError("measured costs require tokens and wall_seconds")
    values = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyError("measured costs must be finite numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise PolicyError("measured costs must be finite and nonnegative")
        values[key] = number
    if values["wall_seconds"] <= 0:
        raise PolicyError("a completed replay needs a measured positive duration")
    return values


def _validated_outcome(card: Mapping[str, Any]) -> float:
    """Recheck a canonical execution record, not a promotion/value boolean.

    Both a reproduced support and a reproduced weakening discriminate a
    hypothesis. Surviving a judge or producing prose earns no capability.
    The input log is an application-owned evidence archive, never model
    output; identities and registered arithmetic are still revalidated.
    """
    from .diagnose import diagnosis_from_payload, judge_probe
    from .evidence import evidence_id

    if (str(card.get("probe_kind") or "").upper() not in WORLD_KINDS
            or str(card.get("epistemic") or "WORLD").upper() != "WORLD"
            or str(card.get("world_role") or "world").lower() in {"generated", "synthetic", "placebo"}
            or card.get("execution_status") != "ran"
            or str(card.get("ledger") or "").lower() != "scientific"
            or card.get("claim_consistent") is not True
            or card.get("cannot_corroborate")
            or (card.get("ablation_required") and card.get("mechanism_identified") is not True)
            or card.get("dv_blind") or card.get("world_consumed") is False
            or card.get("object_absent") or card.get("validation_gaps")):
        raise PolicyError("capability evidence must be validated executed WORLD science")
    keys = ("experiment_digest", "world_digest", "data_digest", "evidence_id")
    if any(not isinstance(card.get(key), str) or len(card[key]) != 64
           or any(ch not in "0123456789abcdef" for ch in card[key]) for key in keys):
        raise PolicyError("execution evidence has missing or malformed immutable identity")
    expected = evidence_id(claim_id=str(card.get("card_id") or ""),
                           experiment_digest=card["experiment_digest"],
                           world_digest=card["world_digest"], data_digest=card["data_digest"])
    if not card.get("card_id") or card["evidence_id"] != expected:
        raise PolicyError("execution EvidenceID does not match its input digests")
    host = card.get("host_record")
    if (not isinstance(host, Mapping) or host.get("status") != "ran"
            or host.get("replication") != "E1_same_instance"
            or any(host.get(key) != card[key] for key in keys)):
        raise PolicyError("unconfirmed WORLD execution is not research capability")
    for key in ("treatment", "control"):
        value = card.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PolicyError("execution arms must be finite measured numbers")
        if host.get(key) != value:
            raise PolicyError("host reproduction changed the measured outcome")
    diagnosis = diagnosis_from_payload(card["card_id"], dict(card.get("diagnosis") or {}))
    verdict = judge_probe(diagnosis, card["treatment"], card["control"])["verdict"]
    if verdict != card.get("verdict"):
        raise PolicyError("recorded verdict differs from pre-registered arithmetic")
    _finite_cost(card.get("measured_cost"))
    return float(verdict in {"supports", "weakens"})


def _check_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping) or set(candidate) - {"weights", "fitted_on"}:
        raise PolicyError("ResearchPolicy may only change supported routing weights")
    weights = candidate.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        raise PolicyError("candidate must declare routing weights")
    for key, value in weights.items():
        if (not isinstance(key, str) or not key or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise PolicyError("routing weights must be finite positive numbers")
    if "fitted_on" in candidate and (isinstance(candidate["fitted_on"], bool)
                                     or not isinstance(candidate["fitted_on"], int)
                                     or candidate["fitted_on"] < 0):
        raise PolicyError("fitted_on must be a nonnegative count")
    return json.loads(json.dumps(candidate))


def evaluate_research_policy(
    log_path: Path,
    policy_path: Path,
    *,
    suite: Mapping[str, list[str]],
    candidate: Mapping[str, Any] | None = None,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
) -> ResearchPolicyEvaluation:
    """Evaluate one immutable proposal through protected mission replay.

    ``suite`` contains disjoint training/nearby/heldout/regression mission
    IDs. ``evaluator(policy, mission, *, partition)`` is trusted application
    code, never an LLM tool argument. It executes an offline mission replay
    and returns completed observations, run ID, policy/source digests, and
    measured costs. This module recomputes capability and matches each
    observation and its cost to the canonical validated execution archive.

    The old ``score_policy`` is reported only as historical routing
    precision. Reweighting old promotion labels cannot authorize admission.
    No evaluation callback means an inadmissible receipt, never a pass.
    """
    path, log_path = Path(policy_path), Path(log_path)
    receipt_id = secrets.token_hex(16)
    payload: dict[str, Any] = {"schema": 1, "receipt_id": receipt_id,
        "policy_path": str(path.resolve()), "log_path": str(log_path.resolve()),
        "baseline_bytes_digest": _policy_bytes_digest(path), "rejections": [], "replays": []}
    try:
        incumbent = load_policy(path)
        missions = settlement_missions(load_log(log_path))
        payload["log_digest"] = _digest(missions)
        payload["baseline_digest"] = _digest(incumbent)
        payload["suite"] = json.loads(json.dumps(suite))
        if not isinstance(suite, Mapping) or set(suite) != set(PARTITIONS):
            raise PolicyError("evaluation suite requires training, nearby, heldout, regression partitions")
        groups = {}
        used = set()
        index = {}
        for mission in missions:
            mid = str(mission.get("mission_id") or "")
            if not mid or mid in index:
                raise PolicyError("evaluation requires distinct immutable mission IDs")
            index[mid] = mission
        for partition in PARTITIONS:
            ids = suite[partition]
            if not isinstance(ids, (list, tuple)) or not ids:
                raise PolicyError(f"missing {partition} mission evidence")
            if any(not isinstance(mid, str) or mid not in index for mid in ids):
                raise PolicyError(f"{partition} contains unknown mission IDs")
            if len(set(ids)) != len(ids) or used.intersection(ids):
                raise PolicyError("training/nearby/heldout/regression mission IDs must be disjoint")
            used.update(ids)
            groups[partition] = [index[mid] for mid in ids]
            for mission in groups[partition]:
                if mission.get("status") != "completed":
                    raise PolicyError("only completed missions may evaluate research policy")
                if not mission.get("cards"):
                    raise PolicyError("completed mission lacks execution evidence")
                for card in mission["cards"]:
                    _validated_outcome(card)
        failure_groups: dict[str, dict[str, Any]] = {}
        for mission in groups["training"]:
            evidence = {card["evidence_id"]: card for card in mission["cards"]}
            for failure in mission.get("failures") or []:
                if str(failure.get("class") or "").upper() not in {"SCIENTIFIC", "SCIENTIFIC_FAILURE"}:
                    continue
                signature = str(failure.get("signature") or "")
                from .evidence import DIAGNOSTIC_KINDS
                diagnostic = signature.lower().replace("-", "_")
                if (diagnostic in DIAGNOSTIC_KINDS
                        or any(word in diagnostic for word in (
                            "infra", "timeout", "sandbox", "implementation", "artifact_missing",
                            "api_error", "quota", "rate_limit", "budget_exhaust"))):
                    continue
                record = evidence.get(failure.get("evidence_id"))
                if (not signature or not failure.get("id") or not record
                        or record.get("verdict") not in {"weakens", "uninformative"}):
                    continue
                group = failure_groups.setdefault(signature, {"ids": set(), "missions": {}})
                group["ids"].add(str(failure["id"]))
                group["missions"][mission["mission_id"]] = mission
        recurring = [group for group in failure_groups.values()
                     if len(group["ids"]) >= 3 and len(group["missions"]) >= 3]
        if not recurring:
            raise PolicyError("at least three distinct recurring scientific failures are required")
        historical = {mid: mission for group in recurring for mid, mission in group["missions"].items()}
        payload["historical_failure_ids"] = sorted(historical)
        proposal = _check_candidate(candidate if candidate is not None else propose_policy(groups["training"]))
        if "fitted_on" in proposal and proposal["fitted_on"] != len(groups["training"]):
            raise PolicyError("candidate fitted_on does not match its training partition")
        payload.update(candidate=proposal, candidate_digest=_digest(proposal),
                       historical_candidate_precision=score_policy(proposal, groups["heldout"]),
                       historical_baseline_precision=score_policy(incumbent, groups["heldout"]))
        if not callable(evaluator):
            raise PolicyError("protected candidate-specific mission replay evaluator is required")
        groups["historical_failure"] = list(historical.values())
        run_ids = set()
        summaries = {}
        for partition in ("nearby", "heldout", "regression", "historical_failure"):
            summaries[partition] = {}
            for label, policy in (("baseline", incumbent), ("candidate", proposal)):
                gains, count = 0.0, 0
                costs = {"tokens": 0.0, "wall_seconds": 0.0}
                mission_metrics = {}
                for mission in groups[partition]:
                    policy_input = json.loads(json.dumps(policy))
                    mission_input = json.loads(json.dumps(mission))
                    replay = evaluator(policy_input, mission_input, partition=partition)
                    if _digest(policy_input) != _digest(policy) or _digest(mission_input) != _digest(mission):
                        raise PolicyError("replay evaluator mutated its policy or mission inputs")
                    if (not isinstance(replay, Mapping) or replay.get("status") != "completed"
                            or replay.get("mission_id") != mission["mission_id"]
                            or replay.get("source_digest") != _digest(mission)
                            or replay.get("policy_digest") != _digest(policy)):
                        raise PolicyError("replay is not bound to the evaluated policy and mission")
                    rid = str(replay.get("run_id") or "")
                    if not rid or rid in run_ids:
                        raise PolicyError("replay run IDs must be distinct")
                    run_ids.add(rid)
                    observations = replay.get("observations")
                    if not isinstance(observations, list) or not observations:
                        raise PolicyError("replay produced no measured scientific observations")
                    canonical = {card["evidence_id"]: card for card in mission["cards"]}
                    if len({row.get("evidence_id") for row in observations}) != len(observations):
                        raise PolicyError("replay duplicates one evidence object")
                    local_gains = 0.0
                    local_cost = {"tokens": 0.0, "wall_seconds": 0.0}
                    for observation in observations:
                        if canonical.get(observation.get("evidence_id")) != observation:
                            raise PolicyError("replay observation or measured cost differs from validated archive")
                        local_gains += _validated_outcome(observation)
                        measured = _finite_cost(observation.get("measured_cost"))
                        for key in local_cost:
                            local_cost[key] += measured[key]
                    reported = _finite_cost(replay.get("measured_cost"))
                    if any(abs(reported[key] - local_cost[key]) > SCORE_TOLERANCE for key in local_cost):
                        raise PolicyError("reported replay costs differ from actual matched observations")
                    metric = local_gains / len(observations)
                    mission_metrics[mission["mission_id"]] = {"capability": metric, "cost": local_cost}
                    gains += metric
                    count += 1
                    for key in costs:
                        costs[key] += local_cost[key]
                    payload["replays"].append({"partition": partition, "policy": label,
                        "mission_id": mission["mission_id"], "run_id": rid, "digest": _digest(replay)})
                _finite_cost(costs)
                summaries[partition][label] = {"capability": gains / count, "cost": costs,
                                               "missions": mission_metrics}
            baseline, changed = summaries[partition]["baseline"], summaries[partition]["candidate"]
            for mid, measured in changed["missions"].items():
                before = baseline["missions"][mid]
                if any(measured["cost"][key] > before["cost"][key] + SCORE_TOLERANCE for key in costs):
                    payload["rejections"].append(f"{partition}: candidate exceeded matched baseline costs")
                if measured["capability"] + SCORE_TOLERANCE < before["capability"]:
                    payload["rejections"].append(f"{partition}: research capability regressed on {mid}")
            if partition == "heldout" and changed["capability"] <= baseline["capability"] + SCORE_TOLERANCE:
                payload["rejections"].append("heldout capability did not strictly improve")
        payload["metrics"] = summaries
        payload["metric"] = "reproduced_hypothesis_discrimination_precision"
    except Exception as exc:
        # An evaluator failure is a recorded rejection, never a default pass.
        payload["rejections"].append(str(exc) or type(exc).__name__)
    receipt_path = path.with_name(path.name + f".evaluation-{receipt_id}.json")
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(raw)
    _ISSUED_EVALUATIONS[receipt_id] = raw
    return ResearchPolicyEvaluation(receipt_id, hashlib.sha256(raw).hexdigest(), str(receipt_path))


def maybe_update(
    log_path: Path,
    policy_path: Path,
    *,
    min_missions: int = MIN_MISSIONS,
    evaluation: ResearchPolicyEvaluation | None = None,
) -> dict[str, Any] | None:
    """End-of-mission production admission; missing evaluation rejects.

    Never accepts caller-provided pass/fail booleans. Every attempt is
    persisted separately; rejecting leaves incumbent policy bytes intact.
    An issued receipt is consumed on its first admission attempt, including
    rejection, and cannot be replayed or revived from its audit JSON.
    """
    from .filelock import file_lock

    path = Path(policy_path)
    decisions_path = path.with_name(path.name + ".decisions.jsonl")
    with file_lock(path):
        reasons = []
        payload = {}
        receipt_id = ""
        if not isinstance(evaluation, ResearchPolicyEvaluation):
            reasons.append("mandatory evaluator-issued evaluation receipt is missing")
        else:
            receipt_id = evaluation.receipt_id
            previous = []
            if decisions_path.is_file():
                previous = [json.loads(line) for line in decisions_path.read_text().splitlines() if line]
            if any(row.get("receipt_id") == receipt_id for row in previous):
                reasons.append("evaluation receipt has already been consumed")
            raw = _ISSUED_EVALUATIONS.pop(receipt_id, None)
            if raw is None:
                reasons.append("evaluation receipt was not issued in this application process")
            else:
                payload = json.loads(raw)
                try:
                    if (hashlib.sha256(raw).hexdigest() != evaluation.digest
                            or Path(evaluation.path).read_bytes() != raw):
                        reasons.append("evaluation receipt was mutated")
                except OSError:
                    reasons.append("evaluation receipt artifact is missing")
                if (payload.get("policy_path") != str(path.resolve())
                        or payload.get("log_path") != str(Path(log_path).resolve())):
                    reasons.append("evaluation belongs to another policy or trajectory log")
                if payload.get("baseline_bytes_digest") != _policy_bytes_digest(path):
                    reasons.append("incumbent bytes changed after evaluation")
                try:
                    if payload.get("baseline_digest") != _digest(load_policy(path)):
                        reasons.append("evaluated baseline differs from current policy")
                    missions = settlement_missions(load_log(log_path))
                    if payload.get("log_digest") != _digest(missions):
                        reasons.append("evaluation mission inputs changed")
                    if len(missions) < min_missions:
                        reasons.append("insufficient completed mission history")
                except (OSError, ValueError, PolicyError) as exc:
                    reasons.append(str(exc))
                reasons.extend(payload.get("rejections") or [])
                if not payload.get("metrics") or not payload.get("replays"):
                    reasons.append("required protected replay results are missing")
                if payload.get("candidate_digest") != _digest(payload.get("candidate")):
                    reasons.append("candidate identity changed")
        committed = not reasons
        decision = {"committed": committed, "receipt_id": receipt_id,
                    "reason": "; ".join(reasons) if reasons else "heldout research capability improved at matched costs",
                    "candidate": payload.get("candidate"), "metrics": payload.get("metrics", {}),
                    "baseline_bytes_digest": _policy_bytes_digest(path)}
        if committed:
            candidate = _check_candidate(payload["candidate"])
            path.write_text(json.dumps({"policy": candidate, "digest": _digest(candidate)},
                                       ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        with decisions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, ensure_ascii=False, sort_keys=True) + "\n")
        return decision
