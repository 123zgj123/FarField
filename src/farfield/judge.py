"""Wave C judgment layer: H_t, pre-declaration, DV, far-field reserve.

This module does not read ``expected_information_gain`` and does not call
``select_portfolio``. Far-zone ordering is first measurable cost only.
Importing this module does not put judgment in the loop; only a Judge that
actually declared and resolved does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from .ledger import content_digest
from .models import (
    WAVE_A_PLACEHOLDERS,
    BudgetEnvelope,
    Evidence,
    TrustClass,
    new_id,
)
from .runtime import FarfieldRuntime

FARFIELD_BUDGET_FRACTION = 0.3
MAX_STRAWMAN_RATE = 0.25
Zone = Literal["near", "far"]
Pair = tuple[str, str]


class JudgeError(ValueError):
    pass


def far_quota(budget: float, fraction: float = FARFIELD_BUDGET_FRACTION) -> int:
    if budget < 1:
        return 0
    return max(1, ceil(fraction * budget))


def near_quota(budget: float, fraction: float = FARFIELD_BUDGET_FRACTION) -> int:
    total = int(budget)
    return max(0, total - far_quota(budget, fraction))


def rank_far_by_first_measurable_cost(
    candidates: Sequence[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Far-reserve sort key: first cost that creates a declarable pair, not DV."""
    return sorted(candidates, key=lambda item: (item[1], item[0]))


@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    zone: Zone
    cheapest_falsifier: str
    falsifier_cost: float
    is_core: bool = False

    def validate(self) -> None:
        if not self.id.strip() or not self.statement.strip():
            raise JudgeError("hypothesis id and statement must be non-empty")
        if self.zone not in {"near", "far"}:
            raise JudgeError("hypothesis zone must be near or far")
        if not self.cheapest_falsifier.strip():
            raise JudgeError("hypothesis cheapest_falsifier must be non-empty")
        if self.falsifier_cost <= 0:
            raise JudgeError("hypothesis falsifier_cost must be positive")


@dataclass
class Purchase:
    e_id: str
    pair_ids: frozenset[Pair]
    cost: float
    zone: Zone
    is_core_falsifier: bool
    unresolved_at_declare: frozenset[Pair]
    realized: frozenset[Pair] | None = None
    trust_class: str | None = None
    killed: tuple[str, ...] = ()


class Judge:
    def __init__(
        self,
        hypotheses: Sequence[Hypothesis],
        *,
        literature_fetches: float,
        far_fraction: float = FARFIELD_BUDGET_FRACTION,
        core_absent_reason: str | None = None,
    ) -> None:
        rows = tuple(hypotheses)
        for hyp in rows:
            hyp.validate()
        ids = [hyp.id for hyp in rows]
        if len(set(ids)) != len(ids):
            raise JudgeError("hypothesis ids must be unique")
        cores = [hyp for hyp in rows if hyp.is_core]
        if len(cores) > 1:
            raise JudgeError("at most one core hypothesis")
        if cores and core_absent_reason:
            raise JudgeError("core_absent_reason given but a core hypothesis exists")
        self.has_core = bool(cores)

        self.core_absent_reason = (core_absent_reason or "").strip() or None
        self.hypotheses = rows
        self.killed = set()
        self.purchases = []
        self.trace = []
        self.budget = literature_fetches
        self.far_cap = far_quota(literature_fetches, far_fraction)
        self.near_cap = near_quota(literature_fetches, far_fraction)
        self.far_spent = 0
        self.near_spent = 0
        self.focus = "core"
        self._focus_changed_by_resolve = False
        self._declare_set_changed_after_resolve = False
        self._resolves = 0
        self._executed_checks = 0
        self.trace.append(
            {
                "event": "judge_init",
                "n_hypotheses": len(rows),
                "far_cap": self.far_cap,
                "near_cap": self.near_cap,
                "has_core": self.has_core,
                "core_absent_reason": self.core_absent_reason,
            }
        )

    def live(self) -> tuple[Hypothesis, ...]:
        return tuple(hyp for hyp in self.hypotheses if hyp.id not in self.killed)

    def unresolved_pairs(self) -> frozenset[Pair]:
        return _pairs_of(hyp.id for hyp in self.live())

    def declare_purchase(
        self,
        e_id: str,
        pair_ids: Iterable[Pair],
        cost: float,
        zone: Zone,
        *,
        is_core_falsifier: bool = False,
    ) -> Purchase:
        if not e_id.strip():
            raise JudgeError("purchase e_id must be non-empty")
        if any(item.e_id == e_id for item in self.purchases):
            raise JudgeError(f"duplicate purchase id: {e_id}")
        if zone not in {"near", "far"}:
            raise JudgeError("purchase zone must be near or far")
        if cost <= 0:
            raise JudgeError("purchase cost must be positive")
        if is_core_falsifier and zone != "near":
            raise JudgeError("core falsifier must be a near purchase")
        if self.purchases and is_core_falsifier:
            raise JudgeError("core falsifier must be the first purchase")
        if not self.purchases and not is_core_falsifier:
            pass

        pairs = _canonical_pairs(pair_ids)
        unresolved = self.unresolved_pairs()
        if not pairs <= unresolved:
            raise JudgeError("declared pairs must be currently unresolved")
        if zone == "far":
            if self.far_spent >= self.far_cap:
                raise JudgeError(
                    "far quota exhausted; unused far quota does not flow back to near"
                )
        else:
            if self.near_spent >= self.near_cap:
                raise JudgeError("near quota exhausted; near cannot borrow far reserve")
        if self._resolves > 0 and self.purchases:
            previous = self.purchases[-1].pair_ids
            if pairs != previous:
                self._declare_set_changed_after_resolve = True
        purchase = Purchase(
            e_id=e_id,
            pair_ids=pairs,
            cost=cost,
            zone=zone,
            is_core_falsifier=is_core_falsifier,
            unresolved_at_declare=unresolved,
        )
        self.purchases.append(purchase)
        if zone == "far":
            self.far_spent += 1
        else:
            self.near_spent += 1
        self.trace.append(
            {
                "event": "declare_purchase",
                "e_id": e_id,
                "zone": zone,
                "cost": cost,
                "pairs": sorted(pairs),
                "is_core_falsifier": is_core_falsifier,
                "focus": self.focus,
                "far_spent": self.far_spent,
                "near_spent": self.near_spent,
            }
        )
        return purchase

    def resolve(
        self,
        e_id: str,
        realized_pair_ids: Iterable[Pair],
        *,
        trust_class: TrustClass | str,
        kill: Sequence[str] = (),
        executed_checks: int = 0,
    ) -> Purchase:
        label = (
            trust_class.value if isinstance(trust_class, TrustClass) else str(trust_class)
        )
        if label != TrustClass.BOUND.value:
            raise JudgeError("resolution requires bound evidence")
        purchase = self._purchase(e_id)
        if purchase.realized is not None:
            raise JudgeError(f"purchase {e_id} already resolved")
        realized = _canonical_pairs(realized_pair_ids)
        purchase.realized = realized
        purchase.trust_class = label
        purchase.killed = tuple(kill)
        for hyp_id in kill:
            self.killed.add(hyp_id)
        self._resolves += 1
        self._executed_checks += max(0, int(executed_checks))
        core_ids = {hyp.id for hyp in self.hypotheses if hyp.is_core}
        if self.focus == "core" and core_ids & set(kill):
            self.focus = "far"
            self._focus_changed_by_resolve = True
        self.trace.append(
            {
                "event": "resolve",
                "e_id": e_id,
                "realized": sorted(realized),
                "killed": list(kill),
                "focus": self.focus,
                "executed_checks": int(executed_checks),
                "dv": self.dv(e_id),
            }
        )
        return purchase

    def dv(self, e_id: str) -> float:
        purchase = self._purchase(e_id)
        if purchase.realized is None:
            raise JudgeError(f"purchase {e_id} has no realized pairs yet")
        hit = purchase.realized & purchase.pair_ids
        return len(hit) / max(1, len(purchase.unresolved_at_declare))

    def strawman_rate(self) -> float:
        if not self.killed:
            return 0.0
        declared = set()
        for purchase in self.purchases:
            for left, right in purchase.pair_ids:
                declared.add(left)
                declared.add(right)
        straw = [hyp_id for hyp_id in self.killed if hyp_id not in declared]
        return len(straw) / len(self.killed)

    def judgment_failed(self) -> bool:
        return self.strawman_rate() > MAX_STRAWMAN_RATE

    def pre_judgment(self) -> bool:
        """True iff judgment is not in the loop for this campaign."""
        if not self.hypotheses:
            return True
        if not self.purchases:
            return True
        if any(item.realized is None for item in self.purchases):
            return True
        if self.has_core:
            if not self.purchases[0].is_core_falsifier:
                return True

            if (
                not self._focus_changed_by_resolve
                and not self._declare_set_changed_after_resolve
            ):
                return True
            return False
        if self.core_absent_reason is None:
            return True

        return self._executed_checks <= 0

    def rating(self, certified_rating_events: Sequence[dict[str, Any]] = ()) -> str:
        if not certified_rating_events:
            return WAVE_A_PLACEHOLDERS["rating"]
        return WAVE_A_PLACEHOLDERS["rating"]

    def _purchase(self, e_id: str) -> Purchase:
        for item in self.purchases:
            if item.e_id == e_id:
                return item
        raise JudgeError(f"unknown purchase: {e_id}")


def acceptance_hypotheses() -> tuple[Hypothesis, ...]:
    return (
        Hypothesis(
            id="hyp_core",
            statement="strawman near: the mean baseline is already optimal",
            zone="near",
            cheapest_falsifier="run OLS against the frozen mean baseline",
            falsifier_cost=1.0,
            is_core=True,
        ),
        Hypothesis(
            id="hyp_near2",
            statement="another near-field linear variant",
            zone="near",
            cheapest_falsifier="ablate the intercept term",
            falsifier_cost=1.0,
        ),
        Hypothesis(
            id="hyp_far_d",
            statement="distant n_d mechanism is worth probing",
            zone="far",
            cheapest_falsifier="inspect whether n_d sits inside the seed 2-hop",
            falsifier_cost=1.0,
        ),
        Hypothesis(
            id="hyp_far_e",
            statement="distant n_e anomaly is worth probing",
            zone="far",
            cheapest_falsifier="inspect whether n_e sits inside the seed 2-hop",
            falsifier_cost=1.0,
        ),
    )


def run_acceptance_judgment() -> Judge:
    judge = Judge(acceptance_hypotheses(), literature_fetches=3.0)
    core_pairs = _pairs_involving(judge.unresolved_pairs(), "hyp_core")
    judge.declare_purchase(
        "ev_core01",
        core_pairs,
        cost=1.0,
        zone="near",
        is_core_falsifier=True,
    )
    judge.resolve(
        "ev_core01",
        core_pairs,
        trust_class=TrustClass.BOUND,
        kill=("hyp_core",),
    )
    far_ranked = rank_far_by_first_measurable_cost(
        [(hyp.id, hyp.falsifier_cost) for hyp in judge.live() if hyp.zone == "far"]
    )
    far_id = far_ranked[0][0]
    far_pairs = _pairs_involving(judge.unresolved_pairs(), far_id)
    judge.declare_purchase("ev_far01", far_pairs, cost=1.0, zone="far")
    judge.resolve("ev_far01", far_pairs, trust_class=TrustClass.BOUND)
    return judge


def commit_judgment(
    runtime: FarfieldRuntime,
    mission_id: str,
    clause_id: str,
    dest: Path,
    judge: Judge,
) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    evidence_ids = []
    for purchase in judge.purchases:
        payload = {
            "e_id": purchase.e_id,
            "zone": purchase.zone,
            "cost": purchase.cost,
            "declared_pairs": [list(pair) for pair in sorted(purchase.pair_ids)],
            "realized_pairs": (
                [list(pair) for pair in sorted(purchase.realized)]
                if purchase.realized is not None
                else []
            ),
            "dv": judge.dv(purchase.e_id) if purchase.realized is not None else None,
            "focus_after": judge.focus,
        }
        path = dest / f"{purchase.e_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence_ids.append(
            runtime.record_evidence(
                Evidence(
                    claim=f"judgment purchase {purchase.e_id}",
                    artifact_uri=path.resolve().as_uri(),
                    verifier_id="judgment-dv-v1",
                    outcome="supports",
                    scope="wave-c-judgment",
                    mission_id=mission_id,
                    clause_ids=(clause_id,),
                    sealed=False,
                    trust_class=TrustClass.BOUND,
                    source_id=f"judgment:{purchase.e_id}",
                ),
                actor="verifier",
            )
        )

    trace_path = dest / "judgment_trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "trace": judge.trace,
                "pre_judgment": judge.pre_judgment(),
                "strawman_rate": judge.strawman_rate(),
                "far_spent": judge.far_spent,
                "near_spent": judge.near_spent,
                "digest": content_digest(judge.trace),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    evidence_ids.append(
        runtime.record_evidence(
            Evidence(
                claim="judgment trace",
                artifact_uri=trace_path.resolve().as_uri(),
                verifier_id="judgment-dv-v1",
                outcome="supports",
                scope="wave-c-judgment",
                mission_id=mission_id,
                clause_ids=(clause_id,),
                sealed=False,
                trust_class=TrustClass.BOUND,
                source_id="judgment:trace",
            ),
            actor="verifier",
        )
    )

    return evidence_ids


def run_judgment_mission(
    project_dir: Path | str,
    *,
    envelope: BudgetEnvelope | None = None,
) -> Judge:
    envelope = envelope or BudgetEnvelope()
    envelope.validate()
    runtime = FarfieldRuntime(project_dir)
    objective = runtime.current_objective()
    mission_id = runtime.export_mission(
        objective_id=objective["id"],
        goal="Wave C: declare, falsify, then spend far-field reserve",
        completion_evidence=["judgment purchases recorded with DV"],
        constraints=["do not price far-field ideas by DV/cost", "do not propose tool updates"],
        budget=envelope.mission_budget(),
        actor="planner",
    )
    contract = runtime.load_mission_contract(mission_id)
    clause_id = contract["completion_clauses"][0]["id"]
    judge = run_acceptance_judgment()
    dest = Path(runtime.project_dir) / ".farfield" / "wave_c" / new_id("camp")
    evidence_ids = commit_judgment(runtime, mission_id, clause_id, dest, judge)
    status = "failed" if judge.judgment_failed() else "done"
    runtime.settle_mission(
        mission_id,
        status=status,
        evidence_ids=evidence_ids,
        actor="reviewer",
    )

    return judge


def _pairs_of(ids: Iterable[str]) -> frozenset[Pair]:
    ordered = sorted(ids)
    return frozenset(
        (ordered[i], ordered[j])
        for i in range(len(ordered))
        for j in range(i + 1, len(ordered))
    )


def _canonical_pairs(pairs: Iterable[Pair]) -> frozenset[Pair]:
    canonical = set()
    for left, right in pairs:
        if left == right:
            raise JudgeError("a pair must name two distinct hypotheses")
        canonical.add((left, right) if left < right else (right, left))
    return frozenset(canonical)


def _pairs_involving(pairs: Iterable[Pair], hyp_id: str) -> frozenset[Pair]:
    return frozenset(pair for pair in pairs if hyp_id in pair)
