"""Falsifiable probing of far-field conjecture cards (INV-2, INV-11).

A probe whose predicate is the generator's own filter cannot fail, and a card
that cannot fail did not survive anything. Membership in the seed k-hop is such
a predicate: `_drift_cards` already discards every path inside the k-hop, so
re-checking membership marks cards `probed` without ever being able to kill one.

Every check below whose `discriminating` flag is true tests a claim the
generator did not filter on, so a card can fail it. Checks with the flag false
are consistency guards (mainly against arbitrary installed operators) and never
by themselves justify a `probed` mark.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .graph import GraphSnapshot, path_edges

Edge = tuple[str, str]


@dataclass(frozen=True)
class ProbeCheck:
    name: str
    claim: str
    passed: bool
    detail: str
    discriminating: bool
    # Whether this check is here because a model asked for it. Without the flag a
    # verdict cannot be attributed: a killed card looks the same whether the
    # kernel's own check or the proposed one did it.
    proposed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "claim": self.claim,
            "passed": self.passed,
            "detail": self.detail,
            "discriminating": self.discriminating,
            "proposed": self.proposed,
        }


@dataclass(frozen=True)
class ProbeResult:
    card_id: str
    mode: str
    probed: bool
    killed: bool
    checks: tuple[ProbeCheck, ...]

    @property
    def killed_without_proposals(self) -> bool:
        """The verdict the kernel would have reached with no model in the loop."""
        return any(not check.passed for check in self.checks if not check.proposed)

    @property
    def proposal_was_decisive(self) -> bool:
        """This card died, and it is a proposed check that killed it.

        Counting proposals, or even counting the checks they add, does not say
        whether a model changed an outcome: a card already dead from its own
        mode's check stays dead whatever else is run against it. This is the
        counterfactual, and it is the only form in which a claim that the model
        contributed to a verdict can be checked.
        """
        return self.killed and not self.killed_without_proposals

    def to_dict(self) -> dict[str, Any]:
        return {
            "card_id": self.card_id,
            "mode": self.mode,
            "probed": self.probed,
            "killed": self.killed,
            "proposal_was_decisive": self.proposal_was_decisive,
            "checks": [check.to_dict() for check in self.checks],
        }


def descendants(graph: GraphSnapshot, node_id: str) -> set[str]:
    seen: set[str] = set()
    queue = deque(graph.outgoing.get(node_id, ()))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(graph.outgoing.get(node, ()))
    return seen


def card_sinks(nodes: Sequence[str], edges: Iterable[Edge]) -> tuple[str, ...]:
    """Nodes with no outgoing edge inside the card's own edge set.

    A single path has one sink. A crossover union has one per parent branch,
    which is how the parent endpoints are recovered without a schema field.
    """
    sources = {src for src, _ in edges}
    return tuple(node for node in nodes if node not in sources)


def _edges_exist(graph: GraphSnapshot, edges: Sequence[Edge]) -> ProbeCheck:
    known = set(graph.edges)
    missing = [edge for edge in edges if tuple(edge) not in known]
    return ProbeCheck(
        name="edges_exist_in_snapshot",
        claim="every edge asserted by the card is an edge of the frozen snapshot",
        passed=not missing,
        detail=(
            "all card edges present"
            if not missing
            else f"fabricated edges: {sorted(missing)}"
        ),
        discriminating=False,
    )


def _mechanism_carries_through(
    graph: GraphSnapshot, endpoint: str, neighborhood: set[str]
) -> ProbeCheck:
    onward = [
        node
        for node in graph.outgoing.get(endpoint, ())
        if node not in neighborhood
    ]
    return ProbeCheck(
        name="mechanism_carries_through",
        claim=(
            f"the mechanism node {endpoint} continues outward, i.e. it has an "
            f"outgoing edge to a node outside the seed k-hop"
        ),
        passed=bool(onward),
        detail=(
            f"onward nodes: {sorted(onward)}"
            if onward
            else f"{endpoint} has no outgoing edge leaving the k-hop; a bridge "
            f"that goes nowhere is not a mechanism bridge"
        ),
        discriminating=True,
    )


def _anomaly_is_leaf(graph: GraphSnapshot, endpoint: str) -> ProbeCheck:
    outgoing = graph.outgoing.get(endpoint, ())
    incoming = graph.incoming.get(endpoint, ())
    passed = not outgoing and len(incoming) >= 1
    return ProbeCheck(
        name="anomaly_is_a_true_leaf",
        claim=(
            f"the anomalous node {endpoint} is a sink with at least one citing "
            f"parent, not an uncited source"
        ),
        passed=passed,
        detail=(
            f"sink with {len(incoming)} parent(s)"
            if passed
            else f"outgoing={sorted(outgoing)} incoming={sorted(incoming)}"
            f"; degree 1 alone does not make a leaf"
        ),
        discriminating=True,
    )


def _falsifier_is_not_near_implied(
    graph: GraphSnapshot, endpoint: str, neighborhood: set[str]
) -> ProbeCheck:
    inside_parents = [
        node for node in graph.incoming.get(endpoint, ()) if node in neighborhood
    ]
    return ProbeCheck(
        name="falsifier_not_implied_by_neighborhood",
        claim=(
            f"the counterexample node {endpoint} is not reachable in one hop "
            f"from inside the seed k-hop, so it is not already implied by it"
        ),
        passed=not inside_parents,
        detail=(
            "no k-hop parent cites it"
            if not inside_parents
            else f"cited directly by in-hood nodes {sorted(inside_parents)}"
            f"; a near-implied counterexample is not a distant falsifier"
        ),
        discriminating=True,
    )


def _falsifier_is_counterexample(graph: GraphSnapshot, endpoint: str) -> ProbeCheck:
    passed = endpoint in graph.counterexample_nodes
    return ProbeCheck(
        name="falsifier_endpoint_is_counterexample",
        claim=f"{endpoint} is a declared counterexample node of the snapshot",
        passed=passed,
        detail="declared counterexample" if passed else "not a counterexample node",
        discriminating=False,
    )


def _bridge_parents_have_contact(
    graph: GraphSnapshot, sinks: Sequence[str]
) -> ProbeCheck:
    if len(sinks) < 2:
        return ProbeCheck(
            name="bridge_parents_have_contact",
            claim="a crossover card must expose at least two parent endpoints",
            passed=False,
            detail=f"recovered sinks: {sorted(sinks)}",
            discriminating=True,
        )
    reach = {node: descendants(graph, node) for node in sinks}
    for index, left in enumerate(sinks):
        for right in sinks[index + 1 :]:
            if right in reach[left] or left in reach[right]:
                return ProbeCheck(
                    name="bridge_parents_have_contact",
                    claim=(
                        "the crossed parent endpoints have structural contact "
                        "beyond their shared prefix"
                    ),
                    passed=True,
                    detail=f"{left} and {right} are connected by descent",
                    discriminating=True,
                )
            shared = reach[left] & reach[right]
            if shared:
                return ProbeCheck(
                    name="bridge_parents_have_contact",
                    claim=(
                        "the crossed parent endpoints have structural contact "
                        "beyond their shared prefix"
                    ),
                    passed=True,
                    detail=f"{left} and {right} share descendants {sorted(shared)}",
                    discriminating=True,
                )
    return ProbeCheck(
        name="bridge_parents_have_contact",
        claim=(
            "the crossed parent endpoints have structural contact "
            "beyond their shared prefix"
        ),
        passed=False,
        detail=(
            f"parents {sorted(sinks)} have neither a connecting path nor a "
            f"common descendant; the union is a stapling, not a bridge"
        ),
        discriminating=True,
    )


HUB_QUANTILE = 0.9


def _in_degree_ceiling(graph: GraphSnapshot, quantile: float = HUB_QUANTILE) -> int:
    """The in-degree at `quantile` of the snapshot, by index on sorted degrees."""
    degrees = sorted(len(graph.incoming.get(node, ())) for node in graph.nodes)
    if not degrees:
        return 0
    return degrees[int(quantile * (len(degrees) - 1))]


def _endpoint_is_not_a_hub(graph: GraphSnapshot, endpoint: str) -> ProbeCheck:
    """Fail a card whose distance is explained by the endpoint's popularity.

    A node the whole corpus cites sits a short path from almost everything in it,
    so arriving at one says more about the citation graph's shape than about the
    seed. This is the failure mode a bibliography-dumping arm exploits, and it is
    not covered by any mode's default checks, which all look at local structure.

    The ceiling is a quantile of this snapshot rather than an absolute degree, so
    the check cannot fail for every card -- above the 0.9 quantile there is by
    construction at most a tenth of the nodes -- which is what keeps it from
    becoming the always-false mirror of an unfalsifiable path.

    The quantile alone is not enough, though. In a snapshot where most nodes are
    uncited it sits at zero, and then a single citation clears it, which would
    call a node with exactly one route in a hub -- the opposite of being near
    everything. So the ceiling never drops below one. On the real corpora it
    lands at 7 or 8 and the floor does not bind; it is there for the sparse case.
    """
    degree = len(graph.incoming.get(endpoint, ()))
    ceiling = max(_in_degree_ceiling(graph), 1)
    passed = degree <= ceiling
    return ProbeCheck(
        name="endpoint_is_not_a_citation_hub",
        claim=(
            f"the endpoint {endpoint} is cited no more than the snapshot's "
            f"{HUB_QUANTILE:g} quantile ({ceiling}), so the card's distance is not "
            f"an artefact of its popularity"
        ),
        passed=passed,
        detail=(
            f"in-degree {degree} at or below the {ceiling} ceiling"
            if passed
            else f"in-degree {degree} exceeds the {ceiling} ceiling; a node the "
            f"corpus cites this heavily is near everything, so reaching it is not "
            f"a far-field result"
        ),
        discriminating=True,
    )


def _path_has_no_shortcut(
    graph: GraphSnapshot,
    nodes: Sequence[str],
    edges: Sequence[Edge],
    endpoint: str,
) -> ProbeCheck:
    """Fail a card whose own chain is longer than the graph requires.

    The card asserts the endpoint is reached through the nodes it lists. If a node
    earlier than the endpoint's own predecessor already cites the endpoint, the
    intermediates are decoration: the endpoint is one hop from somewhere the card
    has already been, and the claimed distance is the card's, not the graph's.
    """
    predecessors = {destination: source for source, destination in edges}
    immediate = predecessors.get(endpoint)
    known = set(graph.edges)
    shortcuts = [
        node
        for node in nodes
        if node not in (endpoint, immediate) and (node, endpoint) in known
    ]
    return ProbeCheck(
        name="path_has_no_shortcut",
        claim=(
            f"no node the card lists reaches {endpoint} more directly than the "
            f"path it asserts"
        ),
        passed=not shortcuts,
        detail=(
            "the asserted path is the only route the card's own nodes offer"
            if not shortcuts
            else f"{sorted(shortcuts)} cite the endpoint directly, so the "
            f"intermediates are decoration and the distance is overstated"
        ),
        discriminating=True,
    )


def _check_by_name(
    name: str,
    graph: GraphSnapshot,
    endpoint: str,
    sinks: Sequence[str],
    neighborhood: set[str],
    nodes: Sequence[str],
    edges: Sequence[Edge],
) -> ProbeCheck | None:
    if name == "mechanism_carries_through":
        return _mechanism_carries_through(graph, endpoint, neighborhood)
    if name == "anomaly_is_a_true_leaf":
        return _anomaly_is_leaf(graph, endpoint)
    if name == "falsifier_not_implied_by_neighborhood":
        return _falsifier_is_not_near_implied(graph, endpoint, neighborhood)
    if name == "bridge_parents_have_contact":
        return _bridge_parents_have_contact(graph, sinks)
    if name == "endpoint_is_not_a_citation_hub":
        return _endpoint_is_not_a_hub(graph, endpoint)
    if name == "path_has_no_shortcut":
        return _path_has_no_shortcut(graph, nodes, edges, endpoint)
    return None


# The first four are also what each mode runs by default. Naming the one for the
# card's own mode is a model agreeing with the kernel: recorded, but it adds no
# check. Naming a different mode's does add one, so `added_checks` was never zero
# by construction -- in the recorded live run a bridge card was handed the
# mechanism check that way. What that could not do is change a verdict, because a
# check picked for another card shape fails for reasons that have nothing to do
# with this card, on a card its own mode's check had usually killed already.
#
# The last two belong to no mode's defaults and are about the card at hand, which
# is what makes it possible for a proposal to be the reason a card dies. Whether
# one ever is remains a measurement: `proposal_was_decisive`.
PROPOSABLE_CHECKS: tuple[str, ...] = (
    "mechanism_carries_through",
    "anomaly_is_a_true_leaf",
    "falsifier_not_implied_by_neighborhood",
    "bridge_parents_have_contact",
    "endpoint_is_not_a_citation_hub",
    "path_has_no_shortcut",
)

# What each mode already runs, as data rather than only as branches in
# `probe_card`. A test pins the two against each other, because a mapping that
# drifted from the code would quietly start offering a model the check its card
# is already facing again.
DEFAULT_CHECKS_BY_MODE: dict[str, tuple[str, ...]] = {
    "mechanism": ("mechanism_carries_through",),
    "anomaly": ("anomaly_is_a_true_leaf",),
    "falsifier": (
        "falsifier_endpoint_is_counterexample",
        "falsifier_not_implied_by_neighborhood",
    ),
    "bridge": ("bridge_parents_have_contact",),
}


def proposable_for(mode: str) -> tuple[str, ...]:
    """The checks worth asking a model about for a card of this mode.

    Offering the whole list gets the mode's own default back, because that check
    is genuinely the likeliest one to fail -- most cards that die, die on it. Two
    live runs answered it on every card, so the model's agreement with the kernel
    is established and costs a call per card to re-establish.

    What is not established is whether a model can pick, out of the checks the
    card is not already facing, one that catches a card the kernel let through.
    That question has an answer only if the default is off the menu, so it is.
    """
    already = set(DEFAULT_CHECKS_BY_MODE.get(mode, ()))
    return tuple(name for name in PROPOSABLE_CHECKS if name not in already)


def probe_card(
    graph: GraphSnapshot,
    card: dict[str, Any],
    neighborhood: set[str],
) -> ProbeResult:
    nodes = [str(node) for node in card.get("nodes") or []]
    mode = str(card.get("mode") or "")
    raw_edges = card.get("edges")
    edges = (
        [(str(src), str(dst)) for src, dst in raw_edges]
        if raw_edges
        else list(path_edges(nodes))
    )
    checks: list[ProbeCheck] = []
    if nodes:
        checks.append(_edges_exist(graph, edges))
    unknown = [node for node in nodes if node not in graph.nodes]
    if unknown:
        checks.append(
            ProbeCheck(
                name="nodes_exist_in_snapshot",
                claim="every node asserted by the card is in the frozen snapshot",
                passed=False,
                detail=f"unknown nodes: {sorted(unknown)}",
                discriminating=False,
            )
        )
    sinks = card_sinks(nodes, edges)
    endpoint = sinks[-1] if sinks else nodes[-1] if nodes else ""
    if not unknown and endpoint:
        if mode == "mechanism":
            checks.append(
                _mechanism_carries_through(graph, endpoint, neighborhood)
            )
        elif mode == "anomaly":
            checks.append(_anomaly_is_leaf(graph, endpoint))
        elif mode == "falsifier":
            checks.append(_falsifier_is_counterexample(graph, endpoint))
            checks.append(
                _falsifier_is_not_near_implied(graph, endpoint, neighborhood)
            )
        elif mode == "bridge":
            checks.append(_bridge_parents_have_contact(graph, sinks))

        already = {check.name for check in checks}
        for name in card.get("proposed_checks") or ():
            if name in already or name not in PROPOSABLE_CHECKS:
                continue
            extra = _check_by_name(
                str(name), graph, endpoint, sinks, neighborhood, nodes, edges
            )
            if extra is not None:
                already.add(extra.name)
                checks.append(replace(extra, proposed=True))
    probed = any(check.discriminating for check in checks)
    killed = any(not check.passed for check in checks)
    return ProbeResult(
        card_id=str(card.get("id") or ""),
        mode=mode,
        probed=probed,
        killed=killed,
        checks=tuple(checks),
    )


def probe_cards(
    graph: GraphSnapshot,
    cards: Sequence[dict[str, Any]],
    neighborhood: set[str],
) -> tuple[ProbeResult, ...]:
    return tuple(probe_card(graph, card, neighborhood) for card in cards)
