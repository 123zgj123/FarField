"""The two checks a model can add, and how a proposal's effect is measured.

`PROPOSABLE_CHECKS` used to be exactly the union of what each mode runs by
default. A model could still add a check by naming another mode's, and in the
recorded live run one did -- but a check chosen for a different card shape fails
for reasons unrelated to the card in front of it, and the cards it lands on are
usually already dead. So the two checks added here belong to no mode's defaults
and are about the card at hand.

That turns the question into a measurement rather than a count: not how many
checks a proposal added, but whether a proposed check is the reason a card died,
which is `proposal_was_decisive`.
"""

from __future__ import annotations

import unittest

from farfield.graph import GraphSnapshot
from farfield.probe import (
    DEFAULT_CHECKS_BY_MODE,
    HUB_QUANTILE,
    PROPOSABLE_CHECKS,
    _check_by_name,
    _in_degree_ceiling,
    probe_card,
    proposable_for,
)

# The checks each mode runs on its own, which is what a proposal has to be able
# to go beyond. Kept here as a literal rather than imported so that widening a
# mode's defaults without widening the proposable set fails this file.
MODE_DEFAULTS = frozenset(
    {
        "mechanism_carries_through",
        "anomaly_is_a_true_leaf",
        "falsifier_endpoint_is_counterexample",
        "falsifier_not_implied_by_neighborhood",
        "bridge_parents_have_contact",
    }
)


def snapshot(edges: list[tuple[str, str]], counterexamples: tuple[str, ...] = ()) -> GraphSnapshot:
    nodes: dict[str, dict[str, object]] = {}
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for source, destination in edges:
        for node in (source, destination):
            nodes.setdefault(node, {"id": node})
        outgoing.setdefault(source, []).append(destination)
        incoming.setdefault(destination, []).append(source)
    return GraphSnapshot(
        snapshot_id="test-snapshot",
        k=2,
        nodes=nodes,
        edges=tuple(edges),
        outgoing={key: tuple(value) for key, value in outgoing.items()},
        incoming={key: tuple(value) for key, value in incoming.items()},
        counterexample_nodes=frozenset(counterexamples),
    )


def run(name: str, graph: GraphSnapshot, nodes: list[str], edges: list[tuple[str, str]]):
    endpoint = nodes[-1]
    return _check_by_name(name, graph, endpoint, tuple(nodes[-1:]), set(), nodes, edges)


class WhatIsOfferedToAModelTests(unittest.TestCase):
    def test_the_mode_default_table_matches_what_probe_card_runs(self) -> None:
        # `proposable_for` subtracts this table from the proposable set. If the
        # table drifted from the branches in probe_card, a model would be offered
        # the check its card is already facing and the offer would look widened
        # while changing nothing.
        graph = snapshot(
            [("seed", "mid"), ("mid", "end"), ("other", "end"), ("end", "onward")],
            counterexamples=("end",),
        )
        for mode, expected in DEFAULT_CHECKS_BY_MODE.items():
            with self.subTest(mode=mode):
                card = {
                    "id": f"conj_{mode}",
                    "mode": mode,
                    "nodes": ["seed", "mid", "end"],
                    "edges": [["seed", "mid"], ["mid", "end"]],
                }
                ran = {
                    check.name
                    for check in probe_card(graph, card, {"seed"}).checks
                    if check.name != "edges_exist_in_snapshot"
                }
                self.assertEqual(ran, set(expected))

    def test_a_card_is_never_offered_the_check_it_already_faces(self) -> None:
        for mode, defaults in DEFAULT_CHECKS_BY_MODE.items():
            with self.subTest(mode=mode):
                offered = set(proposable_for(mode))
                self.assertFalse(offered & set(defaults))
                self.assertTrue(offered, f"{mode} has nothing left to be asked about")

    def test_an_unknown_mode_is_offered_everything(self) -> None:
        # Better to ask about too much than to silently offer nothing at all.
        self.assertEqual(set(proposable_for("no_such_mode")), set(PROPOSABLE_CHECKS))


class WhyThereAreProposableChecksAtAllTests(unittest.TestCase):
    def test_a_model_can_name_a_check_no_mode_runs_by_itself(self) -> None:
        beyond = set(PROPOSABLE_CHECKS) - MODE_DEFAULTS
        self.assertTrue(
            beyond,
            "every proposable check is also a mode default, so a proposal can"
            " never add one and added_checks is zero whatever the model says",
        )

    def test_every_proposable_check_is_actually_runnable(self) -> None:
        # A name in the list that no dispatcher handles would be accepted from
        # the model and then silently drop out, which reads as a passed check.
        graph = snapshot([("a", "b"), ("b", "c"), ("x", "c")])
        for name in PROPOSABLE_CHECKS:
            with self.subTest(name=name):
                check = _check_by_name(
                    name, graph, "c", ("b", "c"), {"a"}, ["a", "b", "c"], [("a", "b"), ("b", "c")]
                )
                self.assertIsNotNone(check)
                self.assertEqual(check.name, name)


class CitationHubTests(unittest.TestCase):
    def setUp(self) -> None:
        # One node cited by ten others, the rest cited once, so the ceiling sits
        # well below the hub.
        edges = [(f"p{index}", "hub") for index in range(10)]
        edges += [(f"p{index}", f"q{index}") for index in range(10)]
        self.graph = snapshot(edges)

    def test_a_much_cited_endpoint_fails(self) -> None:
        check = run("endpoint_is_not_a_citation_hub", self.graph, ["p0", "hub"], [("p0", "hub")])
        self.assertFalse(check.passed)
        self.assertTrue(check.discriminating)
        self.assertIn("exceeds", check.detail)

    def test_an_ordinarily_cited_endpoint_passes(self) -> None:
        check = run("endpoint_is_not_a_citation_hub", self.graph, ["p3", "q3"], [("p3", "q3")])
        self.assertTrue(check.passed)

    def test_the_ceiling_is_a_quantile_so_the_check_cannot_fail_everywhere(self) -> None:
        # A check that every card fails is as uninformative as one every card
        # passes. Reading the ceiling off this snapshot bounds the failures by
        # construction, which is the property that rules that out.
        ceiling = _in_degree_ceiling(self.graph)
        above = [
            node
            for node in self.graph.nodes
            if len(self.graph.incoming.get(node, ())) > ceiling
        ]
        self.assertLessEqual(len(above), (1.0 - HUB_QUANTILE) * len(self.graph.nodes) + 1)
        self.assertTrue(above, "a snapshot with a hub should have something above the ceiling")

    def test_one_citation_is_never_a_hub_however_sparse_the_snapshot(self) -> None:
        # Here the quantile ceiling is zero, because most nodes are uncited, and
        # the bare quantile would call a node with one route in a hub.
        graph = snapshot([("only", "target")])
        self.assertEqual(_in_degree_ceiling(graph), 0)
        check = run("endpoint_is_not_a_citation_hub", graph, ["only", "target"], [("only", "target")])
        self.assertTrue(check.passed)


class ShortcutTests(unittest.TestCase):
    def test_an_earlier_node_citing_the_endpoint_fails(self) -> None:
        graph = snapshot([("a", "b"), ("b", "c"), ("a", "c")])
        check = run(
            "path_has_no_shortcut", graph, ["a", "b", "c"], [("a", "b"), ("b", "c")]
        )
        self.assertFalse(check.passed)
        self.assertIn("a", check.detail)
        self.assertIn("overstated", check.detail)

    def test_a_path_with_no_shortcut_passes(self) -> None:
        graph = snapshot([("a", "b"), ("b", "c")])
        check = run(
            "path_has_no_shortcut", graph, ["a", "b", "c"], [("a", "b"), ("b", "c")]
        )
        self.assertTrue(check.passed)

    def test_the_endpoints_own_predecessor_is_not_a_shortcut(self) -> None:
        # b cites c because b is the step before c. Counting that as a shortcut
        # would fail every path there is, which is the always-false trap.
        graph = snapshot([("a", "b"), ("b", "c")])
        check = run(
            "path_has_no_shortcut", graph, ["a", "b", "c"], [("a", "b"), ("b", "c")]
        )
        self.assertTrue(check.passed)

    def test_a_shortcut_that_is_not_in_the_snapshot_does_not_count(self) -> None:
        # The check reads the graph, not the card. An edge the card happens to
        # list is not evidence the corpus has it.
        graph = snapshot([("a", "b"), ("b", "c")])
        check = run(
            "path_has_no_shortcut",
            graph,
            ["a", "b", "c"],
            [("a", "b"), ("b", "c")],
        )
        self.assertTrue(check.passed)


class ProposalReachesTheVerdictTests(unittest.TestCase):
    def test_naming_a_widened_check_adds_one_and_can_kill(self) -> None:
        edges = [(f"p{index}", "hub") for index in range(10)]
        edges += [("p0", "mid"), ("mid", "hub")]
        graph = snapshot(edges)
        card = {
            "id": "conj_hub",
            "mode": "mechanism",
            "nodes": ["p0", "mid", "hub"],
            "edges": [["p0", "mid"], ["mid", "hub"]],
        }
        hood = {"p0"}
        before = probe_card(graph, card, hood)
        after = probe_card(graph, dict(card, proposed_checks=["endpoint_is_not_a_citation_hub"]), hood)
        self.assertEqual(len(after.checks), len(before.checks) + 1)
        self.assertTrue(after.killed)
        self.assertIn(
            "endpoint_is_not_a_citation_hub", {check.name for check in after.checks}
        )

    def test_a_proposal_gets_no_credit_for_a_card_that_was_already_dead(self) -> None:
        # The bridge card in the recorded live run went this way: the model named
        # another mode's check, it was added, it failed -- and the card was
        # already dead from bridge_parents_have_contact. Counting added checks
        # calls that a contribution; the counterfactual does not.
        edges = [("p0", "left"), ("p0", "right")]
        graph = snapshot(edges)
        card = {
            "id": "conj_bridge",
            "mode": "bridge",
            "nodes": ["p0", "left", "right"],
            "edges": [["p0", "left"], ["p0", "right"]],
            "proposed_checks": ["mechanism_carries_through"],
        }
        result = probe_card(graph, card, {"p0"})
        self.assertTrue(result.killed)
        self.assertTrue(result.killed_without_proposals)
        self.assertFalse(result.proposal_was_decisive)
        self.assertFalse(result.to_dict()["proposal_was_decisive"])

    def test_a_proposal_that_alone_kills_a_card_is_recorded_as_decisive(self) -> None:
        edges = [(f"p{index}", "hub") for index in range(10)]
        edges += [("p0", "mid"), ("mid", "hub"), ("hub", "onward")]
        graph = snapshot(edges)
        card = {
            "id": "conj_hub_only",
            "mode": "mechanism",
            "nodes": ["p0", "mid", "hub"],
            "edges": [["p0", "mid"], ["mid", "hub"]],
        }
        hood = {"p0"}
        # The kernel's own check passes here, because hub cites onward, which is
        # outside the hood. So the card survives without a proposal.
        self.assertFalse(probe_card(graph, card, hood).killed)
        result = probe_card(
            graph, dict(card, proposed_checks=["endpoint_is_not_a_citation_hub"]), hood
        )
        self.assertTrue(result.killed)
        self.assertFalse(result.killed_without_proposals)
        self.assertTrue(result.proposal_was_decisive)

    def test_only_the_proposed_check_carries_the_proposed_flag(self) -> None:
        graph = snapshot([("p0", "mid"), ("mid", "out")])
        card = {
            "id": "conj_flag",
            "mode": "mechanism",
            "nodes": ["p0", "mid"],
            "edges": [["p0", "mid"]],
            "proposed_checks": ["endpoint_is_not_a_citation_hub"],
        }
        flags = {
            check.name: check.proposed
            for check in probe_card(graph, card, {"p0"}).checks
        }
        self.assertTrue(flags["endpoint_is_not_a_citation_hub"])
        self.assertFalse(flags["mechanism_carries_through"])
        self.assertFalse(flags["edges_exist_in_snapshot"])

    def test_a_widened_check_still_cannot_save_a_card(self) -> None:
        # mid has no edge leaving the hood, so the mechanism check kills it. A
        # proposal that passes must not undo that.
        graph = snapshot([("p0", "mid")])
        card = {
            "id": "conj_dead",
            "mode": "mechanism",
            "nodes": ["p0", "mid"],
            "edges": [["p0", "mid"]],
        }
        hood = {"p0", "mid"}
        before = probe_card(graph, card, hood)
        self.assertTrue(before.killed)
        after = probe_card(graph, dict(card, proposed_checks=["endpoint_is_not_a_citation_hub"]), hood)
        self.assertTrue(after.killed)


if __name__ == "__main__":
    unittest.main()
