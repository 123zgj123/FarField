"""The same-anchor control stake: one measure, two node sources.

The fair-two-arm discipline applies to the stake itself: both arms go
through `gate_survival` against the same anchor, and the only difference
is where the far node came from. Nothing here may hand the jump arm a
win by construction.
"""

from __future__ import annotations

import unittest

from farfield.graph import GraphSnapshot
from farfield.extras.baseline import (
    anchor_alienness,
    control_stake,
    gate_survival,
    missing_stake,
)
from farfield.extras.embed import embed_nodes
from farfield.extras.generate import ConceptOracle


def cluster_nodes() -> dict[str, dict[str, object]]:
    nodes: dict[str, dict[str, object]] = {}
    for i in range(10):
        nodes[f"attn:{i}"] = {
            "id": f"attn:{i}",
            "title": f"sparse attention transformer variant{i}",
        }
    for i in range(10):
        nodes[f"prot:{i}"] = {
            "id": f"prot:{i}",
            "title": f"protein folding energy mutant{i}",
        }
    return nodes


def toy_graph(edges: tuple[tuple[str, str], ...] = ()) -> GraphSnapshot:
    nodes = {nid: {"id": nid, "title": str(row["title"])} for nid, row in cluster_nodes().items()}
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)
    return GraphSnapshot(
        snapshot_id="toy-baseline",
        k=2,
        nodes=nodes,
        edges=tuple(edges),
        outgoing={k: tuple(v) for k, v in outgoing.items()},
        incoming={k: tuple(v) for k, v in incoming.items()},
        counterexample_nodes=frozenset(),
    )


class ControlStakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.space = embed_nodes(cluster_nodes())
        cls.near_ids = ("attn:0", "attn:1", "attn:2")

    def test_an_empty_arm_reports_no_stake(self) -> None:
        oracle = ConceptOracle.from_graph(toy_graph())
        self.assertIsNone(
            control_stake(self.space, oracle, self.near_ids, [])
        )

    def test_a_literature_arm_is_not_rewritten_into_a_graph_cousin(self) -> None:
        payload = missing_stake(anchor="attn:0", arm_n=2, reason="literature_landing")
        self.assertEqual(payload["skipped"], "literature_landing")
        self.assertEqual(payload["arm"]["n"], 2)
        self.assertEqual(payload["control"]["n"], 0)
        self.assertIn("Not a verdict", payload["note"])

    def test_the_stake_is_deterministic_under_the_same_seed(self) -> None:
        oracle = ConceptOracle.from_graph(toy_graph())
        arm = ["prot:3", "prot:7"]
        first = control_stake(
            self.space, oracle, self.near_ids, arm, seed=41
        )
        second = control_stake(
            self.space, oracle, self.near_ids, arm, seed=41
        )
        self.assertEqual(first, second)

    def test_both_arms_go_through_the_same_measure(self) -> None:
        # On an edgeless graph every pair passes every gate, so neither
        # arm can score anything but 1.0. A stake that hard-coded a win
        # for the jump arm would separate here; separation would be the
        # stake's own bug.
        oracle = ConceptOracle.from_graph(toy_graph())
        stake = control_stake(
            self.space, oracle, self.near_ids, ["prot:3", "prot:7"], seed=7
        )
        self.assertIsNotNone(stake)
        self.assertEqual(stake["arm"]["rate"], 1.0)
        self.assertEqual(stake["control"]["rate"], 1.0)

    def test_a_bad_jump_arm_is_allowed_to_lose(self) -> None:
        # The anchor is already combined with the arm's far node, so the
        # arm fails the same gate any control would fail. The stake must
        # report the loss, not repair it.
        oracle = ConceptOracle.from_graph(toy_graph((("attn:0", "prot:3"),)))
        stake = control_stake(
            self.space, oracle, self.near_ids, ["prot:3"], seed=7
        )
        self.assertIsNotNone(stake)
        self.assertEqual(stake["arm"]["passed"], 0)
        self.assertGreater(stake["control"]["rate"], stake["arm"]["rate"])

    def test_the_arm_count_matches_an_independent_recount(self) -> None:
        oracle = ConceptOracle.from_graph(toy_graph((("attn:0", "prot:3"),)))
        arm = ["prot:3", "prot:7"]
        stake = control_stake(self.space, oracle, self.near_ids, arm, seed=7)
        recount = sum(gate_survival(oracle, "attn:0", node) for node in arm)
        self.assertEqual(stake["arm"]["passed"], recount)

    def test_alienness_is_one_ruler_for_both_arms(self) -> None:
        near = anchor_alienness(self.space, self.near_ids, "attn:5")
        far = anchor_alienness(self.space, self.near_ids, "prot:5")
        self.assertLess(near, far)

    def test_the_stake_says_what_it_is_not(self) -> None:
        oracle = ConceptOracle.from_graph(toy_graph())
        stake = control_stake(
            self.space, oracle, self.near_ids, ["prot:3"], seed=7
        )
        self.assertIn("Not a verdict", stake["note"])


if __name__ == "__main__":
    unittest.main()
