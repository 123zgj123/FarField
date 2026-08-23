"""P1 of the v2 design: LLM card generation and the concept-pair oracle.

Two boundaries under test. The schema boundary: the model's card is either
verbatim-executable or refused with a record — nothing is fuzzily repaired,
because the refusal rate is a P1 exit criterion and a silent repair would
fake it. The oracle boundary: all three checks can fail and can pass on the
same graph, and the "already combined" check agrees with the novelty
anchor's definition of novel, so the killing floor and the value anchor
cannot drift apart.
"""

from __future__ import annotations

import json
import unittest

from farfield.extras.generate import (
    CHECK_CATALOGUE,
    ConceptOracle,
    GeneratedCard,
    GenerationRefused,
    check_pair,
    generate_card,
    probe_generated,
    refine_card,
)
from farfield.extras.llm import Completion
from farfield.graph import GraphSnapshot
from farfield.novelty import CONCEPT_PAIRS


class FakeClient:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        system: str | None = None,
        logprobs: bool = False,
    ):
        self.prompts.append(prompt)
        text = self.answers.pop(0) if self.answers else ""
        return Completion(
            text=text,
            model="fake-model",
            request_digest="req",
            digest="a1b2c3d4e5f6a7b8",
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=15,
            mode="replay",
        )


LABELS = {
    "sparse attention": "concept:sparse-attention",
    "kv cache": "concept:kv-cache",
    "protein folding": "concept:protein-folding",
    "energy landscape": "concept:energy-landscape",
}


def answer(**overrides: object) -> str:
    payload: dict[str, object] = {
        "claim": "sparse attention structure can prune folding search",
        "mechanism": "both are sparse search problems over combinatorial spaces",
        "pair": ["sparse attention", "protein folding"],
        "falsifier": "pair_not_already_combined",
        "prediction": "a post-2017 work carries both labels",
        "dead_end": "encode the folding path as a dense matrix and multiply at every residue",
        "why_failed": "dense multiply costs quadratic work so the claimed prune never appears",
        "reframe": "treat folding search as a sparse combinatorial walk attention can skip",
        "objection": "protein energy landscapes are not sparse the way token graphs are",
    }
    payload.update(overrides)
    return json.dumps(payload)


def call(client: FakeClient) -> GeneratedCard:
    return generate_card(
        client,
        seed_label="sparse attention",
        near_labels=("kv cache",),
        far_labels=("protein folding", "energy landscape"),
        operator="directional",
        alienness=0.61,
        label_to_node=LABELS,
    )


class SchemaBoundaryTests(unittest.TestCase):
    def test_a_verbatim_card_is_built_and_mapped_to_graph_nodes(self) -> None:
        card = call(FakeClient(answer()))
        self.assertEqual(
            card.pair_nodes, ("concept:sparse-attention", "concept:protein-folding")
        )
        self.assertEqual(card.card_id, "gen_a1b2c3d4e5f6")
        self.assertEqual(card.falsifier, "pair_not_already_combined")

    def test_markdown_fences_around_the_json_are_tolerated(self) -> None:
        card = call(FakeClient("```json\n" + answer() + "\n```"))
        self.assertEqual(card.pair, ("sparse attention", "protein folding"))

    def test_prose_that_is_not_json_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused) as caught:
            call(FakeClient("Here are my thoughts on the matter."))
        self.assertEqual(
            caught.exception.record.missing_capability, "model_card_schema"
        )

    def test_a_paraphrased_concept_is_refused_not_fuzzily_matched(self) -> None:
        """`sparse attentions` is close, and close is exactly what verbatim
        copying exists to refuse: fuzzy matching would make the executable
        pair the parser's choice rather than the model's."""
        bad = answer(pair=["sparse attention", "the folding of proteins"])
        with self.assertRaises(GenerationRefused):
            call(FakeClient(bad))

    def test_both_pair_members_from_one_side_is_refused(self) -> None:
        bad = answer(pair=["protein folding", "energy landscape"])
        with self.assertRaises(GenerationRefused):
            call(FakeClient(bad))

    def test_an_empty_claim_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            call(FakeClient(answer(claim="  ")))

    def test_a_missing_dead_end_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            call(FakeClient(answer(dead_end="too short")))

    def test_a_dead_end_that_restates_the_claim_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            call(
                FakeClient(
                    answer(
                        dead_end=(
                            "sparse attention structure can prune folding"
                            " search in combinatorial spaces"
                        )
                    )
                )
            )

    def test_a_reframe_that_never_lands_in_the_card_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            call(
                FakeClient(
                    answer(
                        reframe=(
                            "replace the imitation game with a wholly"
                            " unrelated thermodynamic ensemble"
                        )
                    )
                )
            )

    def test_a_surviving_card_keeps_the_struggle_off_the_evidence_status(self) -> None:
        card = call(FakeClient(answer()))
        self.assertIn("dead_end", card.to_dict())
        self.assertIn("not evidence", card.to_dict()["status"])
        self.assertTrue(card.dead_end)

    def test_a_falsifier_outside_the_catalogue_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused) as caught:
            call(FakeClient(answer(falsifier="check_the_vibes")))
        self.assertEqual(
            caught.exception.record.missing_capability,
            "model_named_an_executable_check",
        )

    def test_failure_capsules_reach_the_prompt_as_negative_examples(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            capsules=[
                {"context": "attention x weather", "mechanism": "already combined"}
            ],
        )
        self.assertIn("attention x weather", client.prompts[0])
        self.assertIn("do not repeat", client.prompts[0])

    def test_review_criteria_reach_the_prompt_as_standards(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            criteria=(
                "prediction_names_a_measurable_quantity: a prediction with"
                " no quantity cannot be wrong",
            ),
        )
        self.assertIn("Installed reviewers", client.prompts[0])
        self.assertIn("prediction_names_a_measurable_quantity", client.prompts[0])

    def test_research_state_reaches_the_prompt_as_known_ground(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            knowledge=(
                "Established in earlier missions (corroborated): a succinct"
                " index stores pivot history in linear space",
            ),
        )
        self.assertIn("Research state", client.prompts[0])
        self.assertIn("Established in earlier missions", client.prompts[0])
        self.assertNotIn("speculative", client.prompts[0])

    def test_live_context_reaches_the_prompt_as_grounding(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            context=("Faster sparse kernels for long contexts (2026-08-12)",),
        )
        self.assertIn("newest papers", client.prompts[0])
        self.assertIn("Faster sparse kernels", client.prompts[0])
        self.assertIn(
            "for grounding only", client.prompts[0],
            "live context must be marked advisory, never evidence",
        )

    def test_no_criteria_leaves_the_prompt_byte_identical(self) -> None:
        """Old cached prompts must replay unchanged: the criteria slot is
        invisible when empty, so pre-feedback-loop artifacts keep their
        digests."""
        with_slot = FakeClient(answer())
        generate_card(
            with_slot,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
        )
        self.assertNotIn("Installed reviewers", with_slot.prompts[0])
        self.assertIn(
            "(operator: analogy): protein folding\n\nPropose ONE",
            with_slot.prompts[0],
        )

    def test_viable_pairings_reach_the_prompt_and_bite_on_violation(self) -> None:
        # A live mission showed the model habitually pairs with the seed
        # even when only another near concept survives the graph gates with
        # its chosen far concept — so the viable pairings are both shown
        # and enforced.
        client = FakeClient(answer())
        card = generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
            viable_pairings={"protein folding": ("sparse attention",)},
        )
        self.assertIn("may pair with", client.prompts[0])
        self.assertEqual(card.pair, ("sparse attention", "protein folding"))
        with self.assertRaises(GenerationRefused) as ctx:
            generate_card(
                FakeClient(answer()),
                seed_label="sparse attention",
                near_labels=("kv cache",),
                far_labels=("protein folding",),
                operator="analogy",
                alienness=0.5,
                label_to_node=LABELS,
                viable_pairings={"protein folding": ("kv cache",)},
            )
        self.assertIn("structurally viable", ctx.exception.record.unlock_condition)

    def test_a_jump_that_retrieved_nothing_is_refused_before_any_call(self) -> None:
        client = FakeClient(answer())
        with self.assertRaises(GenerationRefused):
            generate_card(
                client,
                seed_label="sparse attention",
                near_labels=(),
                far_labels=(),
                operator="interpolate",
                alienness=0.7,
                label_to_node=LABELS,
            )
        self.assertEqual(client.prompts, [])

    def test_a_claim_that_leaves_the_topic_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused) as caught:
            generate_card(
                FakeClient(answer()),
                seed_label="sparse attention",
                near_labels=("kv cache",),
                far_labels=("protein folding", "energy landscape"),
                operator="directional",
                alienness=0.61,
                label_to_node=LABELS,
                topic="formal verification of safety properties for LLM agents",
            )
        self.assertEqual(
            caught.exception.record.missing_capability, "model_claim_on_topic"
        )

    def test_omitting_topic_leaves_the_prompt_byte_identical(self) -> None:
        client = FakeClient(answer())
        generate_card(
            client,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            far_labels=("protein folding",),
            operator="analogy",
            alienness=0.5,
            label_to_node=LABELS,
        )
        self.assertTrue(client.prompts[0].startswith("A research seed"))
        self.assertNotIn("Researcher's topic:", client.prompts[0])


def concept_graph() -> GraphSnapshot:
    """A small graph with one hub, one linked pair, and quiet corners.

    hub connects to everything; a-b are linked; c and d are unlinked and
    share only the hub, so their Jaccard stays under any sane ceiling.
    """
    nodes = {}
    edges = []
    names = ["hub", "a", "b", "c", "d", "e", "f", "g"]
    for name in names:
        nodes[name] = {"id": name, "title": name}
    for name in names[1:]:
        edges.append(("hub", name))
    edges.append(("a", "b"))
    edges.append(("e", "f"))
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = {}
    for src, dst in edges:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)
    return GraphSnapshot(
        snapshot_id="toy-concepts",
        k=2,
        nodes=nodes,
        edges=tuple(edges),
        outgoing={k: tuple(v) for k, v in outgoing.items()},
        incoming={k: tuple(v) for k, v in incoming.items()},
        counterexample_nodes=frozenset(),
    )


class OracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = concept_graph()
        self.oracle = ConceptOracle.from_graph(self.graph)

    def test_the_oracle_is_deterministic(self) -> None:
        again = ConceptOracle.from_graph(self.graph)
        self.assertEqual(self.oracle.degree_ceiling, again.degree_ceiling)
        self.assertEqual(
            self.oracle.implication_ceiling, again.implication_ceiling
        )

    def test_an_already_combined_pair_is_killed(self) -> None:
        checks = {c.name: c for c in check_pair(self.oracle, "a", "b")}
        self.assertFalse(checks["pair_not_already_combined"].passed)

    def test_a_hub_pairing_is_killed(self) -> None:
        checks = {c.name: c for c in check_pair(self.oracle, "hub", "c")}
        self.assertFalse(checks["endpoint_is_not_a_concept_hub"].passed)

    def test_a_quiet_unlinked_pair_survives_every_check(self) -> None:
        checks = check_pair(self.oracle, "c", "d")
        self.assertTrue(all(c.passed for c in checks))

    def test_already_combined_agrees_with_the_novelty_anchor(self) -> None:
        """One definition of novel, held in two places, checked equal here:
        if the killing floor and the value anchor ever disagree about what
        counts as a new combination, P2's readings become uninterpretable."""
        for u in self.graph.nodes:
            for v in self.graph.nodes:
                if u >= v:
                    continue
                check = {
                    c.name: c for c in check_pair(self.oracle, u, v)
                }["pair_not_already_combined"]
                self.assertEqual(
                    check.passed,
                    CONCEPT_PAIRS.is_novel(self.graph, u, v),
                    (u, v),
                )

    def test_the_verdict_records_where_the_author_expected_to_die(self) -> None:
        card = call(FakeClient(answer(pair=["sparse attention", "protein folding"])))
        oracle = ConceptOracle.from_graph(self.graph)
        # Remap onto toy nodes: a-b is linked, so the card dies there, and the
        # card pre-registered exactly that check.
        remapped = GeneratedCard(
            **{**card.__dict__, "pair_nodes": ("a", "b")}
        )
        verdict = probe_generated(oracle, [remapped])[0]
        self.assertTrue(verdict.killed)
        self.assertIn("pair_not_already_combined", verdict.killed_by)
        self.assertTrue(verdict.falsifier_was_right)

    def test_every_catalogue_check_is_run_on_every_card(self) -> None:
        card = call(FakeClient(answer()))
        remapped = GeneratedCard(**{**card.__dict__, "pair_nodes": ("c", "d")})
        verdict = probe_generated(self.oracle, [remapped])[0]
        self.assertEqual(
            tuple(c.name for c in verdict.checks), CHECK_CATALOGUE
        )
        self.assertFalse(verdict.killed)


class RefineSameIdeaTests(unittest.TestCase):
    def test_a_refine_keeps_the_pair_and_changes_the_prediction(self) -> None:
        parent = call(FakeClient(answer()))
        revised = refine_card(
            FakeClient(
                answer(
                    prediction="later work reports 12 ns queries at n = 10^6 on a public trace"
                )
            ),
            parent,
            seed_label="sparse attention",
            near_labels=("kv cache",),
            label_to_node=LABELS,
            killed_by=["prediction_names_a_measurable_quantity"],
        )
        self.assertEqual(revised.pair, parent.pair)
        self.assertNotEqual(revised.prediction, parent.prediction)

    def test_copying_the_previous_draft_is_refused(self) -> None:
        parent = call(FakeClient(answer()))
        with self.assertRaises(GenerationRefused):
            refine_card(
                FakeClient(answer()),
                parent,
                seed_label="sparse attention",
                near_labels=("kv cache",),
                label_to_node=LABELS,
                killed_by=["prediction_names_a_measurable_quantity"],
            )


if __name__ == "__main__":
    unittest.main()
