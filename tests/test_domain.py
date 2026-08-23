"""Object-domain lock: generation stays on-topic; writing cannot change fields."""

from __future__ import annotations

import unittest

from farfield.extras.domain import (
    claim_covers_topic,
    experiment_stays_on_object,
    measure_stays_on_object,
    topic_payload_terms,
    writeup_retrofits_topic,
)


class PayloadTests(unittest.TestCase):
    def test_payload_drops_the_seed_label(self) -> None:
        payload = topic_payload_terms(
            "formal verification of safety properties for LLM agents",
            "formal verification",
        )
        self.assertIn("agent", payload)
        self.assertIn("safety", payload)
        self.assertNotIn("formal", payload)

    def test_a_claim_that_stays_on_topic_covers_it(self) -> None:
        self.assertTrue(
            claim_covers_topic(
                "safety properties of LLM agents can be model-checked",
                "the tool-use trace is a finite automaton",
                "formal verification of safety properties for LLM agents",
                "formal verification",
            )
        )

    def test_a_graph_hardness_claim_does_not_cover_agent_safety(self) -> None:
        self.assertFalse(
            claim_covers_topic(
                "tight hardness of fine-grained reductions for graph alignment",
                "the certificate is a cut in the alignment graph",
                "formal verification of safety properties for LLM agents",
                "formal verification",
            )
        )


class WriteupTests(unittest.TestCase):
    def test_retitling_hardness_as_tool_use_is_a_retrofit(self) -> None:
        found = writeup_retrofits_topic(
            "Certified Replay for LLM Tool-Use Safety",
            "Verify tool-calling agents with a replay certificate",
            home=(
                "tight hardness of fine-grained reductions for graph alignment"
                " formal verification"
            ),
            topic="formal verification of safety properties for LLM agents and tool-using autonomous agents",
            seed_label="formal verification",
        )
        self.assertGreaterEqual(len(found), 2)
        self.assertTrue({"llm", "tool", "agent", "safety"} & set(found))

    def test_a_title_inside_the_claim_field_is_allowed(self) -> None:
        found = writeup_retrofits_topic(
            "Pivot-history compression for the simplex method",
            "Build a succinct pivot log",
            home="a succinct structure stores simplex pivot history in linear space",
            topic="compress simplex traces",
            seed_label="succinct data structure",
        )
        self.assertEqual(found, ())


class MeasureObjectTests(unittest.TestCase):
    def test_counting_mincut_ops_does_not_test_an_auth_required_claim(self) -> None:
        self.assertFalse(
            measure_stays_on_object(
                "cut-maintenance edge operations",
                "an initial Agent Card that omits AUTH_REQUIRED never inserts the state",
                mechanism="incremental min-cut on the task-state graph",
                topic="A2A protocol security Agent Card JSON-RPC authentication",
            )
        )
        self.assertTrue(
            measure_stays_on_object(
                "AUTH_REQUIRED transitions after Agent Card mutation",
                "an initial Agent Card that omits AUTH_REQUIRED never inserts the state",
                mechanism="incremental min-cut on the task-state graph",
                topic="A2A protocol security Agent Card JSON-RPC authentication",
            )
        )

    def test_empty_claim_is_a_noop_so_old_probes_keep_their_bytes(self) -> None:
        self.assertTrue(measure_stays_on_object("edge hops", "", mechanism="splay"))

    def test_a_diagnosis_that_only_counts_the_far_mechanism_is_refused(self) -> None:
        self.assertFalse(
            experiment_stays_on_object(
                "count min-cut edge updates on the TaskState graph",
                "incremental min-cut on, same graph",
                "Agent Card AUTH_REQUIRED must appear before task execution",
                mechanism="incremental min-cut",
                topic="A2A Agent Card authentication",
            )
        )


if __name__ == "__main__":
    unittest.main()
