"""Object-domain lock: generation stays on-topic; writing cannot change fields."""

from __future__ import annotations

import unittest

from farfield.extras.domain import (
    claim_covers_topic,
    experiment_stays_on_object,
    feed_query_concepts,
    label_covers_topic,
    measure_stays_on_object,
    graph_supplies_mechanisms,
    preferred_object_labels,
    surface_object_labels,
    topic_object_phrases,
    topic_payload_terms,
    without_far_terms,
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

    def test_sharing_metric_is_not_coding_agent_coverage(self) -> None:
        self.assertFalse(
            claim_covers_topic(
                "block z-score distance under a finite metric",
                "uniform metric on token blocks",
                "Coding-agent evaluation: a host-stdlib metric on attested tool-call fields",
                "finite metric",
            )
        )

    def test_two_topic_unigrams_are_not_coverage_when_a_bigram_exists(self) -> None:
        topic = "process supervision of chain-of-thought reasoning traces"
        self.assertFalse(
            claim_covers_topic(
                "reasoning models need process scores",
                "a wavelet tree stores the rank dictionary",
                topic,
                "wavelet tree",
            )
        )
        self.assertTrue(
            claim_covers_topic(
                "chain-of-thought reasoning traces need process supervision",
                "a verifier scores the same traces",
                topic,
                "process supervision",
            )
        )
        self.assertFalse(
            claim_covers_topic(
                "reasoning models need better traces",
                "a wavelet tree stores the rank dictionary",
                topic,
                "wavelet tree",
            )
        )

    def test_surface_labels_fall_back_to_topic_phrases(self) -> None:
        labels = surface_object_labels(
            ("finite metric", "balanced tree"),
            "Coding-agent evaluation on tool-call traces",
        )
        self.assertTrue(any("coding" in item or "tool-call" in item or "evaluation" in item for item in labels))
        self.assertNotIn("finite metric", labels)

    def test_cosine_neighbours_do_not_license_graph_mechanisms(self) -> None:
        self.assertFalse(
            graph_supplies_mechanisms(
                ("finite metric", "balanced tree"),
                "code world models of executable program state",
            )
        )
        self.assertTrue(
            graph_supplies_mechanisms(
                ("succinct data structure", "wavelet tree"),
                "compress genomic sequence collections with succinct data structures",
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


class TopicObjectTests(unittest.TestCase):
    def test_phrases_prefer_bigrams_from_the_named_topic(self) -> None:
        phrases = topic_object_phrases(
            "formal verification of concurrent systems"
        )
        self.assertIn("formal verification", phrases)
        self.assertTrue(all(len(item) >= 5 or " " in item for item in phrases))

    def test_preferred_labels_keep_topic_overlap_only(self) -> None:
        kept = preferred_object_labels(
            (
                "formal verification",
                "golden ratio",
                "concurrent systems",
            ),
            "formal verification of concurrent systems",
        )
        self.assertEqual(
            kept, ("formal verification", "concurrent systems")
        )

    def test_literature_queries_topic_bigrams_not_the_nearest_neighbour(self) -> None:
        query = feed_query_concepts(
            "formal verification of concurrent systems",
            ("formal verification",),
            ("golden ratio", "compression ratio"),
        )
        self.assertIn("formal verification", query)
        self.assertNotIn("golden ratio", query)

    def test_definite_program_does_not_cover_a_code_world_topic(self) -> None:
        topic = (
            "code world models of executable program state as the world "
            "of an agent harness"
        )
        self.assertFalse(label_covers_topic("definite program", topic))
        self.assertFalse(label_covers_topic("feedback edge set", topic))
        self.assertEqual(
            preferred_object_labels(
                ("definite program", "feedback edge set", "self concordant"),
                topic,
            ),
            (),
        )
        surface = surface_object_labels(
            ("definite program", "feedback edge set"),
            topic,
        )
        self.assertTrue(surface)
        self.assertNotIn("definite program", surface)
        self.assertNotIn("feedback edge set", surface)

    def test_far_graph_terms_are_stripped_before_schema_inference(self) -> None:
        stripped = without_far_terms(
            "intercept the write-read cycle instead of a feedback edge set",
            "feedback edge set",
            "code world models of executable program state",
        )
        lowered = stripped.lower()
        self.assertNotIn("feedback", lowered)
        self.assertNotIn("edge", lowered)
        kept = without_far_terms(
            "executable program state with a validator",
            "definite program",
            "code world models of executable program state",
        )
        self.assertIn("program", kept.lower())


if __name__ == "__main__":
    unittest.main()
