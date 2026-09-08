"""Grounded literature transitions and reusable structural research moves."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from farfield.extras import explore
from farfield.extras.livefeed import FreshWork


def paper(identifier: str, year: str, *, target: bool = False, references=()) -> dict:
    fields = {
        "problem": "Coordinate actions under delayed feedback.",
        "bottleneck_gap": "Delayed feedback is not used.",
        "assumptions": "Outcomes arrive after decisions.",
        "representation": "Each worker keeps a local state.",
        "mechanism": "Independent local workers select actions without delayed outcomes.",
        "evidence": "We compare action errors on recorded observations.",
    }
    if target:
        fields["mechanism"] = "A shared controller updates action choice using delayed outcomes."
    return {
        "work_id": identifier,
        "title": "Delayed feedback coordination" if target else "Independent local coordination",
        "published": year,
        "source": "verified_fixture",
        "references": list(references),
        "abstract": " ".join(fields.values()),
        "field_spans": {
            name: {"value": value, "quote": value, "section": "abstract"}
            for name, value in fields.items()
        },
    }


class ExtractionClient:
    def __init__(self, fields: dict) -> None:
        self.fields = fields

    def complete(self, prompt, *, purpose, system=None):
        return SimpleNamespace(text=json.dumps({"fields": self.fields}))


class PaperExtractionTests(unittest.TestCase):
    def test_abstract_alone_does_not_invent_a_mechanism(self) -> None:
        state = explore.extract_paper_state(
            FreshWork("A surprising title", "2020", work_id="paper:a", abstract="We report a study.")
        )
        self.assertEqual(state.mechanism, "unknown")
        self.assertEqual(state.problem, "unknown")
        self.assertEqual(state.cite_id, "paper:a")
        self.assertEqual(state.epistemic, "GENERATED")

    def test_explicit_fields_keep_exact_source_spans(self) -> None:
        raw = paper("paper:a", "2020")
        state = explore.extract_paper_state(raw)
        self.assertEqual(state.mechanism, raw["field_spans"]["mechanism"]["value"])
        span = state.extraction_spans["mechanism"]
        self.assertEqual(raw["abstract"][span["start"]:span["end"]], span["quote"])
        self.assertEqual(span["cite_id"], "paper:a")
        self.assertEqual(state.objective, "unknown")
        self.assertEqual(state.to_dict()["field_epistemics"]["mechanism"], "GENERATED")

    def test_model_cannot_invent_quotes_or_mark_interpretation_world(self) -> None:
        raw = paper("paper:a", "2020")
        raw.pop("field_spans")
        state = explore.extract_paper_state(raw, client=ExtractionClient({
            "mechanism": {
                "value": "Uses a feedback controller",
                "quote": "This quote is not in the supplied paper.",
                "section": "abstract",
                "epistemic": "WORLD",
            },
            "assumptions": {
                "value": "Feedback is delayed",
                "quote": "Outcomes arrive after decisions.",
                "section": "abstract",
                "epistemic": "WORLD",
            },
        }))
        self.assertEqual(state.mechanism, "unknown")
        self.assertEqual(state.assumptions, "Feedback is delayed")
        self.assertEqual(state.field_epistemics["assumptions"], "GENERATED")
        self.assertTrue(any("mechanism" in reason for reason in state.extraction_errors))

    def test_unverified_full_text_cannot_ground_fields(self) -> None:
        raw = paper("paper:a", "2020")
        raw["full_text_spans"] = [{"text": "Uses a verified instrument.", "verified": False}]
        raw["field_spans"]["method"] = {
            "value": "Uses a verified instrument.", "quote": "Uses a verified instrument.",
            "section": "full_text:0",
        }
        self.assertEqual(explore.extract_paper_state(raw).method, "unknown")
        raw["full_text_spans"][0]["verified"] = True
        self.assertEqual(explore.extract_paper_state(raw).method, "Uses a verified instrument.")

    def test_arbitrary_text_sources_do_not_bypass_full_text_verification(self) -> None:
        raw = paper("paper:a", "2020")
        raw["text_sources"] = {"full_text:0": "Unverified invented instrument."}
        raw["field_spans"]["method"] = {
            "value": "Unverified invented instrument.", "quote": "Unverified invented instrument.",
            "section": "full_text:0",
        }
        self.assertEqual(explore.extract_paper_state(raw).method, "unknown")


class TransitionTests(unittest.TestCase):
    def test_different_interpretations_of_identical_quotes_are_not_a_change(self) -> None:
        source = paper("paper:a", "2020")
        target = paper("paper:b", "2021", references=("paper:a",))
        target["field_spans"]["mechanism"]["value"] = "A totally new mechanism invented by interpretation"
        transition = explore.extract_transition(source, target)
        self.assertEqual(transition.status, "blocked")
        self.assertIn("no_grounded_conceptual_change", transition.reasons)

    def test_validated_operator_survives_serialization_and_retrieval(self) -> None:
        source = explore.extract_paper_state(paper("paper:a", "2020"))
        target = explore.extract_paper_state(paper("paper:b", "2021", target=True, references=("paper:a",)))
        transition = explore.extract_transition(source, target, client=ExtractionClient({
            "operator": {
                "value": "feedback_introduction", "quote": target.mechanism,
                "section": "target.abstract",
            },
        }))
        self.assertEqual(transition.operator, "feedback_introduction")
        matches = explore.retrieve_transitions(
            {"bottleneck_gap": "Delayed feedback is not used."}, [transition.to_dict()],
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].operator, "feedback_introduction")

    def test_citation_time_and_grounded_change_are_required(self) -> None:
        source = paper("paper:a", "2020")
        for target, reason in (
            (paper("paper:b", "2021", target=True), "missing_citation_lineage"),
            (paper("paper:b", "2019", target=True, references=("paper:a",)), "unproven_temporal_order"),
            (paper("paper:b", "2021", references=("paper:a",)), "no_grounded_conceptual_change"),
        ):
            with self.subTest(reason=reason):
                transition = explore.extract_transition(source, target)
                self.assertEqual(transition.status, "blocked")
                self.assertIn(reason, transition.reasons)
                self.assertEqual(explore.retrieve_transitions({}, [transition]), [])

    def test_grounded_transition_preserves_both_citations_and_changed_component(self) -> None:
        transition = explore.extract_transition(
            paper("paper:a", "2020"),
            paper("paper:b", "2021", target=True, references=("paper:a",)),
        )
        self.assertEqual(transition.status, "extracted")
        self.assertEqual(transition.changed_component, ("mechanism",))
        self.assertIn("assumptions", transition.preserved_component)
        self.assertEqual(transition.operator, "mechanism_replacement")
        self.assertEqual(transition.source_cite_id, "paper:a")
        self.assertEqual(transition.target_cite_id, "paper:b")
        self.assertEqual(transition.source_published, "2020")
        self.assertEqual(transition.target_published, "2021")
        self.assertEqual(transition.epistemic, "GENERATED")
        self.assertEqual(transition.why_progress_occurred, "unknown")
        self.assertIn("mechanism", transition.extraction_spans["target"])

    def test_same_operator_retrieves_for_structurally_similar_distinct_topics(self) -> None:
        transition = explore.extract_transition(
            paper("paper:a", "2020"),
            paper("paper:b", "2021", target=True, references=("paper:a",)),
        )
        for topic in ("forest irrigation", "distributed database repair"):
            with self.subTest(topic=topic):
                matches = explore.retrieve_transitions(
                    {"topic": topic, "bottleneck_gap": "Delayed feedback is not used."},
                    [transition],
                )
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].operator, "mechanism_replacement")
        self.assertEqual(explore.retrieve_transitions(
            {"topic": "Delayed feedback coordination", "bottleneck_gap": "Insufficient sample diversity."},
            [transition],
        ), [])
        self.assertEqual(explore.retrieve_transitions(
            {"bottleneck_gap": "Delayed feedback is not used."}, [transition], limit=0,
        ), [])

    def test_negated_applicability_condition_is_not_a_structural_match(self) -> None:
        transition = explore.extract_transition(
            paper("paper:a", "2020"),
            paper("paper:b", "2021", target=True, references=("paper:a",)),
        )
        self.assertEqual(explore.retrieve_transitions(
            {"assumptions": "Outcomes do not arrive after decisions."}, [transition],
        ), [])


class HistoricalCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transition = explore.extract_transition(
            paper("paper:a", "2020"),
            paper("paper:b", "2021", target=True, references=("paper:a",)),
        )

    def test_copied_target_mechanism_is_rejected_with_citation(self) -> None:
        reasons = explore.copy_check(
            {"claim": "A new research direction", "mechanism": self.transition.target_state.mechanism},
            self.transition,
        )
        self.assertTrue(any("historical_mechanism_copy" in reason and "paper:b" in reason for reason in reasons))

    def test_lightly_reworded_historical_target_is_rejected(self) -> None:
        reasons = explore.copy_check(
            {"claim": "Shared controller updates action choice using delayed outcomes."},
            [self.transition],
        )
        self.assertTrue(any("historical_target_similarity" in reason and "paper:b" in reason for reason in reasons))

    def test_transfer_with_distinct_mechanism_is_not_a_historical_copy(self) -> None:
        self.assertEqual(explore.copy_check({
            "claim": "Test whether soil saturation changes recovery after irrigation failure.",
            "mechanism": "Randomized withholding isolates root hysteresis from rainfall variation.",
        }, [self.transition]), ())

    def test_apply_x_to_y_remains_a_kill(self) -> None:
        self.assertIn("apply_x_to_y", explore.copy_check({
            "claim": "Apply neural networks to agriculture", "mechanism": "neural networks agriculture",
            "pair": ["neural networks", "agriculture"],
        }, []))


if __name__ == "__main__":
    unittest.main()
