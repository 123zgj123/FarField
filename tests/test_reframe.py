"""Reframe cards: the assumption must be real, grounded, and falsifiable."""

from __future__ import annotations

import json
import unittest

from farfield.extras.generate import GenerationRefused
from farfield.extras.llm import Completion
from farfield.extras.reframe import (
    GRAPH_SKIP_REASON,
    OPERATORS,
    card_from_payload,
    generate_reframe,
    refine_reframe,
)


def payload(**overrides):
    base = {
        "assumption": "worst case analysis",
        "claim": (
            "dropping worst case analysis for succinct indexes yields query"
            " time at most O(log n) on realistic inputs"
        ),
        "mechanism": (
            "worst case analysis prices adversarial inputs that realistic"
            " query logs never produce"
        ),
        "prediction": "later work reports query time below 100 ns",
        "dead_end": "keep worst case analysis and size the index for adversarial query logs",
        "why_failed": "adversarial padding dominates the bound so realistic logs never see it",
        "reframe": "drop worst case analysis and measure realistic query logs instead",
        "objection": "a hidden adversary can still force the logarithmic bound to collapse",
    }
    base.update(overrides)
    return base


def card(**overrides):
    return card_from_payload(
        payload(**overrides),
        seed_label="succinct data structure",
        operator="assumption_removal",
        model="fake",
        artifact_digest="d",
        artifact_uri="file:///dev/null",
        replay_mode="replay",
    )


class ScriptedClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        return Completion(
            text=self.text,
            model="fake",
            request_digest="req",
            digest="d" + "0" * 15,
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            mode="replay",
        )


class SchemaTests(unittest.TestCase):
    def test_a_grounded_reframe_card_is_accepted(self) -> None:
        result = card()
        self.assertEqual(result.operator, "assumption_removal")
        self.assertEqual(result.pair[1], "worst case analysis")
        self.assertIn("does not answer", GRAPH_SKIP_REASON)

    def test_an_assumption_absent_from_the_claim_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            card(
                claim="succinct indexes get query time at most O(log n)",
                mechanism="realistic logs never produce adversarial inputs",
            )

    def test_an_empty_assumption_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            card(assumption="")

    def test_a_missing_dead_end_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            card(dead_end="")

    def test_an_unknown_operator_is_a_caller_error(self) -> None:
        with self.assertRaises(KeyError):
            generate_reframe(
                ScriptedClient(json.dumps(payload())),
                seed_label="s",
                near_labels=("a",),
                operator="vibes_search",
            )


class PromptTests(unittest.TestCase):
    def test_knowledge_and_operator_instruction_ride_in_the_prompt(self) -> None:
        client = ScriptedClient(json.dumps(payload()))
        generate_reframe(
            client,
            seed_label="succinct data structure",
            near_labels=("pivot rule",),
            operator="assumption_removal",
            knowledge=("Open question from earlier missions: whether X",),
        )
        prompt = client.prompts[0]
        self.assertIn(OPERATORS["assumption_removal"].split(",")[0], prompt)
        self.assertIn("whether X", prompt)


class RefineTests(unittest.TestCase):
    def test_a_refine_keeps_the_assumption_and_quotes_the_kill(self) -> None:
        parent = card()
        client = ScriptedClient(json.dumps(payload()))
        revised = refine_reframe(
            client,
            parent,
            seed_label="succinct data structure",
            near_labels=("pivot rule",),
            killed_by=["prediction_names_a_measurable_quantity"],
        )
        self.assertEqual(revised.pair, parent.pair)
        prompt = client.prompts[0]
        self.assertIn("Revise THIS idea", prompt)
        self.assertIn("prediction_names_a_measurable_quantity", prompt)

    def test_a_new_assumption_is_refused(self) -> None:
        parent = card()
        client = ScriptedClient(
            json.dumps(
                payload(
                    assumption="average case analysis",
                    claim=(
                        "dropping average case analysis for succinct indexes"
                        " yields query time at most O(log n) on realistic inputs"
                    ),
                    mechanism=(
                        "average case analysis prices typical inputs that"
                        " adversarial query logs never produce"
                    ),
                    dead_end="keep average case analysis and size the index for typical logs",
                    reframe="drop average case analysis and measure realistic query logs instead",
                )
            )
        )
        with self.assertRaises(GenerationRefused):
            refine_reframe(
                client,
                parent,
                seed_label="succinct data structure",
                near_labels=("pivot rule",),
                killed_by=["apply_x_to_y"],
            )


if __name__ == "__main__":
    unittest.main()
