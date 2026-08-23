"""A surviving card becomes a research-ready briefing, or is refused."""

from __future__ import annotations

import unittest

from farfield.extras.brief import write_brief
from farfield.extras.generate import GeneratedCard, GenerationRefused
from farfield.extras.livefeed import FreshWork
from farfield.extras.llm import Completion

PAPER = FreshWork(
    title="Succinct wavelet trees meet pivot rules",
    published="2026-08-12",
    arxiv_id="2608.01234v1",
    abstract="We compress pivot histories with a wavelet tree.",
)

CARD = GeneratedCard(
    card_id="gen_testbrief01",
    operator="directional",
    claim="a succinct structure stores simplex pivot history in linear space",
    mechanism="the pivot sequence is compressible because it repeats",
    prediction="later work reports a structure with query time below 100 ns",
    falsifier="pair_not_already_combined",
    pair=("succinct data structure", "pivot rule"),
    pair_nodes=("concept:a", "concept:b"),
    alienness=0.4,
    model="fake",
    artifact_digest="d",
    artifact_uri="file:///dev/null",
    replay_mode="replay",
)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        return Completion(
            text=self.text,
            model="fake",
            request_digest="req",
            digest="digestbrief0001",
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            mode="replay",
        )


def answer(**overrides) -> str:
    payload = {
        "title": "Pivot-history compression for the simplex method",
        "gap": "2608.01234v1 compresses pivot histories but never measures query time at n=10^6",
        "idea": "Build a succinct pivot log that answers anticycling queries in sublinear time",
        "approach": "Encode the pivot sequence as a wavelet tree and benchmark against a plain log",
        "first_steps": [
            "Reproduce the wavelet-tree baseline on a 10k-iteration simplex trace",
            "Add rank/select queries for the last repeated pivot",
            "Compare bits per pivot against a gzipped log",
        ],
        "baseline": "an uncompressed circular buffer of recent pivots",
        "risks": "the sequence may not be compressible on real LP instances",
        "read_first": ["2608.01234v1"],
    }
    payload.update(overrides)
    import json

    return json.dumps(payload)


class BriefSchemaTests(unittest.TestCase):
    def test_a_grounded_brief_names_the_retrieved_paper(self) -> None:
        client = FakeClient(answer())
        brief = write_brief(client, CARD, "compress simplex traces", [PAPER])
        self.assertEqual(brief.title.startswith("Pivot-history"), True)
        self.assertEqual(brief.read_first, ("2608.01234v1",))
        self.assertIn("2608.01234v1", client.prompts[0])
        self.assertIn("Succinct wavelet trees", client.prompts[0])

    def test_an_invented_citation_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused) as caught:
            write_brief(
                FakeClient(answer(read_first=["9999.00000"])),
                CARD,
                "topic",
                [PAPER],
            )
        self.assertEqual(
            caught.exception.record.missing_capability, "model_brief_schema"
        )

    def test_a_gap_that_mentions_no_retrieved_paper_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            write_brief(
                FakeClient(answer(gap="the literature has not studied this")),
                CARD,
                "topic",
                [PAPER],
            )

    def test_empty_survey_still_writes_a_program(self) -> None:
        brief = write_brief(
            FakeClient(answer(read_first=[], gap="no live papers were retrieved")),
            CARD,
            "topic",
            [],
        )
        self.assertEqual(brief.read_first, ())
        self.assertGreaterEqual(len(brief.first_steps), 2)

    def test_retitling_into_the_topic_field_is_refused(self) -> None:
        hardness = CARD
        with self.assertRaises(GenerationRefused) as caught:
            write_brief(
                FakeClient(
                    answer(
                        title="Certified Replay for LLM Tool-Use Safety",
                        idea="Verify tool-calling agents with a replay certificate",
                        gap="2608.01234v1 leaves the bound untested",
                    )
                ),
                hardness,
                "formal verification of safety properties for LLM agents and tool-using autonomous agents",
                [PAPER],
            )
        self.assertIn("field", caught.exception.record.unlock_condition)


if __name__ == "__main__":
    unittest.main()
