"""Model review is an opinion; revision must actually change the program."""

from __future__ import annotations

import json
import unittest

from farfield.extras.brief import write_brief
from farfield.extras.generate import GeneratedCard, GenerationRefused
from farfield.extras.livefeed import FreshWork
from farfield.extras.llm import Completion
from farfield.extras.review import critique_card, refine_brief, review_idea

PAPER = FreshWork(
    title="Succinct structure for simplex pivot history",
    published="2026-08-12",
    arxiv_id="2608.01234v1",
    abstract=(
        "A succinct structure stores simplex pivot history in linear space "
        "using a wavelet tree."
    ),
)
CARD = GeneratedCard(
    card_id="gen_testreview01",
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


class ScriptedClient:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.prompts: list[str] = []

    def complete(self, prompt, *, purpose, system=None, logprobs=False):
        self.prompts.append(prompt)
        text = self.texts.pop(0)
        return Completion(
            text=text,
            model="fake",
            request_digest="req",
            digest=f"d{len(self.prompts):015x}",
            artifact_uri="file:///dev/null",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            reasoning_tokens=0,
            mode="replay",
        )


def brief_json() -> str:
    return json.dumps(
        {
            "title": "Pivot-history compression",
            "plain_title": "单纯形法的转轴历史能压缩吗",
            "one_liner": "单纯形法每一步的转轴记录，能否用线性空间存下并快速查询？",
            "why_it_matters": "如果可行，防循环检查就不必保留完整日志，求解器内存占用可以明显下降。",
            "gap": "2608.01234v1 compresses pivot histories but never measures query time",
            "idea": "Build a succinct pivot log",
            "approach": "Encode the sequence as a wavelet tree",
            "first_steps": ["Reproduce the baseline", "Add rank queries", "Compare bits"],
            "baseline": "a plain log",
            "risks": "traces may not compress",
            "read_first": ["2608.01234v1"],
        }
    )


def review_json(**overrides) -> str:
    payload = {
        "novelty": 3,
        "novelty_note": "2608.01234v1 already stores pivot histories, so the log itself is not new",
        "clarity": 4,
        "feasibility": 4,
        "importance": 3,
        "information_gain": 4,
        "transfer_potential": 2,
        "keep_going": True,
        "critique": "name a public trace and a query-time number",
        "must_change": ["first_steps should name a public LP trace"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brief = write_brief(
            ScriptedClient(brief_json()), CARD, "compress simplex traces", [PAPER]
        )

    def test_novelty_is_an_opinion_and_names_a_paper(self) -> None:
        review = review_idea(
            ScriptedClient(review_json()),
            CARD,
            self.brief,
            "topic",
            [PAPER],
        )
        self.assertEqual(review.novelty, 3)
        self.assertIn("cannot kill", review.to_dict()["status"])
        self.assertIn("2608.01234v1", review.novelty_note)

    def test_the_value_vector_is_never_collapsed(self) -> None:
        review = review_idea(
            ScriptedClient(review_json()), CARD, self.brief, "topic", [PAPER]
        )
        vector = review.to_dict()["value_vector"]
        self.assertEqual(vector["importance"], 3)
        self.assertEqual(vector["information_gain"], 4)
        self.assertEqual(vector["transfer_potential"], 2)
        self.assertNotIn("total", vector)
        self.assertNotIn("score", review.to_dict())

    def test_a_review_missing_a_value_dimension_is_refused(self) -> None:
        payload = json.loads(review_json())
        del payload["information_gain"]
        with self.assertRaises(GenerationRefused):
            review_idea(
                ScriptedClient(json.dumps(payload)),
                CARD,
                self.brief,
                "topic",
                [PAPER],
            )

    def test_a_novelty_note_that_cites_nothing_is_refused(self) -> None:
        with self.assertRaises(GenerationRefused):
            review_idea(
                ScriptedClient(review_json(novelty_note="this is quite original")),
                CARD,
                self.brief,
                "topic",
                [PAPER],
            )

    def test_a_revision_must_change_the_program(self) -> None:
        review = review_idea(
            ScriptedClient(review_json()), CARD, self.brief, "topic", [PAPER]
        )
        with self.assertRaises(GenerationRefused):
            refine_brief(
                ScriptedClient(brief_json()),
                CARD,
                self.brief,
                review,
                "topic",
                [PAPER],
            )
        changed = json.loads(brief_json())
        changed["idea"] = "Measure ns/query of a succinct pivot log on MIPLIB traces"
        changed["first_steps"] = [
            "Record pivots on a public MIPLIB instance",
            "Time rank queries",
            "Compare bits against gzip",
        ]
        revised = refine_brief(
            ScriptedClient(json.dumps(changed)),
            CARD,
            self.brief,
            review,
            "topic",
            [PAPER],
        )
        self.assertNotEqual(revised.idea, self.brief.idea)

    def test_critique_card_is_opinion_and_names_must_change(self) -> None:
        client = ScriptedClient(review_json(keep_going=True))
        review = critique_card(
            client,
            CARD,
            "compress simplex traces",
            [PAPER],
        )
        self.assertEqual(review.novelty, 3)
        self.assertTrue(review.keep_going)
        self.assertTrue(review.must_change)
        self.assertIn("cannot kill", review.to_dict()["status"])
        self.assertTrue(
            any("Attack this probe" in prompt for prompt in client.prompts)
        )


if __name__ == "__main__":
    unittest.main()
