"""Literature lineage may ground a constructed world; it may not climb."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.genworld import construct_world, load_world_payload
from farfield.extras.generate import GeneratedCard
from farfield.extras.livefeed import FreshWork
from farfield.extras.world import WorldRequirement, record_world_wishlist, wishlist_path
from farfield.extras.worldpath import path_from_payload, write_world_path


def _card() -> GeneratedCard:
    return GeneratedCard(
        card_id="gen_path",
        operator="directional",
        claim="a safety invariant of the TCP handshake is preserved under loss",
        mechanism="reverse Kolmogorov drift is nonpositive",
        prediction="unreachable unsafe fraction is 0.0",
        falsifier="pair_not_already_combined",
        pair=("formal verification", "zero weight"),
        pair_nodes=("a", "b"),
        alienness=0.4,
        model="test",
        artifact_digest="d",
        artifact_uri="file:///dev/null",
        replay_mode="replay",
    )


PAPERS = [
    FreshWork(
        title="A learned TCP state machine",
        published="2026-01-01",
        arxiv_id="2601.00001v1",
        abstract="We extract a labeled transition system from Linux TCP traces.",
        url="https://arxiv.org/abs/2601.00001",
    )
]


def _valid_brief(**extra) -> dict:
    payload = {
        "lineage": (
            "protocol papers moved from handwritten TCP automata to learned "
            "labeled traces; the open question is loss-induced unsafe reachability"
        ),
        "named_instance": "Linux TCP handshake automaton from 2601.00001v1",
        "object_type": "formula",
        "schema": "symbolic_trace",
        "cite_ids": ["2601.00001v1"],
        "why_this_object": (
            "the claim is about handshake safety, not about the distant drift rule"
        ),
        "freeze_source": "the paper's public DOT trace",
        "freeze_url": "https://arxiv.org/abs/2601.00001",
        "freeze_schema": "symbolic_trace",
    }
    payload.update(extra)
    return payload


class PathGateTests(unittest.TestCase):
    def test_a_grounded_brief_binds(self) -> None:
        path = path_from_payload(
            "c1",
            _valid_brief(),
            allowed_ids={"2601.00001v1"},
            allowed_urls={"https://arxiv.org/abs/2601.00001"},
            required_object="formula",
        )
        self.assertIsNotNone(path)
        self.assertEqual(path.cite_ids, ("2601.00001v1",))
        self.assertEqual(path.freeze_url, "https://arxiv.org/abs/2601.00001")

    def test_an_invented_citation_is_dropped(self) -> None:
        self.assertIsNone(
            path_from_payload(
                "c1",
                _valid_brief(cite_ids=["9999.99999"]),
                allowed_ids={"2601.00001v1"},
                allowed_urls={"https://arxiv.org/abs/2601.00001"},
                required_object="formula",
            )
        )

    def test_object_drift_is_dropped(self) -> None:
        self.assertIsNone(
            path_from_payload(
                "c1",
                _valid_brief(object_type="graph"),
                allowed_ids={"2601.00001v1"},
                allowed_urls=set(),
                required_object="formula",
            )
        )

    def test_an_invented_url_is_stripped_not_fatal(self) -> None:
        path = path_from_payload(
            "c1",
            _valid_brief(freeze_url="https://evil.example/win.json"),
            allowed_ids={"2601.00001v1"},
            allowed_urls={"https://arxiv.org/abs/2601.00001"},
            required_object="formula",
        )
        self.assertIsNotNone(path)
        self.assertEqual(path.freeze_url, "")

    def test_the_prompt_placeholder_is_not_an_object_type(self) -> None:
        # With no requirement the template says "the claim's object"; the
        # model echoed it verbatim and it was stored as object_type.
        path = path_from_payload(
            "c1",
            _valid_brief(object_type="the claim's object", schema="program_state"),
            allowed_ids={"2601.00001v1"},
            allowed_urls=set(),
        )
        self.assertIsNotNone(path)
        self.assertEqual(path.object_type, "executable")
        self.assertEqual(path.schema, "program_state")

    def test_freeze_schema_never_votes_for_the_construction_schema(self) -> None:
        # program_state used to be unfreezable, so the model was forced to
        # name a freezable neighbour (symbolic_trace) as freeze_schema —
        # and wishlist_fields promoted that neighbour to `schema`, turning
        # the mission world into a protocol automaton.
        path = path_from_payload(
            "c1",
            _valid_brief(
                object_type="executable",
                schema="program_state",
                freeze_schema="symbolic_trace",
            ),
            allowed_ids={"2601.00001v1"},
            allowed_urls={"https://arxiv.org/abs/2601.00001"},
            required_object="executable",
        )
        self.assertIsNotNone(path)
        fields = path.wishlist_fields()
        self.assertNotIn("schema", fields)
        self.assertEqual(fields["freeze_schema"], "symbolic_trace")
        self.assertIn("named_instance", fields)


class WritePathTests(unittest.TestCase):
    def test_no_papers_means_no_lineage(self) -> None:
        class Client:
            def complete(self, *args, **kwargs):
                raise AssertionError("must not ask the model without papers")

        self.assertIsNone(
            write_world_path(Client(), _card(), "tcp safety", [], requirement=None)
        )

    def test_a_usable_client_returns_a_path(self) -> None:
        class Client:
            def complete(self, prompt, *, purpose, system=None):
                self.prompt = prompt

                class Done:
                    text = json.dumps(_valid_brief())

                    def assert_usable(self) -> None:
                        return None

                return Done()

        client = Client()
        path = write_world_path(
            client,
            _card(),
            "tcp safety",
            PAPERS,
            requirement=WorldRequirement(object_type="formula"),
        )
        self.assertIsNotNone(path)
        self.assertIn("2601.00001v1", client.prompt)
        self.assertEqual(path.named_instance.startswith("Linux TCP"), True)


class ConstructFromLineageTests(unittest.TestCase):
    def test_the_sealed_world_carries_the_lineage_and_stays_generated(self) -> None:
        path = path_from_payload(
            "gen_path",
            _valid_brief(),
            allowed_ids={"2601.00001v1"},
            allowed_urls={"https://arxiv.org/abs/2601.00001"},
            required_object="formula",
        )
        with tempfile.TemporaryDirectory() as tmp:
            fixture = construct_world(
                None,
                _card(),
                WorldRequirement(object_type="formula"),
                dest=Path(tmp) / "world",
                lineage=path,
            )
            self.assertEqual(fixture.role, "generated")
            payload = load_world_payload(fixture.root)
            self.assertEqual(payload["lineage"]["cite_ids"], ["2601.00001v1"])
            self.assertIn("cannot corroborate", payload["honesty"])

    def test_wishlist_keeps_the_freeze_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record_world_wishlist(
                root,
                [
                    {
                        "object_type": "formula",
                        "schema": "symbolic_trace",
                        "named_instance": "Linux TCP handshake",
                        "freeze_url": "https://arxiv.org/abs/2601.00001",
                        "lineage_cite_ids": ["2601.00001v1"],
                    }
                ],
                topic="tcp safety",
            )
            payload = json.loads(
                wishlist_path(root).read_text(encoding="utf-8")
            )
            entry = payload["entries"][0]
            self.assertEqual(entry["named_instance"], "Linux TCP handshake")
            self.assertEqual(entry["freeze_url"], "https://arxiv.org/abs/2601.00001")


if __name__ == "__main__":
    unittest.main()
