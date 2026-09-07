"""Scientific cache keys isolate world and prompt versions."""

from __future__ import annotations

import unittest

from farfield.extras.claimspec import scientific_cache_fields
from farfield.extras.llm import Backend, complete_call, request_digest


BACKEND = Backend(base_url="https://example.invalid/v1", model="test-model", api_key="k")
MESSAGES = [{"role": "user", "content": "diagnose this claim"}]


class CacheIsolationTests(unittest.TestCase):
    def test_different_worlds_do_not_share_a_scientific_key(self) -> None:
        first = request_digest(
            BACKEND,
            MESSAGES,
            "diagnose:c1",
            scientific=scientific_cache_fields(world_digest="A", workspace_id="mission_a"),
        )
        second = request_digest(
            BACKEND,
            MESSAGES,
            "diagnose:c1",
            scientific=scientific_cache_fields(world_digest="B", workspace_id="mission_b"),
        )
        self.assertNotEqual(first, second)

    def test_prompt_version_changes_the_scientific_key(self) -> None:
        first = request_digest(
            BACKEND,
            MESSAGES,
            "diagnose:c1",
            scientific=scientific_cache_fields(prompt_version="v1", workspace_id="m"),
        )
        second = request_digest(
            BACKEND,
            MESSAGES,
            "diagnose:c1",
            scientific=scientific_cache_fields(prompt_version="v2", workspace_id="m"),
        )
        self.assertNotEqual(first, second)

    def test_omitting_scientific_keeps_legacy_digests_stable(self) -> None:
        first = request_digest(BACKEND, MESSAGES, "diagnose:c1")
        second = request_digest(BACKEND, MESSAGES, "diagnose:c1")
        self.assertEqual(first, second)

    def test_complete_call_forwards_scientific_when_accepted(self) -> None:
        class Aware:
            def __init__(self) -> None:
                self.scientific = None

            def complete(
                self,
                prompt,
                *,
                purpose,
                system=None,
                logprobs=False,
                scientific=None,
            ):
                self.scientific = scientific
                return "ok"

        client = Aware()
        fields = scientific_cache_fields(world_digest="W", workspace_id="m")
        self.assertEqual(
            complete_call(client, "diagnose this", purpose="diagnose:c1", scientific=fields),
            "ok",
        )
        self.assertEqual(client.scientific["world_digest"], "W")

    def test_complete_call_falls_back_for_legacy_clients(self) -> None:
        class Legacy:
            def complete(self, prompt, *, purpose, system=None, logprobs=False):
                return "legacy"

        self.assertEqual(
            complete_call(
                Legacy(),
                "diagnose this",
                purpose="diagnose:c1",
                scientific=scientific_cache_fields(world_digest="W"),
            ),
            "legacy",
        )


if __name__ == "__main__":
    unittest.main()
