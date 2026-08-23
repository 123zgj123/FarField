"""The LLM adapter must not import non-determinism, silent cost, or leaked keys."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from farfield.extras.llm import Backend, LLMClient, LLMUnavailable, TokenLedger

# Every test drives the adapter through a stand-in transport, so nothing here
# ever opens a socket. The endpoint is a .invalid host, which no resolver will
# answer, so a test that escapes the transport fails loudly instead of reaching
# something real; the key is a literal so the artifact test has a string to
# look for in the recorded bytes.
BACKEND = Backend(base_url="https://example.invalid/v1", model="test-model", api_key="secret-key")


def response(
    content: str,
    *,
    finish: str = "stop",
    completion: int = 120,
    logprobs: list[float] | None = None,
) -> bytes:
    choice: dict = {"message": {"content": content}, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = {
            "content": [{"token": f"t{i}", "logprob": lp} for i, lp in enumerate(logprobs)]
        }
    return json.dumps(
        {
            "model": "test-model",
            "choices": [choice],
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": completion,
                "total_tokens": 30 + completion,
                "completion_tokens_details": {"reasoning_tokens": completion - 20},
            },
        }
    ).encode("utf-8")


class Transport:
    """A stand-in endpoint that answers differently every time, like the real one."""

    def __init__(self, *replies: bytes) -> None:
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


class LLMAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "llm"
        self.addCleanup(self._tmp.cleanup)

    def client(self, transport: Transport | None, *, mode: str = "live", budget: float = 1000.0) -> LLMClient:
        return LLMClient(
            BACKEND,
            self.cache,
            ledger=TokenLedger(api_calls=8.0, token_cost=budget),
            mode=mode,
            transport=transport,
        )

    def test_a_reset_connection_is_retried_not_fatal(self) -> None:
        # A live mission died to a single Errno 104 mid-probe; connection
        # weather must not abort paid work.
        class Flaky(Transport):
            def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
                self.calls += 1
                if self.calls < 3:
                    raise ConnectionResetError(104, "Connection reset by peer")
                return self.replies[0]

        import farfield.extras.llm as llm_module

        original = llm_module.TRANSPORT_BACKOFF_SECONDS
        llm_module.TRANSPORT_BACKOFF_SECONDS = 0.0
        self.addCleanup(setattr, llm_module, "TRANSPORT_BACKOFF_SECONDS", original)
        transport = Flaky(response("weathered the storm"))
        completion = self.client(transport).complete("hello", purpose="conjecture")
        self.assertEqual(completion.text, "weathered the storm")
        self.assertEqual(transport.calls, 3)

    def test_a_client_error_is_not_retried(self) -> None:
        import urllib.error
        from io import BytesIO

        class BadRequest(Transport):
            def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
                self.calls += 1
                raise urllib.error.HTTPError(
                    url, 400, "Bad Request", {}, BytesIO(b"model not found")
                )

        transport = BadRequest()
        with self.assertRaises(LLMUnavailable) as ctx:
            self.client(transport).complete("hello", purpose="conjecture")
        self.assertEqual(transport.calls, 1)
        self.assertIn("400", ctx.exception.record.unlock_condition)

    def test_replay_mode_refuses_instead_of_reaching_the_network(self) -> None:
        transport = Transport(response("should never be sent"))
        client = self.client(transport, mode="replay")
        with self.assertRaises(LLMUnavailable) as ctx:
            client.complete("hello", purpose="conjecture")
        self.assertEqual(
            ctx.exception.record.missing_capability, "llm_recorded_response"
        )
        self.assertEqual(transport.calls, 0)

    def test_a_recorded_call_replays_the_same_bytes_a_changed_endpoint_would_not(self) -> None:
        # The transport answers differently the second time, like the real
        # endpoint does, so a replay that reached it would say "second answer".
        transport = Transport(response("first answer"), response("second answer"))
        live = self.client(transport)
        first = live.complete("prompt", purpose="conjecture")
        self.assertEqual(first.mode, "live")
        self.assertEqual(first.text, "first answer")

        replayed = self.client(transport, mode="replay").complete(
            "prompt", purpose="conjecture"
        )
        self.assertEqual(replayed.mode, "replay")
        self.assertEqual(replayed.text, "first answer")
        self.assertEqual(replayed.digest, first.digest)
        self.assertEqual(transport.calls, 1)

    def test_the_api_key_is_absent_from_the_recorded_artifact(self) -> None:
        client = self.client(Transport(response("answer")))
        completion = client.complete("prompt", purpose="conjecture")
        raw = Path(client.artifact_path(completion.request_digest)).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("secret-key", raw)
        self.assertIn("test-model", raw)

    def test_a_tampered_artifact_refuses_to_replay(self) -> None:
        client = self.client(Transport(response("answer")))
        completion = client.complete("prompt", purpose="conjecture")
        path = client.artifact_path(completion.request_digest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["response_raw"] = payload["response_raw"].replace("answer", "ansxer")
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(LLMUnavailable) as ctx:
            self.client(None, mode="replay").complete("prompt", purpose="conjecture")
        self.assertEqual(
            ctx.exception.record.missing_capability, "llm_artifact_integrity"
        )

    def test_a_truncated_completion_is_not_usable_evidence(self) -> None:
        client = self.client(Transport(response("half an ans", finish="length")))
        completion = client.complete("prompt", purpose="conjecture")
        self.assertTrue(completion.truncated)
        with self.assertRaises(LLMUnavailable) as ctx:
            completion.assert_usable()
        self.assertEqual(
            ctx.exception.record.missing_capability, "llm_completion_finished"
        )
        self.assertIn("max_tokens", ctx.exception.record.unlock_condition)

    def test_reasoning_only_completions_are_not_usable_either(self) -> None:
        client = self.client(Transport(response("   ")))
        completion = client.complete("prompt", purpose="conjecture")
        with self.assertRaises(LLMUnavailable) as ctx:
            completion.assert_usable()
        self.assertEqual(
            ctx.exception.record.missing_capability, "llm_visible_content"
        )

    def test_an_exhausted_envelope_refuses_before_the_call_is_made(self) -> None:
        transport = Transport(response("a", completion=400), response("b"))
        client = self.client(transport, budget=100.0)
        client.complete("first", purpose="conjecture")
        self.assertGreater(client.ledger.spent_tokens, 100.0)
        with self.assertRaises(LLMUnavailable) as ctx:
            client.complete("second", purpose="conjecture")
        self.assertEqual(
            ctx.exception.record.missing_capability, "llm_budget_token_cost"
        )
        self.assertEqual(transport.calls, 1)

    def test_a_truncated_answer_still_costs_what_it_burned(self) -> None:
        client = self.client(Transport(response("cut", finish="length", completion=500)))
        client.complete("prompt", purpose="conjecture")
        self.assertEqual(client.ledger.spent_tokens, 530.0)
        self.assertEqual(client.ledger.spent_calls, 1.0)

    def test_arms_are_split_into_equal_allotments(self) -> None:
        # If the far arm could draw more model budget than the near control, a
        # far-beats-near result would be measuring which arm got more tokens.
        ledger = TokenLedger(api_calls=8.0, token_cost=20000.0)
        arms = ledger.split(("near", "far"))
        self.assertEqual(
            {name: arm.token_cost for name, arm in arms.items()},
            {"near": 10000.0, "far": 10000.0},
        )
        self.assertEqual({arm.api_calls for arm in arms.values()}, {4.0})

    def test_resolve_backend_uses_the_callers_model_not_a_builtin_default(self) -> None:
        from farfield.extras.llm import resolve_backend

        backend = resolve_backend(
            model="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            env={},
        )
        self.assertEqual(backend.model, "gpt-4.1")
        self.assertEqual(backend.base_url, "https://api.openai.com/v1")

    def test_a_named_model_overrides_whatever_the_environment_had(self) -> None:
        from farfield.extras.llm import resolve_backend

        backend = resolve_backend(
            model="claude-sonnet-4",
            base_url="https://openrouter.ai/api/v1",
            api_key="or-key",
            env={
                "FARFIELD_LLM_MODEL": "deepseek-v4-pro",
                "FARFIELD_LLM_BASE_URL": "https://api.deepseek.com/v1",
                "FARFIELD_LLM_KEY": "ds-key",
            },
        )
        self.assertEqual(backend.model, "claude-sonnet-4")
        self.assertEqual(backend.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(backend.api_key, "or-key")

    def test_resolve_backend_does_not_invent_a_model_when_none_was_named(self) -> None:
        from farfield.extras.llm import resolve_backend

        with self.assertRaises(LLMUnavailable) as ctx:
            resolve_backend(env={})
        self.assertEqual(ctx.exception.record.missing_capability, "llm_model_id")

    def test_configured_status_never_echoes_the_key(self) -> None:
        from farfield.extras.llm import configured_status

        status = configured_status(
            {
                "FARFIELD_LLM_MODEL": "gpt-4.1",
                "FARFIELD_LLM_KEY": "sk-secret",
                "FARFIELD_LLM_BASE_URL": "https://api.openai.com/v1",
            }
        )
        self.assertTrue(status["ready"])
        self.assertEqual(status["model"], "gpt-4.1")
        dumped = json.dumps(status)
        self.assertNotIn("sk-secret", dumped)
        self.assertTrue(any(row["id"] == "openai" for row in status["presets"]))

    def test_backend_from_env_names_what_is_missing(self) -> None:
        with self.assertRaises(LLMUnavailable) as ctx:
            Backend.from_env({})
        self.assertEqual(ctx.exception.record.missing_capability, "llm_model_id")
        with self.assertRaises(LLMUnavailable) as ctx:
            Backend.from_env({"FARFIELD_LLM_MODEL": "m"})
        self.assertEqual(ctx.exception.record.missing_capability, "llm_api_key")

    def test_backend_reads_the_key_from_a_file_so_it_stays_out_of_argv(self) -> None:
        key_file = Path(self._tmp.name) / "key"
        key_file.write_text("file-key\n", encoding="utf-8")
        backend = Backend.from_env(
            {
                "FARFIELD_LLM_MODEL": "m",
                "FARFIELD_LLM_KEY_FILE": str(key_file),
                "FARFIELD_LLM_BASE_URL": "https://example.invalid/v1/",
            }
        )
        self.assertEqual(backend.api_key, "file-key")
        self.assertEqual(backend.base_url, "https://example.invalid/v1")

    def test_purpose_separates_cache_entries_for_the_same_prompt(self) -> None:
        transport = Transport(response("as proposer"), response("as judge"))
        client = self.client(transport)
        first = client.complete("same text", purpose="propose")
        second = client.complete("same text", purpose="judge")
        self.assertNotEqual(first.request_digest, second.request_digest)
        self.assertEqual(transport.calls, 2)


class LogprobsTest(unittest.TestCase):
    """NLL surprise rides on recorded logprobs; absence must stay a fact."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "llm"
        self.addCleanup(self._tmp.cleanup)

    def client(self, transport: Transport | None, *, mode: str = "live", budget: float = 1000.0) -> LLMClient:
        return LLMClient(
            BACKEND,
            self.cache,
            ledger=TokenLedger(api_calls=8.0, token_cost=budget),
            mode=mode,
            transport=transport,
        )

    def test_returned_logprobs_become_a_mean_nll_and_replay_identically(self) -> None:
        transport = Transport(response("answer", logprobs=[-0.5, -1.5, -1.0]))
        live = self.client(transport)
        completion = live.complete("prompt", purpose="score", logprobs=True)
        self.assertAlmostEqual(completion.mean_nll, 1.0)
        self.assertEqual(completion.nll_tokens, 3)

        replayed = self.client(None, mode="replay").complete(
            "prompt", purpose="score", logprobs=True
        )
        self.assertAlmostEqual(replayed.mean_nll, 1.0)
        self.assertEqual(replayed.digest, completion.digest)

    def test_an_endpoint_that_ignores_the_option_yields_none_not_zero(self) -> None:
        client = self.client(Transport(response("answer")))
        completion = client.complete("prompt", purpose="score", logprobs=True)
        self.assertIsNone(completion.mean_nll)
        self.assertEqual(completion.nll_tokens, 0)

    def test_asking_for_logprobs_is_a_different_request(self) -> None:
        # Every artifact recorded before this option existed keeps its digest;
        # a logprobs call can never silently replay a non-logprobs recording.
        transport = Transport(response("plain"), response("with logprobs"))
        client = self.client(transport)
        plain = client.complete("prompt", purpose="score")
        scored = client.complete("prompt", purpose="score", logprobs=True)
        self.assertNotEqual(plain.request_digest, scored.request_digest)
        self.assertEqual(transport.calls, 2)

    def test_the_live_request_body_carries_the_option_only_when_asked(self) -> None:
        seen: list[bytes] = []

        class Recorder(Transport):
            def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
                seen.append(body)
                return super().__call__(url, body, headers)

        client = self.client(Recorder(response("a"), response("b")))
        client.complete("prompt", purpose="score")
        client.complete("prompt2", purpose="score", logprobs=True)
        self.assertNotIn(b'"logprobs"', seen[0])
        self.assertIn(b'"logprobs": true', seen[1])

    def test_old_artifacts_without_an_nll_block_still_replay(self) -> None:
        client = self.client(Transport(response("answer")))
        completion = client.complete("prompt", purpose="conjecture")
        path = client.artifact_path(completion.request_digest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["nll"]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        replayed = self.client(None, mode="replay").complete(
            "prompt", purpose="conjecture"
        )
        self.assertIsNone(replayed.mean_nll)

    def test_two_live_calls_overlap_on_the_wire(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor, wait

        barrier = threading.Barrier(2)
        lock = threading.Lock()

        class Slow(Transport):
            def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
                barrier.wait(timeout=2)
                with lock:
                    self.calls += 1
                    index = self.calls - 1
                return self.replies[min(index, len(self.replies) - 1)]

        client = self.client(
            Slow(response("one"), response("two")), budget=20000.0
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(client.complete, "alpha", purpose="a"),
                pool.submit(client.complete, "beta", purpose="b"),
            ]
            wait(futures)
            texts = sorted(f.result().text for f in futures)
        self.assertEqual(texts, ["one", "two"])
        self.assertEqual(client.ledger.spent_calls, 2.0)
        self.assertEqual(client.ledger.reserved_calls, 0.0)

    def test_a_one_call_envelope_cannot_be_double_booked(self) -> None:
        import time
        from concurrent.futures import ThreadPoolExecutor, wait

        class Slow(Transport):
            def __call__(self, url: str, body: bytes, headers: dict[str, str]) -> bytes:
                time.sleep(0.05)
                return response("ok")

        client = LLMClient(
            BACKEND,
            self.cache,
            ledger=TokenLedger(api_calls=1.0, token_cost=20000.0),
            mode="live",
            transport=Slow(),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(client.complete, "alpha", purpose="a"),
                pool.submit(client.complete, "beta", purpose="b"),
            ]
            wait(futures)
        outcomes = [f.exception() is None for f in futures]
        self.assertEqual(sum(outcomes), 1)
        self.assertEqual(client.ledger.spent_calls, 1.0)
        self.assertEqual(client.ledger.reserved_calls, 0.0)


if __name__ == "__main__":
    unittest.main()
