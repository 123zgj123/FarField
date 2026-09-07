"""Backend-neutral LLM adapter with recorded artifacts and a spent-as-you-go budget.

Three constraints from the kernel drive every choice here.

Reproducibility. The upstream endpoint is not deterministic: two identical
requests at temperature 0 returned different completions, one of them truncated.
So the artifact, not the endpoint, is the ground truth. Every call writes the raw
request and raw response bytes with their sha256, and the default mode is
`replay`: a cache hit replays the artifact, a cache miss refuses. Reaching the
network takes an explicit `mode="live"`, which is how tests stay offline and how a
re-run of a settled campaign produces the same text it was settled on.

Budget (INV-15). The envelope is declared before the campaign starts, so the
ledger is charged as calls happen and refuses once a dimension is exhausted.
A truncated response still costs what it burned.

Fair arms. `TokenLedger.split` hands each arm an equal allotment. If the far arm
could draw more model budget than the near control, a far-beats-near result would
be measuring which arm got more tokens.

No per-call `max_tokens` is sent. Measured on deepseek-v4-pro, 92-96% of
completion tokens are reasoning tokens, so any cap small enough to be useful
truncates before the visible answer begins. The bound that matters is the
campaign envelope, not the call.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..ledger import content_digest
from ..models import BlockedRecord

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_KEY_FILE = "~/.config/farfield/llm.key"
ENV_BASE_URL = "FARFIELD_LLM_BASE_URL"
ENV_MODEL = "FARFIELD_LLM_MODEL"
ENV_KEY = "FARFIELD_LLM_KEY"
ENV_KEY_FILE = "FARFIELD_LLM_KEY_FILE"
TIMEOUT_SECONDS = 300.0
TRANSPORT_ATTEMPTS = 3
TRANSPORT_BACKOFF_SECONDS = 3.0
# Resume is replay. Every completion is recorded under the cache before it
# is used, and the feed snapshot is recorded in the workspace, so the same
# command in the same workspace replays every recorded step for free and
# spends again only at the first call that was never answered.
RESUME_HINT = (
    " | resume: rerun the same command with the same workspace and state "
    "store; recorded completions and the feed snapshot replay, spend restarts "
    "at the first unrecorded call"
)
_QUOTA_MARKERS = ("insufficient_quota", "credit_balance_exhausted", "no credits remaining")


def quota_exhausted(detail: str) -> bool:
    """A 429 that says the balance is gone, not that the rate is high."""
    text = str(detail or "").lower()
    return any(marker in text for marker in _QUOTA_MARKERS)

# Common OpenAI-compatible hosts. The model id is never implied by the host:
# a researcher picks both, the way AI Scientist exposes --model_writeup.
PRESETS: tuple[dict[str, str], ...] = (
    {"id": "openai", "label": "OpenAI", "base_url": "https://api.openai.com/v1"},
    {"id": "deepseek", "label": "DeepSeek", "base_url": "https://api.deepseek.com/v1"},
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
    },
    {
        "id": "anthropic",
        "label": "Anthropic (OpenAI-compatible)",
        "base_url": "https://api.anthropic.com/v1",
    },
    {"id": "custom", "label": "Custom endpoint", "base_url": ""},
)


class LLMUnavailable(RuntimeError):
    """The adapter cannot answer. Always carries a blocked record to file."""

    def __init__(self, record: BlockedRecord):
        super().__init__(record.unlock_condition)
        self.record = record


@dataclass(frozen=True)
class Backend:
    """An OpenAI-compatible endpoint. The model id is passed through verbatim.

    Guessing a model id is how a run spends a minute to learn `deepseek-v4-pro-0813`
    is not served; `doctor` asks the endpoint which ids it actually offers.
    """

    base_url: str
    model: str
    api_key: str = field(repr=False, default="")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Backend":
        source = dict(os.environ if env is None else env)
        model = (source.get(ENV_MODEL) or "").strip()
        if not model:
            raise LLMUnavailable(
                BlockedRecord(
                    missing_capability="llm_model_id",
                    attempted=f"read the model id from ${ENV_MODEL}",
                    unlock_condition=f"set ${ENV_MODEL} to an id the endpoint serves"
                    " (farfield llm-doctor lists them)",
                )
            )

        key = (source.get(ENV_KEY) or "").strip()
        key_file = (source.get(ENV_KEY_FILE) or "").strip()
        if not key and key_file:
            path = Path(key_file).expanduser()
            if path.is_file():
                key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise LLMUnavailable(
                BlockedRecord(
                    missing_capability="llm_api_key",
                    attempted=f"read the api key from ${ENV_KEY} or ${ENV_KEY_FILE}",
                    unlock_condition=f"set ${ENV_KEY}, or point ${ENV_KEY_FILE} at a"
                    " file containing the key; the key is never written to an artifact",
                )
            )

        base = (source.get(ENV_BASE_URL) or DEFAULT_BASE_URL).strip().rstrip("/")
        return cls(base_url=base, model=model, api_key=key)


def resolve_backend(
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    env: dict[str, str] | None = None,
) -> Backend:
    """Pick an endpoint from what this call said, then from the environment.

    Nothing here names a default model. A missing model or key is a blocked
    record, not a silent DeepSeek. The conventional key file is tried only
    when neither the call nor the env supplied a key — so a local console
    can keep the secret off the form, and a different machine still has to
    say which model it wants.
    """
    source = dict(os.environ if env is None else env)
    chosen_model = (model or source.get(ENV_MODEL) or "").strip()
    if not chosen_model:
        raise LLMUnavailable(
            BlockedRecord(
                missing_capability="llm_model_id",
                attempted="read a model id from the request or $FARFIELD_LLM_MODEL",
                unlock_condition="choose a model the endpoint serves"
                " (OpenAI gpt-4.1, DeepSeek deepseek-v4-pro, or any id"
                " `farfield llm-doctor` lists)",
            )
        )

    key = (api_key or source.get(ENV_KEY) or "").strip()
    key_file = (source.get(ENV_KEY_FILE) or "").strip()
    if not key and not key_file:
        conventional = Path(DEFAULT_KEY_FILE).expanduser()
        if conventional.is_file():
            key_file = str(conventional)
    if not key and key_file:
        path = Path(key_file).expanduser()
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise LLMUnavailable(
            BlockedRecord(
                missing_capability="llm_api_key",
                attempted="read an api key from the request, $FARFIELD_LLM_KEY,"
                " or the conventional key file",
                unlock_condition="paste a key in the console, or set"
                f" ${ENV_KEY} / ${ENV_KEY_FILE}",
            )
        )

    chosen_base = (base_url or source.get(ENV_BASE_URL) or "").strip().rstrip("/")
    if not chosen_base:
        raise LLMUnavailable(
            BlockedRecord(
                missing_capability="llm_base_url",
                attempted="read an endpoint from the request or $FARFIELD_LLM_BASE_URL",
                unlock_condition="choose an OpenAI-compatible endpoint"
                " (api.openai.com, api.deepseek.com, openrouter.ai, or your own)",
            )
        )
    return Backend(base_url=chosen_base, model=chosen_model, api_key=key)


def configured_status(env: dict[str, str] | None = None) -> dict[str, Any]:
    """What the machine already knows, with the key left out."""
    try:
        backend = resolve_backend(env=env)
    except LLMUnavailable as error:
        return {
            "ready": False,
            "missing": error.record.missing_capability,
            "unlock": error.record.unlock_condition,
            "presets": [dict(row) for row in PRESETS],
        }
    return {
        "ready": True,
        "model": backend.model,
        "base_url": backend.base_url,
        "presets": [dict(row) for row in PRESETS],
    }


@dataclass
class TokenLedger:
    """Spend-as-you-go accounting against the declared envelope."""

    api_calls: float
    token_cost: float
    spent_calls: float = 0.0
    spent_tokens: float = 0.0
    reserved_calls: float = 0.0
    charges: list[dict[str, Any]] = field(default_factory=list)

    def remaining(self) -> dict[str, float]:
        return {
            "api_calls": self.api_calls - self.spent_calls - self.reserved_calls,
            "token_cost": self.token_cost - self.spent_tokens,
        }

    def reserve_call(self, purpose: str) -> None:
        """Hold one call slot until `commit_call` or `abort_call`.

        Concurrent live completions reserve before they hit the network so
        two in-flight calls cannot both pass a remaining-calls check of 1.
        """
        self.assert_can_call(purpose)
        self.reserved_calls += 1.0

    def commit_call(self, purpose: str, tokens: float) -> None:
        if self.reserved_calls >= 1.0:
            self.reserved_calls -= 1.0
        self.spent_calls += 1.0
        self.spent_tokens += float(tokens)
        self.charges.append(
            {"purpose": purpose, "tokens": float(tokens), "calls": 1.0}
        )

    def abort_call(self) -> None:
        if self.reserved_calls >= 1.0:
            self.reserved_calls -= 1.0

    def assert_can_call(self, purpose: str) -> None:
        left = self.remaining()
        for key, value in left.items():
            if value <= 0:
                raise LLMUnavailable(
                    BlockedRecord(
                        missing_capability=f"llm_budget_{key}",
                        attempted=f"one model call for {purpose}",
                        unlock_condition=f"declare a larger envelope.{key}"
                        " before the campaign starts; "
                        f"{self.spent_calls:g} call(s) and "
                        f"{self.spent_tokens:g} token(s) are already spent",
                    )
                )

    def charge(self, purpose: str, tokens: float) -> None:
        self.spent_calls += 1.0
        self.spent_tokens += float(tokens)
        self.charges.append(
            {"purpose": purpose, "tokens": float(tokens), "calls": 1.0}
        )

    def split(self, arms: tuple[str, ...]) -> dict[str, "TokenLedger"]:
        """Equal allotments, so a far-vs-near result is not a budget comparison."""
        if not arms:
            raise ValueError("split needs at least one arm")
        share = len(arms)
        return {
            arm: TokenLedger(
                api_calls=self.api_calls / share,
                token_cost=self.token_cost / share,
            )
            for arm in arms
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": {"api_calls": self.api_calls, "token_cost": self.token_cost},
            "spent": {"api_calls": self.spent_calls, "token_cost": self.spent_tokens},
            "remaining": self.remaining(),
            "charges": list(self.charges),
        }


@dataclass(frozen=True)
class Completion:
    """One recorded exchange. `digest` pins the bytes any claim must cite.

    `mean_nll` is the negative mean logprob of the visible completion tokens
    when the call requested logprobs and the endpoint returned them, else
    None. None is a fact about the endpoint, not a zero: the scorer that
    wants an NLL-surprise term must record its absence, never default it.
    """

    text: str
    model: str
    request_digest: str
    digest: str
    artifact_uri: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    mode: str
    mean_nll: float | None = None
    nll_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        return self.finish_reason != "stop"

    def assert_usable(self) -> None:
        """A truncated or empty completion is not evidence of anything."""
        if self.truncated:
            raise LLMUnavailable(
                BlockedRecord(
                    missing_capability="llm_completion_finished",
                    attempted=f"a completion that ends on stop (got"
                    f" {self.finish_reason!r} after "
                    f"{self.completion_tokens} completion tokens, "
                    f"{self.reasoning_tokens} of them reasoning)",
                    unlock_condition="let the model self-terminate; do not send"
                    " max_tokens, and declare an envelope large enough for the"
                    " reasoning it needs",
                )
            )

        if not self.text.strip():
            raise LLMUnavailable(
                BlockedRecord(
                    missing_capability="llm_visible_content",
                    attempted="a completion with non-empty content",
                    unlock_condition="the endpoint returned reasoning but no visible"
                    " content; retry with a prompt that asks for the answer explicitly",
                )
            )


def sampling_is_locked(model: str) -> bool:
    """Reasoning endpoints that reject temperature / optional logprobs.

    DeepSeek and GPT-4 family artifacts still record `temperature: 0`.
    Sending that field to `gpt-5.6-sol` is a 400, so the live body and
    the request digest both omit it for those ids.
    """
    name = str(model or "").strip().lower()
    return name.startswith(("gpt-5.6", "o1", "o3", "o4"))


def _canonical_request(
    backend: Backend,
    messages: list[dict[str, str]],
    purpose: str,
    *,
    logprobs: bool = False,
    scientific: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "base_url": backend.base_url,
        "model": backend.model,
        "messages": messages,
        "purpose": purpose,
    }
    if not sampling_is_locked(backend.model):
        payload["temperature"] = 0
    # Only present when asked for, so every request recorded before this
    # option existed keeps its digest and stays replayable.
    if logprobs and not sampling_is_locked(backend.model):
        payload["logprobs"] = True
    if scientific:
        payload["scientific"] = dict(scientific)
    return payload


def request_digest(
    backend: Backend,
    messages: list[dict[str, str]],
    purpose: str,
    *,
    logprobs: bool = False,
    scientific: dict[str, Any] | None = None,
) -> str:
    payload = _canonical_request(
        backend, messages, purpose, logprobs=logprobs, scientific=scientific
    )
    return content_digest(payload)


def complete_call(
    client: Any,
    prompt: str,
    *,
    purpose: str,
    system: str | None = None,
    logprobs: bool = False,
    scientific: dict[str, Any] | None = None,
) -> Any:
    """Call `complete` with scientific cache fields when the client accepts them.

    Test doubles keep the old signature. A TypeError falls back so FakeClient
    does not have to know about object-identity keys.
    """
    kwargs: dict[str, Any] = {"purpose": purpose}
    if system is not None:
        kwargs["system"] = system
    if logprobs:
        kwargs["logprobs"] = True
    if scientific:
        try:
            return client.complete(prompt, scientific=scientific, **kwargs)
        except TypeError:
            pass
    return client.complete(prompt, **kwargs)


class LLMClient:
    """Records every call under `cache_dir` and replays it forever after."""

    def __init__(
        self,
        backend: Backend,
        cache_dir: Path | str,
        *,
        ledger: TokenLedger | None = None,
        mode: str = "replay",
        transport: Callable[[str, bytes, dict[str, str]], bytes] | None = None,
    ) -> None:
        if mode not in {"replay", "live"}:
            raise ValueError(f"mode must be replay or live, not {mode!r}")
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.ledger = ledger or TokenLedger(api_calls=0.0, token_cost=0.0)
        self.mode = mode
        self._transport = transport or _http_post
        self._lock = threading.Lock()

    def artifact_path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"

    def complete(
        self,
        prompt: str,
        *,
        purpose: str,
        system: str | None = None,
        logprobs: bool = False,
        scientific: dict[str, Any] | None = None,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        digest = request_digest(
            self.backend,
            messages,
            purpose,
            logprobs=logprobs,
            scientific=scientific,
        )
        path = self.artifact_path(digest)
        with self._lock:
            if path.is_file():
                return _completion_from_artifact(
                    json.loads(path.read_text(encoding="utf-8")), path, mode="replay"
                )
            if self.mode != "live":
                raise LLMUnavailable(
                    BlockedRecord(
                        missing_capability="llm_recorded_response",
                        attempted=f"replay a recorded response for {purpose}"
                        f" ({digest[:12]})",
                        unlock_condition="run once with mode='live' to record the"
                        " artifact, then every later run replays those exact bytes",
                    )
                )
            self.ledger.reserve_call(purpose)
        request_body: dict[str, Any] = {
            "model": self.backend.model,
            "messages": messages,
            "stream": False,
        }
        if not sampling_is_locked(self.backend.model):
            request_body["temperature"] = 0
        if logprobs and not sampling_is_locked(self.backend.model):
            request_body["logprobs"] = True
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.backend.api_key}",
        }
        started = time.time()
        # A reset connection or a 5xx is weather, not a verdict on the
        # mission: live APIs shed connections routinely, and one Errno 104
        # must not abort half an hour of paid work. Client errors other
        # than 429 stay fatal — retrying a bad request buys nothing.
        # The lock is not held across the socket: parallel experiments
        # share a client and must not queue on each other's round-trip.
        raw = b""
        try:
            for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
                try:
                    raw = self._transport(
                        f"{self.backend.base_url}/chat/completions", body, headers
                    )
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                    # A spent balance is a 429 that no backoff will change;
                    # three sleeps on it only delay the operator.
                    exhausted = exc.code == 429 and quota_exhausted(detail)
                    retryable = (exc.code == 429 and not exhausted) or exc.code >= 500
                    if not retryable or attempt == TRANSPORT_ATTEMPTS:
                        raise LLMUnavailable(
                            BlockedRecord(
                                missing_capability="llm_endpoint",
                                attempted=f"POST {self.backend.base_url}/chat/completions",
                                unlock_condition=(
                                    f"endpoint returned HTTP {exc.code}: {detail}"
                                    + RESUME_HINT
                                ),
                            )
                        ) from exc
                    time.sleep(TRANSPORT_BACKOFF_SECONDS * attempt)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    if attempt == TRANSPORT_ATTEMPTS:
                        raise LLMUnavailable(
                            BlockedRecord(
                                missing_capability="llm_endpoint",
                                attempted=f"POST {self.backend.base_url}/chat/completions",
                                unlock_condition=f"endpoint is reachable: {exc}" + RESUME_HINT,
                            )
                        ) from exc
                    time.sleep(TRANSPORT_BACKOFF_SECONDS * attempt)
        except Exception:
            with self._lock:
                self.ledger.abort_call()
            raise
        elapsed = time.time() - started
        artifact = _artifact(
            _canonical_request(
                self.backend,
                messages,
                purpose,
                logprobs=logprobs,
                scientific=scientific,
            ),
            raw,
            elapsed,
            digest,
        )

        with self._lock:
            if path.is_file():
                # Another worker recorded the same request while this one
                # was on the wire. Keep the first artifact; do not charge twice.
                self.ledger.abort_call()
                return _completion_from_artifact(
                    json.loads(path.read_text(encoding="utf-8")), path, mode="replay"
                )
            self.ledger.commit_call(purpose, artifact["usage"]["total_tokens"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return _completion_from_artifact(artifact, path, mode="live")

    def doctor(self) -> dict[str, Any]:
        """Ask the endpoint which model ids it serves, Argus `--doctor` style."""
        url = f"{self.backend.base_url}/models"
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.backend.api_key}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            return {
                "reachable": False,
                "configured_model": self.backend.model,
                "served": [],
                "configured_model_is_served": False,
                "reason": str(exc),
            }
        served = sorted(str(item.get("id")) for item in payload.get("data") or [])
        return {
            "reachable": True,
            "configured_model": self.backend.model,
            "served": served,
            "configured_model_is_served": self.backend.model in served,
            "reason": (
                "configured model is served"
                if self.backend.model in served
                else f"{self.backend.model!r} is not in the served catalog"
            ),
        }


def _http_post(url: str, body: bytes, headers: dict[str, str]) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _nll_summary(choice: dict[str, Any]) -> dict[str, Any]:
    """Negative log-likelihood of the visible completion, from token logprobs.

    OpenAI-compatible shape: choices[0].logprobs.content is a list of
    per-token records with a `logprob` field, covering the visible content
    (never the reasoning tokens). An endpoint that ignores the request
    option returns no such list, and that absence is recorded as
    `present: false` rather than as a zero surprise.
    """
    records = (choice.get("logprobs") or {}).get("content")
    if not isinstance(records, list) or not records:
        return {"present": False, "tokens": 0, "mean_nll": None}
    values = [
        -float(item["logprob"])
        for item in records
        if isinstance(item, dict) and item.get("logprob") is not None
    ]
    if not values:
        return {"present": False, "tokens": 0, "mean_nll": None}
    return {
        "present": True,
        "tokens": len(values),
        "mean_nll": round(sum(values) / len(values), 8),
        "sum_nll": round(sum(values), 6),
    }


def _artifact(
    request: dict[str, Any], raw: bytes, elapsed: float, digest: str
) -> dict[str, Any]:
    text = raw.decode("utf-8", "replace")
    payload = json.loads(text)
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "request": request,
        "request_digest": digest,
        "response_raw": text,
        "response_digest": content_digest(text),
        "content": str(message.get("content") or ""),
        "finish_reason": str(choice.get("finish_reason") or "unknown"),
        "model": str(payload.get("model") or request["model"]),
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        "nll": _nll_summary(choice),
        "elapsed_seconds": round(elapsed, 3),
    }


def _completion_from_artifact(
    artifact: dict[str, Any], path: Path, *, mode: str
) -> Completion:
    stored = str(artifact.get("response_raw") or "")
    if content_digest(stored) != artifact.get("response_digest"):
        raise LLMUnavailable(
            BlockedRecord(
                missing_capability="llm_artifact_integrity",
                attempted=f"replay {path.name}",
                unlock_condition="the recorded response bytes no longer match their"
                " digest; delete the artifact and record it again",
            )
        )

    usage = artifact.get("usage") or {}
    # Artifacts recorded before the nll block existed replay with it absent,
    # which is exactly what was true of those calls.
    nll = artifact.get("nll") or {"present": False, "tokens": 0, "mean_nll": None}
    return Completion(
        text=str(artifact.get("content") or ""),
        model=str(artifact.get("model") or ""),
        request_digest=str(artifact.get("request_digest") or ""),
        digest=str(artifact.get("response_digest") or ""),
        artifact_uri=path.resolve().as_uri(),
        finish_reason=str(artifact.get("finish_reason") or "unknown"),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        mode=mode,
        mean_nll=nll.get("mean_nll"),
        nll_tokens=int(nll.get("tokens") or 0),
    )
