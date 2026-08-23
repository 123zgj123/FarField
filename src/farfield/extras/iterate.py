"""Evidence-driven experiment redesign.

A high-level researcher does not freeze the first protocol and hope. They
run it, name the bottleneck, and change the *experiment* — not the claim
to match the numbers. Argus's loop is plan → build → experiment → read
evidence → review → retry; FarField already has bounded roles for those
steps. What it lacked was a second attempt when the first test failed to
speak.

The stop rule is the evaluation function:

- `supports` or `weakens` is informative. Stop. Redesigning after a
  weakening to hunt a support is p-hacking.
- a sandbox refusal or a crash is an *implementation* bottleneck: rewrite
  the script, keep the pre-registration.
- `uninformative` is a *method* bottleneck: pre-register a sharper
  two-arm design. Quote the failed experiment text. Never quote the two
  numbers — they are withheld so the redesign cannot chase a sign.
- a generated world that cannot execute is a *world_model* bottleneck:
  adapt the bound world in place for this same registration. Do not
  spawn a sidecar fixture. An attested freeze is never rewritten.
- a cap (default two extra attempts) is the time budget. The model does
  not vote for more experiments. Elo does not vote either.

Self-evolution of a hypothesis is in-mission idea refine (`refine_card`).
This module only iterates the test of a card that already entered.
Each experiment iteration constructs or adapts the world that
registration needs. Diagnose is the verb; the world is the object.
"""

from __future__ import annotations

from typing import Any, Mapping

from .explore import GRAPH_PAIR_MARKS


EXPERIMENT_AUTO = -1
EXPERIMENT_AUTO_CAP = 2
DEFAULT_EXPERIMENT_ROUNDS = EXPERIMENT_AUTO

INFORMATIVE = frozenset({"supports", "weakens"})


def resolve_experiment_rounds(requested: int) -> tuple[int, str]:
    """Map the user-facing knob to (extra attempts, mode).

    `-1` means auto: up to `EXPERIMENT_AUTO_CAP` redesigns after the first
    registered run. `0` is the one-shot used by tests. A positive integer
    is an explicit extra-attempt cap.
    """
    n = int(requested)
    if n < 0:
        return EXPERIMENT_AUTO_CAP, "auto"
    return max(0, n), "fixed"


WORLD_UNUSABLE_MARKS = (
    "world.json",
    "data/world",
    "symbolic_trace",
)


def world_unusable(error: str) -> bool:
    """True when the crash is the bound constructed world, not the script."""
    text = str(error or "").lower()
    if any(mark in text for mark in WORLD_UNUSABLE_MARKS):
        return True
    if "keyerror" in text and any(
        key in text for key in ("states", "transitions", "invariant")
    ):
        return True
    if "jsondecode" in text or "json.decoder" in text:
        return True
    return False


def classify_bottleneck(
    *,
    refused: str | None = None,
    status: str | None = None,
    verdict: str | None = None,
    error: str | None = None,
    world_role: str | None = None,
    last_action: str | None = None,
) -> str | None:
    """Attested bottleneck, never a model self-report.

    `None` means the experiment discriminated. `implementation` means the
    script did not produce two arms. `method` means it ran and the arms
    did not separate, or the diagnosis itself could not be registered.
    `world_model` means the constructed experimental world failed to
    execute; adapt it in place. An attested freeze never takes this path.
    """
    if verdict in INFORMATIVE:
        return None
    if refused == "diagnosis":
        return "method"
    if refused == "world":
        return "world"
    generated = str(world_role or "") == "generated"
    crashed = bool(status and status != "ran")
    if (
        generated
        and crashed
        and world_unusable(error or "")
        and last_action != "adapt_world"
    ):
        return "world_model"
    if refused == "probe" or crashed:
        return "implementation"
    if verdict == "uninformative":
        return "method"
    return "implementation"


GRAPH_KILL_MARKS = GRAPH_PAIR_MARKS


def pipeline_bottleneck(record: Mapping[str, Any]) -> str:
    """Where this card stopped, for Verifier memory. Not a value score.

    Experiment bottlenecks stay `implementation` / `method` on the retry
    loop. This name is the card-level attribution the next mission logs.
    """
    if record.get("prior_kills"):
        return "prior"
    killed_by = [str(item) for item in (record.get("killed_by") or [])]
    if record.get("killed") or killed_by:
        lowered = " ".join(killed_by).lower()
        if any(mark in lowered for mark in GRAPH_KILL_MARKS):
            return "graph"
        return "predicate"
    experiment = str(record.get("experiment_bottleneck") or "")
    if experiment == "implementation":
        return "probe_implementation"
    if experiment == "method":
        return "probe_method"
    if experiment == "world_model":
        return "world_model"
    if record.get("world_incompatible") and not record.get("generated_world"):
        return "world"
    if record.get("host_ok") is False:
        return "host"
    if record.get("verdict") in INFORMATIVE:
        return "informative"
    if record.get("verdict") == "uninformative":
        return "probe_method"
    return "unresolved"


def decide_retry(
    *,
    attempt: int,
    cap: int,
    bottleneck: str | None,
) -> dict[str, Any]:
    """Whether to spend another experiment attempt on this card.

    `attempt` is the 0-based attempt that just finished. `cap` is how many
    extra attempts are allowed after the first. The rule reads only the
    attested bottleneck. It does not ask the model how many more tests it
    wants, and it does not use Elo.
    """
    cap = max(0, int(cap))
    attempt = max(0, int(attempt))
    if bottleneck is None:
        return {
            "continue": False,
            "action": "stop",
            "reason": "informative",
            "detail": (
                "the probe discriminated (supports or weakens); "
                "redesigning after that would chase a preferred sign"
            ),
        }
    if bottleneck == "world":
        return {
            "continue": False,
            "action": "stop",
            "reason": "world_incompatible",
            "bottleneck": "world",
            "detail": (
                "no freeze matched and no experimental world was constructed; "
                "rewriting the script cannot create a matching world"
            ),
        }
    if attempt >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "bottleneck": bottleneck,
            "detail": (
                f"attempt {attempt + 1} of {cap + 1} still {bottleneck}"
            ),
        }
    if bottleneck == "world_model":
        return {
            "continue": True,
            "action": "adapt_world",
            "reason": "world_model",
            "bottleneck": "world_model",
            "detail": (
                "the constructed world failed execution; adapt that same "
                "world in place, keep the claim and the registration"
            ),
        }
    if bottleneck == "implementation":
        return {
            "continue": True,
            "action": "rewrite_probe",
            "reason": "implementation",
            "bottleneck": "implementation",
            "detail": (
                "the script failed before two arms existed; rewrite it, "
                "keep the pre-registration"
            ),
        }
    return {
        "continue": True,
        "action": "rewrite_diagnosis",
        "reason": "method",
        "bottleneck": "method",
        "detail": (
            "the design did not discriminate; pre-register a sharper "
            "experiment, not a new claim"
        ),
    }
