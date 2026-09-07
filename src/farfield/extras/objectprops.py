"""What a world *is*, beyond its schema — read off the bytes, matched to
what the claim *needs*, read off the claim.

Schema match plus domain words bound a claim about "a fixed evaluator
ranking candidate programs" to a Live-SWE self-modification history
whose validators the agent rewrites: same schema (`program_state`), same
generic words (`program`, `executable`), opposite object. Words are not
the object. These properties are:

    has_oracle          an outcome field independent of the gate varies
    validators_mutable  some update rewrites a validator (the agent owns
                        part of its own gate)
    gate_endogenous     synonym of validators_mutable for program_state:
                        acceptance is decided by something the program
                        can write
    acceptance_varies   both accepted and rejected updates exist
    self_modifying      the program writes its own cells / creates tools

Each is computed deterministically from the frozen payload, written on
the manifest at freeze time (`object_properties`), and recomputed on the
fly for older manifests. A claim states requirements through a small
lexicon (`required_properties`); a world that contradicts a stated
requirement is not this claim's object, whatever its schema says.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROPERTY_NAMES = (
    "has_oracle",
    "validators_mutable",
    "gate_endogenous",
    "acceptance_varies",
    "self_modifying",
)


def object_properties(schema: str, payload: dict[str, Any]) -> dict[str, bool]:
    """Properties of one world payload. Empty for schemas without them."""
    kind = str(schema or "")
    if kind == "program_state":
        updates = [u for u in (payload.get("updates") or []) if isinstance(u, dict)]
        if not updates:
            return {}
        returns = {json.dumps(u.get("task_return"), sort_keys=True) for u in updates}
        accepted = {bool(u.get("accepted")) for u in updates}
        mutable = any(u.get("validator_writes") for u in updates)
        oracle = payload.get("oracle") if isinstance(payload.get("oracle"), dict) else {}
        # An outcome the gate itself computed (an evaluator's score) is not
        # an oracle: the world says so in `oracle.independent_of_gate`.
        independent = bool(oracle.get("independent_of_gate", True))
        return {
            "has_oracle": len(returns) >= 2 and independent,
            "validators_mutable": mutable,
            "gate_endogenous": mutable,
            "acceptance_varies": len(accepted) == 2,
            "self_modifying": any(u.get("writes") for u in updates),
        }
    if kind == "labeled_traces":
        traces = [t for t in (payload.get("traces") or []) if isinstance(t, dict)]
        if not traces:
            return {}
        labels = {str(t.get("label")) for t in traces}
        return {
            "has_oracle": len(labels) >= 2,
            "validators_mutable": False,
            "gate_endogenous": False,
            "acceptance_varies": False,
            "self_modifying": any(t.get("created_tools") for t in traces),
        }
    return {}


def properties_of_path(schema: str, world_json: Path) -> dict[str, bool]:
    try:
        payload = json.loads(Path(world_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return object_properties(schema, payload) if isinstance(payload, dict) else {}


# claim text -> required property values. Order matters only for readability.
_REQUIREMENT_LEXICON: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (
        re.compile(
            r"(fixed|frozen|external|static|immutable)[\s-]+(evaluator|validator|verifier|judge|gate|test suite)",
            re.IGNORECASE,
        ),
        "validators_mutable",
        False,
    ),
    (
        re.compile(
            r"(rewrit|modif|mutat|edit|updat|evolv|co-?adapt|learn)\w*\s+(its|their|own|the agent'?s)?\s*(own\s+)?"
            r"(validator|evaluator|verifier|test|gate|judge)s?|"
            r"(mutable|self-?written|self-?modif\w*|continuously[\s-]+updated|evolving)[\s-]+"
            r"(validator|evaluator|verifier|gate|test)s?|validator[\s_-]*(writes|mutation|updates)",
            re.IGNORECASE,
        ),
        "validators_mutable",
        True,
    ),
    (
        re.compile(
            r"(false[\s_-]*accept|oracle|ground[\s_-]*truth|hidden[\s_-]*test|held[\s_-]*out|task[\s_-]*return|resolved)",
            re.IGNORECASE,
        ),
        "has_oracle",
        True,
    ),
    (
        re.compile(
            r"(self-?improv|self-?modif|recursive self|rewrites? (its|their) own (code|cells|harness|tools)|creates? (its|their) own tools)",
            re.IGNORECASE,
        ),
        "self_modifying",
        True,
    ),
)


def required_properties(*texts: str) -> dict[str, bool]:
    """Property requirements a claim/topic states. Contradictory statements
    (both fixed and mutable validators) cancel: nothing is required."""
    blob = " ".join(str(t or "") for t in texts)
    wanted: dict[str, set[bool]] = {}
    for pattern, name, value in _REQUIREMENT_LEXICON:
        if pattern.search(blob):
            wanted.setdefault(name, set()).add(value)
    out: dict[str, bool] = {}
    for name, values in wanted.items():
        if len(values) == 1:
            out[name] = next(iter(values))
    return out


def properties_compatible(required: dict[str, bool], actual: dict[str, bool]) -> tuple[bool, list[str]]:
    """Every stated requirement the world can answer must hold.

    A world with no properties (schema without them) answers nothing and
    is not contradicted. A world that has the property and disagrees is
    the wrong object.
    """
    if not required or not actual:
        return True, []
    conflicts = [
        f"{name}: claim needs {value}, world has {actual[name]}"
        for name, value in required.items()
        if name in actual and bool(actual[name]) != bool(value)
    ]
    return not conflicts, conflicts


def property_overlap(required: dict[str, bool], actual: dict[str, bool]) -> int:
    """How many stated requirements this world positively satisfies."""
    return sum(1 for name, value in required.items() if name in actual and bool(actual[name]) == bool(value))
