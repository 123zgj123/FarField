"""Is the dependent variable measuring the object the claim names?

The cwm-iclr2027 mission spent three WORLD probes on a claim about
*false acceptance* — an update the gate accepted that the oracle
rejects — with a script whose measure never read the oracle field
(`task_return`). Every design was "uninformative"; the honest verdict
was that the quantity was blind to its object.

The gate here is a placebo on the oracle alone. Shuffle the oracle field
across records (everything else untouched), rerun the registered script,
and compare: a measure of an oracle-relative quantity must move when the
oracle is scrambled. If neither arm moves, the measure is blind to the
oracle — `dv_blind` — and the run is booked as `object_absent`, not as
an experiment that failed to discriminate. That keeps a blind measure
from making the plan "executable", from spending the experiment budget,
and from ever producing a `supports` about a quantity it does not read.

The check applies only when the claim's quantity is oracle-relative
(`oracle_dependent`): a claim about `accepted_fraction` legitimately
ignores the oracle.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .world import WorldFixture

# Per schema: where the oracle lives and which observables depend on it.
_ORACLE_FIELD: dict[str, tuple[str, str]] = {
    # schema -> (collection key, per-record oracle field)
    "program_state": ("updates", "task_return"),
    "labeled_traces": ("traces", "label"),
}

_ORACLE_WORDS = re.compile(
    r"(false[\s_-]*accept|false[\s_-]*positive|oracle|ground[\s_-]*truth|hidden[\s_-]*test|"
    r"held[\s_-]*out|task[\s_-]*return|task[\s_-]*success|resolved|unresolved|"
    r"mean[\s_-]*accepted[\s_-]*return|label|misjudg|wrongly accepted|regret)",
    re.IGNORECASE,
)

EPS = 1e-9


def oracle_field(schema: str) -> tuple[str, str] | None:
    return _ORACLE_FIELD.get(str(schema or ""))


def oracle_dependent(*texts: str) -> bool:
    """The claim's quantity is defined relative to an oracle / label."""
    blob = " ".join(str(t or "") for t in texts)
    return bool(_ORACLE_WORDS.search(blob))


def materialize_oracle_placebo(
    fixture: WorldFixture, dest: Path, *, seed: int = 0
) -> WorldFixture | None:
    """Copy the freeze with only the oracle field permuted across records.

    Returns None when the schema has no oracle field or the payload has
    fewer than two distinct oracle values (a shuffle would be identity).
    """
    where = oracle_field(fixture.schema)
    if where is None:
        return None
    collection, field = where
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    shuffled_any = False
    for name in fixture.files:
        src = fixture.root / name
        out = dest / name
        if not src.is_file():
            continue
        if name == "world.json":
            payload = json.loads(src.read_text(encoding="utf-8"))
            rows = payload.get(collection) if isinstance(payload, dict) else None
            if isinstance(rows, list) and rows:
                values = [row.get(field) for row in rows if isinstance(row, dict)]
                if len({json.dumps(v, sort_keys=True) for v in values}) >= 2:
                    rng = random.Random(int(seed))
                    permuted = list(values)
                    rng.shuffle(permuted)
                    # A permutation that happens to be identity is retried once.
                    if permuted == values:
                        rng.shuffle(permuted)
                    index = 0
                    for row in rows:
                        if isinstance(row, dict):
                            row[field] = permuted[index]
                            index += 1
                    shuffled_any = True
            out.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            continue
        out.write_bytes(src.read_bytes())
    if not shuffled_any:
        return None
    return WorldFixture(
        id=f"{fixture.id}-oracle-placebo",
        title=f"{fixture.title} (oracle field {field} permuted)",
        source=f"oracle placebo of {fixture.id}",
        retrieved_at=fixture.retrieved_at,
        digest="",
        files=fixture.files,
        root=dest,
        domains=fixture.domains,
        role="placebo",
        schema=fixture.schema,
        load_hint=fixture.load_hint,
    )


def dv_blind(
    real: tuple[float, float],
    placebo: tuple[float, float],
    *,
    eps: float = EPS,
) -> bool:
    """Neither arm moved when the oracle was scrambled."""
    (rt, rc), (pt, pc) = real, placebo
    return abs(float(rt) - float(pt)) <= eps and abs(float(rc) - float(pc)) <= eps


def blind_reason(schema: str) -> str:
    where = oracle_field(schema)
    field = where[1] if where else "the oracle field"
    return (
        f"dv_blind: the claim names an oracle-relative quantity, but permuting "
        f"`{field}` across every record left both arms unchanged — the measure "
        f"never reads the oracle. Define the dependent variable on `{field}` "
        f"(e.g. accepted-but-oracle-failed) before another design."
    )


def summarize(record: dict[str, Any]) -> str:
    return (
        f"oracle placebo: real ({record.get('treatment')}, {record.get('control')}) vs "
        f"permuted ({record.get('placebo_treatment')}, {record.get('placebo_control')})"
    )
