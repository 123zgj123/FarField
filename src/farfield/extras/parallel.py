"""Concurrent jobs whose scientific fate does not depend on each other.

A mission's cards are independent experiments once the jump landings are
fixed: generating candidate k does not need candidate k+1's text, one
card's survey does not wait on another's probe, and a pairwise debate
does not need the other pairings' verdicts before it can be judged.
This module runs those jobs on a thread pool.

Ordering that *is* scientific stays serial here by construction:

- workers return in submission order, so Elo is applied in the same pair
  order a serial tournament would have used
- the event stream is yielded by the caller in jump / candidate order
- diagnosis still precedes the probe that implements it; those two share
  a card and are not concurrent with each other

`workers=1` is a pure sequential map, so tests that pin prompt bytes or
fake-client call order can opt out without a second code path.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence, TypeVar

DEFAULT_WORKERS = 8
ENV_WORKERS = "FARFIELD_PARALLEL"

T = TypeVar("T")
R = TypeVar("R")


def resolve_workers(value: int | None = None) -> int:
    """Positive worker count. Explicit argument wins; else the env; else 8."""
    if value is not None:
        return max(1, int(value))
    raw = (os.environ.get(ENV_WORKERS) or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return DEFAULT_WORKERS
    return DEFAULT_WORKERS


def map_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int,
) -> list[R]:
    """Apply `fn` to every item. Order of results matches order of items."""
    rows: Sequence[T] = list(items)
    if not rows:
        return []
    n = max(1, int(workers))
    if n == 1 or len(rows) == 1:
        return [fn(item) for item in rows]
    with ThreadPoolExecutor(max_workers=min(n, len(rows))) as pool:
        return list(pool.map(fn, rows))
