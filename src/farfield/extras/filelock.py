"""Exclusive flock for shared JSON stores.

Cross-mission writers (trajectory log, research state, judge bench, routing
policy) are whole-file read-modify-write. Concurrent missions must serialize
those updates or the last writer silently drops the other mission's record.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock named after `path` until the block exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
