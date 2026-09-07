"""Frozen workspace inputs and deterministic, explicitly exploratory summaries."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from ..world import WorldFixture, load_fixture, digest_files


def is_attested(fixture: WorldFixture) -> bool:
    return bool(
        fixture.role == "world"
        and fixture.provenance in {"", "derived", "harvested"}
        and fixture.source and fixture.retrieved_at and fixture.digest
        and (fixture.root / "manifest.json").is_file()
        and json.loads((fixture.root / "manifest.json").read_text()).get("digest")
        and digest_files(fixture.root, fixture.files) == fixture.digest
    )


def bind_fixture(workspace: Path, world: Any, world_id: str = "") -> WorldFixture:
    """Resolve a catalog id or fixture object, verify it, then freeze a local copy."""
    from ..freeze import default_catalog

    if isinstance(world, str) or world is None:
        wid = str(world or world_id)
        if not wid or Path(wid).name != wid or wid in {".", ".."}:
            raise ValueError("world must be a catalog id")
        fixture = load_fixture(default_catalog() / wid)
    else:
        fixture = load_fixture(Path(world.root))
        if fixture.digest != world.digest or fixture.id != world.id:
            raise ValueError("world identity differs from its manifest")
    manifest = json.loads((fixture.root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("digest"):
        raise ValueError("explicit world requires a recorded digest; freeze the input first")
    if world_id and fixture.id != world_id:
        raise ValueError("requested world id differs from its manifest")
    if Path(fixture.id).name != fixture.id or fixture.id in {".", ".."}:
        raise ValueError("invalid world id in manifest")
    dest = workspace / "worlds" / fixture.id
    if dest.exists():
        existing = load_fixture(dest)
        if existing.to_dict() != fixture.to_dict():
            raise ValueError("workspace world differs from the requested freeze; use a new workspace")
        return existing
    for name in fixture.files:
        src = fixture.root / name
        if not src.resolve().is_relative_to(fixture.root.resolve()):
            raise ValueError("world artifact escapes its freeze")
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(fixture.to_dict(), indent=2) + "\n")
    return load_fixture(dest)


def summarize(fixture: WorldFixture) -> dict[str, Any]:
    """Describe available bytes. These numbers do not test a topic's hypothesis."""
    out: dict[str, Any] = {"schema": fixture.schema, "files": list(fixture.files)}
    for rel in fixture.files:
        if not rel.endswith(".json") or rel in {"origin.json", "acquisition.json", "provenance.json"}:
            continue
        payload = json.loads((fixture.root / rel).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        if fixture.schema == "numeric_table" and isinstance(payload.get("rows"), list):
            rows = payload["rows"]
            out["n_rows"] = len(rows)
            numeric: dict[str, list[float]] = {}
            for row in rows:
                fields = row.items() if isinstance(row, dict) else enumerate(row) if isinstance(row, list) else ()
                for key, value in fields:
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                        numeric.setdefault(str(key), []).append(float(value))
            out["numeric_columns"] = {
                key: {"n": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}
                for key, values in numeric.items()
            }
        for field in ("nodes", "edges", "traces", "updates", "records", "tokens"):
            if isinstance(payload.get(field), list):
                out[f"n_{field}"] = len(payload[field])
    if fixture.schema == "fasta":
        sequences = [fixture.root / rel for rel in fixture.files if rel.endswith((".fasta", ".fa", ".fna"))]
        lines = [line for path in sequences for line in path.read_text().splitlines() if line.strip()]
        out["n_sequences"] = sum(line.startswith(">") for line in lines)
        out["n_bases"] = sum(len(line.strip()) for line in lines if not line.startswith(">"))
    return out
