"""The experimental world: attested fixtures, not invented datasets.

Literature already works this way. `RecordingFeed` freezes every paper a
mission consumes *before* prior-overlap is computed. Experiments did not:
the probe prompt said "invent a small constructed dataset", the sandbox
had no `data/`, and a SYNTHETIC `supports` still climbed to corroborated.

A world is a directory of files with a manifest (id, source, digest). The
mission copies it into the probe cwd as `data/` *before* `experiment.py`
runs. The script may read those bytes; it may not fetch, and it may not
invent a stand-in and still call the result corroboration.

Kinds:

- `SYNTHETIC` — the script invented its own data. Arithmetic is a
  coherence check. It can weaken a claim. It cannot corroborate.
- `GENERATED` — the mission constructed the experimental world for this
  iteration (no matching freeze). Readable, can weaken, cannot corroborate.
- `WORLD` — a fixture was bound and the script actually reads `data/`.
  Only this kind may climb the ladder.

Opening the network inside the sandbox would let the author pick a
dataset after seeing the claim. That is not a real world; it is a
larger invented one. Real means *already retrieved, hashed, and frozen*.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .prior import content_tokens
from .schemas import OBJECT_SCHEMA, SCHEMAS

# Claim families that no schema can attest yet. Auto must not substitute
# another schema for these. Formula claims left this table when
# `symbolic_trace` gained a freeze slicer: they now bind an attested
# learned automaton or miss honestly like everyone else.
_INCOMPATIBLE_FAMILIES = {
    "io": frozenset(
        {
            "external-memory",
            "externalmemory",
            "disk-based",
            "i/o",
            "io-complexity",
            "buffer-pool",
            "b-tree",
            "btree",
        }
    ),
}

KIND_SYNTHETIC = "SYNTHETIC"
KIND_WORLD = "WORLD"
WORLD_KINDS = frozenset({KIND_WORLD, "REAL", "FIXTURE"})

_DATA_REF = re.compile(
    r"""['\"]data/|Path\(\s*['\"]data['\"]|Path\(\s*['\"]data/"""
)


@dataclass(frozen=True)
class WorldRequirement:
    """The scientific object a claim must acquire and then measure.

    Binding, construction, levers, and the probe measure are the same
    predicate: this object, not a schema-shaped substitute. WORLD only
    when an attested fixture *is* this object. A miss is
    WORLD_INCOMPATIBLE — never a silent substitute dataset.
    """

    object_type: str
    schema: str = ""
    operations: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    scale: int = 0
    environment: tuple[str, ...] = ()
    # Claim/topic content tokens. Schema match is necessary; a non-empty
    # overlap with fixture.domains is what makes it the same scientific
    # object. Empty means "no claim-side tokens yet" and is not a license
    # to bind by schema alone when the fixture itself has domain tags.
    anchors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_type": self.object_type,
            "schema": self.schema,
            "operations": list(self.operations),
            "metrics": list(self.metrics),
            "scale": self.scale,
            "environment": list(self.environment),
            "anchors": list(self.anchors),
        }


@dataclass(frozen=True)
class WorldFixture:
    id: str
    title: str
    source: str
    retrieved_at: str
    digest: str
    files: tuple[str, ...]
    root: Path
    domains: tuple[str, ...] = ()
    role: str = "world"
    schema: str = ""
    load_hint: str = ""
    source_url: str = ""
    source_digest: str = ""
    slice_rule: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "retrieved_at": self.retrieved_at,
            "digest": self.digest,
            "files": list(self.files),
            "domains": list(self.domains),
            "role": self.role,
            "schema": self.schema,
            "load_hint": self.load_hint,
        }
        if self.source_url:
            payload["source_url"] = self.source_url
        if self.source_digest:
            payload["source_digest"] = self.source_digest
        if self.slice_rule:
            payload["slice_rule"] = dict(self.slice_rule)
        payload["capabilities"] = sorted(fixture_capabilities(self))
        return payload


def incompatible_family(object_type: str) -> bool:
    """True when no schema can attest this claim family yet.

    The one place that answers the question — mission binding and
    requirement matching must not keep their own lists.
    """
    return str(object_type or "") in _INCOMPATIBLE_FAMILIES


def reads_world_data(source: str, world: Any = None) -> bool:
    """True when the script reads the bound scientific object under `data/`.

    Naming `Path('data') / 'seed.json'` is not enough: that still invents
    the instance. When `world` is given, the script must name a payload
    file from the fixture (not seed/origin/manifest).
    """
    if not _DATA_REF.search(source or ""):
        return False
    if world is None:
        return True
    skip = {"seed.json", "origin.json", "manifest.json"}
    payload = [
        str(name)
        for name in (getattr(world, "files", ()) or [])
        if Path(str(name)).name not in skip
    ]
    if not payload:
        return True
    text = source or ""
    return any(Path(name).name in text for name in payload)


def attested_freeze(world: Any) -> bool:
    """True when `world` is a catalog freeze, not a constructed stand-in."""
    if world is None:
        return False
    role = str(getattr(world, "role", "") or "").strip().lower()
    return role not in {"generated", "placebo", "synthetic"}


def world_attested(outcome: dict[str, Any] | None) -> bool:
    """Whether this outcome may climb. Missing kind is SYNTHETIC."""
    kind = str((outcome or {}).get("probe_kind") or KIND_SYNTHETIC).upper()
    return kind in WORLD_KINDS


def digest_files(root: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for name in files:
        payload = (root / name).read_bytes()
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(payload)
        h.update(b"\0")
    return h.hexdigest()


def load_fixture(directory: Path) -> WorldFixture:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json in {directory}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("world manifest must be an object")
    files = [
        str(name)
        for name in (payload.get("files") or [])
        if str(name) and str(name) != "manifest.json"
    ]
    if not files:
        files = sorted(
            p.name
            for p in directory.iterdir()
            if p.is_file() and p.name != "manifest.json"
        )
    for name in files:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"world file missing: {directory / name}")
    files = sorted(files)
    recorded = str(payload.get("digest") or "").strip()
    digest = digest_files(directory, files)
    if recorded and recorded != digest:
        raise ValueError(
            f"world {payload.get('id')!r} digest mismatch: "
            f"manifest {recorded[:12]}… vs files {digest[:12]}…"
        )
    role = str(payload.get("role") or "world").strip().lower() or "world"
    return WorldFixture(
        id=str(payload.get("id") or directory.name).strip(),
        title=str(payload.get("title") or directory.name).strip(),
        source=str(payload.get("source") or "").strip(),
        retrieved_at=str(payload.get("retrieved_at") or "").strip(),
        digest=digest,
        files=tuple(files),
        root=directory,
        domains=tuple(
            str(item).strip()
            for item in (payload.get("domains") or [])
            if str(item).strip()
        ),
        role=role,
        schema=str(payload.get("schema") or "").strip(),
        load_hint=str(payload.get("load_hint") or "").strip(),
        source_url=str(payload.get("source_url") or "").strip(),
        source_digest=str(payload.get("source_digest") or "").strip(),
        slice_rule=_slice_rule(payload.get("slice_rule")),
    )


def _slice_rule(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict) and raw:
        return {str(key): value for key, value in raw.items()}
    text = str(raw or "").strip()
    if not text:
        return None
    return {"kind": text}


def worlds_dir(root: Path) -> Path:
    """Frozen attested fixtures: `worlds/<id>/manifest.json`."""
    return Path(root) / "worlds"


def wishlist_path(root: Path) -> Path:
    """Unmatched requirements. Runtime ledger, not an attested fixture."""
    return Path(root) / "var" / "world_wishlist.json"


def load_wishlist(root: Path) -> dict[str, Any]:
    path = wishlist_path(root)
    if not path.is_file():
        return {"entries": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"entries": []}
    return payload if isinstance(payload, dict) else {"entries": []}


def in_mission_constructable(req: WorldRequirement | None) -> bool:
    """True when this object has a same-family in-mission constructor.

    A registered schema may be built as GENERATED for simulation.
    That still cannot corroborate. `io` has no schema — constructing a
    formula automaton for it would be a substitute object.
    """
    if req is None:
        return False
    if incompatible_family(req.object_type):
        return False
    schema = req.schema or OBJECT_SCHEMA.get(req.object_type, "")
    family = schema_family(schema) or schema
    return family in SCHEMAS or req.object_type in OBJECT_SCHEMA


def refine_requirement(
    req: WorldRequirement | None,
    *texts: str,
) -> WorldRequirement | None:
    """Fill an empty requirement from claim, prediction, or diagnosis.

    A named object_type / schema wins. Otherwise infer from the texts
    so construction can follow the registered experiment's object.
    """
    if req is not None and (req.object_type or req.schema):
        if req.schema and not req.object_type:
            spec = SCHEMAS.get(schema_family(req.schema) or req.schema)
            if spec is not None:
                return WorldRequirement(
                    object_type=spec.object_type,
                    schema=spec.name,
                    operations=req.operations,
                    environment=req.environment,
                    anchors=req.anchors,
                )
        if req.object_type and not req.schema:
            mapped = OBJECT_SCHEMA.get(req.object_type, "")
            if mapped:
                return WorldRequirement(
                    object_type=req.object_type,
                    schema=mapped,
                    operations=req.operations,
                    environment=req.environment,
                    anchors=req.anchors,
                )
        return req
    return infer_requirement(*texts)


def record_world_wishlist(
    root: Path,
    requirements: list[dict[str, Any]],
    *,
    topic: str = "",
    seen_at: str = "",
) -> Path | None:
    """Merge this mission's unmatched `WorldRequirement`s into the wishlist.

    Runtime file under `var/`. The next mission may freeze a named
    public source. Finished WORLD_INCOMPATIBLE verdicts stay as recorded.
    """
    rows = [req for req in requirements if isinstance(req, dict)]
    if not rows:
        return None
    path = wishlist_path(root)
    entries: dict[str, dict[str, Any]] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        for row in existing.get("entries") or []:
            if isinstance(row, dict) and row.get("key"):
                entries[str(row["key"])] = row
    stamp = str(seen_at or "").strip() or date.today().isoformat()
    for req in rows:
        object_type = str(req.get("object_type") or "").strip()
        schema = str(req.get("schema") or "").strip()
        if not object_type and not schema:
            continue
        key = f"{object_type}|{schema}"
        entry = entries.get(key) or {
            "key": key,
            "object_type": object_type,
            "schema": schema,
            "misses": 0,
            "operations": [],
        }
        entry["misses"] = int(entry.get("misses") or 0) + 1
        ops = set(str(op) for op in (entry.get("operations") or []))
        ops.update(str(op) for op in (req.get("operations") or []) if str(op).strip())
        entry["operations"] = sorted(ops)[:24]
        if topic:
            entry["last_topic"] = str(topic)[:200]
        entry["last_seen"] = stamp
        for field in (
            "named_instance",
            "lineage",
            "freeze_source",
            "freeze_url",
            "freeze_schema",
        ):
            value = str(req.get(field) or "").strip()
            if value:
                entry[field] = value[:400]
        cites = [
            str(item).strip()
            for item in (req.get("lineage_cite_ids") or [])
            if str(item).strip()
        ]
        if cites:
            entry["lineage_cite_ids"] = cites[:8]
        entries[key] = entry
    if not entries:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "unmatched world requirements from recorded missions; freeze a "
            "matching schema host-side, then new claims can bind it — this "
            "list never reopens finished verdicts"
        ),
        "entries": sorted(
            entries.values(), key=lambda row: (-int(row.get("misses") or 0), row["key"])
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_catalog(root: Path) -> dict[str, WorldFixture]:
    """Every `worlds/<id>/manifest.json` under `root`."""
    catalog: dict[str, WorldFixture] = {}
    worlds = worlds_dir(root)
    if not worlds.is_dir():
        return catalog
    for child in sorted(worlds.iterdir()):
        if child.name.startswith(".") or not child.is_dir():
            continue
        if not (child / "manifest.json").is_file():
            continue
        fixture = load_fixture(child)
        catalog[fixture.id] = fixture
    return catalog


# Named graphs are the same scientific family as undirected graphs.
# They keep their own schema name for loaders; matching uses the family.
_SCHEMA_FAMILY = {
    "undirected_named_graph": "undirected_graph",
}

# Same-schema instance collisions only. Cross-schema tokens are already
# refused by `requirement_compatible`'s family check — listing them here
# as well turned a mixed formula+genomic claim into WORLD_INCOMPATIBLE
# even after `symbolic_trace` was inferred. Schema hints (sketch, hash,
# cache) stay out: they are experiment verbs, not instance identity.
_FOREIGN_OBJECTS = {
    "text_stream": frozenset(
        {
            "agent",
            "distill",
            "distillation",
            "sft",
            "tool",
            "tooluse",
            "student",
            "llm",
            "chatbot",
            "agentcard",
            "jsonrpc",
            "a2a",
        }
    ),
    "undirected_graph": frozenset(
        {
            "agentcard",
            "a2a",
            "jsonrpc",
        }
    ),
    "fasta": frozenset(
        {
            "agent",
            "agentcard",
            "a2a",
            "tool",
        }
    ),
    "numeric_table": frozenset(
        {
            "agent",
            "agentcard",
            "a2a",
            "tool",
        }
    ),
    "labeled_traces": frozenset(
        {
            "pride",
            "austen",
            "phage",
            "fasta",
            "genome",
        }
    ),
    "symbolic_trace": frozenset(
        {
            "agent",
            "agentcard",
            "a2a",
            "jsonrpc",
            "authentication",
            "authorization",
            "distill",
            "distillation",
            "sft",
        }
    ),
}


def schema_family(name: str) -> str:
    """Canonical family for matching. Named graphs sit with undirected graphs."""
    return _SCHEMA_FAMILY.get(str(name or ""), str(name or ""))


def fixture_capabilities(fixture: WorldFixture) -> frozenset[str]:
    """What this schema can attest — not what the instance *is*.

    Instance `domains` used to be mixed in here, so sketch/hash/cache on a
    novel became capabilities and then eligibility. Capabilities rank and
    gate operations; they do not bind.
    """
    spec = SCHEMAS.get(schema_family(fixture.schema)) or SCHEMAS.get(fixture.schema)
    caps = set(spec.capabilities) if spec is not None else set()
    if fixture.schema:
        caps.add(fixture.schema)
        caps.add(schema_family(fixture.schema))
    return frozenset(caps)


def _anchors_of(wanted: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(wanted))[:64]


def infer_requirement(*texts: str) -> WorldRequirement | None:
    """Read object type off the claim and topic. The mechanism does not vote.

    A distant concept supplies the intervention, not a new field. Passing
    the mechanism into this function would let a min-cut jump turn an
    Agent Card claim into a SNAP graph world.
    """
    blob = " ".join(str(item or "") for item in texts).lower()
    wanted = content_tokens(blob)
    if any(
        needle in blob
        for needle in (
            "external memory",
            "external-memory",
            "i/o complexity",
            "io complexity",
            "disk-based",
            "buffer pool",
        )
    ):
        return WorldRequirement(
            object_type="io",
            environment=("io",),
            anchors=_anchors_of(wanted),
        )
    if any(
        needle in blob
        for needle in (
            "smt ",
            "smt solver",
            "sat solver",
            "formal verification",
            "model checking",
            "theorem prover",
        )
    ) or blob.strip() in {"smt", "sat"}:
        trace = SCHEMAS["symbolic_trace"]
        return WorldRequirement(
            object_type=trace.object_type,
            schema=trace.name,
            operations=tuple(sorted(trace.capabilities & wanted)),
            environment=("formula",),
            anchors=_anchors_of(wanted),
        )
    scores = {
        spec.object_type: len(wanted & spec.hints) for spec in SCHEMAS.values()
    }
    best = max(scores, key=lambda key: (scores[key], key))
    if scores[best] <= 0:
        return None
    schema = OBJECT_SCHEMA[best]
    spec = SCHEMAS[schema]
    return WorldRequirement(
        object_type=best,
        schema=schema,
        operations=tuple(sorted(spec.capabilities & wanted)),
        anchors=_anchors_of(wanted),
    )


def requirement_tokens(req: WorldRequirement) -> frozenset[str]:
    """Claim-side tokens the fixture must scientifically match."""
    parts = list(req.anchors) + list(req.operations) + list(req.environment)
    return content_tokens(" ".join(parts)) if parts else frozenset()


def domain_hits(fixture: WorldFixture, wanted: frozenset[str]) -> int:
    """How many claim/topic tokens land on the fixture's domain tags."""
    if not wanted:
        return 0
    tags = content_tokens(" ".join(fixture.domains))
    hits = 0
    for token in wanted:
        if any(_tag_match(token, tag) for tag in tags):
            hits += 1
    return hits


def _tag_match(token: str, tag: str) -> bool:
    """Plural fold leaves `sketches` as `sketche`; tags stay `sketch`."""
    if token == tag:
        return True
    shorter, longer = (token, tag) if len(token) <= len(tag) else (tag, token)
    return len(shorter) >= 5 and longer.startswith(shorter)


def foreign_conflict(fixture: WorldFixture, wanted: frozenset[str]) -> bool:
    """True when the claim names a different scientific object.

    `agent` on an A2A automaton is instance identity, not foreign: own
    domain tags are subtracted before the check.
    """
    family = schema_family(fixture.schema)
    banned = _FOREIGN_OBJECTS.get(family) or _FOREIGN_OBJECTS.get(fixture.schema)
    if not banned or not wanted:
        return False
    own = content_tokens(" ".join(fixture.domains))
    return bool((wanted & banned) - own)


def domain_compatible(fixture: WorldFixture, wanted: frozenset[str]) -> bool:
    """Instance identity, not experiment verbs.

    `domains` say what the fixture *is* (prose, phage, tcp). Schema
    capabilities (sketch, hash, cache) do not vote. A claim that names a
    foreign object (agent-tool traces vs Pride) is refused even when the
    schema family matches. A generic claim of this family with no foreign
    object may bind without an instance-tag hit — otherwise every
    text-stream experiment would have to say "Austen".
    """
    if not wanted:
        return True
    return not foreign_conflict(fixture, wanted)


def lineage_conflicts(
    fixture: WorldFixture,
    *,
    named_instance: str = "",
    lineage_schema: str = "",
) -> str:
    """Why a literature lineage unbinds this freeze, or empty if it may stay.

    Schema name is not scientific identity. `domain_compatible` already
    binds by instance tags. If the papers name an object, that object
    decides — Pride vs agent-tool traces unbinds; an A2A automaton vs a
    lineage that says labeled_traces does not, because the named instance
    is not foreign to this freeze. Schema family is only consulted when
    the lineage did not name an instance.
    """
    named = content_tokens(named_instance)
    if named and foreign_conflict(fixture, named):
        return "the named instance is a different scientific object than this freeze"
    if named:
        return ""
    if lineage_schema and schema_family(lineage_schema) != schema_family(
        fixture.schema
    ):
        return (
            f"lineage schema {lineage_schema} is not this fixture's family "
            f"{fixture.schema}"
        )
    return ""


def requirement_compatible(req: WorldRequirement, fixture: WorldFixture) -> bool:
    if req.object_type in _INCOMPATIBLE_FAMILIES:
        return False
    req_schema = schema_family(req.schema)
    fix_schema = schema_family(fixture.schema)
    if req_schema and fix_schema and req_schema != fix_schema:
        return False
    mapped = OBJECT_SCHEMA.get(req.object_type) if req.object_type else ""
    if mapped and schema_family(mapped) not in {"", fix_schema}:
        return False
    needed = set(req.operations)
    if needed and not needed.issubset(fixture_capabilities(fixture)):
        return False
    if req.scale and _fixture_scale(fixture) and _fixture_scale(fixture) < req.scale:
        return False
    tokens = requirement_tokens(req)
    if tokens and not domain_compatible(fixture, tokens):
        return False
    return True


def match_world(
    req: WorldRequirement | None,
    catalog: dict[str, WorldFixture],
    *,
    preferred: WorldFixture | None = None,
) -> WorldFixture | None:
    """Bind a fixture that satisfies the claim, or None (WORLD_INCOMPATIBLE)."""
    if req is None:
        return None
    if preferred is not None and requirement_compatible(req, preferred):
        return preferred
    eligible = [
        fixture
        for fixture in catalog.values()
        if fixture.role not in {"test", "fixture", "synthetic"}
        and requirement_compatible(req, fixture)
    ]
    if not eligible:
        return None
    tokens = requirement_tokens(req)
    eligible.sort(
        key=lambda item: (-domain_hits(item, tokens), -_fixture_scale(item), item.id)
    )
    return eligible[0]


def pick_world(topic: str, catalog: dict[str, WorldFixture]) -> WorldFixture | None:
    """Choose a real attested world for this topic. Test fixtures stay out.

    Domain tags on the manifest are the scientific bind. Schema hints
    may rank among tagged fixtures; they cannot create eligibility on
    their own — otherwise a distillation claim that says "token" binds
    Pride and Prejudice. Title/id prose still cannot bind. A graph topic
    binds a graph (medium SNAP slices beat the toy clubs when the topic
    names connectivity); a genomic/succinct topic binds a sequence; an
    index/text topic binds prose; a protocol-security topic binds the
    matching `symbolic_trace`, not whichever automaton sorts first. No
    positive domain match is WORLD_INCOMPATIBLE — auto never falls back
    to Zachary karate or any other substitute schema. `path-trace` is a
    test fixture and is never auto-picked. External-memory claims still
    have no schema and must not run as WORLD on a social graph.
    """
    req = infer_requirement(topic)
    if req is not None and req.object_type in _INCOMPATIBLE_FAMILIES:
        return None
    eligible = [
        fixture
        for fixture in catalog.values()
        if fixture.role not in {"test", "fixture", "synthetic"}
    ]
    if not eligible:
        return None
    wanted = content_tokens(topic)
    ranked: list[tuple[int, str, WorldFixture]] = []
    for fixture in eligible:
        if req is not None and not requirement_compatible(req, fixture):
            continue
        extra = content_tokens(
            " ".join(fixture.domains) + " " + fixture.title + " " + fixture.id
        )
        spec = SCHEMAS.get(fixture.schema)
        schema_hits = wanted & (spec.hints if spec is not None else frozenset())
        hits = domain_hits(fixture, wanted)
        if not domain_compatible(fixture, wanted):
            # Foreign-object tokens refuse a schema-matched substitute.
            # Schema hints and title prose only rank; they do not bind
            # an agent-tool claim to a novel.
            continue
        if hits <= 0 and not schema_hits:
            # Title/id prose must not create eligibility — otherwise
            # "language model training" binds a TCP automaton because
            # the title says "learned model".
            continue
        score = 2 * hits + len(schema_hits) + len(wanted & extra)
        ranked.append((score, fixture.id, fixture))
    if not ranked:
        return None
    ranked.sort(key=lambda row: (-row[0], -_fixture_scale(row[2]), row[1]))
    if ranked[0][0] <= 0:
        return None
    return ranked[0][2]


def _fixture_scale(fixture: WorldFixture) -> int:
    """Among equal topic matches, the larger attested slice wins.

    Slice `n` is comparable (edges, bases, tokens, rows). File bytes are
    not: a 14 kB iris table must not beat a 10k-row vector slice.
    """
    rule = fixture.slice_rule or {}
    n = rule.get("n")
    if isinstance(n, int) and n > 0:
        return n
    return 0


def resolve_world(
    name: str | None,
    catalog: dict[str, WorldFixture],
    *,
    topic: str = "",
) -> WorldFixture | None:
    text = str(name or "").strip()
    if not text or text.lower() in {"none", "synthetic", "off", "0"}:
        return None
    if text.lower() == "auto":
        return pick_world(topic, catalog)
    if text not in catalog:
        raise KeyError(text)
    return catalog[text]


def bind_world(work_dir: Path, fixture: WorldFixture) -> Path:
    """Copy attested files into `work_dir/data/` and write `world.json`.

    The probe process sees only this cwd. Copying is the freeze: later
    edits to the catalog cannot change what this run measured.
    """
    work_dir = Path(work_dir)
    dest = work_dir / "data"
    dest.mkdir(parents=True, exist_ok=True)
    for name in fixture.files:
        shutil.copy2(fixture.root / name, dest / name)
    (work_dir / "world.json").write_text(
        json.dumps(fixture.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dest
