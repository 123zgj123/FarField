"""Host-side ingest of attested experimental worlds.

Network is allowed here, on the operator's machine, before any claim
exists. The probe sandbox still has no network. Fetching a dataset after
seeing a claim is not this command: it is a larger invented world.

Freezable schemas live in `schemas.FREEZABLE_SCHEMAS`: undirected_graph
(SNAP edgelist), fasta, text_stream, numeric_table, and symbolic_trace
(a labeled DOT digraph — e.g. a learned protocol state machine). A
freeze records the retrieved bytes' digest, a pre-registered slice rule,
and the sliced files. Missing source is blocked.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from .schemas import FREEZABLE_SCHEMAS, SCHEMAS
from .world import WorldFixture, digest_files, load_fixture, load_wishlist, wishlist_path


class FreezeError(ValueError):
    """Ingest refused. The catalog is unchanged."""


_ID = re.compile(r"^[a-z][a-z0-9-]*$")
_GZIP_MAGIC = b"\x1f\x8b"
_USER_AGENT = "FarField-freeze/0.1 (attested catalog ingest)"


def default_catalog() -> Path:
    """Repo `worlds/` when freeze.py lives in this tree."""
    repo = Path(__file__).resolve().parents[3]
    worlds = repo / "worlds"
    if worlds.is_dir():
        return worlds
    return Path.cwd() / "worlds"


def parse_slice_rule(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw or raw.lower() in {"all", "identity", "full"}:
        return {"kind": "all"}
    if ":" in raw:
        kind, _, rest = raw.partition(":")
        kind = kind.strip().lower()
        counted = {
            "first_edges",
            "first_bases",
            "first_tokens",
            "first_chars",
            "first_rows",
            "first_states",
        }
        if kind in counted:
            try:
                n = int(rest.strip())
            except ValueError as exc:
                raise FreezeError(f"slice {kind} needs an integer, got {rest!r}") from exc
            if n < 1:
                raise FreezeError(f"slice {kind} n must be >= 1")
            return {"kind": kind, "n": n}
    raise FreezeError(
        f"unknown slice {raw!r}; use all, first_edges:<n>, first_bases:<n>, "
        "first_tokens:<n>, first_chars:<n>, first_rows:<n>, or first_states:<n> "
        "(pre-registered official order, not a claim-conditioned subset)"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def retrieve_bytes(url: str, *, timeout: float = 60.0) -> bytes:
    """Fetch on the host. Never call this from a probe process."""
    target = str(url or "").strip()
    if not target:
        raise FreezeError("source URL is empty")
    request = urllib.request.Request(
        target, headers={"User-Agent": _USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise FreezeError(f"retrieve failed for {target}: {exc}") from exc


def decode_payload(retrieved: bytes) -> bytes:
    if retrieved.startswith(_GZIP_MAGIC):
        return gzip.decompress(retrieved)
    return retrieved


def parse_snap_edgelist(text: str) -> list[tuple[int, int]]:
    """Official SNAP listing order, including both directions and loops."""
    edges: list[tuple[int, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise FreezeError(f"edgelist line is not a pair: {raw!r}")
        try:
            src = int(parts[0])
            dst = int(parts[1])
        except ValueError as exc:
            raise FreezeError(f"edgelist line is not integer ids: {raw!r}") from exc
        edges.append((src, dst))
    if not edges:
        raise FreezeError("edgelist contained no edges")
    return edges


def unique_undirected(listed: list[tuple[int, int]]) -> tuple[list[list[int]], int]:
    """First occurrence of each unordered pair; skip self-loops."""
    seen: set[tuple[int, int]] = set()
    edges: list[list[int]] = []
    loops = 0
    for src, dst in listed:
        if src == dst:
            loops += 1
            continue
        key = (src, dst) if src < dst else (dst, src)
        if key in seen:
            continue
        seen.add(key)
        edges.append([key[0], key[1]])
    return edges, loops


def parse_fasta_sequence(text: str) -> tuple[str, str]:
    header = ""
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not header:
                header = line[1:].strip()
            continue
        chunks.append(line.upper())
    seq = "".join(chunks)
    if not seq:
        raise FreezeError("FASTA contained no bases")
    return header, seq


def tokenize_text(text: str) -> list[str]:
    return text.split()


_DOT_EDGE = re.compile(
    r"^\s*\"?([\w.+-]+)\"?\s*->\s*\"?([\w.+-]+)\"?\s*"
    r"(?:\[label=\"([^\"]*)\"[^\]]*\])?\s*;?\s*$"
)
_DOT_NODE = re.compile(r"^\s*\"?([\w.+-]+)\"?\s*(\[[^\]]*\])?\s*;?\s*$")
_DOT_LABEL = re.compile(r"label=\"([^\"]*)\"")


def parse_dot_machine(
    text: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Parse a labeled DOT digraph into states, transitions, and a start id.

    This is the interchange format of published learned state machines
    (Automata Wiki, protocol state fuzzing). States keep the file's
    listing order — the pre-registered order `first_states` slices by.
    An unlabeled edge from a `__start`-style pseudo node marks the
    initial state; pseudo nodes themselves are dropped.
    """
    states: list[dict[str, str]] = []
    seen: set[str] = set()
    transitions: list[dict[str, str]] = []
    start = ""

    def note_state(name: str, label: str = "") -> None:
        if name.startswith("__") or name in seen:
            return
        seen.add(name)
        states.append({"id": name, "label": label or name})

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("digraph", "graph", "//", "#", "}", "{")):
            continue
        edge = _DOT_EDGE.match(line)
        if edge:
            src, dst, label = edge.group(1), edge.group(2), edge.group(3)
            if src.startswith("__"):
                start = start or dst
                note_state(dst)
                continue
            note_state(src)
            note_state(dst)
            action, _, output = (label or "").partition("/")
            transitions.append(
                {
                    "src": src,
                    "dst": dst,
                    "action": action.strip() or "step",
                    "output": output.strip(),
                }
            )
            continue
        node = _DOT_NODE.match(line)
        if node and "=" not in node.group(1):
            attrs = node.group(2) or ""
            label_match = _DOT_LABEL.search(attrs)
            note_state(node.group(1), label_match.group(1) if label_match else "")
    if len(states) < 2:
        raise FreezeError("DOT digraph contained fewer than two states")
    if not transitions:
        raise FreezeError("DOT digraph contained no labeled transitions")
    return states, transitions, start or states[0]["id"]


def slice_machine(
    states: list[dict[str, str]],
    transitions: list[dict[str, str]],
    rule: dict[str, Any],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """`first_states:<n>` keeps the first n listed states and the
    transitions among them. Listing order is the artifact's, fixed at
    freeze time — never a claim-conditioned subset."""
    kind = str(rule.get("kind") or "all").strip().lower()
    if kind == "all":
        return list(states), list(transitions)
    kept = states[: int(rule["n"])]
    ids = {state["id"] for state in kept}
    edges = [
        edge for edge in transitions if edge["src"] in ids and edge["dst"] in ids
    ]
    return kept, edges


def machine_invariant(
    states: list[dict[str, str]], transitions: list[dict[str, str]]
) -> str:
    """A structural fact computed from the artifact, not from any claim."""
    fired: set[tuple[str, str]] = set()
    deterministic = True
    for edge in transitions:
        key = (edge["src"], edge["action"])
        if key in fired:
            deterministic = False
            break
        fired.add(key)
    if deterministic:
        return (
            "deterministic labeled transition system: each (state, action) "
            "pair fires at most one transition"
        )
    return "labeled transition system; transition structure is nondeterministic"


def parse_numeric_table(text: str) -> tuple[list[str], list[dict[str, Any]]]:
    sample = text.lstrip("\ufeff")
    reader = csv.reader(io.StringIO(sample))
    rows_raw = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows_raw:
        raise FreezeError("table contained no rows")
    header = rows_raw[0]
    body = rows_raw[1:]
    header_is_numeric = all(_number_or_none(cell) is not None for cell in header[1:])
    if header_is_numeric or not any(ch.isalpha() for ch in "".join(header[1:])):
        width = len(header)
        columns = ["label"] + [f"f{i}" for i in range(1, width)]
        body = rows_raw
    else:
        columns = [str(name).strip() or f"c{i}" for i, name in enumerate(header)]
    parsed: list[dict[str, Any]] = []
    for raw in body:
        if len(raw) < len(columns):
            raw = list(raw) + [""] * (len(columns) - len(raw))
        row: dict[str, Any] = {}
        for name, cell in zip(columns, raw):
            number = _number_or_none(cell)
            row[name] = number if number is not None else str(cell).strip()
        parsed.append(row)
    if not parsed:
        raise FreezeError("table contained no data rows")
    return columns, parsed


def _number_or_none(cell: str) -> int | float | None:
    text = str(cell).strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return None


def apply_count_slice(items: list[Any], rule: dict[str, Any]) -> list[Any]:
    kind = str(rule.get("kind") or "all").strip().lower()
    if kind == "all":
        return list(items)
    n = int(rule["n"])
    if n < 1:
        raise FreezeError("slice n must be >= 1")
    return items[:n]


def graph_payload(
    world_id: str,
    edges: list[list[int]],
    *,
    slice_rule: dict[str, Any],
    parent_m: int,
    parent_n: int,
    listed_m: int,
    loops: int,
) -> dict[str, Any]:
    nodes = sorted({node for edge in edges for node in edge})
    return {
        "schema": "undirected_graph",
        "id": world_id,
        "nodes": nodes,
        "edges": edges,
        "n": len(nodes),
        "m": len(edges),
        "slice": {
            **dict(slice_rule),
            "parent_n": parent_n,
            "parent_m": parent_m,
            "listed_m": listed_m,
            "self_loops": loops,
        },
    }


def freeze_world(
    *,
    world_id: str,
    schema: str,
    slice_rule: dict[str, Any] | str,
    catalog: Path | None = None,
    url: str = "",
    source_file: Path | None = None,
    title: str = "",
    source: str = "",
    domains: tuple[str, ...] | list[str] = (),
    retrieved_at: str = "",
    load_hint: str = "",
    force: bool = False,
    timeout: float = 60.0,
    cache_dir: Path | None = None,
) -> WorldFixture:
    """Write `catalog/<id>/` from retrieved bytes. Host-side only."""
    ident = str(world_id or "").strip()
    if not _ID.match(ident):
        raise FreezeError(
            f"world id {world_id!r} must be lowercase letters, digits, and dashes"
        )
    kind = str(schema or "").strip()
    spec = SCHEMAS.get(kind)
    if spec is None or not spec.freezable:
        raise FreezeError(
            f"freeze schema {kind!r} is not implemented; "
            f"use {list(FREEZABLE_SCHEMAS)}"
        )
    rule = (
        dict(slice_rule)
        if isinstance(slice_rule, dict)
        else parse_slice_rule(str(slice_rule))
    )
    if str(rule.get("kind") or "") not in spec.slice_kinds:
        raise FreezeError(
            f"slice {rule.get('kind')!r} does not apply to schema {kind}"
        )
    catalog_root = Path(catalog) if catalog is not None else default_catalog()
    catalog_root.mkdir(parents=True, exist_ok=True)
    dest = catalog_root / ident
    if dest.exists() and not force:
        raise FreezeError(f"{dest} already exists; pass force to replace")
    retrieved, source_url = _load_source(
        url=url, source_file=source_file, timeout=timeout
    )
    if cache_dir is None:
        cache_dir = catalog_root / ".cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".gz" if retrieved.startswith(_GZIP_MAGIC) else ".bin"
    (cache_dir / f"{ident}.retrieved{suffix}").write_bytes(retrieved)
    text = decode_payload(retrieved).decode("utf-8")
    when = str(retrieved_at or date.today().isoformat()).strip()
    tags = tuple(
        str(item).strip()
        for item in (domains or spec.default_domains)
        if str(item).strip()
    )
    citation = str(source or "").strip() or _default_source(source_url, source_file)
    heading = str(title or "").strip() or ident
    files, origin, hint = _slice_payload(
        ident,
        kind,
        text,
        retrieved=retrieved,
        rule=rule,
        source_url=source_url,
        when=when,
    )
    hint = str(load_hint or "").strip() or hint
    return _commit_world(
        dest,
        ident=ident,
        heading=heading,
        citation=citation,
        source_url=source_url,
        source_digest=origin["source_digest"],
        when=when,
        tags=tags,
        schema=kind,
        hint=hint,
        rule=rule,
        files=files,
        origin=origin,
        force=force,
    )


def reconstruct_parent(fixture: WorldFixture, dest: Path) -> dict[str, Any]:
    """Rebuild the unsliced parent into dest/data from the cached retrieval.

    Missing cache is blocked — silently using the slice would pretend the
    host run saw the mother trace.
    """
    cache_dir = fixture.root.parent / ".cache"
    matches = sorted(cache_dir.glob(f"{fixture.id}.retrieved*"))
    if not matches:
        raise FreezeError(
            f"parent cache missing for {fixture.id}; freeze again or run the slice only"
        )
    retrieved = matches[0].read_bytes()
    recorded = str(fixture.source_digest or "").strip()
    digest = sha256_bytes(retrieved)
    if recorded and recorded != digest:
        raise FreezeError(
            f"parent cache digest mismatch for {fixture.id}: "
            f"manifest {recorded[:12]}… vs cache {digest[:12]}…"
        )
    text = decode_payload(retrieved).decode("utf-8")
    rule = {"kind": "all"}
    files, origin, _hint = _slice_payload(
        fixture.id,
        fixture.schema or "undirected_graph",
        text,
        retrieved=retrieved,
        rule=rule,
        source_url=fixture.source_url,
        when=fixture.retrieved_at,
    )
    data = Path(dest) / "data"
    data.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (data / name).write_text(payload, encoding="utf-8")
    return origin


def _slice_payload(
    ident: str,
    schema: str,
    text: str,
    *,
    retrieved: bytes,
    rule: dict[str, Any],
    source_url: str,
    when: str,
) -> tuple[dict[str, str], dict[str, Any], str]:
    origin_base = {
        "url": source_url,
        "retrieved_at": when,
        "source_digest": sha256_bytes(retrieved),
        "text_digest": sha256_bytes(text.encode("utf-8")),
        "slice_rule": rule,
    }
    load_hint = SCHEMAS[schema].load_hint if schema in SCHEMAS else ""
    if schema == "undirected_graph":
        listed = parse_snap_edgelist(text)
        undirected, loops = unique_undirected(listed)
        parent_nodes = {node for edge in undirected for node in edge}
        sliced = apply_count_slice(undirected, rule)
        if not sliced:
            raise FreezeError("slice produced no edges")
        graph = graph_payload(
            ident,
            sliced,
            slice_rule=rule,
            parent_m=len(undirected),
            parent_n=len(parent_nodes),
            listed_m=len(listed),
            loops=loops,
        )
        origin = {
            **origin_base,
            "parent": {
                "n": len(parent_nodes),
                "m": len(undirected),
                "listed_m": len(listed),
                "self_loops": loops,
                "format": "snap_edgelist",
            },
            "slice": {"n": graph["n"], "m": graph["m"]},
        }
        files = {
            "graph.json": json.dumps(graph, ensure_ascii=False, separators=(",", ":")),
            "origin.json": json.dumps(origin, ensure_ascii=False, indent=2) + "\n",
        }
        return files, origin, load_hint
    if schema == "fasta":
        header, seq = parse_fasta_sequence(text)
        kind = str(rule.get("kind") or "all")
        if kind == "first_bases":
            sliced = seq[: int(rule["n"])]
        else:
            sliced = seq
        if not sliced:
            raise FreezeError("slice produced no bases")
        fasta = f">{ident} {header}\n" + "\n".join(
            sliced[i : i + 70] for i in range(0, len(sliced), 70)
        ) + "\n"
        meta = {
            "id": ident,
            "header": header,
            "parent_bp": len(seq),
            "bp": len(sliced),
            "slice_rule": rule,
        }
        origin = {
            **origin_base,
            "parent": {"bp": len(seq), "header": header, "format": "fasta"},
            "slice": {"bp": len(sliced)},
        }
        files = {
            "sequence.fasta": fasta,
            "meta.json": json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            "origin.json": json.dumps(origin, ensure_ascii=False, indent=2) + "\n",
        }
        return files, origin, load_hint
    if schema == "text_stream":
        tokens = tokenize_text(text)
        kind = str(rule.get("kind") or "all")
        if kind == "first_chars":
            sliced_text = text[: int(rule["n"])]
            sliced_tokens = tokenize_text(sliced_text)
        elif kind == "first_tokens":
            sliced_tokens = tokens[: int(rule["n"])]
            sliced_text = " ".join(sliced_tokens)
        else:
            sliced_tokens = tokens
            sliced_text = text
        if not sliced_tokens:
            raise FreezeError("slice produced no tokens")
        keys: list[str] = []
        seen: set[str] = set()
        for token in sliced_tokens:
            if token not in seen:
                seen.add(token)
                keys.append(token)
        stream = {"schema": "text_stream", "id": ident, "tokens": sliced_tokens, "n": len(sliced_tokens)}
        key_doc = {"schema": "text_keys", "id": ident, "keys": keys, "n": len(keys)}
        origin = {
            **origin_base,
            "parent": {
                "chars": len(text),
                "tokens": len(tokens),
                "format": "utf8_text",
            },
            "slice": {"chars": len(sliced_text), "tokens": len(sliced_tokens), "keys": len(keys)},
        }
        files = {
            "text.txt": sliced_text if sliced_text.endswith("\n") else sliced_text + "\n",
            "stream.json": json.dumps(stream, ensure_ascii=False, separators=(",", ":")),
            "keys.json": json.dumps(key_doc, ensure_ascii=False, separators=(",", ":")),
            "origin.json": json.dumps(origin, ensure_ascii=False, indent=2) + "\n",
        }
        return files, origin, load_hint
    if schema == "numeric_table":
        columns, rows = parse_numeric_table(text)
        sliced = apply_count_slice(rows, rule)
        if not sliced:
            raise FreezeError("slice produced no rows")
        table = {
            "schema": "numeric_table",
            "id": ident,
            "columns": columns,
            "rows": sliced,
            "n": len(sliced),
            "d": max(0, len(columns) - 1),
            "slice": {**dict(rule), "parent_n": len(rows)},
        }
        origin = {
            **origin_base,
            "parent": {"n": len(rows), "d": table["d"], "format": "csv"},
            "slice": {"n": len(sliced), "d": table["d"]},
        }
        files = {
            "table.json": json.dumps(table, ensure_ascii=False, separators=(",", ":")),
            "origin.json": json.dumps(origin, ensure_ascii=False, indent=2) + "\n",
        }
        return files, origin, load_hint
    if schema == "symbolic_trace":
        states, transitions, start = parse_dot_machine(text)
        sliced_states, sliced_edges = slice_machine(states, transitions, rule)
        if len(sliced_states) < 2 or not sliced_edges:
            raise FreezeError("slice produced no runnable machine")
        world = {
            "schema": "symbolic_trace",
            "id": ident,
            "object_type": "formula",
            "invariant": machine_invariant(sliced_states, sliced_edges),
            "initial": start if any(s["id"] == start for s in sliced_states) else sliced_states[0]["id"],
            "states": sliced_states,
            "transitions": sliced_edges,
            "n": len(sliced_states),
            "m": len(sliced_edges),
            "slice": {**dict(rule), "parent_n": len(states), "parent_m": len(transitions)},
        }
        origin = {
            **origin_base,
            "parent": {
                "n": len(states),
                "m": len(transitions),
                "initial": start,
                "format": "dot_labeled_digraph",
            },
            "slice": {"n": world["n"], "m": world["m"]},
        }
        files = {
            "world.json": json.dumps(world, ensure_ascii=False, separators=(",", ":")),
            "origin.json": json.dumps(origin, ensure_ascii=False, indent=2) + "\n",
        }
        return files, origin, load_hint
    raise FreezeError(f"schema {schema!r} has no slicer")


def _commit_world(
    dest: Path,
    *,
    ident: str,
    heading: str,
    citation: str,
    source_url: str,
    source_digest: str,
    when: str,
    tags: tuple[str, ...],
    schema: str,
    hint: str,
    rule: dict[str, Any],
    files: dict[str, str],
    origin: dict[str, Any],
    force: bool,
) -> WorldFixture:
    catalog_root = dest.parent
    staging = Path(tempfile.mkdtemp(prefix=f".{ident}.", dir=str(catalog_root)))
    try:
        names = sorted(files)
        for name, payload in files.items():
            (staging / name).write_text(payload, encoding="utf-8")
        digest = digest_files(staging, names)
        manifest = {
            "id": ident,
            "title": heading,
            "source": citation,
            "source_url": source_url,
            "source_digest": source_digest,
            "retrieved_at": when,
            "domains": list(tags),
            "files": names,
            "role": "world",
            "schema": schema,
            "load_hint": hint,
            "slice_rule": rule,
            "digest": digest,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if dest.exists():
            if not force:
                raise FreezeError(f"{dest} already exists; pass force to replace")
            shutil.rmtree(dest)
        staging.rename(dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    fixture = load_fixture(dest)
    # The chain proves *when* the freeze happened relative to later
    # registrations and executions, not just what was frozen.
    from . import chain

    chain.append_event(
        catalog_root / "chain.jsonl",
        chain.FREEZE_WORLD,
        {
            "world_id": ident,
            "digest": fixture.digest,
            "source_digest": source_digest,
            "schema": schema,
            "slice_rule": dict(rule),
        },
    )
    return fixture


def _load_source(
    *, url: str, source_file: Path | None, timeout: float
) -> tuple[bytes, str]:
    path = Path(source_file) if source_file is not None else None
    target = str(url or "").strip()
    if path is not None:
        if not path.is_file():
            raise FreezeError(f"source file missing: {path}")
        return path.read_bytes(), target or str(path)
    if not target:
        raise FreezeError(
            "missing source: pass --url (host fetch) or --file (already retrieved bytes)"
        )
    return retrieve_bytes(target, timeout=timeout), target


def _default_source(url: str, source_file: Path | None) -> str:
    if url:
        return url
    if source_file is not None:
        return str(source_file)
    return ""


def _wish_id(entry: dict[str, Any]) -> str:
    raw = (
        f"wish-{entry.get('object_type') or 'obj'}-"
        f"{entry.get('freeze_schema') or entry.get('schema') or 'x'}"
    )
    ident = re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")
    if not _ID.match(ident):
        ident = "wish-pending"
    return ident[:48]


def resolve_pending_worlds(root: Path, *, catalog: Path | None = None) -> list[dict[str, Any]]:
    """Run wishlist freeze recipes that already name a public source.

    Host-side acquire of the same object the last mission could not bind.
    Failures stay on the list. A later claim may bind the new freeze;
    finished WORLD_INCOMPATIBLE verdicts are not reopened.
    """
    payload = load_wishlist(root)
    entries = [
        row for row in (payload.get("entries") or []) if isinstance(row, dict)
    ]
    if not entries:
        return []
    catalog_root = Path(catalog) if catalog is not None else (Path(root) / "worlds")
    acquired: list[dict[str, Any]] = []
    changed = False
    for entry in entries:
        if entry.get("acquired_id"):
            continue
        schema = str(entry.get("freeze_schema") or entry.get("schema") or "").strip()
        url = str(entry.get("freeze_url") or "").strip()
        if not url or schema not in FREEZABLE_SCHEMAS:
            continue
        ident = _wish_id(entry)
        dest = catalog_root / ident
        if dest.is_dir():
            entry["acquired_id"] = ident
            changed = True
            continue
        try:
            fixture = freeze_world(
                world_id=ident,
                schema=schema,
                slice_rule="all",
                catalog=catalog_root,
                url=url,
                title=str(entry.get("named_instance") or ident),
                source=str(entry.get("lineage") or entry.get("freeze_source") or url),
            )
        except FreezeError:
            continue
        except Exception:
            continue
        entry["acquired_id"] = fixture.id
        entry["acquired_digest"] = fixture.digest
        changed = True
        acquired.append(
            {
                "world_id": fixture.id,
                "digest": fixture.digest,
                "schema": fixture.schema,
                "key": entry.get("key"),
            }
        )
    if changed:
        path = wishlist_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return acquired
