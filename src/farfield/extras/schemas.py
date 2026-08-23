"""Single registry of experimental-world schemas.

Matching (`world.py`), host-side freezing (`freeze.py`), and in-mission
construction (`genworld.py`) used to keep five parallel tables — hints,
capabilities, object families, default domains, slice kinds — that had
to agree by hand. They now agree by construction: one `SchemaSpec` per
schema, everything else is derived. Adding a schema is one entry here
plus a slicer branch in `freeze._slice_payload`.

The registry carries no contract exceptions. Whether a probe may climb
is still decided by probe kind (WORLD vs GENERATED vs SYNTHETIC), never
by anything written here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaSpec:
    """One experimental-world schema, end to end.

    `name` is the manifest `schema`. `object_type` is the claim-side
    object family (`infer_requirement` output, from topic+claim only —
    the far-field mechanism does not vote). `hints` are claim/topic
    words that point at this schema; `capabilities` are what a fixture of
    this schema can attest. `slice_kinds` are the pre-registered slices
    `farfield freeze` accepts. `freezable=False` marks a schema that can
    only be constructed in-mission (GENERATED) until a slicer exists.
    """

    name: str
    object_type: str
    capabilities: frozenset[str]
    hints: frozenset[str]
    default_domains: tuple[str, ...]
    slice_kinds: frozenset[str]
    load_hint: str
    freezable: bool = True


_SPECS = (
    SchemaSpec(
        name="undirected_graph",
        object_type="graph",
        capabilities=frozenset({"graph", "nodes", "edges", "undirected"}),
        hints=frozenset(
            {"graph", "network", "connectivity", "edge", "community", "social"}
        ),
        default_domains=("graph", "network", "connectivity"),
        slice_kinds=frozenset({"all", "first_edges"}),
        load_hint=(
            "payload = json.loads(Path('data/graph.json').read_text(encoding='utf-8')); "
            "nodes = payload['nodes']; edges = payload['edges']  "
            "# undirected int pairs; attested SNAP prefix, do not fetch"
        ),
    ),
    SchemaSpec(
        name="fasta",
        object_type="sequence",
        capabilities=frozenset({"sequence", "bases", "fasta"}),
        hints=frozenset(
            {"sequence", "genomic", "dna", "fasta", "genome", "phage", "succinct"}
        ),
        default_domains=("sequence", "genomic", "dna", "fasta"),
        slice_kinds=frozenset({"all", "first_bases"}),
        load_hint=(
            "seq = ''.join(line.strip() for line in "
            "Path('data/sequence.fasta').read_text(encoding='utf-8').splitlines() "
            "if not line.startswith('>'))"
        ),
    ),
    SchemaSpec(
        name="text_stream",
        object_type="stream",
        capabilities=frozenset({"stream", "tokens", "text"}),
        hints=frozenset(
            {"text", "prose", "stream", "token", "sketch", "hash", "cache", "dictionary"}
        ),
        default_domains=("text", "prose", "stream", "english"),
        slice_kinds=frozenset({"all", "first_tokens", "first_chars"}),
        load_hint=(
            "text = Path('data/text.txt').read_text(encoding='utf-8'); "
            "tokens = json.loads(Path('data/stream.json').read_text(encoding='utf-8'))['tokens']; "
            "keys = json.loads(Path('data/keys.json').read_text(encoding='utf-8'))['keys']"
        ),
    ),
    SchemaSpec(
        name="numeric_table",
        object_type="table",
        capabilities=frozenset({"table", "rows", "numeric"}),
        hints=frozenset(
            {"vector", "similarity", "numeric", "table", "dimensional", "range"}
        ),
        default_domains=("table", "numeric", "vector"),
        slice_kinds=frozenset({"all", "first_rows"}),
        load_hint=(
            "table = json.loads(Path('data/table.json').read_text(encoding='utf-8')); "
            "rows = table['rows']; columns = table['columns']"
        ),
    ),
    SchemaSpec(
        name="labeled_traces",
        object_type="trace",
        capabilities=frozenset(
            {"trace", "traces", "steps", "labels", "order", "trajectory"}
        ),
        hints=frozenset(
            {
                "trajectory",
                "trajectories",
                "denoising",
                "bypass",
                "tool-call",
                "toolcall",
                "tool",
            }
        ),
        default_domains=("trace", "trajectory", "authorization", "tool"),
        slice_kinds=frozenset({"all"}),
        load_hint=(
            "payload = json.loads(Path('data/world.json').read_text(encoding='utf-8')); "
            "traces = payload['traces']  "
            "# GENERATED labeled traces; not a freeze"
        ),
        freezable=False,
    ),
    SchemaSpec(
        name="symbolic_trace",
        object_type="formula",
        capabilities=frozenset(
            {"formula", "symbolic", "states", "transitions", "trace", "automaton"}
        ),
        # "model" and "state" stay out: they collide with "language model"
        # and everyday CS prose, and a wrong formula requirement turns a
        # bindable claim into WORLD_INCOMPATIBLE.
        hints=frozenset(
            {
                "automaton",
                "automata",
                "invariant",
                "transition",
                "reachability",
                "verifier",
                "verification",
                "formal",
                "symbolic",
                "proof",
                "certificate",
                "temporal",
                "safety",
                "protocol",
            }
        ),
        default_domains=(
            "formal",
            "verification",
            "automaton",
            "protocol",
            "symbolic",
            "trace",
        ),
        slice_kinds=frozenset({"all", "first_states"}),
        load_hint=(
            "payload = json.loads(Path('data/world.json').read_text(encoding='utf-8')); "
            "states = payload['states']; transitions = payload['transitions']  "
            "# attested learned automaton; do not fetch"
        ),
    ),
)

SCHEMAS: dict[str, SchemaSpec] = {spec.name: spec for spec in _SPECS}

# Claim-side object family -> schema name. One family, one schema.
OBJECT_SCHEMA: dict[str, str] = {spec.object_type: spec.name for spec in _SPECS}

FREEZABLE_SCHEMAS: tuple[str, ...] = tuple(
    sorted(spec.name for spec in _SPECS if spec.freezable)
)
