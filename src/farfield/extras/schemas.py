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
    # Where named fields live on the freeze. Empty when the load_hint
    # already names every key. A probe that looks a field up on the
    # wrong object (created_tools on a step) is not measuring it.
    attested_layout: str = ""


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
            {
                "trace",
                "traces",
                "steps",
                "labels",
                "order",
                "trajectory",
                "execution",
                "patch",
                "tool",
            }
        ),
        hints=frozenset(
            {
                "trace",
                "traces",
                "trajectory",
                "trajectories",
                "execution",
                "runtime",
                "software",
                "swe",
                "patch",
                "self-play",
                "denoising",
                "bypass",
                "tool-call",
                "toolcall",
                "tool",
                # "agent" stays out: bargaining-game and A2A topics say
                # agent without being execution traces. harness / llm are
                # the words that name the trace-producing runtime.
                "harness",
                "llm",
            }
        ),
        default_domains=(
            "trace",
            "trajectory",
            "software",
            "swe",
            "execution",
            "agent",
            "tool",
        ),
        slice_kinds=frozenset({"all", "first_traces"}),
        load_hint=(
            "payload = json.loads(Path('data/world.json').read_text(encoding='utf-8')); "
            "traces = payload['traces']  "
            "# each trace: created_tools (list on the TRACE), steps, label; "
            "# each step: t, action, returncode — not created_tools, not duration"
        ),
        attested_layout=(
            "Each trace in payload['traces'] has id, label, created_tools "
            "(a list on the TRACE, not on steps), steps, and derived."
            "created_tool_count. Each step has t, action, returncode, "
            "action_chars, output_chars, output_excerpt. Steps do not "
            "carry created_tools or wall-clock duration. Read "
            "trace['created_tools'] or derived['created_tool_count']; "
            "step.get('created_tools') is always empty on this freeze."
        ),
    ),
    SchemaSpec(
        name="program_state",
        object_type="executable",
        capabilities=frozenset(
            {
                "program",
                "cells",
                "updates",
                "validators",
                "acceptance",
                "replay",
                "executable",
            }
        ),
        # "state" and "model" stay out (language-model / everyday-CS
        # collisions). "program" is admitted here because a topic that
        # says program together with executable / validator / recursive
        # is naming running code, not a linear program.
        hints=frozenset(
            {
                "executable",
                "program",
                "interpreter",
                "bytecode",
                "validator",
                "validators",
                "acceptance",
                "rollback",
                "recursive",
                "self-improvement",
                "self-modifying",
            }
        ),
        default_domains=(
            "program",
            "executable",
            "update",
            "validator",
            "acceptance",
        ),
        slice_kinds=frozenset({"all", "first_updates"}),
        load_hint=(
            "payload = json.loads(Path('data/world.json').read_text(encoding='utf-8')); "
            "cells = payload['cells']; validators = payload['validators']; "
            "updates = payload['updates']  "
            "# ordered self-modification epochs; each update: id, epoch, writes, "
            "# reads, validator_writes, accepted, divergent, task_return"
        ),
        # Freezable since `freeze.parse_program_state_source`: a host
        # exports a published self-modification history (JSON or JSONL)
        # and slices `first_updates:<n>`. While this was False, the only
        # freezable neighbour a lineage could name was symbolic_trace,
        # and that neighbour leaked into construction.
        attested_layout=(
            "payload['cells'] is the mutable program state (list of cell ids). "
            "payload['validators'] is a list of {id, reads} — each validator "
            "reads named cells to accept an update. payload['updates'] is the "
            "ordered self-modification history; each update has id, epoch, "
            "writes (cell ids), reads (cell ids), validator_writes (validator "
            "ids this update mutates), accepted (bool), divergent (bool: the "
            "update's replayed state diverges from the pre-update baseline), "
            "task_return (float). An update whose validator_writes names a "
            "validator that reads one of the update's own written cells sits "
            "on a write–read dependency cycle: it can approve itself."
        ),
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


def attested_layout(schema: str) -> str:
    """Where named fields live on this freeze. Empty when unspecified."""
    spec = SCHEMAS.get(str(schema or "").strip())
    return spec.attested_layout if spec is not None else ""
