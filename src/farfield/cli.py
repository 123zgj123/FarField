"""Farfield CLI: product path is `research`. Ledger commands stay for INV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extras.llm import Backend, LLMClient, LLMUnavailable
from .runtime import FarfieldRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farfield")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create an immutable standing intent")
    init.add_argument("project", type=Path)
    init.add_argument("--intent", required=True)

    objective = subparsers.add_parser(
        "objective", help="declare the first research question"
    )
    objective.add_argument("project", type=Path)
    objective.add_argument("--text", required=True)
    objective.add_argument("--success", action="append", required=True)
    objective.add_argument("--constraint", action="append", default=[])

    verify = subparsers.add_parser("verify", help="verify the event hash chain")
    verify.add_argument("project", type=Path)

    status = subparsers.add_parser("status", help="print a compact derived view")
    status.add_argument("project", type=Path)

    research = subparsers.add_parser(
        "research",
        help="OpenWorld local observations (default); --legacy-pipeline for LLM research packets",
        description=(
            "Default OpenWorld writes SCIENTIFIC_STATE/EVENTS and RESEARCH_STATUS.md. "
            "It does not yet run the LLM/literature pipeline or complete a research study. "
            "Use --legacy-pipeline for generation, review, registered experiments and packets. "
            "Model, API budget, review, paper, and state-store flags apply to legacy only."
        ),
    )
    research.add_argument("project", type=Path)
    research.add_argument("--topic", required=True)
    research.add_argument(
        "--jumps",
        type=int,
        default=4,
        help="far-field cap; auto-explore stops when the plan is executable",
    )
    research.add_argument(
        "--explore",
        choices=("auto", "fixed"),
        default="auto",
        help="auto: open slots until the plan is executable or the cap; fixed: generate all jumps",
    )
    research.add_argument("--candidates", type=int, default=1)
    research.add_argument("--reframes", type=int, default=1)
    research.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="concurrent LLM/literature jobs (1 = serial; env FARFIELD_PARALLEL)",
    )
    research.add_argument(
        "--polish-rounds",
        type=int,
        default=0,
        help="-1 auto (up to 2 reviews); 0 skip (product default); polish is not idea evolution",
    )
    research.add_argument(
        "--experiment-rounds",
        type=int,
        default=-1,
        help=(
            "-1 auto (up to 2 extra experiment attempts after the first); "
            "0 one-shot; iterate until the probe discriminates, not until it supports"
        ),
    )
    research.add_argument(
        "--idea-rounds",
        type=int,
        default=-1,
        help=(
            "-1 auto (up to 2 extra rewrites of the same pair after a text-gate death); "
            "0 one-shot; not a new far jump and not briefing polish"
        ),
    )
    research.add_argument(
        "--venue",
        default="",
        help=(
            "declared conference profile (neurips, icml, cav, osdi, stoc, …); "
            "review questions join the value rubric and cannot kill; "
            "domain lock still wins. empty skips"
        ),
    )
    research.add_argument(
        "--world",
        default="auto",
        help=(
            "explicit attested fixture id under worlds/; "
            "OpenWorld auto/none = no external freeze; "
            "legacy auto = construct GENERATED from topic/papers, "
            "legacy none = SYNTHETIC coherence check (cannot corroborate)"
        ),
    )
    research.add_argument(
        "--world-sim",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="world_sim",
        help=(
            "after diagnosis, run three-tier world rehearsal under "
            "ideas/<name>/world-sim/ and rewrite the plan once on "
            "structural holes; imagined numbers cannot climb "
            "(default on; --no-world-sim to skip)"
        ),
    )
    research.add_argument(
        "--min-ideas",
        type=int,
        default=3,
        dest="min_ideas",
        help=(
            "breadth floor: open at least this many distinct far landings this "
            "mission (bounded by --jumps) before an executable plan may stop the "
            "spray (default 3; 1 restores stop-at-first-plan)"
        ),
    )
    research.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "explore again on a topic whose stored program already has an executable "
            "plan (default: such a topic drops to execute-only / continue)"
        ),
    )
    research.add_argument(
        "--human-gate",
        choices=("none", "after_review"),
        default="none",
        dest="human_gate",
        help=(
            "after_review: stop after the adversarial review with HUMAN_GATE.md and "
            "spend on evidence only for cards passed via --approve (rerun the same "
            "command; recorded steps replay)"
        ),
    )
    research.add_argument(
        "--approve",
        action="append",
        default=[],
        help="card id to admit through the human gate (repeatable), or 'all'",
    )
    research.add_argument(
        "--paper-rounds",
        type=int,
        default=1,
        dest="paper_rounds",
        help=(
            "after the packets, write ideas/<name>/paper/RESEARCH_PROPOSAL.md and "
            "EXPERIMENT_DESIGN.md from the attested fact sheet with N reviewer→revision "
            "rounds (default 1; -1 skips). DRAFT: citations audited, ladder untouched"
        ),
    )
    research.add_argument(
        "--corpus",
        default="",
        help=(
            "concept corpus id. empty uses the mixed harvest when "
            "G_full.json exists, otherwise ds-arxiv-concepts-2026"
        ),
    )
    research.add_argument("--model", default="")
    research.add_argument("--base-url", default="")
    research.add_argument("--api-key", default="")
    research.add_argument(
        "--api-calls",
        type=float,
        default=None,
        help=(
            "LLM call envelope for this mission. Default is the "
            "jump/idea/polish formula (~40 calls). Declare a larger "
            "envelope before opening a neighbourhood state-store if "
            "the previous run died on llm_budget_api_calls"
        ),
    )
    research.add_argument(
        "--policy-log",
        type=Path,
        default=None,
        help="trajectory log for this arm; keep Qwen and DeepSeek on separate files",
    )
    research.add_argument(
        "--state-store",
        type=Path,
        default=None,
        help="research state for this arm; do not share with another model",
    )
    research.add_argument(
        "--judges-file",
        type=Path,
        default=None,
        help="shadow/lethal bench for this arm",
    )
    research.add_argument(
        "--policy-file",
        type=Path,
        default=None,
        help="routing policy file for this arm",
    )
    research.add_argument(
        "--legacy-pipeline",
        action="store_true",
        dest="legacy_pipeline",
        help="compat: fixed generate→review→evidence mission stages",
    )
    research.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="OpenWorld ticks. Default is --jumps. Hard budget cap only.",
    )

    freeze = subparsers.add_parser(
        "freeze",
        help=(
            "ingest an attested world on the host: retrieve, hash, slice "
            "by a pre-registered rule, write worlds/<id>/"
        ),
    )
    freeze.add_argument("--id", required=True, dest="world_id")
    freeze.add_argument(
        "--url",
        default="",
        help="source URL recorded on the manifest; fetched only when --file is omitted",
    )
    freeze.add_argument(
        "--file",
        type=Path,
        dest="source_file",
        default=None,
        help="already-retrieved bytes (offline replay). Do not pick this after seeing a claim",
    )
    freeze.add_argument(
        "--dir",
        type=Path,
        dest="run_dir",
        default=None,
        help=(
            "a published self-improvement run directory (with --format): "
            "dgm output_dgm/<run>/ or an openevolve checkpoint_<n>/; "
            "frozen as program_state"
        ),
    )
    freeze.add_argument(
        "--format",
        dest="history_format",
        default="",
        help="layout of --dir: dgm | openevolve",
    )
    freeze.add_argument(
        "--derive-from",
        dest="derive_from",
        default="",
        help=(
            "re-encode an existing catalog world as --schema by a registered "
            "derivation (e.g. labeled_traces created_tools → program_state); "
            "manifest says provenance: derived"
        ),
    )
    freeze.add_argument(
        "--schema",
        default="undirected_graph",
        help=(
            "SNAP edgelist (undirected_graph), fasta, text_stream, "
            "numeric_table, labeled_traces (JSON/JSONL or Live-SWE zip), "
            "symbolic_trace (labeled DOT digraph), or program_state "
            "(JSON/JSONL self-modification history)"
        ),
    )
    freeze.add_argument(
        "--slice",
        default="first_edges:10000",
        dest="slice_rule",
        help=(
            "pre-registered rule: all, first_edges:<n>, first_bases:<n>, "
            "first_tokens:<n>, first_rows:<n>, first_states:<n>, "
            "first_traces:<n>"
        ),
    )
    freeze.add_argument("--title", default="")
    freeze.add_argument("--source", default="", help="citation / provenance sentence")
    freeze.add_argument("--domain", action="append", default=[])
    freeze.add_argument("--retrieved-at", default="")
    freeze.add_argument("--load-hint", default="")
    freeze.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="worlds directory (default: repo worlds/)",
    )
    freeze.add_argument(
        "--force",
        action="store_true",
        help="replace an existing world id",
    )

    paper = subparsers.add_parser(
        "paper",
        help=(
            "write the paper-level RESEARCH_PROPOSAL.md and EXPERIMENT_DESIGN.md "
            "for one idea folder from its attested fact sheet, with one ICLR-style "
            "reviewer round; DRAFT, citations audited, ladder untouched"
        ),
    )
    paper.add_argument("idea_dir", type=Path, help="workspace/ideas/<name>/")
    paper.add_argument("--card", default="", help="candidates/<card_id> (default: read from AGENT_PACKET.md)")
    paper.add_argument("--rounds", type=int, default=1, help="reviewer→revision rounds (default 1)")
    paper.add_argument("--lang", choices=("zh", "en"), default="zh")

    analyze = subparsers.add_parser(
        "analyze",
        help=(
            "exploratory analysis on an attested world: run a stdlib script (or the "
            "schema's builtin descriptives) on a bound copy, record results with world "
            "and script digests under var/exploratory/ and on the chain. Post hoc; "
            "cannot corroborate"
        ),
    )
    analyze.add_argument("world_id")
    analyze.add_argument("--script", type=Path, default=None, help="analysis.py writing analysis.json (default: builtin descriptives)")
    analyze.add_argument("--label", default="")
    analyze.add_argument("--catalog", type=Path, default=None)

    harvest = subparsers.add_parser(
        "harvest",
        help=(
            "world by execution: register a harness recipe on the chain, run "
            "it on the host (no shell, no probe), import the exported "
            "self-improvement history, freeze it as program_state with "
            "provenance: harvested"
        ),
    )
    harvest.add_argument("recipe", type=Path, help="JSON recipe: world_id, harness, command, format, output, seed, iterations, timeout")
    harvest.add_argument("--catalog", type=Path, default=None, help="worlds directory (default: repo worlds/)")
    harvest.add_argument("--force", action="store_true", help="replace an existing world id")
    harvest.add_argument(
        "--reuse-run",
        action="store_true",
        help="skip the harness when the chain records a successful run of this exact recipe whose output still exists; import and freeze that run",
    )
    harvest.add_argument(
        "--register-only",
        action="store_true",
        help="write the recipe to the chain and stop; run later with the same file",
    )

    execute = subparsers.add_parser(
        "execute",
        help=(
            "host-run a compiled protocol.json against the attested parent "
            "(same digest, stdlib whitelist, no network). protocol_executed "
            "is not a discovery"
        ),
    )
    execute.add_argument("folder", type=Path)
    execute.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="override seconds; default follows the protocol's pre-registered compute_tier",
    )

    world_sim_cmd = subparsers.add_parser(
        "world-sim",
        help=(
            "three-tier idea rehearsal under ideas/<name>/world-sim/; "
            "imagined numbers cannot corroborate or rewrite expected_direction"
        ),
    )
    world_sim_cmd.add_argument("idea_dir", type=Path)
    world_sim_cmd.add_argument(
        "--review-mode",
        choices=("simple", "full"),
        default="simple",
        help="simple = one-pass review; full = two sequential review passes",
    )
    world_sim_cmd.add_argument(
        "--smart-skip",
        action="store_true",
        help="reuse the last report when brief + plan + wiki did not change",
    )
    world_sim_cmd.add_argument(
        "--what-if",
        action="append",
        default=[],
        dest="what_if",
        help="JSON object with name, claim, description, estimated_gpu_hours (repeatable)",
    )

    run = subparsers.add_parser(
        "run",
        help=(
            "dispatch a compiled protocol through a named runner "
            "(host, host-heavy, external). external emits a SkyPilot "
            "sky_task.yaml + receipt; it cannot corroborate"
        ),
    )
    run.add_argument("folder", type=Path)
    run.add_argument(
        "--runner",
        default="",
        help="host | host-heavy | external; default: protocol host.runner / compute_tier",
    )
    run.add_argument("--timeout", type=float, default=None)
    run.add_argument(
        "--on-slice",
        action="store_true",
        help="bind the frozen slice instead of reconstructing the parent",
    )

    campaign = subparsers.add_parser(
        "campaign",
        help=(
            "ledger of missions against a frozen intent: init, append, "
            "rollback, status. rollback does not rewrite finished verdicts"
        ),
    )
    campaign.add_argument("action", choices=("init", "append", "rollback", "status", "close"))
    campaign.add_argument("--id", required=True, dest="campaign_id")
    campaign.add_argument("--root", type=Path, default=None, help="repo root (default: cwd)")
    campaign.add_argument("--intent", default="")
    campaign.add_argument("--mission", type=Path, default=None)
    campaign.add_argument("--reason", default="")
    campaign.add_argument("--note", default="")
    campaign.add_argument(
        "--authority",
        default="operator",
        help="operator|manager|planner|engineer|reviewer — recorded, not an LLM role",
    )

    artifact = subparsers.add_parser(
        "compile-artifact",
        help=(
            "dump attested.json for an existing writing skill "
            "(jin-s13/ai-research-writing-skill). not a paper, not a discovery"
        ),
    )
    artifact.add_argument("folder", type=Path)
    artifact.add_argument(
        "--venue",
        default="",
        help="declared venue id written into WRITING.md; not a paper compiler",
    )

    ingest = subparsers.add_parser(
        "ingest-papers",
        help=(
            "bank retrieved papers into the neighbourhood wiki after "
            "arXiv / CrossRef / Scholar verification "
            "(limitation sentences become agenda, not solved claims)"
        ),
    )
    ingest.add_argument(
        "--state-store",
        type=Path,
        required=True,
        help="research_state.json for this neighbourhood",
    )
    ingest.add_argument("--anchor", required=True, help="seed / near concept label")
    ingest.add_argument(
        "--file",
        type=Path,
        required=True,
        dest="papers_file",
        help="JSON list of {title, arxiv_id|cite_id, abstract, ...}",
    )
    ingest.add_argument("--corpus", default="", help="corpus id (default: production)")
    ingest.add_argument("--seen-at", default="", help="YYYY-MM-DD; default today")
    ingest.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "bank rows without arXiv/CrossRef/Scholar checks "
            "(test fixtures and already-attested dumps only)"
        ),
    )

    attest = subparsers.add_parser(
        "attest-run",
        help=(
            "record an external replication of a compiled protocol: submit "
            "the two arm numbers from a run the sandbox cannot host (GPU, "
            "cluster); the pre-registered arithmetic judges them. "
            "externally_replicated is not corroborated"
        ),
    )
    attest.add_argument("folder", type=Path)
    attest.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help='JSON file with {"treatment": <num>, "control": <num>}; a verdict key is refused',
    )
    attest.add_argument(
        "--environment",
        default="",
        help="where it ran: hardware, framework, scale (recorded verbatim)",
    )
    attest.add_argument("--runner", default="", help="who ran it (recorded verbatim)")
    execute.add_argument(
        "--on-slice",
        action="store_true",
        help="bind the in-loop slice instead of reconstructing the parent",
    )

    research.add_argument(
        "--host-execute",
        action="store_true",
        default=True,
        help="after compiling the plan, run farfield execute on each live card folder",
    )
    research.add_argument(
        "--no-host-execute",
        action="store_false",
        dest="host_execute",
        help="compile the plan only; do not host-run protocol.json",
    )
    research.add_argument(
        "--host-timeout",
        type=float,
        default=None,
        help=(
            "override seconds for each in-mission host run (no network); "
            "default follows each protocol's pre-registered compute_tier. "
            "Scale via tiers and farfield freeze, never by opening the sandbox"
        ),
    )

    subparsers.add_parser(
        "llm-doctor",
        help="ask the configured endpoint which model ids it actually serves",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        runtime = FarfieldRuntime.create(args.project, args.intent)
        print(json.dumps({"intent": runtime.standing_intent()}, ensure_ascii=False))
        return 0
    if args.command == "llm-doctor":
        try:
            backend = Backend.from_env()
        except LLMUnavailable as exc:
            print(json.dumps({"ok": False, **exc.record.to_dict()}, indent=2))
            return 2
        report = LLMClient(backend, Path.home() / ".cache" / "farfield" / "llm").doctor()
        print(json.dumps({"ok": report["configured_model_is_served"], **report}, indent=2))
        return 0 if report["configured_model_is_served"] else 2
    if args.command == "freeze":
        from .extras.freeze import (
            FreezeError,
            derive_world,
            freeze_history,
            freeze_world,
        )

        try:
            if args.derive_from:
                slice_rule = args.slice_rule
                if slice_rule == "first_edges:10000":
                    slice_rule = "all"
                fixture = derive_world(
                    world_id=args.world_id,
                    parent=args.derive_from,
                    schema=args.schema,
                    slice_rule=slice_rule,
                    catalog=args.catalog,
                    title=args.title,
                    domains=args.domain,
                    force=args.force,
                )
            elif args.run_dir is not None:
                if not args.history_format:
                    raise FreezeError("--dir needs --format (dgm | openevolve)")
                slice_rule = args.slice_rule
                if slice_rule == "first_edges:10000":
                    slice_rule = "all"
                fixture = freeze_history(
                    world_id=args.world_id,
                    run_dir=args.run_dir,
                    fmt=args.history_format,
                    slice_rule=slice_rule,
                    catalog=args.catalog,
                    title=args.title,
                    source=args.source,
                    domains=args.domain,
                    retrieved_at=args.retrieved_at,
                    force=args.force,
                )
            else:
                fixture = freeze_world(
                    world_id=args.world_id,
                    schema=args.schema,
                    slice_rule=args.slice_rule,
                    catalog=args.catalog,
                    url=args.url,
                    source_file=args.source_file,
                    title=args.title,
                    source=args.source,
                    domains=args.domain,
                    retrieved_at=args.retrieved_at,
                    load_hint=args.load_hint,
                    force=args.force,
                )
        except FreezeError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(
            json.dumps(
                {"ok": True, **fixture.to_dict(), "root": str(fixture.root)},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "paper":
        from .extras.llm import TokenLedger
        from .extras.paperplan import PaperError, card_id_for, compile_paper

        idea_dir = Path(args.idea_dir).resolve()
        workspace = idea_dir.parent.parent
        try:
            card_id = args.card or card_id_for(idea_dir)
            backend = Backend.from_env()
            client = LLMClient(
                backend,
                Path.home() / ".cache" / "farfield" / "llm",
                ledger=TokenLedger(api_calls=3.0 + 2.0 * max(0, args.rounds), token_cost=float("inf")),
                mode="live",
            )
            record = compile_paper(
                workspace, idea_dir, card_id, client=client, rounds=args.rounds, lang=args.lang
            )
        except (PaperError, LLMUnavailable) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"ok": True, **record}, ensure_ascii=False))
        return 0
    if args.command == "analyze":
        from .extras.exploratory import AnalysisError, analyze_world
        from .extras.freeze import default_catalog

        catalog = args.catalog if args.catalog is not None else default_catalog()
        try:
            source = args.script.read_text(encoding="utf-8") if args.script is not None else None
            record = analyze_world(args.world_id, catalog_root=catalog, source=source, label=args.label)
        except (AnalysisError, OSError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"ok": True, **record}, ensure_ascii=False))
        return 0
    if args.command == "harvest":
        from .extras.freeze import default_catalog
        from .extras.harvest import (
            HarvestError,
            load_recipe,
            register_recipe,
            registered,
            run_harvest,
            sibling_seeds,
        )

        catalog = args.catalog if args.catalog is not None else default_catalog()
        try:
            recipe = load_recipe(args.recipe)
            if args.register_only:
                if registered(recipe, catalog=catalog):
                    print(json.dumps({"ok": True, "registered": True, "recipe_digest": recipe.digest(), "already": True}))
                    return 0
                event = register_recipe(recipe, catalog=catalog)
                print(json.dumps({"ok": True, "registered": True, "recipe_digest": recipe.digest(), "seq": event.get("seq")}))
                return 0
            result = run_harvest(recipe, catalog=catalog, force=args.force, reuse_run=args.reuse_run)
        except HarvestError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        result["confirming_seeds"] = sibling_seeds(recipe.world_id, catalog=catalog)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "execute":
        from .extras.hostexp import ExecuteError, execute_protocol

        try:
            record = execute_protocol(
                args.folder,
                timeout_seconds=args.timeout,
                prefer_parent=not args.on_slice,
            )
        except ExecuteError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(record, ensure_ascii=False))
        return 0 if record.get("ok") else 2
    if args.command == "world-sim":
        from .extras.worldsim import WhatIfExperiment, WorldSimError, run_world_sim

        extras = []
        for raw in args.what_if or []:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
                return 2
            if not isinstance(payload, dict):
                print(
                    json.dumps(
                        {"ok": False, "error": "--what-if must be a JSON object"},
                        ensure_ascii=False,
                    )
                )
                return 2
            extras.append(WhatIfExperiment.from_dict(payload))
        try:
            backend = Backend.from_env()
            client = LLMClient(backend, Path.home() / ".cache" / "farfield" / "llm")
            record = run_world_sim(
                args.idea_dir,
                client=client,
                review_mode=args.review_mode,
                smart_skip=args.smart_skip,
                what_if=extras,
            )
        except (WorldSimError, LLMUnavailable) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(record, ensure_ascii=False))
        return 0 if record.get("ok") else 2
    if args.command == "run":
        from .extras.hostexp import ExecuteError
        from .extras.plugins import PluginHost, repo_root
        from .extras.runners import dispatch_run

        try:
            record = dispatch_run(
                args.folder,
                runner=args.runner or None,
                timeout_seconds=args.timeout,
                prefer_parent=not args.on_slice,
                plugin_host=PluginHost.load(repo_root()),
            )
        except ExecuteError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(record, ensure_ascii=False))
        return 0 if record.get("ok") or record.get("status") == "awaiting_metrics" else 2
    if args.command == "campaign":
        from .extras.campaign import (
            CampaignError,
            append_mission,
            close_campaign,
            init_campaign,
            load_campaign,
            rollback_stage,
        )

        root = args.root or Path.cwd()
        try:
            if args.action == "init":
                payload = init_campaign(
                    root, args.campaign_id, intent=args.intent, authority=args.authority
                )
            elif args.action == "append":
                if args.mission is None:
                    raise CampaignError("append needs --mission")
                payload = append_mission(
                    root,
                    args.campaign_id,
                    args.mission,
                    authority=args.authority,
                    note=args.note,
                )
            elif args.action == "rollback":
                payload = rollback_stage(
                    root,
                    args.campaign_id,
                    reason=args.reason,
                    authority=args.authority,
                )
            elif args.action == "close":
                payload = close_campaign(
                    root, args.campaign_id, authority=args.authority, note=args.note
                )
            else:
                payload = load_campaign(root, args.campaign_id)
        except CampaignError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"ok": True, **payload}, ensure_ascii=False))
        return 0
    if args.command == "compile-artifact":
        from .extras.artifact import ArtifactError, compile_artifact
        from .extras.plugins import PluginHost, repo_root

        try:
            manifest = compile_artifact(
                args.folder,
                plugin_host=PluginHost.load(repo_root()),
                venue=args.venue,
            )
        except ArtifactError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"ok": True, **manifest}, ensure_ascii=False))
        return 0
    if args.command == "ingest-papers":
        from datetime import date

        from .extras.knowledge import admit_works
        from .extras.livefeed import FreshWork
        from .extras.mission import resolve_production_corpus
        from .extras.state import record_wiki

        try:
            rows = json.loads(args.papers_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        if not isinstance(rows, list):
            print(json.dumps({"ok": False, "error": "papers file must be a JSON list"}, ensure_ascii=False))
            return 2
        incoming = [row for row in rows if isinstance(row, dict)]
        harvest = admit_works(
            [FreshWork.from_dict(row) for row in incoming],
            verify=not args.skip_verify,
        )
        added = record_wiki(
            args.state_store,
            resolve_production_corpus(requested=args.corpus),
            args.anchor,
            harvest.works,
            seen_at=args.seen_at or date.today().isoformat(),
        )
        print(
            json.dumps(
                {"ok": True, "added": added, **harvest.to_dict()},
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "attest-run":
        from .extras.hostexp import ExecuteError, attest_external_run

        try:
            record = attest_external_run(
                args.folder,
                metrics_path=args.metrics,
                environment=args.environment,
                runner=args.runner,
            )
        except ExecuteError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(record, ensure_ascii=False))
        return 0
    if args.command == "research":
        from .extras.mission import resolve_production_corpus, run_mission
        from .extras.openworld.runtime import run_research
        from .extras.workspace import jsonable, mission_dir, var_dir
        import sys

        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

        dest = mission_dir(args.project, args.topic)
        dest.mkdir(parents=True, exist_ok=True)
        runtime = var_dir(args.project)
        last = None
        try:
            if args.legacy_pipeline:
                stream = run_mission(
                    args.topic,
                    jumps=args.jumps,
                    candidates=args.candidates,
                    reframes=args.reframes,
                    parallel=args.parallel,
                    polish_rounds=args.polish_rounds,
                    explore=args.explore,
                    experiment_rounds=args.experiment_rounds,
                    idea_rounds=args.idea_rounds,
                    venue=args.venue,
                    world=args.world,
                    world_sim=args.world_sim,
                    paper_rounds=args.paper_rounds,
                    min_ideas=args.min_ideas,
                    human_gate=args.human_gate,
                    approved=tuple(args.approve),
                    fresh=args.fresh,
                    host_execute=args.host_execute,
                    host_timeout=args.host_timeout,
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    api_calls=args.api_calls,
                    corpus_id=resolve_production_corpus(requested=args.corpus),
                    workspace=dest,
                    policy_log=args.policy_log or (runtime / "policy_log.json"),
                    state_store=args.state_store or (runtime / "research_state.json"),
                    judges_file=args.judges_file or (runtime / "research_judges.json"),
                    policy_file=args.policy_file or (runtime / "research_policy.json"),
                )
            else:
                stream = run_research(
                    args.topic,
                    workspace=dest,
                    horizon=args.horizon if args.horizon is not None else args.jumps,
                    world=args.world if args.world not in {"auto", "none", ""} else None,
                    world_id="" if args.world in {"auto", "none", ""} else str(args.world),
                    corpus_id=resolve_production_corpus(requested=args.corpus),
                )
            for event in stream:
                last = event
                print(json.dumps(jsonable(event), ensure_ascii=False), flush=True)
        except Exception as exc:
            import traceback

            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        return 0 if last and last.get("stage") == "done" else 2
    runtime = FarfieldRuntime(args.project)
    if args.command == "objective":
        objective_id = runtime.declare_initial_objective(
            text=args.text,
            success_criteria=args.success,
            constraints=args.constraint,
        )
        print(json.dumps({"objective_id": objective_id}))
        return 0
    if args.command == "verify":
        ok, message = runtime.verify()
        print(json.dumps({"ok": ok, "message": message}))
        return 0 if ok else 2
    if args.command == "status":
        events = runtime.events()
        payload = {
            "standing_intent": runtime.standing_intent(),
            "event_count": len(events),
            "active_update_ids": runtime.active_update_ids(),
            "quarantined_updates": sum(
                event["event_type"] == "update_quarantined" for event in events
            ),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
