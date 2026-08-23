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
        help="AI Scientist: topic → ideas evolved in-mission → evidence → research plan",
    )
    research.add_argument("project", type=Path)
    research.add_argument("--topic", required=True)
    research.add_argument(
        "--jumps",
        type=int,
        default=4,
        help="far-field cap; auto-explore may stop earlier on evidence",
    )
    research.add_argument(
        "--explore",
        choices=("auto", "fixed"),
        default="auto",
        help="auto: open slots until a research plan exists or the cap; fixed: generate all jumps",
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
            "-1 auto (up to 2 extra rewrites of the same idea after the first draft); "
            "0 one-shot; evolve the hypothesis in this mission, not a new far jump"
        ),
    )
    research.add_argument(
        "--world",
        default="auto",
        help=(
            "attested fixture id under worlds/; "
            "auto = pick a real frozen trace by topic; none = SYNTHETIC "
            "(coherence check, cannot corroborate)"
        ),
    )
    research.add_argument("--model", default="")
    research.add_argument("--base-url", default="")
    research.add_argument("--api-key", default="")
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
        "--schema",
        default="undirected_graph",
        help=(
            "SNAP edgelist (undirected_graph), fasta, text_stream, "
            "numeric_table, or symbolic_trace (labeled DOT digraph)"
        ),
    )
    freeze.add_argument(
        "--slice",
        default="first_edges:10000",
        dest="slice_rule",
        help=(
            "pre-registered rule: all, first_edges:<n>, first_bases:<n>, "
            "first_tokens:<n>, first_rows:<n>, first_states:<n>"
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
        from .extras.freeze import FreezeError, freeze_world

        try:
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
                args.folder, plugin_host=PluginHost.load(repo_root())
            )
        except ArtifactError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps({"ok": True, **manifest}, ensure_ascii=False))
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
        from .extras.mission import run_mission
        from .extras.workspace import mission_dir, var_dir
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
            for event in run_mission(
                args.topic,
                jumps=args.jumps,
                candidates=args.candidates,
                reframes=args.reframes,
                parallel=args.parallel,
                polish_rounds=args.polish_rounds,
                explore=args.explore,
                experiment_rounds=args.experiment_rounds,
                idea_rounds=args.idea_rounds,
                world=args.world,
                host_execute=args.host_execute,
                host_timeout=args.host_timeout,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                workspace=dest,
                policy_log=args.policy_log or (runtime / "policy_log.json"),
                state_store=args.state_store or (runtime / "research_state.json"),
                judges_file=args.judges_file or (runtime / "research_judges.json"),
                policy_file=args.policy_file or (runtime / "research_policy.json"),
            ):
                last = event
                print(json.dumps(event, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
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
