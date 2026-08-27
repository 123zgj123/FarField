"""The production mission: one AI Scientist step, as an event stream.

One mission is one pass of a single loop. Every module hangs off it:

    read the Research State (known / open / contradicted / rejected / elites /
    generation program / literature wiki)
    → plan an agenda (when a program exists, the first far slot is
      exploitation and remaining slots stay as exploration fallback;
      auto-explore opens them only while attested evidence says there is
      no research plan a colleague can start)
    → generate (few cards, constrained by the working program)
    → gates decide who ENTERS research (ineligible / not selected / entered)
    → on a *text* gate kill, refine the same pair against working H
      (idea_rounds). Graph-dead pairs open a new landing instead.
    → cheap evidence: survey → prior → diagnosis → two-arm probe on
      an attested freeze (or a SYNTHETIC coherence check if the operator
      asked for no world); WORLD_INCOMPATIBLE refuses a substitute freeze
      and does not construct a GENERATED stand-in for verification
    → optional recorded value tournament among live cards (colleague
      opinion only; Elo cannot pick the lead or steer routing)
    → only ideas the evidence did not weaken get a compiled research plan
      (briefing LLM is optional; protocol is compiled either way)
    → lexicographic reading order: WORLD discriminant, wiki-gap aim, host
      execute, then Elo as a tie-break
    → this idea's H is compiled after each landing *including evidence*
      so the next refine of THE SAME pair sees the probe and diagnosis;
      a new far jump does not inherit another idea's program
    → routing may update on held-out WORLD-supports / promotions

This is the product path. There is no control-arm campaign: `farfield research`
/ this module is the only scientist loop. Two loops only: rewrite the same
idea after a text-gate death, and rewrite the same test after a crash or
an uninformative probe. Crossover spray is not on this path. Persisting H
at mission end is for an optional later run the user starts; it is not the
finish line. LLM value
judgment, when it runs, is recorded. It never enters the promotion ladder,
never overrides a gate or a probe, and does not choose the program's lead.
Skills are PluginHost hooks (trusted plugin.py) plus optional prompt colour;
DeepSeek Harness is one plugin that launches `dsh` when present.

Exits are distinct and named, because they mean different things:

- `ineligible`      a gate killed the card; it never entered research.
                    Banked as a rejection capsule (failure memory).
- `not_selected`    lost the same-jump elimination; not a scientific verdict.
- `closed_by_prior` the claim already appears in a paper this mission
                    retrieved; reading beats researching.
- `supports` / `weakens` / `uninformative`  the probe's arithmetic verdict
                    against the pre-registered direction.
- ladder statuses   `speculative → corroborated → verified`, down moves
                    `weakened` / `closed_by_prior`; nothing reaches
                    `verified` without support in two missions.

Missions are LLM-first: every card is written live, every call recorded
for audit; an unreachable endpoint yields `blocked` with an unlock
condition instead of canned answers. With a workspace, every live
retrieval is frozen into `evidence_snapshot/` the moment it arrives.

Stages, in stream order:

- `corpus`      knowledge base: slice, freshness, judges on the bench
- `anchor`      the topic anchored to its nearest concepts
- `state`       what earlier missions established, left open, contradicted,
                and rejected in this neighbourhood — the one long-term memory
- `fresh`       the newest submissions near the anchor (live feed)
- `agenda`      this mission's plan: slots are a cap; `explore=auto` opens them
                only while attested evidence says there is no live line
- `parallel`    a wave of independent jobs (generate / evidence / debate / writeup)
- `jump`        one generation slot landed (far-field or reframe)
- `explore_decision`  auto-explore: continue or stop, with the attested reason
- `generating` / `card` / `refused`   the model writing a candidate
- `verdict`     the gates ruled: graph checks, text kills, and an `outcome`
                (`entered` / `ineligible` / `not_selected`)
- `fresh_check` an entered card's pair probed against today's indexes (advisory)
- `surveying` / `survey`   papers pulled around the pair (frozen to snapshot)
- `prior`       claim-level span support; can close the idea as already stated
- `wiki`        retrieved papers compiled into the neighbourhood research state
- `diagnosing` / `diagnosis`   competing explanation + pre-registered two-arm
                design (direction, margin) — before any write-up
- `probing` / `probe`   the experiment as a whitelisted stdlib script
- `evidence`    supports / weakens / uninformative, by arithmetic alone
- `experiment_decision`  iterate the test, not the claim: rewrite the
                script on implementation failure, rewrite the diagnosis
                on an uninformative design, stop when the probe
                discriminates or the attempt cap is hit
- `heavy_probing` / `heavy_probe` / `heavy_evidence`   the confirmation
                tier: a corroborated line's second support triggers a 10x,
                seed-replicated rerun — verified is granted only past it
- `briefing` / `brief`   the research-ready write-up (only for ideas the
                evidence did not weaken)
- `reviewing` / `review`   a six-dimension value vector, opinion only
- `refining` / `polish_stop`   optional revision rounds
- `note`        markdown + compilable LaTeX + next-week protocol, all
                compiled from attested fields (not a new generation)
- `program`     the committed generation brief for the next mission
                (why this line, what the next card must do)
- `baseline`    same-anchor control stake: gate survival of the jump arm vs
                matched-alienness random draws, one measure both arms
- `ranked`      reading order by evidence status, never by a collapsed score
- `promotion`   each entered hypothesis's move on the ladder, with reason
- `policy`      the routing meta-loop ran: candidate, held-out score, decision
- `verifier`    the judge meta-loop ran: shadow judges promoted to lethal
                only on held-out WORLD-weakens catches
- `shadow_judge` a repeated failure earned one predicate proposal onto the
                shadow bench (or a named refusal); it cannot kill from there
- `brief_refused` / `probe_refused` / `diagnosis_refused`   schema teeth
- `blocked`     the mission cannot continue, with unlock condition
- `done`        mission packet: ranked do-next / closed / weakened / open,
                plus the books (cards, entered, questions, spend)
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from ..graph import GraphSnapshot, load_graph
from .embed import EmbeddingSpace, cosine_distance, embed_nodes
from .brief import ResearchBrief, compile_brief, write_brief
from .archive import QDArchive
from .farspace import JUMP_OPERATORS, sample_jumps
from .generate import (
    ConceptOracle,
    GeneratedCard,
    GenerationRefused,
    check_pair,
    generate_card,
    probe_generated,
    refine_card,
)
from .livefeed import CompositeFeed, FeedBlocked, FreshWork
from .baseline import control_stake
from .diagnose import judge_probe, judge_replicated, write_diagnosis
from .routing import append_mission, load_policy, maybe_update, preferred_operators
from .verifier import (
    SHADOW_TRIGGER,
    compile_bench,
    load_bench,
    maybe_promote as maybe_promote_judges,
    propose_shadow,
    shadow_flags,
)
from .value import (
    DEFAULT_BETA,
    archive_fitness,
    attach_observations,
    run_tournament,
    score_live,
    selection_key,
    tournament_pool,
    value_lines,
)
from .prior import apply_x_to_y, strongest_prior
from .probeexp import (
    HEAVY_TIMEOUT_SECONDS,
    ProbeRefused,
    replication_seeds,
    run_probe,
    write_probe,
)
from .parallel import map_parallel, resolve_workers
from .reframe import GRAPH_SKIP_REASON, OPERATORS as REFRAME_OPERATORS, generate_reframe, refine_reframe
from .review import refine_brief, review_idea
from .snapshot import RecordingFeed, SNAPSHOT_DIR
from .skills import distill_skill, select_skills, skills_prompt_block
from .harness import persist_skill
from .plugins import PluginHost, bind_host, get_host
from .state import (
    apply_outcomes,
    elites_for,
    hypothesis_id,
    line_status,
    load as load_state,
    probe_history,
    program_for,
    prompt_lines,
    record_elites,
    record_program,
    record_rejections,
    record_wiki,
    rejections_for,
    summary as state_summary,
    topic_key,
    wiki_for,
    wiki_works,
)
from .llm import Backend, LLMClient, LLMUnavailable, TokenLedger, resolve_backend
from .predicates import (
    INSTALLED_PREDICATES,
    Predicate,
    PredicateError,
    criteria_lines,
    run as run_predicate,
)
from .explore import (
    DEFAULT_EXPLORE,
    DEFAULT_JUMP_CAP,
    IDEA_AUTO,
    decide_idea_refine,
    decide_next,
    has_research_plan,
    normalize_explore,
    resolve_idea_rounds,
    resolve_polish_rounds,
)
from .iterate import (
    DEFAULT_EXPERIMENT_ROUNDS,
    classify_bottleneck,
    decide_retry,
    resolve_experiment_rounds,
)
from .packet import render_mission_packet, render_protocol
from .program import compile_program, prompt_lines as program_prompt_lines, target_line as program_target
from .wiki import merge_works, prompt_lines as wiki_prompt_lines, gap_lines as wiki_gap_lines
from .writeup import compile_tex, render_note
from .workspace import (
    EVENTS_FILE,
    append_recorded_event,
    candidate_dir,
    llm_cache_dir,
    seal_candidate,
    var_dir,
    write_idea,
    write_summary,
)
from .world import (
    KIND_SYNTHETIC,
    KIND_WORLD,
    attested_freeze,
    bind_world,
    incompatible_family,
    infer_requirement,
    lineage_conflicts,
    load_catalog,
    match_world,
    reads_world_data,
    record_world_wishlist,
    resolve_world,
)
from .genworld import (
    HONESTY,
    KIND_GENERATED,
    construct_world,
    execute_world,
    load_world_payload,
)
from . import chain
from .dynworld import (
    RESPONSE_FLOOR,
    arm_separation,
    forward_simulate,
    has_dynamics,
    levers_of,
    materialize_placebo,
    response_card,
    scout_summary,
    world_consumed,
)
from .worldpath import write_world_path
from .evidence import (
    TIER_SAME_INSTANCE,
    TIER_SYNTHETIC,
    ablation_required,
    campaign_funnel,
    evidence_id,
    hypothesis_revision_id,
    sha256_text,
)

PRODUCTION_CORPUS = "ds-arxiv-concepts-2026"
# Clone-sized graph that ships in git. Production G_full.json exceeds GitHub's
# 100 MB file limit; load_assets falls back here when that file is absent.
SHIPPED_CORPUS = "attn-concepts-s1"
NEAR_LABELS = 6
DEFAULT_JUMPS = DEFAULT_JUMP_CAP
DEFAULT_CANDIDATES = 1
DEFAULT_POLISH = 0
DEFAULT_EXPERIMENT = DEFAULT_EXPERIMENT_ROUNDS
DEFAULT_IDEA = IDEA_AUTO
# Far retrieve stays outside the seed neighborhood but inside the topic ball.
DOMAIN_QUANTILE = 0.4
TOKENS_PER_CALL = 40000.0


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _operator_allows_synthetic(world_request: str) -> bool:
    """True when the operator asked for no freeze: SYNTHETIC coherence is allowed."""
    text = str(world_request or "").strip().lower()
    return text in {"", "none", "synthetic", "off", "0"}


def _probe_kind_for(world: Any, source: str) -> str:
    """GENERATED fixtures are readable but cannot corroborate."""
    if world is None or not reads_world_data(source, world):
        return KIND_SYNTHETIC
    if str(getattr(world, "role", "") or "") == "generated":
        return KIND_GENERATED
    return KIND_WORLD


def _world_work_dir(workspace: Path | None, card_id: str) -> Path:
    if workspace is not None:
        dest = candidate_dir(workspace, card_id) / "world"
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    return Path(tempfile.mkdtemp(prefix="ff-world-"))


def _construct_iteration_world(
    client: Any,
    card: GeneratedCard,
    requirement: Any,
    dest: Path,
    *,
    diagnosis: Any = None,
    previous: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    adapted: bool = False,
    lineage: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build or adapt the experimental world for this registration.

    One dest. Not a sidecar diagnostic fixture.
    """
    fixture = construct_world(
        client,
        card,
        requirement or {},
        dest=dest,
        diagnosis=diagnosis,
        previous=previous,
        errors=errors,
        lineage=lineage,
    )
    payload = load_world_payload(fixture.root)
    if payload.get("acquisition") == "incomplete":
        return fixture, [
            {
                "stage": "world_constructed",
                "card_id": card.card_id,
                "world": fixture.to_dict(),
                "origin": "generated",
                "honesty": payload.get("honesty") or HONESTY,
                "executable": False,
                "acquisition": "incomplete",
                "errors": ["acquisition incomplete"],
                "adapted": False,
                "lineage": lineage.to_dict() if lineage is not None and hasattr(lineage, "to_dict") else None,
            }
        ]
    exec_errors = execute_world(payload)
    events = [
        {
            "stage": "world_constructed",
            "card_id": card.card_id,
            "world": fixture.to_dict(),
            "origin": "generated",
            "honesty": HONESTY,
            "executable": not exec_errors,
            "errors": list(exec_errors),
            "adapted": bool(adapted),
            "lineage": lineage.to_dict() if lineage is not None and hasattr(lineage, "to_dict") else None,
        }
    ]
    if exec_errors and not adapted:
        fixture = construct_world(
            client,
            card,
            requirement or {},
            dest=dest,
            diagnosis=diagnosis,
            previous=payload,
            errors=exec_errors,
            lineage=lineage,
        )
        payload = load_world_payload(fixture.root)
        exec_errors = execute_world(payload)
        events.append(
            {
                "stage": "world_constructed",
                "card_id": card.card_id,
                "world": fixture.to_dict(),
                "origin": "generated",
                "honesty": HONESTY,
                "executable": not exec_errors,
                "errors": list(exec_errors),
                "adapted": True,
                "lineage": lineage.to_dict() if lineage is not None and hasattr(lineage, "to_dict") else None,
            }
        )
    return fixture, events


def _resolve_card_world(
    card: GeneratedCard,
    topic: str,
    catalog: dict,
    bound_world: Any,
    world_request: str,
) -> tuple[Any, dict[str, Any] | None]:
    """Per-claim world. Auto never substitutes an incompatible fixture."""
    text = str(world_request or "").strip().lower()
    if text in {"", "none", "synthetic", "off", "0"}:
        return bound_world, None
    if text != "auto":
        return bound_world, None
    req = infer_requirement(topic, card.claim, card.prediction)
    if req is not None and incompatible_family(req.object_type):
        return None, {
            "stage": "world_incompatible",
            "card_id": card.card_id,
            "requirement": req.to_dict(),
            "reason": (
                f"no attested fixture can satisfy {req.object_type}; "
                "refusing to bind a substitute world"
            ),
        }
    fixture = match_world(req, catalog, preferred=bound_world)
    if fixture is None:
        return None, {
            "stage": "world_incompatible",
            "card_id": card.card_id,
            "requirement": req.to_dict() if req is not None else {"object_type": ""},
            "reason": (
                "no attested fixture matches this claim; "
                "refusing to bind a substitute world"
            ),
        }
    return fixture, None


def _fold_record_into_outcome(outcome: dict[str, Any], record: dict[str, Any]) -> None:
    """Copy attested fields after host execute, before the promotion ladder."""
    outcome["prior_kills"] = record.get("prior_kills")
    outcome["verdict"] = record.get("verdict")
    outcome["alternative"] = record.get("alternative") or ""
    outcome["experiment"] = record.get("experiment") or ""
    outcome["failed_experiments"] = list(record.get("failed_experiments") or [])
    outcome["heavy_confirmed"] = record.get("heavy_confirmed")
    outcome["heavy"] = record.get("heavy")
    outcome["chain_path"] = str(record.get("chain_path") or "")
    outcome["probe_kind"] = record.get("probe_kind") or KIND_SYNTHETIC
    outcome["value_vector"] = record.get("value_vector")
    outcome["host_ok"] = record.get("host_ok")
    outcome["evidence_id"] = record.get("evidence_id")
    outcome["host_evidence_id"] = record.get("host_evidence_id")
    outcome["experiment_digest"] = record.get("experiment_digest")
    outcome["world_digest"] = record.get("world_digest")
    outcome["data_digest"] = record.get("data_digest")
    outcome["host_experiment_digest"] = record.get("host_experiment_digest")
    outcome["host_world_digest"] = record.get("host_world_digest")
    outcome["host_data_digest"] = record.get("host_data_digest")
    outcome["host_replication"] = record.get("host_replication")
    outcome["world_incompatible"] = bool(record.get("world_incompatible"))
    outcome["world_binding"] = str(record.get("world_binding") or "")
    if "world_consumed" in record:
        outcome["world_consumed"] = bool(record["world_consumed"])
    if "world_has_dynamics" in record:
        outcome["world_has_dynamics"] = bool(record["world_has_dynamics"])
    if "world_forecast_agrees" in record:
        outcome["world_forecast_agrees"] = bool(record["world_forecast_agrees"])
    if isinstance(record.get("idea_analysis"), dict):
        outcome["idea_analysis"] = dict(record["idea_analysis"])
    outcome["mechanism_identified"] = record.get("mechanism_identified", True)
    outcome["protocol_complete"] = record.get("protocol_complete")
    outcome["ledger"] = record.get("ledger") or ""
    outcome["ablation_required"] = bool(record.get("ablation_required"))
    outcome["host_error"] = record.get("host_status") or record.get("host_error")


@dataclass(frozen=True)
class MissionAssets:
    """Everything a mission reads that does not depend on the topic."""

    corpus_uid: str
    fresh_until: str
    graph: GraphSnapshot
    oracle: ConceptOracle
    space: EmbeddingSpace
    label_of: dict[str, str]
    label_to_node: dict[str, str]
    judges: tuple[Predicate, ...]
    standards: tuple[str, ...]


_ASSETS: dict[str, MissionAssets] = {}


def installed_judges(root: Path) -> tuple[Predicate, ...]:
    """Kernel plus the two structure judges that earned installation."""
    del root
    return INSTALLED_PREDICATES


def knowledge_end(root: Path, corpus_id: str) -> str:
    """The last record datestamp in the recorded harvest: how fresh the
    oracle's 'already combined' actually is."""
    corpus_dir = root / "concepts" / corpus_id
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    pages = manifest.get("pages") or []
    raw_name = ""
    if pages:
        raw_name = str(pages[-1].get("file") or "")
    if raw_name:
        raw = root / "concepts" / manifest["spec"]["corpus_id"] / "raw"
        last_page = raw / raw_name
        if last_page.is_file():
            stamps = re.findall(
                r"<datestamp>(\d{4}-\d\d-\d\d)</datestamp>", last_page.read_text()
            )
            if stamps:
                return max(stamps)
    return str(manifest.get("fetched_at", ""))[:10]


def load_assets(corpus_id: str = PRODUCTION_CORPUS, root: Path | None = None) -> MissionAssets:
    """Load once, reuse across missions: the corpus is heavy, topics are not.

    A git clone ships `SHIPPED_CORPUS` only. Asking for `PRODUCTION_CORPUS`
    without its graph is not a crash: the same mission path runs on the
    shipped slice so `farfield research` and the unit tests work offline.
    Rebuild the production graph with `scripts/build_concepts.py --replay`.
    """
    root = root or _root()
    if corpus_id in _ASSETS:
        return _ASSETS[corpus_id]
    registry = json.loads((root / "concepts/arxiv_registry.json").read_text())
    snap = next(
        item for item in registry["snapshots"] if item["snapshot_id"] == corpus_id
    )
    graph_path = root / snap["full_graph"]
    if not graph_path.is_file():
        if corpus_id == SHIPPED_CORPUS:
            raise FileNotFoundError(
                f"shipped concept graph missing: {graph_path}"
            )
        return load_assets(SHIPPED_CORPUS, root=root)
    graph = load_graph(graph_path)
    label_of = {nid: str(node["title"]) for nid, node in graph.nodes.items()}
    judges = installed_judges(root)
    assets = MissionAssets(
        corpus_uid=graph.snapshot_id,
        fresh_until=knowledge_end(root, corpus_id),
        graph=graph,
        oracle=ConceptOracle.from_graph(graph),
        space=embed_nodes(graph.nodes),
        label_of=label_of,
        label_to_node={label: nid for nid, label in label_of.items()},
        judges=judges,
        standards=criteria_lines(judges),
    )
    _ASSETS[corpus_id] = assets
    return assets


def default_backend(
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> Backend:
    """Whatever this call named, falling back to the machine's env.

    No model is implied. A console that does not say which model to use
    is blocked until the researcher picks one — same discipline as
    Sakana's `--model_writeup`.
    """
    return resolve_backend(model=model, base_url=base_url, api_key=api_key)


def default_client(
    calls: float,
    root: Path | None = None,
    *,
    backend: Backend | None = None,
) -> LLMClient:
    """A live client sized for this mission; every call is still recorded."""
    root = root or _root()
    return LLMClient(
        backend or default_backend(),
        llm_cache_dir(root),
        ledger=TokenLedger(api_calls=calls, token_cost=calls * TOKENS_PER_CALL),
        mode="live",
    )


def rank_ideas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order surviving ideas by attested process signals, not a vibe score.

    1. Closed and weakened lines last.
    2. Evidence class: WORLD-supports, WORLD-uninformative, SYNTHETIC-supports.
    3. Aimed at a wiki limitation sentence (aiming is not solving).
    4. Host execute succeeded.
    5. Still unseen on today's indexes.
    6. A written briefing, then more retrieved papers.
    7. Recorded Elo breaks remaining ties only.
    This is a reading order, not a claim that the first idea is the best
    science. The reviewer's value vector rides along untouched.
    """

    ranked = []
    for index, row in enumerate(sorted(rows, key=selection_key), start=1):
        why = []
        if row.get("prior_kills"):
            why.append("检索到的论文已经把主张写进去了，先读那篇")
        elif row.get("open_today") is True:
            why.append("今天的文献里还没见到这个组合")
        elif row.get("open_today") is False:
            why.append("今天已有论文同时写到这两个词，先读再决定")
        else:
            why.append("没能复核今天的文献")
        kind = str(row.get("probe_kind") or KIND_SYNTHETIC).upper()
        if row.get("verdict") == "supports":
            if kind in {"WORLD", "REAL", "FIXTURE"}:
                why.append("WORLD 双臂探针按预登记方向支持了机制")
            elif kind == "GENERATED":
                why.append("本轮构造世界上的探针支持了机制（自洽，不能佐证）")
            else:
                why.append("SYNTHETIC 探针支持了机制（自洽，不是佐证）")
        elif row.get("verdict") == "weakens":
            why.append("双臂探针反预登记方向，机制被削弱")
        elif row.get("verdict") == "uninformative":
            if kind in {"WORLD", "REAL", "FIXTURE"}:
                why.append("WORLD 探针没分出两臂差别，问题仍开着")
            elif kind == "GENERATED":
                why.append("本轮构造世界上的探针没分出两臂差别，问题仍开着")
            else:
                why.append("探针没分出两臂差别，问题仍开着")
        if row.get("wiki_gap_hit"):
            why.append("瞄准了文献局限句（瞄准不是已经解决）")
        if row.get("host_ok") is True:
            why.append("宿主按同一 digest 执行过协议（不是发现）")
        elif row.get("host_ok") is False:
            why.append("宿主执行被挡住，下场应 freeze/execute")
        if float(row.get("value_score") or 0) > 0:
            why.append("价值辩论只用于并列打破，不是科学价值")
        if row.get("has_brief"):
            n = int(row.get("papers") or 0)
            why.append(f"对照了 {n} 篇近期论文" if n else "写了调研备忘")
        else:
            why.append("还只有假设，调研没写成")
        if row.get("novelty"):
            why.append(f"审稿人给新颖性 {row['novelty']}/5（意见，不是判决）")
        bottleneck = str(row.get("pipeline_bottleneck") or "")
        if bottleneck and bottleneck not in {"informative", "unresolved"}:
            why.append(f"管线停在 {bottleneck}")
        ranked.append(
            {
                "rank": index,
                "card_id": row["card_id"],
                "recommended": index == 1,
                "why": why,
                "pair": list(row.get("pair") or []),
                "value_vector": row.get("value_vector") or None,
                "verdict": row.get("verdict"),
                "claim": row.get("claim") or "",
                "title": row.get("title") or "",
                "experiment": row.get("experiment") or "",
                "failed_experiments": list(row.get("failed_experiments") or []),
                "probe_kind": row.get("probe_kind") or KIND_SYNTHETIC,
                "track": row.get("track") or "",
                "prior_kills": bool(row.get("prior_kills")),
                "has_brief": bool(row.get("has_brief")),
                "generated_world": bool(row.get("generated_world")),
                "wiki_gap_hit": bool(row.get("wiki_gap_hit")),
                "host_ok": row.get("host_ok"),
                "pipeline_bottleneck": row.get("pipeline_bottleneck") or "",
                "mechanism": row.get("mechanism") or "",
                "prediction": row.get("prediction") or "",
                "alternative": row.get("alternative") or "",
                "treatment_arm": row.get("treatment_arm") or "",
                "control_arm": row.get("control_arm") or "",
                "expected_direction": row.get("expected_direction") or "",
                "first_steps": list(row.get("first_steps") or []),
            }
        )
    return ranked


def _rotate(items: tuple[str, ...], k: int) -> tuple[str, ...]:
    """Candidate k sees the same far menu in a different order — a
    meaning-preserving change that gives each candidate its own prompt
    bytes, hence its own recorded call instead of a cache echo."""
    if not items or k % len(items) == 0:
        return items
    shift = k % len(items)
    return items[shift:] + items[:shift]


def _screen_far(
    assets: MissionAssets,
    near: tuple[str, ...],
    near_ids: tuple[str, ...],
    far_candidates: tuple[tuple[str, str], ...],
) -> tuple[
    list[str],
    dict[str, tuple[str, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Graph-prescreen far concepts before paying the generator."""
    viable: list[str] = []
    partners: dict[str, tuple[str, ...]] = {}
    prescreened: list[dict[str, Any]] = []
    kills: list[dict[str, Any]] = []
    for label, node in far_candidates:
        ok_near: list[str] = []
        best_kills: list[str] | None = None
        for near_label, near_node in zip(near, near_ids):
            failed = [
                check.name
                for check in check_pair(assets.oracle, near_node, node)
                if not check.passed
            ]
            if not failed:
                ok_near.append(near_label)
            elif best_kills is None or len(failed) < len(best_kills):
                best_kills = failed
        if ok_near:
            viable.append(label)
            partners[label] = tuple(ok_near)
        else:
            prescreened.append({"label": label, "killed_by": best_kills or []})
            kills.append(
                {"pair": [near[0], label], "killed_by": list(best_kills or [])}
            )
    return viable, partners, prescreened, kills


def _run_entry(
    client: Any,
    feed: Any,
    card: GeneratedCard,
    topic: str,
    *,
    polish_rounds: int,
    run_probes: bool,
    workspace: Path | None,
    prior_probe: dict[str, Any] | None,
    prior_status: str | None,
    skill_blocks: dict[str, str] | None = None,
    experiment_rounds: int = 0,
    world: Any = None,
    wiki_works_pool: list | None = None,
    program: str | tuple[str, ...] = "",
    catalog: dict | None = None,
    world_request: str = "",
    bound_world: Any = None,
) -> dict[str, Any]:
    """One entered card's cheap evidence, collected for a parallel wave.

    Shared mission state is not touched here: the caller folds events into
    records on the main thread, in enter order.
    """
    bundle: dict[str, Any] = {
        "works": [],
        "probe_payload": None,
        "prior": None,
        "skip_writeup": False,
    }
    events: list[dict[str, Any]] = []
    if feed is not None:
        probe = feed.pair_recently_combined(*card.pair)
        event: dict[str, Any] = {
            "stage": "fresh_check",
            "card_id": card.card_id,
            "pair": list(card.pair),
        }
        if isinstance(probe, FeedBlocked):
            event["blocked"] = probe.to_dict()
        else:
            event.update(probe)
        events.append(event)
    card_world, world_event = _resolve_card_world(
        card, topic, catalog or {}, bound_world, world_request
    )
    world_requirement = None
    world_dest = None
    pipeline_world = card_world
    world_gap = world_event is not None
    if world_event is not None:
        events.append(world_event)
        bundle["skip_writeup"] = False
        world_requirement = world_event.get("requirement") or {}
        world_dest = _world_work_dir(workspace, card.card_id)
        pipeline_world = None
    elif pipeline_world is None:
        pipeline_world = world
    for event in _evidence_pipeline(
        client,
        feed,
        card,
        topic,
        polish_rounds=0,
        run_probes=run_probes,
        workspace=workspace,
        prior_probe=prior_probe,
        prior_status=prior_status,
        writeup=False,
        bundle=bundle,
        skill_blocks=skill_blocks,
        experiment_rounds=experiment_rounds,
        world=pipeline_world,
        wiki_works_pool=wiki_works_pool,
        program=program,
        world_requirement=world_requirement,
        world_dest=world_dest,
        world_request=world_request,
        world_gap=world_gap,
    ):
        events.append(event)
        if event["stage"] == "blocked":
            break
    return {
        "events": events,
        "bundle": bundle,
        "world": bundle.get("world") or pipeline_world,
    }


def run_mission(
    topic: str,
    *,
    jumps: int = DEFAULT_JUMPS,
    candidates: int = DEFAULT_CANDIDATES,
    polish_rounds: int = DEFAULT_POLISH,
    explore: str = DEFAULT_EXPLORE,
    experiment_rounds: int = DEFAULT_EXPERIMENT,
    idea_rounds: int = DEFAULT_IDEA,
    world: str = "",
    world_root: Path | None = None,
    reframes: int = 0,
    parallel: int | None = None,
    run_probes: bool = True,
    host_execute: bool = False,
    host_timeout: float | None = None,
    client: Any = None,
    feed: Any = None,
    backend: Backend | None = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    corpus_id: str = PRODUCTION_CORPUS,
    root: Path | None = None,
    workspace: Path | None = None,
    state_store: Path | None = None,
    policy_log: Path | None = None,
    policy_file: Path | None = None,
    judges_file: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield the mission's events, in order, ending with `done` or `blocked`.

    `client` and `feed` exist for tests and embedding callers; a production
    mission builds a live LLM client and a live arXiv feed itself. Passing
    an explicit `client` without a `feed` runs without live knowledge —
    the deterministic setting the test-bench needs.
    """
    root = root or _root()
    host = PluginHost.load(root, workspace)
    log = None
    if workspace is not None:
        dest = Path(workspace)
        dest.mkdir(parents=True, exist_ok=True)
        log = dest / EVENTS_FILE
        log.write_text("", encoding="utf-8")
    with bind_host(host):
        for event in _run_mission_body(
            topic,
            jumps=jumps,
            candidates=candidates,
            polish_rounds=polish_rounds,
            explore=explore,
            experiment_rounds=experiment_rounds,
            idea_rounds=idea_rounds,
            world=world,
            world_root=world_root,
            reframes=reframes,
            parallel=parallel,
            run_probes=run_probes,
            host_execute=host_execute,
            host_timeout=host_timeout,
            client=client,
            feed=feed,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            corpus_id=corpus_id,
            root=root,
            workspace=workspace,
            state_store=state_store,
            policy_log=policy_log,
            policy_file=policy_file,
            judges_file=judges_file,
            host=host,
        ):
            if log is not None:
                append_recorded_event(log, event)
            yield event


def _run_mission_body(
    topic: str,
    *,
    jumps: int = DEFAULT_JUMPS,
    candidates: int = DEFAULT_CANDIDATES,
    polish_rounds: int = DEFAULT_POLISH,
    explore: str = DEFAULT_EXPLORE,
    experiment_rounds: int = DEFAULT_EXPERIMENT,
    idea_rounds: int = DEFAULT_IDEA,
    world: str = "",
    world_root: Path | None = None,
    reframes: int = 0,
    parallel: int | None = None,
    run_probes: bool = True,
    host_execute: bool = False,
    host_timeout: float | None = None,
    client: Any = None,
    feed: Any = None,
    backend: Backend | None = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    corpus_id: str = PRODUCTION_CORPUS,
    root: Path | None = None,
    workspace: Path | None = None,
    state_store: Path | None = None,
    policy_log: Path | None = None,
    policy_file: Path | None = None,
    judges_file: Path | None = None,
    host: PluginHost | None = None,
) -> Iterator[dict[str, Any]]:
    root = root or _root()
    host = host or PluginHost.load(root, workspace)
    assets = load_assets(corpus_id, root)
    runtime = var_dir(root)
    state_path = state_store or (runtime / "research_state.json")
    log_path = policy_log or (runtime / "policy_log.json")
    policy_path = policy_file or (runtime / "research_policy.json")
    judges_path = judges_file or (runtime / "research_judges.json")
    policy = load_policy(policy_path)
    # The Verifier bench: judges promoted on held-out missions kill like
    # the installed set; shadow judges watch every card but cannot kill.
    bench = load_bench(judges_path)
    earned_judges = compile_bench(bench["lethal"])
    shadow_judges = compile_bench(bench["shadow"])
    lethal_judges = assets.judges + earned_judges
    standards = assets.standards + criteria_lines(earned_judges)
    beta = float((policy or {}).get("beta", DEFAULT_BETA))
    rubric = value_lines()
    workers = resolve_workers(parallel)
    explore_mode = normalize_explore(explore)
    polish_budget, polish_mode = resolve_polish_rounds(polish_rounds)
    experiment_budget, experiment_mode = resolve_experiment_rounds(
        experiment_rounds
    )
    idea_budget, idea_mode = resolve_idea_rounds(idea_rounds)
    world_request = world
    catalog_root = world_root or root
    try:
        from .freeze import resolve_pending_worlds

        acquired = resolve_pending_worlds(catalog_root)
    except Exception:
        acquired = []
    for row in acquired:
        yield {
            "stage": "world_wishlist",
            "acquired": row.get("world_id"),
            "digest": row.get("digest"),
            "schema": row.get("schema"),
        }
    try:
        world_catalog = load_catalog(catalog_root)
        bound_world = resolve_world(
            world_request, world_catalog, topic=topic
        )
    except KeyError:
        yield {
            "stage": "blocked",
            "missing_capability": "attested_world",
            "unlock_condition": (
                f"world {world_request!r} is not in worlds/; "
                "bind a frozen fixture or pass none"
            ),
        }
        return
    slots = jumps + max(0, reframes)
    if client is None:
        try:
            chosen = backend or default_backend(
                model=model, base_url=base_url, api_key=api_key
            )
            client = default_client(
                float(
                    jumps * candidates
                    + max(0, reframes)
                    + max(0, idea_budget) * slots
                    + slots * (3 + 2 * polish_budget)
                    + 8
                    + 2
                    + 1  # one shadow-judge proposal at mission end
                ),
                root,
                backend=chosen,
            )
        except LLMUnavailable as exc:
            yield {
                "stage": "blocked",
                "missing_capability": exc.record.missing_capability,
                "unlock_condition": exc.record.unlock_condition,
            }
            return
        if feed is None:
            feed = CompositeFeed()
    if workspace is not None and feed is not None:
        # Freshness and reproducibility at once: every live answer this
        # mission consumes is frozen into the workspace the moment it
        # arrives, and every literature-facing verdict is a function of
        # those recorded bytes.
        feed = RecordingFeed(feed, Path(workspace) / SNAPSHOT_DIR)

    yield {
        "stage": "corpus",
        "corpus": assets.corpus_uid,
        "knowledge_until": assets.fresh_until,
        "concepts": len(assets.graph.nodes),
        "judges": [judge.name for judge in lethal_judges],
        "shadow_judges": [judge.name for judge in shadow_judges],
        "model": getattr(getattr(client, "backend", None), "model", None),
        "base_url": getattr(getattr(client, "backend", None), "base_url", None),
        "candidates_per_jump": candidates,
        "polish_rounds": polish_budget,
        "polish_mode": polish_mode,
        "experiment_rounds": experiment_budget,
        "experiment_mode": experiment_mode,
        "idea_rounds": idea_budget,
        "idea_mode": idea_mode,
        "world": bound_world.to_dict() if bound_world is not None else None,
        "explore": explore_mode,
        "jump_cap": jumps,
        "parallel": workers,
    }

    catalog = host.catalog
    skill_blocks = {
        stage: skills_prompt_block(
            select_skills(catalog, topic=topic, stage=stage)
        )
        for stage in ("generate", "diagnose", "probe", "writeup", "intern")
    }
    yield {
        "stage": "skills",
        "loaded": [skill.to_dict() for skill in catalog],
        "injected": {
            stage: bool(block) for stage, block in skill_blocks.items()
        },
        "plugins": host.summary(),
        "dsh": host.dsh_status(),
    }

    topic_vector = assets.space.embed(topic)
    anchored = sorted(
        (cosine_distance(topic_vector, vector), nid)
        for nid, vector in assets.space.vectors.items()
    )
    near_ids = tuple(nid for _, nid in anchored[:NEAR_LABELS])
    near = tuple(assets.label_of[n] for n in near_ids)
    yield {
        "stage": "anchor",
        "anchors": [
            {"concept": assets.label_of[nid], "distance": round(dist, 4)}
            for dist, nid in anchored[:NEAR_LABELS]
        ],
    }

    # One long-term memory: the Research State carries what is known, what
    # is open, what contradicted, and which directions already died.
    topics_now = load_state(state_path)
    research = state_summary(topics_now, corpus_id, near[0])
    lessons = rejections_for(topics_now, corpus_id, near[0])
    stored_elites = elites_for(topics_now, corpus_id, near[0])
    stored_program = program_for(topics_now, corpus_id, near[0])
    stored_wiki = wiki_for(topics_now, corpus_id, near[0])
    program_lines = program_prompt_lines(stored_program)
    bucket = topics_now.get(topic_key(corpus_id, near[0])) or {}
    if any(
        entry.get("must_switch_mechanism")
        for entry in (bucket.get("hypotheses") or {}).values()
    ):
        program_lines = program_lines + (
            "Stagnation rule: two consecutive UNINFORMATIVE experiments on the "
            "active line. The next hypothesis MUST change the mechanism flag, "
            "the metric, the scale, the competing explanation, or the world "
            "slice. Rewriting the same experiment in new words is forbidden.",
        )
    wiki_lines = wiki_prompt_lines(stored_wiki)
    knowledge = prompt_lines(research)
    wiki_pool = list(wiki_works(topics_now, corpus_id, near[0]))
    if knowledge or lessons or stored_elites or stored_program or stored_wiki:
        yield {
            "stage": "state",
            "anchor": near[0],
            "known": research["known"],
            "unknowns": research["unknowns"],
            "contradictions": research["contradictions"],
            "rejected": [capsule["context"] for capsule in lessons],
            "elites": [
                {"pair": row.get("pair"), "score": row.get("score")}
                for row in stored_elites
            ],
            "program": stored_program,
            "wiki": [
                {"cite_id": row.get("cite_id"), "title": row.get("title")}
                for row in stored_wiki[:12]
            ],
            "hypotheses_tracked": research["hypotheses"],
        }

    context: tuple[str, ...] = ()
    if feed is not None:
        fresh = feed.recent_in_field(near[:3])
        if isinstance(fresh, FeedBlocked):
            yield {"stage": "fresh", "blocked": fresh.to_dict()}
        else:
            context = tuple(work.line() for work in fresh)
            yield {
                "stage": "fresh",
                "works": [work.to_dict() for work in fresh],
            }

    # The agenda: the state's open questions become this mission's targets,
    # and the installed routing policy (held-out gated) orders the operator
    # cycles. Contradictions outrank unknowns — a conflict on the books is
    # the cheapest place to gain information. Targets go to reframe slots
    # first, because reframes exist to question the field.
    targets = [f"conflict to resolve: {line}" for line in research["contradictions"]]
    targets += [f"open question: {line}" for line in research["unknowns"]]
    targets += [f"literature gap: {line}" for line in wiki_gap_lines(stored_wiki)]
    committed = program_target(stored_program)
    farfield_cycle = preferred_operators(policy, JUMP_OPERATORS)
    reframe_cycle = preferred_operators(policy, tuple(sorted(REFRAME_OPERATORS)))
    # A committed program means the first far slot is exploitation.
    # Remaining slots stay as exploration fallback when that landing is
    # invalid, uninformative, or an implementation failure.
    explore_jumps = jumps
    slot_plan: list[dict[str, Any]] = [
        {
            "track": "farfield",
            "operator": farfield_cycle[i % len(farfield_cycle)],
            "target": None,
            "role": "exploit" if stored_program and i == 0 else "explore",
        }
        for i in range(explore_jumps)
    ] + [
        {"track": "reframe", "operator": reframe_cycle[i % len(reframe_cycle)],
         "target": None}
        for i in range(max(0, reframes))
    ]
    # Contradictions / unknowns / wiki gaps go to reframe first — those
    # slots exist to question the field. The committed line is not a
    # field question; it occupies the exploit jump.
    receivers = [s for s in slot_plan if s["track"] == "reframe"]
    receivers += [s for s in slot_plan if s["track"] == "farfield"]
    if committed:
        for slot_entry in slot_plan:
            if slot_entry.get("role") == "exploit":
                slot_entry["target"] = committed
                break
        receivers = [s for s in receivers if s.get("role") != "exploit"]
    for slot_entry, target in zip(receivers, targets):
        slot_entry["target"] = target
    yield {
        "stage": "agenda",
        "targets": targets,
        "slots": list(slot_plan),
        "deepen": bool(stored_program),
        "explore": explore_mode,
        "explore_jumps": explore_jumps,
        "reframe_cap": max(0, reframes),
    }

    def slot_knowledge(entry: dict[str, Any]) -> tuple[str, ...]:
        if entry.get("target"):
            return knowledge + (
                f"This mission targets one gap on the books — {entry['target']}."
                " Prefer a claim that would move it.",
            )
        return knowledge

    rng_seed = "mission:" + hashlib.sha256(topic.encode()).hexdigest()[:12]
    total_cards = 0
    survivors = 0
    far_opened = 0
    reframe_opened = 0
    all_kills: list[dict[str, Any]] = []
    found: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    writeup_stash: list[dict[str, Any]] = []
    entered_cards: dict[str, GeneratedCard] = {}
    # W3/W4 bookkeeping: which shadow judges crashed (a crashing judge is
    # never promotable), and which far nodes the jump arm actually chose
    # (the control stake draws its matched arm from these).
    shadow_error_names: set[str] = set()
    arm_far_nodes: list[str] = []

    def new_record(card: GeneratedCard, track: str) -> dict[str, Any]:
        record = {
            "card_id": card.card_id,
            "operator": card.operator,
            "track": track,
            "alienness": card.alienness,
            "open_today": None,
            "papers": 0,
            "has_brief": False,
            "title": "",
            "novelty": 0,
            "value_vector": None,
            "prior_kills": False,
            "probe_ran": False,
            "probe_kind": KIND_SYNTHETIC,
            "verdict": None,
            "alternative": "",
            "experiment": "",
            "failed_experiments": [],
            "heavy_confirmed": False,
            "heavy": None,
            "chain_path": "",
            "value_score": 0.0,
            "wiki_gap_hit": False,
            "wiki_gap": "",
            "pipeline_bottleneck": "unresolved",
            "experiment_bottleneck": "",
            "killed_by": [],
            "world_id": getattr(bound_world, "id", "") or "",
            "world_schema": getattr(bound_world, "schema", "") or "",
            "host_ok": None,
            "world_incompatible": False,
            "ledger": "",
            "evidence_id": "",
            "experiment_digest": "",
            "world_digest": "",
            "data_digest": "",
            "host_experiment_digest": "",
            "host_world_digest": "",
            "host_data_digest": "",
            "host_replication": "",
            "pair": list(card.pair),
            "pair_nodes": list(card.pair_nodes),
            "claim": card.claim,
            "mechanism": card.mechanism,
            "prediction": card.prediction,
            "treatment_arm": "",
            "control_arm": "",
            "expected_direction": "",
            "first_steps": [],
            "killed": False,
        }
        found.append(record)
        return record

    def watch(event: dict[str, Any], record: dict[str, Any]) -> None:
        """Fold one pipeline event into the survivor's checkable record."""
        stage = event["stage"]
        if stage == "prior" and event.get("kills"):
            record["prior_kills"] = True
            record["open_today"] = False
        elif stage == "survey" and event.get("works"):
            added = record_wiki(
                state_path,
                corpus_id,
                near[0],
                list(event.get("works") or []),
                pair=record.get("pair") or [],
                seen_at=assets.fresh_until,
            )
            event["wiki_added"] = added
            wiki_pool[:] = wiki_works(load_state(state_path), corpus_id, near[0])
        elif stage == "brief" and int(event.get("round") or 0) == 0:
            record["has_brief"] = True
            record["papers"] = len(event.get("papers") or [])
            record["title"] = str(event.get("title") or "")
            record["first_steps"] = list(event.get("first_steps") or [])
        elif stage == "review":
            record["novelty"] = int(event.get("novelty") or 0)
            record["value_vector"] = event.get("value_vector")
        elif stage == "diagnosis":
            record["alternative"] = str(event.get("alternative") or "")
            record["experiment"] = str(event.get("experiment") or "")
            record["treatment_arm"] = str(event.get("treatment_arm") or "")
            record["control_arm"] = str(event.get("control_arm") or "")
            record["expected_direction"] = str(event.get("expected_direction") or "")
        elif stage == "experiment_decision":
            discarded = [
                str(item).strip()
                for item in (event.get("discarded") or [])
                if str(item or "").strip()
            ]
            if discarded:
                record["failed_experiments"] = discarded
            if event.get("bottleneck"):
                record["experiment_bottleneck"] = str(event["bottleneck"])
        elif stage == "probe":
            if event.get("kind"):
                record["probe_kind"] = event["kind"]
            if event.get("status") == "ran":
                record["probe_ran"] = True
        elif stage == "evidence":
            record["verdict"] = event.get("verdict")
            if event.get("evidence_id"):
                record["evidence_id"] = str(event["evidence_id"])
            if event.get("experiment_digest"):
                record["experiment_digest"] = str(event["experiment_digest"])
            if event.get("world_digest"):
                record["world_digest"] = str(event["world_digest"])
            if event.get("data_digest"):
                record["data_digest"] = str(event["data_digest"])
            if event.get("mechanism_identified") is False:
                record["mechanism_identified"] = False
            if event.get("ablation_required"):
                record["ablation_required"] = True
        elif stage == "heavy_evidence":
            record["heavy_confirmed"] = event.get("verdict") == "supports"
            record["heavy"] = {
                key: event.get(key)
                for key in (
                    "verdict",
                    "replicated",
                    "runner",
                    "seeds",
                    "invariant",
                    "n",
                    "mean_separation",
                    "ci_low",
                    "ci_high",
                    "margin",
                    "expected_direction",
                    "reason",
                    "evidence_id",
                    "experiment_digest",
                    "world_digest",
                    "replication",
                )
            }
        elif stage == "host_execute":
            record["host_ok"] = bool(event.get("ok"))
            record["host_bound"] = str(event.get("bound") or "")
            record["host_status"] = str(
                event.get("error") or event.get("status") or ""
            )
            # Host provenance stays under host_* keys. Overwriting the
            # probe's digests here would make the fail-closed comparison
            # in host_confirms compare the host run against itself.
            record["host_evidence_id"] = str(event.get("evidence_id") or "")
            record["host_experiment_digest"] = str(
                event.get("experiment_digest") or ""
            )
            record["host_world_digest"] = str(event.get("world_digest") or "")
            record["host_data_digest"] = str(event.get("data_digest") or "")
            record["host_replication"] = str(event.get("replication") or "")
            record["chain_path"] = str(event.get("chain_path") or "")
            record["protocol_complete"] = bool(event.get("protocol_complete", event.get("ok")))
            if event.get("verdict"):
                record["verdict"] = event["verdict"]
            if not record["host_ok"]:
                record["ledger"] = "diagnostic"
        elif stage == "world_incompatible":
            record["world_incompatible"] = True
            record["ledger"] = "diagnostic"
            if isinstance(event.get("requirement"), dict):
                record["world_requirement"] = dict(event["requirement"])
        elif stage == "world_path":
            extra = {
                key: event[key]
                for key in (
                    "named_instance",
                    "lineage",
                    "freeze_source",
                    "freeze_url",
                    "freeze_schema",
                )
                if event.get(key)
            }
            if event.get("cite_ids"):
                extra["lineage_cite_ids"] = list(event["cite_ids"])
            if extra:
                current = dict(record.get("world_requirement") or {})
                current.update(extra)
                record["world_requirement"] = current
                record["world_path"] = extra
        elif stage == "world_surrogate":
            record["world_binding"] = "surrogate"
        elif stage == "world_forecast":
            record["world_forecast_agrees"] = bool(event.get("agrees"))
            if isinstance(event.get("analysis"), dict):
                record["idea_analysis"] = dict(event["analysis"])
        elif stage == "probe_skipped":
            record["experiment_bottleneck"] = str(event.get("bottleneck") or "world")
            record["ledger"] = "diagnostic"
            if isinstance(event.get("analysis"), dict):
                record["idea_analysis"] = dict(event["analysis"])
        elif stage == "world_placebo":
            record["world_has_dynamics"] = True
            record["world_consumed"] = bool(event.get("consumed"))
        elif stage in {"world_generated", "world_constructed"}:
            record["generated_world"] = True
            record["ledger"] = "diagnostic"
            world_row = event.get("world") if isinstance(event.get("world"), dict) else {}
            if world_row.get("id"):
                record["world_id"] = str(world_row["id"])
            record["world_schema"] = str(world_row.get("schema") or "symbolic_trace")

    # Sample landings (CPU, deterministic). `jumps` / `reframes` are a cap.
    # In fixed mode every landing is generated before any evidence. In auto
    # mode each landing is generated, gated, and evidenced before the next
    # one opens — the stop rule reads attested records, not a preset count.
    jump_meta: list[dict[str, Any]] = []
    for jump_index, jump in enumerate(
        sample_jumps(
            assets.space,
            topic,
            rng_seed=rng_seed,
            count=explore_jumps,
            operators=farfield_cycle,
            domain_quantile=DOMAIN_QUANTILE,
        )
    ):
        far_candidates = tuple(
            (assets.label_of[nid], nid)
            for nid, _ in jump.retrieved
            if nid not in near_ids
        )
        viable, partners, prescreened, extra_kills = _screen_far(
            assets, near, near_ids, far_candidates
        )
        all_kills.extend(extra_kills)
        far = tuple(viable)
        jump_meta.append(
            {
                "index": jump_index,
                "jump": jump,
                "far": far,
                "partners": partners,
                "prescreened": prescreened,
            }
        )

    def _far_jobs(meta: dict[str, Any]) -> list[dict[str, Any]]:
        if not meta["far"]:
            return []
        return [
            {
                "kind": "farfield",
                "jump_index": meta["index"],
                "k": k,
                "jump": meta["jump"],
                "far": meta["far"],
                "partners": meta["partners"],
                "slot": slot_plan[meta["index"]],
            }
            for k in range(candidates)
        ]

    def _reframe_job(slot: int) -> dict[str, Any]:
        entry = slot_plan[explore_jumps + slot]
        return {
            "kind": "reframe",
            "jump_index": explore_jumps + slot,
            "k": 0,
            "entry": entry,
            "operator": entry["operator"],
        }

    def _jump_event_far(meta: dict[str, Any]) -> dict[str, Any]:
        jump = meta["jump"]
        return {
            "stage": "jump",
            "index": meta["index"],
            "operator": jump.operator,
            "track": "farfield",
            "target": slot_plan[meta["index"]].get("target"),
            "alienness": round(jump.alienness, 4),
            "far": list(meta["far"]),
            "prescreened_out": meta["prescreened"],
        }

    def _jump_event_reframe(slot: int) -> dict[str, Any]:
        entry = slot_plan[explore_jumps + slot]
        return {
            "stage": "jump",
            "index": explore_jumps + slot,
            "operator": entry["operator"],
            "track": "reframe",
            "target": entry.get("target"),
            "alienness": 0.0,
            "far": [],
        }

    def _program_for_refine(current: GeneratedCard) -> tuple[str, ...]:
        this_found = [row for row in found if row.get("card_id") == current.card_id]
        this_kills = [
            kill
            for kill in all_kills
            if list(kill.get("pair") or [])[:2] == list(current.pair)[:2]
        ]
        previous = program_for(topics_now, corpus_id, near[0], pair=current.pair)
        working = compile_program(
            topic,
            found=this_found,
            kills=this_kills,
            previous=previous,
            world=bound_world,
        )
        return program_prompt_lines(working) if working else ()

    def _refresh_program() -> None:
        # Explore generate never inherits another idea's H. Exploit uses
        # the stored WORLD commit loaded at mission start.
        return

    def _refine_until_enter(
        current: GeneratedCard,
        kills: list[str],
        jump_index: int,
        track: str,
    ) -> Iterator[dict[str, Any]]:
        """Text-gate deaths only. Graph-dead pairs do not burn idea_rounds."""
        nonlocal total_cards, survivors
        if idea_budget <= 0:
            return
        for attempt in range(idea_budget):
            decision = decide_idea_refine(
                attempt=attempt,
                cap=idea_budget,
                killed_by=kills,
                alive=False,
            )
            yield {
                "stage": "idea_decision",
                "jump": jump_index,
                "card_id": current.card_id,
                "attempt": attempt,
                "pair": list(current.pair),
                "track": track,
                **decision,
            }
            if not decision["continue"]:
                return
            _refresh_program()
            yield {
                "stage": "idea_refining",
                "jump": jump_index,
                "attempt": attempt,
                "pair": list(current.pair),
                "killed_by": kills,
                "track": track,
            }
            try:
                if track == "reframe":
                    current = refine_reframe(
                        client,
                        current,
                        seed_label=near[0],
                        near_labels=near[1:],
                        killed_by=kills,
                        context=context,
                        knowledge=slot_knowledge({"target": None, "track": track}),
                        criteria=standards,
                        topic=topic,
                        skills=skill_blocks["generate"],
                        program=_program_for_refine(current),
                        wiki=wiki_lines,
                    )
                else:
                    current = refine_card(
                        client,
                        current,
                        seed_label=near[0],
                        near_labels=near[1:],
                        label_to_node=assets.label_to_node,
                        killed_by=kills,
                        capsules=lessons,
                        logprobs=True,
                        criteria=standards,
                        context=context,
                        knowledge=slot_knowledge({"target": None, "track": track}),
                        value_criteria=rubric,
                        topic=topic,
                        skills=skill_blocks["generate"],
                        program=_program_for_refine(current),
                        wiki=wiki_lines,
                    )
            except GenerationRefused as exc:
                rec = exc.record
                yield {
                    "stage": "refused",
                    "jump": jump_index,
                    "candidate": attempt + 1,
                    "missing_capability": rec.missing_capability,
                    "unlock_condition": rec.unlock_condition,
                }
                continue
            except LLMUnavailable as exc:
                yield {
                    "stage": "blocked",
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
                }
                return
            total_cards += 1
            yield {
                "stage": "card",
                "jump": jump_index,
                "candidate": attempt + 1,
                "card_id": current.card_id,
                "operator": current.operator,
                "pair": list(current.pair),
                "claim": current.claim,
                "refined": True,
            }
            text_kills: list[str] = []
            graph_killed_by: list[str] = []
            graph_checks: list[dict[str, Any]] = []
            if track != "reframe":
                graph_verdict = probe_generated(assets.oracle, [current])[0]
                graph_killed_by = list(graph_verdict.killed_by)
                graph_checks = [check.to_dict() for check in graph_verdict.checks]
            view = {
                "claim": current.claim,
                "mechanism": current.mechanism,
                "prediction": current.prediction,
                "falsifier": current.falsifier,
                "concept_a": current.pair[0],
                "concept_b": current.pair[1],
                "alienness": current.alienness,
                "plausibility_signal": (
                    0.0
                    if track == "reframe"
                    else assets.oracle.jaccard(*current.pair_nodes)
                ),
            }
            if apply_x_to_y(current.claim, current.mechanism, current.pair):
                text_kills.append("apply_x_to_y")
            for judge in lethal_judges:
                try:
                    if not run_predicate(judge, view):
                        text_kills.append(judge.name)
                except PredicateError:
                    pass
            shadow, shadow_errs = shadow_flags(shadow_judges, view)
            shadow_error_names.update(shadow_errs)
            alive = not graph_killed_by and not text_kills
            survivors += alive
            kills = list(graph_killed_by) + text_kills
            if kills:
                all_kills.append({"pair": list(current.pair), "killed_by": kills})
            yield {
                "stage": "verdict",
                "jump": jump_index,
                "candidate": attempt + 1,
                "chosen": True,
                "card_id": current.card_id,
                "operator": current.operator,
                "alienness": round(current.alienness, 4),
                "pair": list(current.pair),
                "claim": current.claim,
                "mechanism": current.mechanism,
                "prediction": current.prediction,
                "preregistered_falsifier": current.falsifier,
                **current.struggle_event(),
                "graph_killed_by": graph_killed_by,
                "graph_checks": graph_checks,
                "text_kills": text_kills,
                "shadow_kills": shadow,
                "judge_errors": [],
                "alive": alive,
                "outcome": "entered" if alive else "ineligible",
                "refined": True,
            }
            attempts.append(
                {
                    "card_id": current.card_id,
                    "operator": current.operator,
                    "track": track,
                    "killed": not alive,
                    "killed_by": kills if not alive else [],
                    "shadow_kills": shadow,
                }
            )
            if alive:
                entered_jobs.append((current, track))
                return

    def _do_gen(job: dict[str, Any]) -> dict[str, Any]:
        slot = job.get("slot") or job.get("entry") or {}
        generate_program = program_lines if slot.get("role") == "exploit" else ()
        try:
            if job["kind"] == "farfield":
                card = generate_card(
                    client,
                    seed_label=near[0],
                    near_labels=near[1:],
                    far_labels=_rotate(job["far"], job["k"]),
                    operator=job["jump"].operator,
                    alienness=job["jump"].alienness,
                    label_to_node=assets.label_to_node,
                    logprobs=True,
                    criteria=standards,
                    context=context,
                    knowledge=slot_knowledge(job["slot"]),
                    capsules=lessons,
                    viable_pairings=job["partners"],
                    value_criteria=rubric,
                    topic=topic,
                    skills=skill_blocks["generate"],
                    program=generate_program,
                    wiki=wiki_lines,
                )
            else:
                card = generate_reframe(
                    client,
                    seed_label=near[0],
                    near_labels=near[1:],
                    operator=job["operator"],
                    context=context,
                    knowledge=slot_knowledge(job["entry"]),
                    criteria=standards,
                    topic=topic,
                    skills=skill_blocks["generate"],
                    program=generate_program,
                    wiki=wiki_lines,
                )
            return {"status": "ok", "card": card}
        except GenerationRefused as exc:
            return {"status": "refused", "record": exc.record}
        except LLMUnavailable as exc:
            return {"status": "blocked", "record": exc.record}

    def _blocked_of(results: list[dict[str, Any]]) -> dict[str, Any] | None:
        for result in results:
            if result.get("status") == "blocked":
                rec = result["record"]
                return {
                    "stage": "blocked",
                    "missing_capability": rec.missing_capability,
                    "unlock_condition": rec.unlock_condition,
                }
        return None

    entered_jobs: list[tuple[GeneratedCard, str]] = []
    far_opened = 0
    reframe_opened = 0

    def _ingest_far(
        meta: dict[str, Any],
        rows: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> Iterator[dict[str, Any]]:
        nonlocal total_cards, survivors
        jump_index = meta["index"]
        jump_cards: list[tuple[int, GeneratedCard]] = []
        for job, result in rows:
            k = job["k"]
            yield {"stage": "generating", "jump": jump_index, "candidate": k}
            if result["status"] == "refused":
                rec = result["record"]
                yield {
                    "stage": "refused",
                    "jump": jump_index,
                    "candidate": k,
                    "missing_capability": rec.missing_capability,
                    "unlock_condition": rec.unlock_condition,
                }
                continue
            card = result["card"]
            total_cards += 1
            jump_cards.append((k, card))
            yield {
                "stage": "card",
                "jump": jump_index,
                "candidate": k,
                "card_id": card.card_id,
                "operator": card.operator,
                "pair": list(card.pair),
                "claim": card.claim,
            }
        if not jump_cards:
            return
        rulings = []
        graph_verdicts = probe_generated(
            assets.oracle, [card for _, card in jump_cards]
        )
        for (k, card), verdict in zip(jump_cards, graph_verdicts):
            view = {
                "claim": card.claim,
                "mechanism": card.mechanism,
                "prediction": card.prediction,
                "falsifier": card.falsifier,
                "concept_a": card.pair[0],
                "concept_b": card.pair[1],
                "alienness": card.alienness,
                "plausibility_signal": assets.oracle.jaccard(*card.pair_nodes),
            }
            text_kills = []
            judge_errors = []
            if apply_x_to_y(card.claim, card.mechanism, card.pair):
                text_kills.append("apply_x_to_y")
            for judge in lethal_judges:
                try:
                    if not run_predicate(judge, view):
                        text_kills.append(judge.name)
                except PredicateError as error:
                    judge_errors.append({"judge": judge.name, "error": str(error)})
            shadow, shadow_errs = shadow_flags(shadow_judges, view)
            shadow_error_names.update(shadow_errs)
            rulings.append(
                {
                    "candidate": k,
                    "card": card,
                    "graph": verdict,
                    "text_kills": text_kills,
                    "shadow_kills": shadow,
                    "judge_errors": judge_errors,
                    "kills": len(verdict.killed_by) + len(text_kills),
                }
            )
        chosen = min(rulings, key=lambda r: (r["kills"], r["candidate"]))
        for ruling in rulings:
            card = ruling["card"]
            verdict = ruling["graph"]
            is_chosen = ruling is chosen
            alive = ruling["kills"] == 0 and is_chosen
            if alive:
                exit_name = "entered"
            elif ruling["kills"] > 0:
                exit_name = "ineligible"
            else:
                exit_name = "not_selected"
            survivors += alive
            if ruling["kills"] > 0:
                all_kills.append(
                    {
                        "pair": list(card.pair),
                        "killed_by": list(verdict.killed_by) + ruling["text_kills"],
                    }
                )
            yield {
                "stage": "verdict",
                "jump": jump_index,
                "candidate": ruling["candidate"],
                "chosen": is_chosen,
                "card_id": card.card_id,
                "operator": card.operator,
                "alienness": round(card.alienness, 4),
                "pair": list(card.pair),
                "claim": card.claim,
                "mechanism": card.mechanism,
                "prediction": card.prediction,
                "preregistered_falsifier": card.falsifier,
                **card.struggle_event(),
                "graph_killed_by": list(verdict.killed_by),
                "graph_checks": [check.to_dict() for check in verdict.checks],
                "text_kills": ruling["text_kills"],
                "shadow_kills": ruling["shadow_kills"],
                "judge_errors": ruling["judge_errors"],
                "alive": alive,
                "outcome": exit_name,
            }
            if is_chosen:
                arm_far_nodes.append(card.pair_nodes[1])
                attempts.append(
                    {
                        "card_id": card.card_id,
                        "operator": card.operator,
                        "track": "farfield",
                        "killed": not alive,
                        "killed_by": (
                            list(verdict.killed_by) + ruling["text_kills"]
                            if not alive
                            else []
                        ),
                        "shadow_kills": ruling["shadow_kills"],
                    }
                )
            if alive:
                entered_jobs.append((card, "farfield"))

        if chosen["kills"] > 0:
            for event in _refine_until_enter(
                chosen["card"],
                list(chosen["graph"].killed_by) + list(chosen["text_kills"]),
                jump_index,
                "farfield",
            ):
                yield event
                if event["stage"] == "blocked":
                    return
        _refresh_program()

    def _ingest_reframe(
        index: int,
        rows: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> Iterator[dict[str, Any]]:
        nonlocal total_cards, survivors
        if not rows:
            return
        _job, result = rows[0]
        yield {"stage": "generating", "jump": index, "candidate": 0}
        if result["status"] == "refused":
            rec = result["record"]
            yield {
                "stage": "refused",
                "jump": index,
                "candidate": 0,
                "missing_capability": rec.missing_capability,
                "unlock_condition": rec.unlock_condition,
            }
            return
        card = result["card"]
        total_cards += 1
        yield {
            "stage": "card",
            "jump": index,
            "candidate": 0,
            "card_id": card.card_id,
            "operator": card.operator,
            "pair": list(card.pair),
            "claim": card.claim,
        }
        view = {
            "claim": card.claim,
            "mechanism": card.mechanism,
            "prediction": card.prediction,
            "falsifier": card.falsifier,
            "concept_a": card.pair[0],
            "concept_b": card.pair[1],
            "alienness": card.alienness,
            "plausibility_signal": 0.0,
        }
        text_kills = []
        judge_errors = []
        if apply_x_to_y(card.claim, card.mechanism, card.pair):
            text_kills.append("apply_x_to_y")
        for judge in lethal_judges:
            try:
                if not run_predicate(judge, view):
                    text_kills.append(judge.name)
            except PredicateError as error:
                judge_errors.append({"judge": judge.name, "error": str(error)})
        shadow, shadow_errs = shadow_flags(shadow_judges, view)
        shadow_error_names.update(shadow_errs)
        alive = not text_kills
        survivors += alive
        if text_kills:
            all_kills.append({"pair": list(card.pair), "killed_by": list(text_kills)})
        yield {
            "stage": "verdict",
            "jump": index,
            "candidate": 0,
            "chosen": True,
            "card_id": card.card_id,
            "operator": card.operator,
            "alienness": 0.0,
            "pair": list(card.pair),
            "claim": card.claim,
            "mechanism": card.mechanism,
            "prediction": card.prediction,
            "preregistered_falsifier": card.falsifier,
            **card.struggle_event(),
            "graph_killed_by": [],
            "graph_checks": [],
            "graph_skipped": GRAPH_SKIP_REASON,
            "text_kills": text_kills,
            "shadow_kills": shadow,
            "judge_errors": judge_errors,
            "alive": alive,
            "outcome": "entered" if alive else "ineligible",
        }
        attempts.append(
            {
                "card_id": card.card_id,
                "operator": card.operator,
                "track": "reframe",
                "killed": not alive,
                "killed_by": list(text_kills) if not alive else [],
                "shadow_kills": shadow,
            }
        )
        if alive:
            entered_jobs.append((card, "reframe"))
        elif text_kills:
            for event in _refine_until_enter(card, list(text_kills), index, "reframe"):
                yield event
                if event["stage"] == "blocked":
                    return
        _refresh_program()

    def _do_entry(item: tuple[GeneratedCard, str]) -> dict[str, Any]:
        card, _track = item
        return _run_entry(
            client,
            feed,
            card,
            topic,
            polish_rounds=0,
            run_probes=run_probes,
            workspace=workspace,
            prior_probe=probe_history(topics_now, corpus_id, near[0], card.pair),
            prior_status=line_status(topics_now, corpus_id, near[0], card.pair),
            skill_blocks=skill_blocks,
            experiment_rounds=experiment_budget,
            world=bound_world,
            wiki_works_pool=wiki_pool,
            program=program_prompt_lines(
                program_for(topics_now, corpus_id, near[0], pair=card.pair)
            ),
            catalog=world_catalog,
            world_request=world_request,
            bound_world=bound_world,
        )

    def _emit_evidence(
        batch: list[tuple[GeneratedCard, str]],
    ) -> Iterator[dict[str, Any]]:
        if not batch:
            return
        yield {
            "stage": "parallel",
            "wave": "evidence",
            "jobs": len(batch),
            "workers": workers,
        }
        # Hypotheses are registered in the research ledger BEFORE the
        # evidence wave runs: the ledger order proves the claim existed
        # before any result it could have been rewritten to fit. Promotion
        # later refuses a line whose (line, revision) was never registered.
        ledger_path = Path(state_path).with_suffix(".chain.jsonl")
        for card, track in batch:
            line_id = hypothesis_id(near[0], card.pair)
            chain.append_event(
                ledger_path,
                chain.REGISTER_HYPOTHESIS,
                {
                    "topic": f"{corpus_id}::{near[0]}",
                    "line_id": line_id,
                    "revision_id": hypothesis_revision_id(line_id, card.claim),
                    "card_id": card.card_id,
                    "claim_digest": sha256_text(card.claim),
                    "operator": card.operator,
                    "track": track,
                },
            )
        entry_results = map_parallel(_do_entry, batch, workers)
        for (card, track), payload in zip(batch, entry_results):
            record = new_record(card, track)
            outcome = {
                "card_id": card.card_id,
                "operator": card.operator,
                "track": track,
                "pair": list(card.pair),
                "claim": card.claim,
                "prior_kills": False,
                "verdict": None,
                "alternative": "",
                "experiment": "",
                "failed_experiments": [],
                "heavy_confirmed": False,
                "heavy": None,
                "chain_path": "",
                "probe_kind": KIND_SYNTHETIC,
                "value_vector": None,
            }
            outcomes.append(outcome)
            for event in payload["events"]:
                if event["stage"] == "fresh_check" and "combined" in event:
                    record["open_today"] = not bool(event.get("combined"))
                yield event
                watch(event, record)
                if event["stage"] == "blocked":
                    return
            outcome["prior_kills"] = record["prior_kills"]
            outcome["verdict"] = record["verdict"]
            outcome["alternative"] = record["alternative"]
            outcome["experiment"] = record["experiment"]
            outcome["failed_experiments"] = list(
                record.get("failed_experiments") or []
            )
            outcome["heavy_confirmed"] = record["heavy_confirmed"]
            outcome["probe_kind"] = record.get("probe_kind") or KIND_SYNTHETIC
            outcome["value_vector"] = record["value_vector"]
            card_world = payload.get("world")
            if card_world is not None:
                record["world_id"] = getattr(card_world, "id", "") or ""
                record["world_schema"] = getattr(card_world, "schema", "") or ""
            _fold_record_into_outcome(outcome, record)
            writeup_stash.append(
                {
                    "card": card,
                    "track": track,
                    "bundle": payload["bundle"],
                    "record": record,
                    "world": card_world,
                }
            )
            entered_cards[card.card_id] = card

    def _generate_wave(
        jobs: list[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        if jobs:
            yield {
                "stage": "parallel",
                "wave": "generate",
                "jobs": len(jobs),
                "workers": workers,
            }
        results = map_parallel(_do_gen, jobs, workers) if jobs else []
        blocked = _blocked_of(results)
        if blocked is not None:
            yield blocked
            return
        yield {"stage": "_gen_results", "jobs": jobs, "results": results}

    if explore_mode == "fixed":
        gen_jobs: list[dict[str, Any]] = []
        for meta in jump_meta:
            yield _jump_event_far(meta)
            far_opened += 1
            gen_jobs.extend(_far_jobs(meta))
        for slot in range(max(0, reframes)):
            yield _jump_event_reframe(slot)
            reframe_opened += 1
            gen_jobs.append(_reframe_job(slot))
        gen_jobs_local = gen_jobs
        gen_results: list[dict[str, Any]] = []
        for event in _generate_wave(gen_jobs_local):
            if event["stage"] == "_gen_results":
                gen_results = event["results"]
                continue
            yield event
            if event["stage"] == "blocked":
                return
        by_jump: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for job, result in zip(gen_jobs_local, gen_results):
            by_jump.setdefault(job["jump_index"], []).append((job, result))
        for meta in jump_meta:
            yield from _ingest_far(meta, by_jump.get(meta["index"], []))
        for slot in range(max(0, reframes)):
            index = explore_jumps + slot
            yield from _ingest_reframe(index, by_jump.get(index, []))
        for event in _emit_evidence(list(entered_jobs)):
            yield event
            if event["stage"] == "blocked":
                return
        _refresh_program()
    else:
        evidenced = 0
        for meta in jump_meta:
            yield _jump_event_far(meta)
            jobs = _far_jobs(meta)
            if jobs:
                gen_results = []
                for event in _generate_wave(jobs):
                    if event["stage"] == "_gen_results":
                        gen_results = event["results"]
                        continue
                    yield event
                    if event["stage"] == "blocked":
                        return
                yield from _ingest_far(meta, list(zip(jobs, gen_results)))
                batch = entered_jobs[evidenced:]
                evidenced = len(entered_jobs)
                for event in _emit_evidence(batch):
                    yield event
                    if event["stage"] == "blocked":
                        return
                _refresh_program()
            far_opened += 1
            decision = decide_next(
                opened=far_opened,
                cap=len(jump_meta),
                records=found,
            )
            yield {
                "stage": "explore_decision",
                "track": "farfield",
                "opened": far_opened,
                "cap": len(jump_meta),
                **decision,
            }
            if not decision["continue"]:
                break
        if not has_research_plan(found):
            for slot in range(max(0, reframes)):
                yield _jump_event_reframe(slot)
                job = _reframe_job(slot)
                gen_results = []
                for event in _generate_wave([job]):
                    if event["stage"] == "_gen_results":
                        gen_results = event["results"]
                        continue
                    yield event
                    if event["stage"] == "blocked":
                        return
                index = explore_jumps + slot
                yield from _ingest_reframe(index, list(zip([job], gen_results)))
                batch = entered_jobs[evidenced:]
                evidenced = len(entered_jobs)
                for event in _emit_evidence(batch):
                    yield event
                    if event["stage"] == "blocked":
                        return
                _refresh_program()
                reframe_opened += 1
                decision = decide_next(
                    opened=reframe_opened,
                    cap=max(0, reframes),
                    records=found,
                )
                yield {
                    "stage": "explore_decision",
                    "track": "reframe",
                    "opened": reframe_opened,
                    "cap": max(0, reframes),
                    **decision,
                }
                if not decision["continue"]:
                    break

    def _observe_found() -> None:
        gaps = wiki_gap_lines(wiki_for(load_state(state_path), corpus_id, near[0]))
        attach_observations(found, gaps)

    # Recorded tournament: colleague opinion only. Selection and routing
    # read evidence class, not this score.
    records_by_id = {record["card_id"]: record for record in found}
    ranked_value: list[dict[str, Any]] = []
    if found:
        _observe_found()
        live_n = len(tournament_pool(found))
        debate_n = live_n * (live_n - 1) // 2
        if debate_n:
            yield {
                "stage": "parallel",
                "wave": "debate",
                "jobs": debate_n,
                "workers": workers,
            }
        table, debates = run_tournament(
            client, found, rng_seed=rng_seed + ":value", workers=workers
        )
        ranked_value = score_live(found, table.ratings, beta=beta)
        by_score = {row["card_id"]: row["score"] for row in ranked_value}
        for record in found:
            record["value_score"] = float(by_score.get(record["card_id"], 0.0))
        yield {
            "stage": "value",
            "beta": beta,
            "debates": debates,
            "ranking": ranked_value,
        }
        archive = QDArchive(assets.space, near_ids)
        admitted = []
        for record in found:
            nodes = tuple(record.get("pair_nodes") or ())
            if len(nodes) != 2 or nodes[1] not in assets.space.vectors:
                continue
            if record.get("prior_kills") or record.get("verdict") == "weakens":
                continue
            decision = archive.admit(
                {
                    "card_id": record["card_id"],
                    "pair_nodes": nodes,
                    "pair": record["pair"],
                    "claim": record.get("claim") or "",
                    "mechanism": record.get("mechanism") or "",
                    "prediction": record.get("prediction") or "",
                    "alienness": record["alienness"],
                    "killed": False,
                },
                archive_fitness(record),
            )
            admitted.append(decision.to_dict())
        yield {
            "stage": "archive",
            "admitted": admitted,
            "occupancy": archive.occupancy(),
        }
        elite_rows = []
        for row in archive.elites():
            elite_rows.append(
                {
                    "card_id": row["card_id"],
                    "pair": row.get("pair") or [],
                    "pair_nodes": list(row["pair_nodes"]),
                    "claim": row.get("claim") or "",
                    "operator": records_by_id.get(row["card_id"], {}).get("operator") or "",
                    "alienness": row.get("alienness") or 0,
                    "score": row.get("archive_score") or 0,
                }
            )
        record_elites(state_path, corpus_id, near[0], elite_rows)

    # Expensive writing last, and only for ideas evidence did not weaken.
    ready_writeup = [
        stash for stash in writeup_stash if not stash["bundle"].get("skip_writeup")
    ]

    def _do_writeup(stash: dict[str, Any]) -> list[dict[str, Any]]:
        bundle = stash["bundle"]
        events: list[dict[str, Any]] = []
        for event in _writeup_pipeline(
            client,
            stash["card"],
            topic,
            works=list(bundle.get("works") or []),
            probe_payload=bundle.get("probe_payload"),
            prior=bundle.get("prior"),
            polish_rounds=polish_budget,
            workspace=workspace,
            skill_blocks=skill_blocks,
            seed_label=near[0],
            diagnosis=bundle.get("diagnosis"),
            skill_catalog=state_path.parent / "skills",
            world=stash.get("world", bound_world),
            host_execute=host_execute,
            host_timeout=host_timeout,
        ):
            events.append(event)
            if event["stage"] == "blocked":
                break
        return events

    if ready_writeup:
        yield {
            "stage": "parallel",
            "wave": "writeup",
            "jobs": len(ready_writeup),
            "workers": workers,
        }
        writeup_results = map_parallel(_do_writeup, ready_writeup, workers)
        for stash, events in zip(ready_writeup, writeup_results):
            record = stash.get("record")
            for event in events:
                yield event
                if record is not None:
                    watch(event, record)
                if event["stage"] == "blocked":
                    return

    records_by_id = {record["card_id"]: record for record in found}
    if found:
        _observe_found()
    for outcome in outcomes:
        rec = records_by_id.get(outcome["card_id"])
        if rec is not None:
            _fold_record_into_outcome(outcome, rec)

    # The same-anchor control stake: matched-alienness random draws through
    # the same graph gates the jump arm faced. A measurement, not a verdict;
    # it reaches the log and the event stream, never the ranking or state.
    stake = control_stake(
        assets.space,
        assets.oracle,
        near_ids,
        arm_far_nodes,
        seed=int(hashlib.sha256(rng_seed.encode()).hexdigest()[:8], 16),
    )
    if stake is not None:
        yield {"stage": "baseline", **stake}

    ranked = rank_ideas(found) if found else []
    if ranked:
        yield {"stage": "ranked", "ideas": ranked}

    # Belief update: the executable promotion ladder moves only hypotheses
    # that entered research; gate kills are banked as rejections instead.
    promotions = apply_outcomes(state_path, corpus_id, near[0], outcomes)
    promoted_status = {event["card_id"]: event["status"] for event in promotions}
    for event in promotions:
        yield {"stage": "promotion", **event}

    next_program = compile_program(
        topic,
        ranked=ranked,
        found=found,
        kills=all_kills,
        previous=stored_program,
        world=bound_world,
    )
    if next_program is not None:
        record_program(state_path, corpus_id, near[0], next_program)
        yield {"stage": "program", **next_program}

    records_by_id = {record["card_id"]: record for record in found}
    for attempt in attempts:
        record = records_by_id.get(attempt["card_id"])
        trajectory.append(
            {
                "operator": attempt["operator"],
                "track": attempt["track"],
                "killed": attempt["killed"],
                "briefed": bool(record and record["has_brief"]),
                "verdict": record["verdict"] if record else None,
                "value_score": float(record["value_score"]) if record else 0.0,
                "promoted": promoted_status.get(attempt["card_id"]),
                "probe_kind": (record.get("probe_kind") if record else None),
                "host_ok": (record.get("host_ok") if record else None),
                "ledger": (record.get("ledger") if record else None),
                "world_incompatible": (
                    bool(record.get("world_incompatible")) if record else False
                ),
                "evidence_id": (record.get("evidence_id") if record else None),
                "wiki_gap_hit": bool(record.get("wiki_gap_hit")) if record else False,
                "pipeline_bottleneck": (
                    str(record.get("pipeline_bottleneck") or "")
                    if record
                    else ("gate" if attempt.get("killed") else "unresolved")
                ),
                "killed_by": list(
                    attempt.get("killed_by")
                    or (record or {}).get("killed_by")
                    or []
                ),
                "shadow_kills": list(attempt.get("shadow_kills") or []),
            }
        )
    log_size = append_mission(
        log_path,
        {
            "topic_digest": hashlib.sha256(topic.encode()).hexdigest()[:16],
            "corpus": corpus_id,
            "anchor": near[0],
            "model": getattr(getattr(client, "backend", None), "model", None),
            "base_url": getattr(getattr(client, "backend", None), "base_url", None),
            "cards": trajectory,
            "shadow_errors": sorted(shadow_error_names),
            "baseline": stake,
        },
    )
    banked = record_rejections(state_path, corpus_id, near[0], all_kills)

    # The meta loop runs unattended at every mission end: it can only
    # install a routing change that beat the incumbent on missions it was
    # not fitted to, so running it often costs nothing and hides nothing.
    decision = maybe_update(log_path, policy_path)
    if decision is not None:
        yield {"stage": "policy", **decision}

    # The Verifier meta loop, same discipline: a shadow judge becomes
    # lethal only after held-out missions show its flags anticipated
    # WORLD-weakens and never hit a WORLD-supports.
    settlement = maybe_promote_judges(log_path, judges_path)
    if settlement is not None:
        yield {"stage": "verifier", **settlement}

    # The proposal source for that loop: when this mission's gates passed
    # cards the evidence then beat (weakens, or a probe-method death), ask
    # for one predicate that would have flagged them. It lands on the
    # shadow bench with no power; a bad proposal is a recorded refusal.
    failed_views: list[dict[str, Any]] = []
    passed_views: list[dict[str, Any]] = []
    for record in found:
        card = entered_cards.get(record["card_id"])
        if card is None:
            continue
        view = {
            "claim": card.claim,
            "mechanism": card.mechanism,
            "prediction": card.prediction,
            "falsifier": card.falsifier,
            "concept_a": card.pair[0],
            "concept_b": card.pair[1],
            "alienness": card.alienness,
            "plausibility_signal": 0.0,
        }
        beaten = record.get("verdict") == "weakens" or (
            record.get("pipeline_bottleneck") == "probe_method"
        )
        if beaten:
            failed_views.append(view)
        elif record.get("verdict") == "supports":
            passed_views.append(view)
    if len(failed_views) >= SHADOW_TRIGGER:
        try:
            proposal = propose_shadow(
                client,
                failures=failed_views,
                passes=passed_views,
                bench_path=judges_path,
            )
        except Exception as exc:  # a broken proposer must not end the mission
            proposal = {"admitted": False, "reason": f"proposer failed: {exc}"}
        yield {"stage": "shadow_judge", **proposal}

    # Unmatched requirements feed the freeze wishlist. Guidance for the
    # host operator only — a later freeze binds new claims, never these.
    unmatched = [
        record["world_requirement"]
        for record in found
        if record.get("world_incompatible")
        and isinstance(record.get("world_requirement"), dict)
    ]
    if unmatched:
        try:
            wish_path = record_world_wishlist(
                world_root or root, unmatched, topic=topic
            )
        except OSError:
            wish_path = None
        if wish_path is not None:
            yield {
                "stage": "world_wishlist",
                "unmatched": len(unmatched),
                "path": str(wish_path),
            }

    briefs = sum(1 for record in found if record["has_brief"])
    statuses = list(promoted_status.values())
    ledger = getattr(client, "ledger", None)
    funnel = campaign_funnel(attempts, found, promotions)
    questions = {
        "targeted": len(targets),
        "entered": survivors,
        "closed_by_prior": statuses.count("closed_by_prior"),
        "world_incompatible": sum(
            1 for row in outcomes if row.get("world_incompatible")
        ),
        "weakened": statuses.count("weakened"),
        "still_open": statuses.count("speculative"),
    }
    spent = (
        {"api_calls": ledger.spent_calls, "tokens": ledger.spent_tokens}
        if ledger is not None
        else None
    )
    packet_md = render_mission_packet(
        topic,
        ranked=ranked,
        found=found,
        workspace=str(workspace) if workspace is not None else None,
        program=next_program,
        funnel=funnel,
        kills=all_kills,
        questions=questions,
        spent=spent,
        knowledge_until=assets.fresh_until,
        cards_generated=total_cards,
        entered=survivors,
    )
    done = {
        "stage": "done",
        "cards": total_cards,
        "entered": survivors,
        "survivors": survivors,
        "briefs": briefs,
        "corroborated": sum(
            1 for status in statuses if status in ("corroborated", "verified")
        ),
        "funnel": funnel,
        "questions": questions,
        "lessons_banked": banked,
        "trajectories_logged": log_size,
        "spent": spent,
        "knowledge_until": assets.fresh_until,
        "workspace": str(workspace) if workspace is not None else None,
        "packet": (
            str(Path(workspace) / "RESEARCH_PACKET.md")
            if workspace is not None
            else None
        ),
        "packet_markdown": packet_md,
        "program": next_program,
        "explore": explore_mode,
        "jumps_opened": far_opened,
        "reframes_opened": reframe_opened,
        "polish_mode": polish_mode,
        "experiment_mode": experiment_mode,
        "experiment_rounds": experiment_budget,
        "idea_mode": idea_mode,
        "idea_rounds": idea_budget,
    }
    if workspace is not None:
        write_summary(
            workspace,
            topic,
            {k: v for k, v in done.items() if k not in ("stage", "packet_markdown")},
            packet_md=packet_md,
        )
    yield done


def _brief_event(brief: ResearchBrief, card: GeneratedCard, round_no: int) -> dict[str, Any]:
    event = brief.to_dict()
    event["stage"] = "brief"
    event["round"] = round_no
    event["pair"] = list(card.pair)
    event["claim"] = card.claim
    return event


def _run_probe_attempt(
    spec: Any,
    workspace: Path | None,
    card_id: str,
    attempt: int,
    world: Any = None,
) -> Any:
    if workspace is not None:
        dest = candidate_dir(workspace, card_id) / "probe" / f"attempt-{attempt}"
        dest.mkdir(parents=True, exist_ok=True)
        if world is not None:
            bind_world(dest, world)
        return run_probe(spec, dest)
    with tempfile.TemporaryDirectory(prefix="ffprobe-") as tmp:
        dest = Path(tmp)
        if world is not None:
            bind_world(dest, world)
        return run_probe(spec, dest)


def _evidence_pipeline(
    client: Any,
    feed: Any,
    card: GeneratedCard,
    topic: str,
    *,
    polish_rounds: int = 0,
    run_probes: bool = True,
    workspace: Path | None = None,
    prior_probe: dict[str, Any] | None = None,
    prior_status: str | None = None,
    writeup: bool = True,
    bundle: dict[str, Any] | None = None,
    skill_blocks: dict[str, str] | None = None,
    experiment_rounds: int = 0,
    world: Any = None,
    wiki_works_pool: list | None = None,
    program: str | tuple[str, ...] = "",
    world_requirement: Any = None,
    world_dest: Path | None = None,
    world_request: str = "",
    world_gap: bool = False,
) -> Iterator[dict[str, Any]]:
    """Cheap evidence first, expensive writing last.

    Survey → prior check → pre-registered diagnosis → two-arm probe on
    an attested freeze (or SYNTHETIC coherence if the operator asked for
    no world). An uninformative or crashed test can be redesigned up to
    `experiment_rounds` extra times. The stop rule is discrimination
    (`supports` / `weakens`) or the cap — never a preferred sign. A
    catalog miss or honest `none` skips the probe rather than inventing
    data. Write-up is a separate pass (`_writeup_pipeline`) so a value
    tournament can rank live cards before any briefing is paid for.
    `writeup=False` is the production path; `writeup=True` keeps the old
    single-card callers working.

    `prior_status` is the hypothesis line's current ladder rung. When a
    corroborated line draws a second supporting verdict, this pipeline
    immediately runs the confirmation tier — the heavier, seed-replicated
    probe that stands between corroborated and verified.
    """
    works: list = []
    if feed is not None and hasattr(feed, "survey_around"):
        yield {"stage": "surveying", "card_id": card.card_id, "pair": list(card.pair)}
        try:
            result = feed.survey_around(*card.pair, claim=card.claim, topic=topic)
        except TypeError:
            try:
                result = feed.survey_around(*card.pair, claim=card.claim)
            except TypeError:
                result = feed.survey_around(*card.pair)
        if isinstance(result, FeedBlocked):
            yield {"stage": "survey", "card_id": card.card_id, "blocked": result.to_dict()}
        else:
            works = list(result)
            yield {
                "stage": "survey",
                "card_id": card.card_id,
                "works": [work.to_dict() for work in works],
            }
    prior = strongest_prior(
        card.claim, merge_works(works, wiki_works_pool or [])
    ) if works or wiki_works_pool else None
    if prior is not None:
        event = prior.to_dict()
        event["stage"] = "prior"
        event["card_id"] = card.card_id
        yield event
        if prior.kills:
            # Reading beats researching: the claim is already on record.
            if bundle is not None:
                bundle["skip_writeup"] = True
                bundle["prior"] = prior
            return

    probe_payload: dict[str, Any] | None = None
    verdict: str | None = None
    diagnosis = None
    extra_rounds = max(0, int(experiment_rounds))
    attempts_log: list[dict[str, Any]] = []
    discarded: list[str] = []
    last_bottleneck: str | None = None
    last_action: str | None = None
    impl_failure: dict[str, Any] | None = None
    prior_for_diag = prior_probe
    diagnose_skills = (skill_blocks or {}).get("diagnose", "")
    probe_skills = (skill_blocks or {}).get("probe", "")
    allow_synthetic = _operator_allows_synthetic(world_request)
    world_path = None
    paper_pool = list(works) + list(wiki_works_pool or [])
    if paper_pool:
        try:
            world_path = write_world_path(
                client,
                card,
                topic,
                paper_pool,
                requirement=world_requirement,
            )
        except Exception:
            world_path = None
        if world_path is not None:
            yield {
                "stage": "world_path",
                "card_id": card.card_id,
                **world_path.to_dict(),
            }
            if isinstance(world_requirement, dict):
                world_requirement = {
                    **world_requirement,
                    **world_path.wishlist_fields(),
                }
            if bundle is not None:
                bundle["world_path"] = world_path.to_dict()
            if (
                world is not None
                and str(getattr(world, "role", "") or "") not in {"generated", "placebo"}
            ):
                reason = lineage_conflicts(
                    world,
                    named_instance=world_path.named_instance,
                    lineage_schema=world_path.schema,
                )
                if reason:
                    yield {
                        "stage": "world_surrogate",
                        "card_id": card.card_id,
                        "world_id": str(getattr(world, "id", "") or ""),
                        "reason": reason,
                    }
                    world = None

    # Forward-simulate the bound world before any diagnosis: the runtime
    # rolls the frozen bytes under every declared lever and measures every
    # observable. The diagnosis designs against a measured dose-response
    # reading instead of a byte blob. Scout evidence only — it informs and
    # unbinds; it never climbs.
    dynamics_card: dict[str, Any] | None = None
    if run_probes and world is not None:
        try:
            dynamics_card = response_card(world)
        except Exception:
            dynamics_card = None
        if dynamics_card is not None:
            yield {
                "stage": "world_scout",
                "card_id": card.card_id,
                "world_id": dynamics_card["world_id"],
                "levers": sorted(dynamics_card["levers"]),
                "observables": list(dynamics_card["observables"]),
                "responses": dict(dynamics_card["levers"]),
                "inert": list(dynamics_card["inert"]),
                "report_digest": dynamics_card["report_digest"],
            }
            if workspace is not None:
                chain.append_event(
                    candidate_dir(Path(workspace), card.card_id) / "chain.jsonl",
                    chain.SIMULATE_WORLD,
                    {
                        "card_id": card.card_id,
                        "phase": "scout",
                        "world_digest": dynamics_card["world_digest"],
                        "report_digest": dynamics_card["report_digest"],
                        "levers": sorted(dynamics_card["levers"]),
                    },
                )

    if run_probes:
        for attempt in range(extra_rounds + 1):
            inert_forecast = False
            need_new_diagnosis = diagnosis is None or last_bottleneck == "method"
            if need_new_diagnosis:
                yield {
                    "stage": "diagnosing",
                    "card_id": card.card_id,
                    "attempt": attempt,
                }
                try:
                    diagnosis = write_diagnosis(
                        client,
                        card,
                        topic,
                        prior_probe=prior_for_diag,
                        failed_experiments=discarded,
                        skills=diagnose_skills,
                        world=world,
                        works=works,
                        program=program,
                        requirement=world_requirement if world is None else None,
                        dynamics=dynamics_card,
                        world_path=world_path,
                    )
                except GenerationRefused as exc:
                    yield {
                        "stage": "diagnosis_refused",
                        "card_id": card.card_id,
                        "attempt": attempt,
                        "missing_capability": exc.record.missing_capability,
                        "unlock_condition": exc.record.unlock_condition,
                    }
                    last_bottleneck = classify_bottleneck(refused="diagnosis")
                    decision = decide_retry(
                        attempt=attempt,
                        cap=extra_rounds,
                        bottleneck=last_bottleneck,
                    )
                    last_action = str(decision.get("action") or "")
                    yield {
                        "stage": "experiment_decision",
                        "card_id": card.card_id,
                        "attempt": attempt,
                        "discarded": list(discarded),
                        **decision,
                    }
                    if not decision["continue"]:
                        break
                    continue
                except LLMUnavailable as exc:
                    yield {
                        "stage": "blocked",
                        "missing_capability": exc.record.missing_capability,
                        "unlock_condition": exc.record.unlock_condition,
                    }
                    return
                event = diagnosis.to_dict()
                event["stage"] = "diagnosis"
                event["attempt"] = attempt
                yield event

                if dynamics_card is not None and world is not None:
                    lever = diagnosis.world_lever or "none"
                    if lever == "none" and attested_freeze(world):
                        analysis = {
                            "stop_reason": "no_handle",
                            "room_to_move": False,
                            "n_steps": 0,
                            "summary": (
                                f"On {getattr(world, 'id', '') or 'this object'}, "
                                "the registered prediction has no runtime handle; "
                                "the idea cannot be executed here."
                            ),
                        }
                        yield {
                            "stage": "probe_skipped",
                            "card_id": card.card_id,
                            "world_id": str(getattr(world, "id", "") or ""),
                            "reason": (
                                "the registered mechanism has no lever in this "
                                "world's runtime dynamics; skip the probe rather "
                                "than invent data"
                            ),
                            "bottleneck": "world",
                            "analysis": analysis,
                        }
                        last_bottleneck = classify_bottleneck(refused="world")
                        decision = decide_retry(
                            attempt=attempt,
                            cap=extra_rounds,
                            bottleneck=last_bottleneck,
                        )
                        last_action = str(decision.get("action") or "")
                        yield {
                            "stage": "experiment_decision",
                            "card_id": card.card_id,
                            "attempt": attempt,
                            "discarded": list(discarded),
                            **decision,
                        }
                        break
                    else:
                        # The prediction check: roll the world forward under
                        # the registered lever before the experiment exists.
                        # An analysis aid on the record, never a verdict.
                        try:
                            forecast = forward_simulate(world, lever)
                        except Exception:
                            forecast = None
                        if forecast is not None:
                            delta = forecast["response"].get(
                                diagnosis.world_observable
                            )
                            simulated = (
                                "flat"
                                if delta is None or abs(delta) < RESPONSE_FLOOR
                                else "treatment_lower"
                                if delta < 0
                                else "treatment_higher"
                            )
                            inert_forecast = simulated == "flat"
                            yield {
                                "stage": "world_forecast",
                                "card_id": card.card_id,
                                "lever": lever,
                                "observable": diagnosis.world_observable,
                                "simulated_delta": delta,
                                "simulated_direction": simulated,
                                "registered_direction": diagnosis.expected_direction,
                                "agrees": simulated == diagnosis.expected_direction,
                                "report_digest": forecast["report_digest"],
                                "rewrite": inert_forecast,
                                "analysis": dict(forecast.get("analysis") or {}),
                            }
                            if workspace is not None:
                                chain.append_event(
                                    candidate_dir(Path(workspace), card.card_id)
                                    / "chain.jsonl",
                                    chain.SIMULATE_WORLD,
                                    {
                                        "card_id": card.card_id,
                                        "phase": "forecast",
                                        "world_digest": dynamics_card["world_digest"],
                                        "report_digest": forecast["report_digest"],
                                        "lever": lever,
                                        "observable": diagnosis.world_observable,
                                        "simulated_delta": delta,
                                    },
                                )

            if diagnosis is None:
                break
            if inert_forecast:
                last_bottleneck = "method"
                if diagnosis.experiment and diagnosis.experiment not in discarded:
                    discarded.append(diagnosis.experiment)
                prior_for_diag = {
                    "experiment": diagnosis.experiment,
                    "verdict": "uninformative",
                }
                if dynamics_card is not None:
                    prior_for_diag["world_scout"] = scout_summary(
                        dynamics_card, diagnosis.world_lever
                    )
                decision = decide_retry(
                    attempt=attempt,
                    cap=extra_rounds,
                    bottleneck=last_bottleneck,
                )
                last_action = str(decision.get("action") or "")
                yield {
                    "stage": "experiment_decision",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "discarded": list(discarded),
                    **decision,
                    "reason": "registered lever is inert in the bound world",
                }
                if not decision["continue"]:
                    break
                continue
            if world is None and not allow_synthetic:
                analysis = {
                    "stop_reason": "no_object",
                    "room_to_move": False,
                    "n_steps": 0,
                    "summary": (
                        "No attested freeze matches this claim; inventing "
                        "data is not verification and is not an idea analysis."
                    ),
                }
                yield {
                    "stage": "probe_skipped",
                    "card_id": card.card_id,
                    "reason": (
                        "no attested freeze is bound; skip the probe rather "
                        "than invent or construct a substitute"
                    ),
                    "bottleneck": "world",
                    "analysis": analysis,
                    "world_gap": world_gap,
                }
                last_bottleneck = classify_bottleneck(refused="world")
                decision = decide_retry(
                    attempt=attempt,
                    cap=extra_rounds,
                    bottleneck=last_bottleneck,
                )
                last_action = str(decision.get("action") or "")
                yield {
                    "stage": "experiment_decision",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "discarded": list(discarded),
                    **decision,
                }
                break
            yield {
                "stage": "probing",
                "card_id": card.card_id,
                "attempt": attempt,
            }
            try:
                spec = write_probe(
                    client,
                    card,
                    diagnosis,
                    topic,
                    skills=probe_skills,
                    prior_failure=impl_failure,
                    world=world,
                )
            except (GenerationRefused, ProbeRefused) as exc:
                yield {
                    "stage": "probe_refused",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
                }
                last_bottleneck = classify_bottleneck(refused="probe")
                impl_failure = {
                    "status": "refused",
                    "error": exc.record.unlock_condition,
                }
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "bottleneck": last_bottleneck,
                        "status": "refused",
                        "verdict": None,
                        "experiment": diagnosis.experiment,
                        "error": exc.record.unlock_condition,
                    }
                )
                decision = decide_retry(
                    attempt=attempt,
                    cap=extra_rounds,
                    bottleneck=last_bottleneck,
                )
                last_action = str(decision.get("action") or "")
                yield {
                    "stage": "experiment_decision",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "discarded": list(discarded),
                    **decision,
                }
                if not decision["continue"]:
                    break
                continue
            except LLMUnavailable as exc:
                yield {
                    "stage": "blocked",
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
                }
                return

            experiment_digest = sha256_text(spec.source)
            if workspace is not None:
                chain.append_event(
                    candidate_dir(Path(workspace), card.card_id) / "chain.jsonl",
                    chain.REGISTER_PROTOCOL,
                    {
                        "card_id": card.card_id,
                        "experiment_digest": experiment_digest,
                        "world_digest": str(getattr(world, "digest", "") or ""),
                    },
                )

            result = _run_probe_attempt(
                spec, workspace, card.card_id, attempt, world=world
            )
            probe_payload = result.to_dict()
            probe_payload["attempt"] = attempt
            probe_payload["experiment_digest"] = sha256_text(spec.source)
            if world is not None:
                probe_payload["world_digest"] = world.digest
                probe_payload["data_digest"] = world.digest
                probe_payload["evidence_id"] = evidence_id(
                    claim_id=card.card_id,
                    experiment_digest=probe_payload["experiment_digest"],
                    world_digest=world.digest,
                    data_digest=world.digest,
                )
            if diagnosis is not None and ablation_required(
                diagnosis.alternative,
                diagnosis.experiment,
                levers=levers_of(world) if world is not None else (),
                world_lever=diagnosis.world_lever,
            ):
                probe_payload["ablation_required"] = True
                probe_payload["mechanism_identified"] = False
            probe_payload["kind"] = _probe_kind_for(world, spec.source)
            probe_payload["world_has_dynamics"] = bool(
                world is not None and has_dynamics(world)
            )
            event = dict(probe_payload)
            event["stage"] = "probe"
            event["card_id"] = card.card_id
            yield event
            if result.status != "ran":
                last_bottleneck = classify_bottleneck(
                    status=result.status,
                    error=result.error,
                    world_role=str(getattr(world, "role", "") or ""),
                    last_action=last_action,
                )
                impl_failure = {
                    "status": result.status,
                    "error": result.error,
                }
                attempts_log.append(
                    {
                        "attempt": attempt,
                        "bottleneck": last_bottleneck,
                        "status": result.status,
                        "verdict": None,
                        "experiment": diagnosis.experiment,
                        "error": result.error,
                    }
                )
                decision = decide_retry(
                    attempt=attempt,
                    cap=extra_rounds,
                    bottleneck=last_bottleneck,
                )
                last_action = str(decision.get("action") or "")
                yield {
                    "stage": "experiment_decision",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "discarded": list(discarded),
                    **decision,
                }
                if not decision["continue"]:
                    break
                continue

            evidence = judge_probe(diagnosis, result.treatment, result.control)
            probe_payload.update(evidence)
            verdict = str(evidence.get("verdict") or "") or None
            if (
                verdict == "supports"
                and probe_payload.get("kind") == KIND_WORLD
                and world is not None
                and has_dynamics(world)
            ):
                sep_real = arm_separation(
                    float(result.treatment), float(result.control)
                )
                placebo_dir = (
                    candidate_dir(Path(workspace), card.card_id)
                    / "probe"
                    / f"attempt-{attempt}-placebo"
                    if workspace is not None
                    else Path(tempfile.mkdtemp(prefix="ff-placebo-"))
                )
                try:
                    raw = placebo_dir / "raw"
                    dest = placebo_dir / "run"
                    dest.mkdir(parents=True, exist_ok=True)
                    placebo = materialize_placebo(world, raw)
                    bind_world(dest, placebo)
                    placebo_run = run_probe(spec, dest)
                except Exception as exc:
                    placebo_run = None
                    probe_payload["world_consumed"] = False
                    probe_payload["placebo_error"] = str(exc)
                if placebo_run is not None and placebo_run.status == "ran":
                    sep_placebo = arm_separation(
                        float(placebo_run.treatment), float(placebo_run.control)
                    )
                    consumed = world_consumed(
                        sep_real, sep_placebo, diagnosis.margin
                    )
                    probe_payload["world_consumed"] = consumed
                    probe_payload["placebo_separation"] = round(sep_placebo, 6)
                    yield {
                        "stage": "world_placebo",
                        "card_id": card.card_id,
                        "consumed": consumed,
                        "separation": round(sep_real, 6),
                        "placebo_separation": round(sep_placebo, 6),
                        "margin": diagnosis.margin,
                    }
                    if workspace is not None:
                        chain.append_event(
                            candidate_dir(Path(workspace), card.card_id)
                            / "chain.jsonl",
                            chain.EXECUTE_PLACEBO,
                            {
                                "card_id": card.card_id,
                                "consumed": consumed,
                                "experiment_digest": probe_payload.get(
                                    "experiment_digest"
                                ),
                                "world_digest": world.digest,
                                "placebo_digest": placebo.digest,
                            },
                        )
                    if not consumed:
                        verdict = "uninformative"
                        evidence["verdict"] = verdict
                        evidence["reason"] = (
                            "placebo control: |sep_real - sep_placebo| "
                            f"is below the pre-registered margin {diagnosis.margin}; "
                            "the experiment did not consume the bound world"
                        )
                        probe_payload.update(evidence)
                elif placebo_run is None or placebo_run.status != "ran":
                    probe_payload["world_consumed"] = False
                    verdict = "uninformative"
                    evidence["verdict"] = verdict
                    evidence["reason"] = (
                        "placebo control failed; consumption cannot be attested"
                    )
                    probe_payload.update(evidence)
            event = dict(evidence)
            event["stage"] = "evidence"
            event["card_id"] = card.card_id
            event["attempt"] = attempt
            event["experiment_digest"] = probe_payload.get("experiment_digest")
            event["world_digest"] = probe_payload.get("world_digest")
            event["data_digest"] = probe_payload.get("data_digest")
            event["evidence_id"] = probe_payload.get("evidence_id")
            if probe_payload.get("mechanism_identified") is False:
                event["mechanism_identified"] = False
            if probe_payload.get("ablation_required"):
                event["ablation_required"] = True
            yield event
            last_bottleneck = classify_bottleneck(
                status=result.status,
                verdict=verdict,
                world_role=str(getattr(world, "role", "") or ""),
                last_action=last_action,
            )
            row = {
                "attempt": attempt,
                "bottleneck": last_bottleneck,
                "status": result.status,
                "verdict": verdict,
                "experiment": diagnosis.experiment,
                "treatment": evidence.get("treatment"),
                "control": evidence.get("control"),
                "error": "",
            }
            attempts_log.append(row)
            get_host().notify(
                "after_evidence",
                probe=probe_payload,
                card=card,
                diagnosis=diagnosis,
            )
            if last_bottleneck == "method" and diagnosis.experiment:
                if diagnosis.experiment not in discarded:
                    discarded.append(diagnosis.experiment)
                prior_for_diag = {
                    "experiment": diagnosis.experiment,
                    "verdict": "uninformative",
                }
                if dynamics_card is not None:
                    # World facts for the redesign: which levers the runtime
                    # measured responsive, and whether the registered one was
                    # inert. No treatment/control numbers exist in this quote.
                    prior_for_diag["world_scout"] = scout_summary(
                        dynamics_card, diagnosis.world_lever
                    )
                impl_failure = None
            decision = decide_retry(
                attempt=attempt,
                cap=extra_rounds,
                bottleneck=last_bottleneck,
            )
            last_action = str(decision.get("action") or "")
            yield {
                "stage": "experiment_decision",
                "card_id": card.card_id,
                "attempt": attempt,
                "discarded": list(discarded),
                **decision,
            }
            if not decision["continue"]:
                if verdict == "supports" and prior_status == "corroborated":
                    for heavy_event in _heavy_confirmation(
                        card, diagnosis, spec, workspace, world=world
                    ):
                        yield heavy_event
                break

    if probe_payload is not None:
        probe_payload["attempts"] = attempts_log
        probe_payload["discarded"] = list(discarded)
    if bundle is not None:
        bundle["works"] = works
        bundle["probe_payload"] = probe_payload
        bundle["prior"] = prior
        bundle["diagnosis"] = diagnosis
        bundle["skip_writeup"] = verdict == "weakens"
    if verdict == "weakens":
        # The evidence moved against the mechanism; the ladder will record
        # it. Spending review rounds writing this idea up would be paying
        # to decorate a claim we just learned to distrust.
        return
    if not writeup:
        return

    yield from _writeup_pipeline(
        client,
        card,
        topic,
        works=works,
        probe_payload=probe_payload,
        prior=prior,
        polish_rounds=polish_rounds,
        workspace=workspace,
        skill_blocks=skill_blocks,
        seed_label=card.pair[0],
        diagnosis=diagnosis,
        skill_catalog=None,
        world=world,
        host_execute=False,
    )


def _writeup_pipeline(
    client: Any,
    card: GeneratedCard,
    topic: str,
    *,
    works: list,
    probe_payload: dict[str, Any] | None,
    prior: Any,
    polish_rounds: int = 0,
    workspace: Path | None = None,
    skill_blocks: dict[str, str] | None = None,
    seed_label: str = "",
    diagnosis: Any = None,
    skill_catalog: Path | None = None,
    world: Any = None,
    host_execute: bool = False,
    host_timeout: float | None = None,
) -> Iterator[dict[str, Any]]:
    """The expensive writing pass: briefing, optional polish, note, protocol.

    Called only for ideas the evidence did not weaken or close. Diagnosis
    never reads this output, so a later polish cannot touch the registered
    experiment. The protocol is compiled from attested fields; it is not a
    new generation.
    """
    yield {"stage": "briefing", "card_id": card.card_id, "round": 0}
    compiled_fallback = False
    try:
        brief = write_brief(
            client,
            card,
            topic,
            works,
            skills=(skill_blocks or {}).get("writeup", ""),
        )
    except GenerationRefused as exc:
        yield {
            "stage": "brief_refused",
            "card_id": card.card_id,
            "missing_capability": exc.record.missing_capability,
            "unlock_condition": exc.record.unlock_condition,
            "compiled_fallback": True,
        }
        brief = compile_brief(card, works, diagnosis=diagnosis)
        compiled_fallback = True
    except LLMUnavailable as exc:
        yield {
            "stage": "blocked",
            "missing_capability": exc.record.missing_capability,
            "unlock_condition": exc.record.unlock_condition,
        }
        return
    yield _brief_event(brief, card, 0)
    last_brief = brief
    if not compiled_fallback:
        for event in polish_idea(
            client, card, brief, topic, works, rounds=polish_rounds
        ):
            yield event
            if event["stage"] == "blocked":
                return
            if event["stage"] == "brief":
                last_brief = brief_from_dict(event, works)
    note = render_note(
        topic,
        card,
        last_brief,
        works,
        probe=probe_payload,
        prior=prior.to_dict() if prior is not None else None,
    )
    protocol = render_protocol(
        topic,
        card,
        last_brief,
        works,
        probe=probe_payload,
        diagnosis=diagnosis,
        prior=prior.to_dict() if prior is not None else None,
        world=world,
    )
    pdf = None
    if workspace is not None:
        pdf = compile_tex(note.latex, candidate_dir(Path(workspace), card.card_id) / "proposal")
        write_idea(
            workspace,
            card_id=card.card_id,
            brief=last_brief.to_dict(),
            works=[work.to_dict() for work in works],
            note_md=note.markdown,
            note_tex=note.latex,
            probe=probe_payload,
            protocol_md=protocol.markdown,
            protocol=protocol.payload,
            readme_md=protocol.readme,
        )
        if host_execute and str(getattr(world, "role", "") or "") == "generated":
            yield {
                "stage": "host_skipped",
                "card_id": card.card_id,
                "reason": "generated_world",
                "honesty": (
                    "a constructed world cannot host-execute as an attested freeze"
                ),
            }
        elif host_execute:
            from .hostexp import ExecuteError, execute_protocol

            folder = candidate_dir(Path(workspace), card.card_id)
            try:
                # In-loop confirmation is E1 same-instance reproduction:
                # the host reruns the exact bytes the probe saw, so the
                # EvidenceIDs can match. The parent-scale run (E2) stays a
                # separate, honestly-labelled act via `farfield execute`.
                host_record = execute_protocol(
                    folder, timeout_seconds=host_timeout, prefer_parent=False
                )
            except ExecuteError as exc:
                host_record = {
                    "ok": False,
                    "stage": "host_execute",
                    "error": str(exc),
                    "honesty": (
                        "protocol_executed was blocked; the cheap probe "
                        "is unchanged. This is not a discovery."
                    ),
                }
                (folder / "host_run.json").write_text(
                    json.dumps(host_record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            yield {**host_record, "card_id": card.card_id}
        seal_candidate(candidate_dir(Path(workspace), card.card_id))
    yield {
        "stage": "note",
        "card_id": card.card_id,
        "title": last_brief.title,
        "markdown": note.markdown,
        "latex": note.latex,
        "pdf": pdf,
        "protocol": protocol.markdown,
        "protocol_json": protocol.payload,
    }

    if (
        workspace is not None
        and probe_payload
        and probe_payload.get("verdict") == "supports"
        and diagnosis is not None
    ):
        minted = distill_skill(
            topic=topic,
            seed_label=seed_label or card.pair[0],
            card=card,
            diagnosis=diagnosis,
            probe=probe_payload,
            brief=last_brief,
        )
        if minted is not None:
            written = persist_skill(
                minted,
                workspace=candidate_dir(Path(workspace), card.card_id),
                catalog=None,
            )
            yield {
                "stage": "skill_distilled",
                "card_id": card.card_id,
                "name": minted.name,
                "digest": minted.digest,
                "paths": written,
                "executable": False,
            }


HEAVY_REPLICATES = 5


def _heavy_confirmation(
    card: GeneratedCard,
    diagnosis: Any,
    spec: Any,
    workspace: Path | None,
    *,
    world: Any = None,
    replicates: int = HEAVY_REPLICATES,
) -> Iterator[dict[str, Any]]:
    """The confirmation tier between corroborated and verified.

    Executor-owned replication: the model wrote the registered script
    once; it is not asked for a "heavier" version, and it does not report
    its own replicates. The trusted executor reruns the *registered*
    bytes under seeds derived from the evidence identity itself
    (`seed_i = H(experiment_digest ‖ world_digest ‖ i)` — the script
    cannot know its seeds at writing time), records every execution and
    the aggregation in the candidate's event chain, and computes the
    statistics. `supports` requires the entire 95% confidence interval
    of the per-seed separations to clear the pre-registered margin. Any
    failed rerun aborts the tier — the line simply stays corroborated.
    """
    experiment_digest = sha256_text(spec.source)
    world_digest = str(getattr(world, "digest", "") or "")
    seeds = replication_seeds(experiment_digest, world_digest, replicates)
    yield {
        "stage": "heavy_probing",
        "card_id": card.card_id,
        "runner": "executor",
        "seeds": list(seeds),
    }
    chain_path = (
        candidate_dir(Path(workspace), card.card_id) / "chain.jsonl"
        if workspace is not None
        else None
    )
    pairs: list[tuple[float, float]] = []
    parents: list[str] = []
    with tempfile.TemporaryDirectory(prefix="ffheavy-") as tmp:
        base = (
            Path(workspace) / f"{card.card_id}-heavy"
            if workspace is not None
            else Path(tmp)
        )
        for index, seed in enumerate(seeds):
            dest = base / f"seed-{index}"
            dest.mkdir(parents=True, exist_ok=True)
            if world is not None:
                bind_world(dest, world)
            result = run_probe(
                spec, dest, timeout_seconds=HEAVY_TIMEOUT_SECONDS, seed=seed
            )
            yield {
                "stage": "heavy_probe",
                "card_id": card.card_id,
                "seed": seed,
                "status": result.status,
                "treatment": result.treatment,
                "control": result.control,
                "error": result.error,
            }
            if result.status != "ran":
                # Fail-closed and quiet: a rerun the executor could not
                # complete is a missing confirmation, not a refutation.
                return
            pairs.append((float(result.treatment), float(result.control)))
            execution_id = evidence_id(
                claim_id=card.card_id,
                experiment_digest=experiment_digest,
                world_digest=world_digest,
                data_digest=world_digest,
                environment_digest=f"seed:{seed}",
            )
            if chain_path is not None:
                record = chain.append_event(
                    chain_path,
                    chain.EXECUTE_HEAVY_SEED,
                    {
                        "card_id": card.card_id,
                        "seed": seed,
                        "execution_id": execution_id,
                        "experiment_digest": experiment_digest,
                        "world_digest": world_digest,
                        "treatment": result.treatment,
                        "control": result.control,
                    },
                )
                parents.append(str(record["digest"]))
    heavy = judge_replicated(diagnosis, pairs)
    aggregate_eid = evidence_id(
        claim_id=card.card_id,
        experiment_digest=experiment_digest,
        world_digest=world_digest,
        data_digest=world_digest,
        environment_digest="heavy:" + ",".join(str(seed) for seed in seeds),
    )
    heavy.update(
        {
            "stage": "heavy_evidence",
            "card_id": card.card_id,
            "runner": "executor",
            "seeds": list(seeds),
            "treatment": round(sum(t for t, _ in pairs) / len(pairs), 6),
            "control": round(sum(c for _, c in pairs) / len(pairs), 6),
            "experiment_digest": experiment_digest,
            "world_digest": world_digest,
            "evidence_id": aggregate_eid,
            "replication": (
                TIER_SAME_INSTANCE if world is not None else TIER_SYNTHETIC
            ),
        }
    )
    if chain_path is not None:
        chain.append_event(
            chain_path,
            chain.AGGREGATE_HEAVY,
            {
                "card_id": card.card_id,
                "evidence_id": aggregate_eid,
                "experiment_digest": experiment_digest,
                "world_digest": world_digest,
                "verdict": heavy["verdict"],
                "n": heavy["n"],
                "margin": heavy["margin"],
                "ci_low": heavy.get("ci_low"),
                "ci_high": heavy.get("ci_high"),
                "seeds": list(seeds),
                "parents": parents,
            },
        )
    yield heavy


def polish_idea(
    client: Any,
    card: GeneratedCard,
    brief: ResearchBrief,
    topic: str,
    works: list[FreshWork],
    *,
    rounds: int,
) -> Iterator[dict[str, Any]]:
    """Review and revise an idea `rounds` times, or until the reviewer stops."""
    current = brief
    for round_no in range(1, max(0, rounds) + 1):
        yield {
            "stage": "reviewing",
            "card_id": card.card_id,
            "round": round_no,
        }
        try:
            review = review_idea(client, card, current, topic, works)
        except GenerationRefused as exc:
            yield {
                "stage": "review_refused",
                "card_id": card.card_id,
                "round": round_no,
                "missing_capability": exc.record.missing_capability,
                "unlock_condition": exc.record.unlock_condition,
            }
            return
        except LLMUnavailable as exc:
            yield {
                "stage": "blocked",
                "missing_capability": exc.record.missing_capability,
                "unlock_condition": exc.record.unlock_condition,
            }
            return
        event = review.to_dict()
        event["stage"] = "review"
        event["round"] = round_no
        yield event
        if not review.keep_going:
            yield {
                "stage": "polish_stop",
                "card_id": card.card_id,
                "round": round_no,
                "reason": review.critique,
            }
            return
        yield {
            "stage": "refining",
            "card_id": card.card_id,
            "round": round_no,
        }
        try:
            current = refine_brief(client, card, current, review, topic, works)
        except GenerationRefused as exc:
            yield {
                "stage": "refine_refused",
                "card_id": card.card_id,
                "round": round_no,
                "missing_capability": exc.record.missing_capability,
                "unlock_condition": exc.record.unlock_condition,
            }
            return
        except LLMUnavailable as exc:
            yield {
                "stage": "blocked",
                "missing_capability": exc.record.missing_capability,
                "unlock_condition": exc.record.unlock_condition,
            }
            return
        yield _brief_event(current, card, round_no)


def works_from_dicts(rows: list[dict[str, Any]]) -> list[FreshWork]:
    works = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cite = str(row.get("arxiv_id") or row.get("work_id") or "").strip()
        if not cite:
            continue
        works.append(
            FreshWork(
                title=str(row.get("title") or ""),
                published=str(row.get("published") or ""),
                arxiv_id=str(row.get("arxiv_id") or ""),
                abstract=str(row.get("abstract") or ""),
                source=str(row.get("source") or "arxiv"),
                work_id=cite,
                url=str(row.get("url") or ""),
                venue=str(row.get("venue") or ""),
            )
        )
    return works


def card_from_dict(row: dict[str, Any]) -> GeneratedCard:
    pair = row.get("pair") or ["", ""]
    return GeneratedCard(
        card_id=str(row.get("card_id") or "gen_unknown"),
        operator=str(row.get("operator") or "directional"),
        claim=str(row.get("claim") or ""),
        mechanism=str(row.get("mechanism") or ""),
        prediction=str(row.get("prediction") or ""),
        falsifier=str(row.get("falsifier") or "pair_not_already_combined"),
        pair=(str(pair[0]), str(pair[1])),
        pair_nodes=("concept:a", "concept:b"),
        alienness=float(row.get("alienness") or 0),
        model=str(row.get("model") or ""),
        artifact_digest="",
        artifact_uri="file:///dev/null",
        replay_mode="live",
        dead_end=str(row.get("dead_end") or ""),
        why_failed=str(row.get("why_failed") or ""),
        reframe=str(row.get("reframe") or ""),
        objection=str(row.get("objection") or ""),
    )


def brief_from_dict(row: dict[str, Any], works: list[FreshWork]) -> ResearchBrief:
    return ResearchBrief(
        card_id=str(row.get("card_id") or ""),
        title=str(row.get("title") or ""),
        gap=str(row.get("gap") or ""),
        idea=str(row.get("idea") or ""),
        approach=str(row.get("approach") or ""),
        first_steps=tuple(row.get("first_steps") or ()),
        baseline=str(row.get("baseline") or ""),
        risks=str(row.get("risks") or ""),
        read_first=tuple(row.get("read_first") or ()),
        papers=tuple(work.to_dict() for work in works),
    )
