"""One scientist step. Product path is `farfield research` / `run_mission`.

System map (do not collapse the layers):

    explore.py   predicates: is_live_program, is_continue_program,
                 decide_next (spray), decide_idea_refine (text-gate)
    state.py     apply_outcomes writes WORLD-only mechanism debt;
                 record_program seats the neighbourhood continue slot;
                 program_for / idea_programs is per-pair H
    program.py   compile_program builds H; continue_duty_lines locks the
                 pair against spray so execute can run the plan
    diagnose.py  two-arm registration; SWITCH_MECHANISM in this mission
    venue.py     review questions join the value rubric; cannot kill
    wiki.py      neighbourhood literature; only verified rows
    knowledge.py retrieve → verify_papers → wiki (host, not sandbox)
    this module  one pass: state → fresh harvest → agenda → generate → gates → evidence
                 (world forward-sim + world-sim rehearsal) → executable packet.

One mission:

    read Research State (known / open / contradicted / rejected / wiki /
    neighbourhood pair lock)
    → agenda: if a WORLD line already has an executable plan, jumps are
      zero — remaining work is farfield execute, not a deepen generate.
      auto-explore stops when the current idea's plan is executable
      (WORLD support, uninformative, or SYNTHETIC support). Weakened
      lines may open a new landing until the cap.
    → generate: each landing is a trajectory-informed jump (recovered
      development lines → observed move → far menu); later jumps read
      the last landing's support/novelty/value/feasibility/distance.
    → gates: graph + text. Text-gate death rewrites the same pair
      (idea_rounds). Graph-dead pairs land again. Enter stops refine.
    → evidence: survey → prior → world forward-sim → diagnosis →
      world-sim on brief+plan (structural holes rewrite the plan once)
      → two-arm probe. A second WORLD uninformative switches the lever
      in this mission. WORLD only on an attested freeze the script reads.
    → compile H for THIS pair after evidence; persist at mission end.
    → packet is an execute order, not a later research generate.

Two in-mission loops, not a handoff scientist: same-pair text refine;
same-test experiment iteration until the plan is executable. A stored
WORLD line locks spray so the intern executes. Crossover spray is not
on this path. LLM value judgment is recorded; it cannot kill, climb,
pick the lead, or rewrite expected_direction.

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
- `agenda`      slots are a cap; a stored executable WORLD plan sets
                jumps to zero (execute, do not generate)
- `explore_decision`  auto-explore: continue or stop; stop reason is
                `plan_executable` when remaining work is execute
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

from ..graph import GraphSnapshot, load_graph
from .embed import EmbeddingSpace, cosine_distance, embed_nodes
from .brief import ResearchBrief, compile_brief, write_brief
from .archive import QDArchive
from .farspace import Jump, select_near_anchors
from .researchspace import (
    JUMP_MOVES,
    evaluate_landing,
    initial_state,
    propose_jump,
    recover_trajectories,
    trajectory_prompt_lines,
)
from .domain import (
    feed_query_concepts,
    graph_supplies_mechanisms,
    preferred_object_labels,
    surface_object_labels,
    topic_object_phrases,
    without_far_terms,
)
from .generate import (
    ConceptOracle,
    GeneratedCard,
    GenerationRefused,
    LITERATURE_NODE_PREFIX,
    check_pair,
    generate_card,
    graph_endpoint,
    literature_labels,
    probe_generated,
    refine_card,
)
from .livefeed import CompositeFeed, FeedBlocked, FreshWork, as_work
from .baseline import control_stake, missing_stake
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
    idea_usefulness,
    run_tournament,
    score_live,
    selection_key,
    tournament_pool,
    value_lines,
)
from .ideaworld import analyze_idea_world
from .prior import apply_x_to_y, strongest_prior
from .probeexp import (
    HEAVY_TIMEOUT_SECONDS,
    ProbeRefused,
    floor_tier,
    probe_timeout,
    replication_seeds,
    resolve_tier,
    run_probe,
    write_probe,
)
from .parallel import map_parallel, resolve_workers
from .reframe import GRAPH_SKIP_REASON, OPERATORS as REFRAME_OPERATORS, generate_reframe, refine_reframe
from .review import critique_card, refine_brief, review_idea
from .snapshot import RecordingFeed, SNAPSHOT_DIR
from .skills import (
    attach_idea_skills,
    distill_skill,
    select_skills,
    skills_from_payloads,
    skills_prompt_block,
)
from .harness import persist_skill, write_dsh_skill
from .plugins import PluginHost, bind_host, get_host
from .state import (
    apply_outcomes,
    elites_for,
    hypothesis_id,
    line_status,
    load as load_state,
    probe_history,
    idea_programs_for,
    program_for,
    prompt_lines,
    record_elites,
    record_program,
    record_rejections,
    record_wiki,
    rejections_for,
    summary as state_summary,
    wiki_for,
    wiki_works,
)
from .knowledge import harvest_recent, harvest_survey
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
    ExplorationPrior,
    decide_compile_refine,
    decide_idea_refine,
    continue_reason,
    DEFAULT_MIN_LANDINGS,
    decide_next,
    has_research_plan,
    is_continue_program,
    is_live_program,
    landing_category,
    normalize_explore,
    stored_plan_executable,
    resolve_idea_rounds,
    resolve_polish_rounds,
)
from .iterate import (
    DEFAULT_EXPERIMENT_ROUNDS,
    classify_bottleneck,
    decide_retry,
    resolve_experiment_rounds,
    switch_mechanism_now,
)
from .packet import (
    compile_live_idea,
    idea_folder_name,
    live_packet_records,
    render_idea_report,
    render_mission_packet,
    render_protocol,
)
from .program import (
    compile_program,
    continue_duty_lines,
    prompt_lines as program_prompt_lines,
)
from .wiki import (
    gap_lines as wiki_gap_lines,
    merge_works,
    prompt_lines as wiki_prompt_lines,
)
from .writeup import compile_tex, render_note
from .workspace import (
    EVENTS_FILE,
    append_progress,
    append_recorded_event,
    candidate_dir,
    enqueue_world_expansion,
    llm_cache_dir,
    seal_candidate,
    var_dir,
    write_archive,
    write_idea,
    write_idea_report,
    write_mission_manifest,
    write_question_board,
    write_summary,
    prune_idea_packets,
)
from .ideakind import (
    ACQUIRE,
    QUESTION,
    THEORY,
    idea_kind_of,
    render_archive,
    render_question_board,
    runs_probe,
    skip_writeup_on_needs_world,
)
from .lifecycle import (
    ACQUIRE_BOUND,
    HARVEST_SUCCESS,
    LOCAL_ORIGIN_LABEL,
    ORIGIN_LOCAL,
    RESEARCH_STATE_FILE,
    bind_acquired_world,
    child_probe_freeze,
    diagnose_goal,
    harvest_acquired_world,
    load_notebook,
    load_world_lineage,
    opportunity_from_menu,
    persist_acquire,
    propose_acquire,
    save_notebook,
    typed_evidence_role,
    validate_acquire_recipe,
    version_from_fixture,
    append_world_version,
)
from .claimspec import (
    ContractViolation,
    MissionManifest,
    cards_eligible_for_evidence,
    claim_spec_from_payload,
    compile_disposition_of,
    in_world_testable,
    project_evidence_to_claim,
    scientific_for,
    should_queue_world_expansion,
    world_expansion_record,
)
from .world import (
    KIND_SYNTHETIC,
    KIND_WORLD,
    attested_freeze,
    bind_world,
    in_mission_constructable,
    incompatible_family,
    infer_requirement,
    lineage_conflicts,
    load_catalog,
    load_fixture,
    pick_acquired_world,
    reads_world_data,
    record_world_wishlist,
    resolve_world,
    schema_family,
    wishlist_requirements,
)
from .schemas import OBJECT_SCHEMA, SCHEMAS
from .genworld import (
    HONESTY,
    KIND_GENERATED,
    construct_world,
    execute_world,
    load_world_payload,
    topic_lock_card,
)
from . import chain
from .dynworld import (
    RESPONSE_FLOOR,
    arm_separation,
    compose_simulation,
    forward_simulate,
    has_dynamics,
    levers_of,
    materialize_placebo,
    response_card,
    scout_summary,
    world_consumed,
)
from .dvsanity import (
    blind_reason,
    dv_blind,
    materialize_oracle_placebo,
    oracle_dependent,
    oracle_field,
)
from .farcompile import menu_from_card
from .worldpath import write_world_path
from .evidence import (
    TIER_SAME_INSTANCE,
    TIER_SYNTHETIC,
    ablation_required,
    campaign_funnel,
    competing_explanation_missing,
    evidence_id,
    hypothesis_revision_id,
    sha256_text,
)

PRODUCTION_CORPUS = "ds-arxiv-concepts-2026"
# One harvest, one T, several arXiv sets. Not a union of existing graphs.
# load_assets / the CLI switch here only when G_full.json is on disk.
MIXED_CORPUS = "cs-mixed-arxiv-concepts-2026"
# Clone-sized graph that ships in git. Production G_full.json exceeds GitHub's
# 100 MB file limit; load_assets falls back here when that file is absent.
SHIPPED_CORPUS = "attn-concepts-s1"
# Continue of a stored trajectory pair keeps the same two labels.
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


def resolve_production_corpus(
    root: Path | None = None, requested: str = ""
) -> str:
    """Default corpus: mixed harvest when it exists, else the DS snapshot.

    A missing mixed graph is not a silent union of older slices. The
    researcher may still pass `--corpus` to pick any registered id.
    """
    if str(requested or "").strip():
        return str(requested).strip()
    base = root or _root()
    mixed = base / "concepts" / MIXED_CORPUS / "G_full.json"
    if mixed.is_file():
        return MIXED_CORPUS
    return PRODUCTION_CORPUS


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


def _redesign_prior(
    diagnosis: Any,
    *,
    verdict: str,
    dynamics_card: dict[str, Any] | None = None,
    must_switch_mechanism: bool = False,
    design_notes: list[str] | None = None,
    probe_kind: str = "",
) -> dict[str, Any]:
    """Quote the failed design for the next in-mission diagnosis.

    Treatment/control numbers stay out. An unnamed competing explanation
    is a method gap on this pair, not a new far jump. A second WORLD
    uninformative in this mission sets `must_switch_mechanism` so the
    next diagnosis switches the lever here, not in a later research run.
    """
    alternative = str(getattr(diagnosis, "alternative", "") or "")
    world_lever = str(getattr(diagnosis, "world_lever", "") or "")
    prior: dict[str, Any] = {
        "experiment": str(getattr(diagnosis, "experiment", "") or ""),
        "verdict": verdict,
        "alternative": alternative,
        "world_lever": world_lever,
    }
    if probe_kind:
        prior["probe_kind"] = probe_kind
    if must_switch_mechanism:
        prior["must_switch_mechanism"] = True
    notes = [str(item).strip() for item in (design_notes or []) if str(item or "").strip()]
    if notes:
        prior["design_notes"] = notes[:6]
    levers: tuple[str, ...] = ()
    if dynamics_card is not None:
        levers = tuple(
            str(name)
            for name in (dynamics_card.get("levers") or {})
            if str(name or "").strip() and str(name) != "none"
        )
        prior["world_scout"] = scout_summary(
            dynamics_card, getattr(diagnosis, "world_lever", "")
        )
    if competing_explanation_missing(
        alternative, levers=levers, world_lever=world_lever
    ):
        prior["competing_explanation_missing"] = True
        if levers:
            prior["world_levers"] = list(levers)
    return prior


def rehearse_entered_idea(
    *,
    workspace: Path,
    topic: str,
    card: GeneratedCard,
    diagnosis: Any,
    enabled: bool,
    client: Any,
    wiki: list,
    world_lever: str = "",
    schema: str = "",
    scout: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Write brief+plan, run world-sim, rewrite the plan once on structural holes.

    Imagined numbers cannot climb. Failure cannot block execute.
    """
    from .livefeed import FreshWork
    from .worldsim import WORLD_SIM_RESIM_CAP, maybe_world_sim_after_diagnosis

    rec: dict[str, Any] = {
        "card_id": card.card_id,
        "claim": card.claim,
        "mechanism": card.mechanism,
        "prediction": card.prediction,
        "pair": list(card.pair),
        "experiment": str(getattr(diagnosis, "experiment", "") or ""),
        "treatment_arm": str(getattr(diagnosis, "treatment_arm", "") or ""),
        "control_arm": str(getattr(diagnosis, "control_arm", "") or ""),
        "expected_direction": str(getattr(diagnosis, "expected_direction", "") or ""),
        "alternative": str(getattr(diagnosis, "alternative", "") or ""),
        "world_lever": world_lever or str(getattr(diagnosis, "world_lever", "") or ""),
        "world_schema": schema,
        "world_id": str((extra or {}).get("world_id") or ""),
        "probe_kind": str((extra or {}).get("probe_kind") or ""),
        "works": list(wiki or []),
    }
    rec.update({k: v for k, v in (extra or {}).items() if v is not None})
    rec["idea_name"] = idea_folder_name(rec)
    folder = compile_live_idea(workspace, topic, rec)
    rows: list[FreshWork] = []
    for work in wiki or []:
        if isinstance(work, FreshWork):
            rows.append(work)
        elif isinstance(work, dict):
            rows.append(FreshWork.from_dict(work))
    payload = maybe_world_sim_after_diagnosis(
        folder,
        enabled=enabled,
        client=client,
        wiki=rows,
        claim=str(rec.get("claim") or ""),
        mechanism=str(rec.get("mechanism") or ""),
        world_lever=str(rec.get("world_lever") or ""),
        schema=schema,
        scout=scout,
        review_mode="simple",
        smart_skip=True,
    )
    if not payload:
        return
    payload = {**payload, "card_id": card.card_id, "stage": "world-sim"}
    structural = [
        str(item.get("summary") or "").strip()
        for item in (payload.get("issues") or [])
        if isinstance(item, dict)
        and item.get("type") == "structural"
        and str(item.get("summary") or "").strip()
    ]
    if structural:
        rec["world_sim_issues"] = list(payload.get("issues") or [])
        rec["world_sim_h"] = dict(payload.get("h") or {})
        rec["design_notes"] = structural[:6]
        compile_live_idea(workspace, topic, rec)
        if enabled and WORLD_SIM_RESIM_CAP >= 1:
            again = maybe_world_sim_after_diagnosis(
                folder,
                enabled=True,
                client=client,
                wiki=rows,
                claim=str(rec.get("claim") or ""),
                mechanism=str(rec.get("mechanism") or ""),
                world_lever=str(rec.get("world_lever") or ""),
                schema=schema,
                scout=scout,
                review_mode="simple",
                smart_skip=False,
            )
            if again:
                payload = {**again, "card_id": card.card_id, "stage": "world-sim", "resim": True}
                payload["design_notes"] = structural[:6]
    yield payload


def _world_work_dir(workspace: Path | None, card_id: str) -> Path:
    if workspace is not None:
        dest = candidate_dir(workspace, card_id) / "world"
        dest.mkdir(parents=True, exist_ok=True)
        return dest
    return Path(tempfile.mkdtemp(prefix="ff-world-"))


def _mission_world_dir(workspace: Path | None) -> Path | None:
    """One bound or constructed world for the whole mission, not per card."""
    if workspace is None:
        return None
    dest = Path(workspace) / "world"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _want_task_world(world_request: str) -> bool:
    """auto/none get a per-task world; empty string stays SYNTHETIC invention."""
    return str(world_request or "").strip().lower() in {"auto", "none"}


def _ensure_task_world(
    topic: str,
    bound_world: Any,
    world_request: str,
    workspace: Path | None,
) -> Any:
    """Construct a same-family GENERATED stub from the topic object.

    Auto never binds a catalog cousin (Pride, phage, A2A). An explicit
    ``--world <id>`` already sits in ``bound_world``. Later cards reuse
    this dest. Retrieved papers ground the stub in the evidence pipeline.
    """
    if bound_world is not None:
        return bound_world
    if not _want_task_world(world_request):
        return None
    req = infer_requirement(topic)
    if not in_mission_constructable(req):
        return None
    dest = _mission_world_dir(workspace)
    if dest is None:
        dest = Path(tempfile.mkdtemp(prefix="ff-task-world-"))
    fixture = construct_world(
        None,
        topic_lock_card(topic),
        req or {},
        dest=dest,
        topic=topic,
    )
    try:
        payload = load_world_payload(fixture.root)
    except Exception:
        return None
    if payload.get("acquisition") == "incomplete":
        return None
    return fixture


def _write_world_readme(dest: Path, fixture: Any, *, honesty: str = "") -> None:
    payload = fixture.to_dict() if hasattr(fixture, "to_dict") else {}
    role = str(getattr(fixture, "role", "") or payload.get("role") or "")
    kind = (
        "GENERATED"
        if role == "generated"
        else ("WORLD" if role not in {"", "test", "fixture", "synthetic"} else role)
    )
    lines = [
        f"# Mission world: `{payload.get('id') or getattr(fixture, 'id', '')}`",
        "",
        f"- kind: `{kind}`",
        f"- schema: `{payload.get('schema') or getattr(fixture, 'schema', '')}`",
        f"- role: `{role}`",
        "",
    ]
    if kind == "GENERATED":
        lines.extend(
            [
                "This world was constructed because no attested freeze matched "
                "the topic object. Probes on it can weaken a claim. They cannot "
                "corroborate. A later `farfield freeze` of the same schema is "
                "required before WORLD.",
                "",
            ]
        )
    else:
        provenance = str(
            getattr(fixture, "provenance", "") or payload.get("provenance") or ""
        )
        if provenance:
            lines.insert(4, f"- provenance: `{provenance}`")
            lines.extend(
                [
                    {
                        "derived": (
                            "This world was **derived** from another catalog freeze by a "
                            "registered rule (see `origin.json → derived_from`). The bytes "
                            "are a re-encoding of attested data, not a new observation."
                        ),
                        "harvested": (
                            "This world was **harvested**: the host ran a harness under a "
                            "pre-registered recipe and froze the exported history (see "
                            "`origin.json → harness_run`). It is a real run, not GENERATED; "
                            "a second seed of the same recipe is the confirmation."
                        ),
                        "wishlist": (
                            "This world was acquired from a freeze recipe the last mission "
                            "left in `var/world_wishlist.json`."
                        ),
                    }.get(provenance, f"provenance: {provenance}"),
                    "",
                ]
            )
        lines.extend(
            [
                "This freeze is the **fixture** (attested ROM) for probes that "
                "must read `data/`. It is not the claim's title object and not "
                "a paper to follow.",
                "",
                "Three layers, do not collapse them:",
                "",
                "- **Fixture** — these hashed bytes. Probe WORLD requires reading `data/`.",
                "- **Dynamics** — runtime levers (`dynworld`) on evolving state.",
                "- **World rehearsal** — `ideas/<name>/world-sim/` three-tier idea "
                "simulation (`farfield world-sim`). Imagined numbers cannot corroborate.",
                "",
                "A far concept does not get to rebind Pride, a graph, or fasta "
                "in place of this fixture.",
                "",
            ]
        )
    if honesty:
        lines.extend([honesty.strip(), ""])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "WORLD.md").write_text("\n".join(lines), encoding="utf-8")


def _requirement_family(requirement: dict[str, Any] | None) -> str:
    req = requirement or {}
    schema = schema_family(str(req.get("schema") or ""))
    if schema:
        return schema
    return schema_family(OBJECT_SCHEMA.get(str(req.get("object_type") or ""), ""))


def _hold_task_schema(
    task_world: Any,
    requirement: dict[str, Any] | None,
    world_path: Any = None,
) -> dict[str, Any] | None:
    """Keep this mission's world in the topic's schema family.

    The GENERATED task world was constructed from the topic before any
    card existed. A card's requirement that names another family — via
    its prose, or via a lineage's `freeze_schema` chosen from the
    freezable list — must not rebuild `workspace/world` as that family:
    the probe would then read a protocol automaton while the claim is
    about executable program state, and 0/0 gets booked as
    `object_absent`. Returns None when nothing has to be held.
    """
    if task_world is None or str(getattr(task_world, "role", "") or "") != "generated":
        return None
    task_schema = schema_family(str(getattr(task_world, "schema", "") or ""))
    if task_schema not in SCHEMAS:
        return None
    req = dict(requirement or {})
    requested = _requirement_family(req)
    source = "requirement"
    if not requested or requested == task_schema:
        wished = ""
        if world_path is not None:
            wished = schema_family(
                str(getattr(world_path, "freeze_schema", "") or "")
            ) or schema_family(str(getattr(world_path, "schema", "") or ""))
        if not wished or wished == task_schema:
            return None
        requested = wished
        source = "world_path"
    spec = SCHEMAS[task_schema]
    req["schema"] = spec.name
    req["object_type"] = spec.object_type
    return {
        "requirement": req,
        "schema": spec.name,
        "requested": requested,
        "source": source,
    }


def _active_probe_world(world: Any, sim_world: Any) -> Any:
    """Prefer the in-mission constructed instance; attested freeze if that is it.

    A catalog miss used to skip the probe (`probe_executed=0`) even after
    a GENERATED world had been scouted. That empty funnel was an operator
    shortcut, not a scientific verdict. Construction still cannot climb.
    """
    generated = None
    if sim_world is not None and str(getattr(sim_world, "role", "") or "") == "generated":
        generated = sim_world
    elif world is not None and str(getattr(world, "role", "") or "") == "generated":
        generated = world
    if generated is not None:
        try:
            payload = load_world_payload(generated.root)
        except Exception:
            return generated
        if payload.get("acquisition") == "incomplete":
            return None
        return generated
    if world is not None:
        return world
    return None


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
    topic: str = "",
    idea_world: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Build or adapt the experimental world for this registration.

    One dest. Not a sidecar diagnostic fixture. `topic` locks the object
    so a far-field pair cannot rename the world.
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
        topic=topic,
        idea_world=idea_world,
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
            topic=topic,
            idea_world=idea_world,
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
    *,
    lock_world: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    """Per-claim world. Auto constructs; it does not match the catalog.

    A continue of a WORLD line, or an explicit ``--world <id>``, keeps
    that freeze. ``io`` and other unmatched families are still
    WORLD_INCOMPATIBLE for corroboration — construction must not invent
    a substitute family.
    """
    del catalog
    if lock_world:
        return bound_world, None
    text = str(world_request or "").strip().lower()
    if text in {"", "none", "synthetic", "off", "0", "auto"}:
        far_label = card.pair[1] if len(card.pair) > 1 else ""
        req = infer_requirement(
            topic,
            without_far_terms(card.claim, far_label, topic),
            without_far_terms(card.prediction, far_label, topic),
        )
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
        return bound_world, None
    return bound_world, None


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
    outcome["world_provenance"] = record.get("world_provenance") or ""
    outcome["world_id"] = record.get("world_id") or ""
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
    outcome["pair_nodes"] = list(record.get("pair_nodes") or [])
    outcome["design_notes"] = list(record.get("design_notes") or [])
    outcome["world_id"] = str(record.get("world_id") or "")
    if isinstance(record.get("distilled_skill"), dict):
        outcome["distilled_skill"] = dict(record["distilled_skill"])


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
                "plain_title": row.get("plain_title") or "",
                "one_liner": row.get("one_liner") or "",
                "why_it_matters": row.get("why_it_matters") or "",
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
    object_labels: tuple[str, ...] = (),
) -> tuple[
    list[str],
    dict[str, tuple[str, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Graph-prescreen far concepts before paying the generator.

    When the topic still has object labels, the near side of every
    offered pairing is locked to them here — the same lock that
    `generate_card` enforces at parse time — so the generator is never
    handed a menu it is forbidden to satisfy.     A far concept whose only
    surviving partners lie outside the object labels is prescreened out
    as `object_mismatch`: an honest graph fact, and in auto explore an
    empty menu is a graph-dead landing that falls through to the next
    jump instead of burning generation calls on guaranteed refusals.

    A far label with no graph node (verified-paper mechanism) is not
    prescreened by the oracle. The near side is still locked to the
    topic object; novelty for that pair is prior-kill.
    """
    obj = set(object_labels)
    viable: list[str] = []
    partners: dict[str, tuple[str, ...]] = {}
    prescreened: list[dict[str, Any]] = []
    kills: list[dict[str, Any]] = []
    for label, node in far_candidates:
        if not graph_endpoint(assets.oracle, node):
            allowed = [item for item in (object_labels or near) if item]
            if allowed:
                viable.append(label)
                partners[label] = tuple(allowed)
            else:
                prescreened.append(
                    {"label": label, "killed_by": ["object_mismatch"]}
                )
            continue
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
        allowed = [item for item in ok_near if not obj or item in obj]
        if allowed:
            viable.append(label)
            partners[label] = tuple(allowed)
        elif ok_near:
            # Structurally alive, but only off the topic's object: offering
            # it would contradict the object lock the generator enforces.
            prescreened.append({"label": label, "killed_by": ["object_mismatch"]})
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
    lock_world: bool = False,
    hold_world: bool = False,
    dynamics: dict[str, Any] | None = None,
    world_sim: bool = True,
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
        try:
            probe = feed.pair_recently_combined(*card.pair)
        except Exception as exc:
            probe = FeedBlocked(
                attempted="pair_recently_combined", reason=str(exc)
            )
        event: dict[str, Any] = {
            "stage": "fresh_check",
            "card_id": card.card_id,
            "pair": list(card.pair),
        }
        if isinstance(probe, FeedBlocked):
            event["blocked"] = probe.to_dict()
        elif isinstance(probe, dict):
            event.update(probe)
        elif isinstance(probe, list):
            event["combined"] = bool(probe)
            evidence = []
            for item in probe:
                work = as_work(item)
                if work is not None:
                    evidence.append(work.to_dict())
                elif isinstance(item, dict):
                    evidence.append(item)
            event["evidence"] = evidence
        else:
            event["blocked"] = {
                "attempted": "pair_recently_combined",
                "reason": f"unexpected feed shape: {type(probe).__name__}",
            }
        events.append(event)
    card_world, world_event = _resolve_card_world(
        card, topic, catalog or {}, bound_world, world_request, lock_world=lock_world
    )
    world_requirement = None
    world_dest = None
    pipeline_world = card_world
    world_gap = world_event is not None
    if world_event is not None:
        events.append(world_event)
        bundle["skip_writeup"] = False
        world_requirement = world_event.get("requirement") or {}
        world_dest = _mission_world_dir(workspace) or _world_work_dir(
            workspace, card.card_id
        )
        pipeline_world = None
    elif pipeline_world is None:
        pipeline_world = world
    from .openworld import after_card

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
        lock_world=hold_world,
        dynamics=dynamics,
        world_sim=world_sim,
    ):
        events.append(event)
        if event["stage"] == "blocked":
            break
    events.extend(
        after_card(
            workspace,
            bundle.get("card") or card,
            bundle,
            world=bundle.get("world") or pipeline_world,
        )
    )
    return {
        "events": events,
        "bundle": bundle,
        "world": bundle.get("world") or pipeline_world,
        "card": bundle.get("card") or card,
    }


def run_research(topic: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Production research path. OpenWorld owns the next scientific action."""
    from .openworld.runtime import run_research as _run

    yield from _run(topic, **kwargs)


def run_mission(
    topic: str,
    *,
    jumps: int = DEFAULT_JUMPS,
    candidates: int = DEFAULT_CANDIDATES,
    polish_rounds: int = DEFAULT_POLISH,
    explore: str = DEFAULT_EXPLORE,
    experiment_rounds: int = DEFAULT_EXPERIMENT,
    idea_rounds: int = DEFAULT_IDEA,
    venue: str = "",
    world: str = "",
    world_root: Path | None = None,
    reframes: int = 0,
    parallel: int | None = None,
    run_probes: bool = True,
    host_execute: bool = False,
    host_timeout: float | None = None,
    world_sim: bool = True,
    paper_rounds: int = -1,
    min_ideas: int = DEFAULT_MIN_LANDINGS,
    human_gate: str = "none",
    approved: tuple[str, ...] = (),
    fresh: bool = False,
    client: Any = None,
    feed: Any = None,
    backend: Backend | None = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    api_calls: float | None = None,
    corpus_id: str = "",
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
    corpus_id = resolve_production_corpus(root, requested=corpus_id)
    host = PluginHost.load(root, workspace)
    log = None
    if workspace is not None:
        dest = Path(workspace)
        dest.mkdir(parents=True, exist_ok=True)
        log = dest / EVENTS_FILE
        log.write_text("", encoding="utf-8")
        from datetime import datetime, timezone

        write_mission_manifest(
            dest,
            MissionManifest(
                mission_id=dest.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                topic=topic,
            ).to_dict(),
        )
    with bind_host(host):
        for event in _run_mission_body(
            topic,
            jumps=jumps,
            candidates=candidates,
            polish_rounds=polish_rounds,
            explore=explore,
            experiment_rounds=experiment_rounds,
            idea_rounds=idea_rounds,
            venue=venue,
            world=world,
            world_root=world_root,
            reframes=reframes,
            parallel=parallel,
            run_probes=run_probes,
            host_execute=host_execute,
            host_timeout=host_timeout,
            world_sim=world_sim,
            paper_rounds=paper_rounds,
            min_ideas=min_ideas,
            human_gate=human_gate,
            approved=tuple(approved or ()),
            fresh=fresh,
            client=client,
            feed=feed,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            api_calls=api_calls,
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
    venue: str = "",
    world: str = "",
    world_root: Path | None = None,
    reframes: int = 0,
    parallel: int | None = None,
    run_probes: bool = True,
    host_execute: bool = False,
    host_timeout: float | None = None,
    world_sim: bool = True,
    paper_rounds: int = -1,
    min_ideas: int = DEFAULT_MIN_LANDINGS,
    human_gate: str = "none",
    approved: tuple[str, ...] = (),
    fresh: bool = False,
    client: Any = None,
    feed: Any = None,
    backend: Backend | None = None,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    api_calls: float | None = None,
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
    try:
        from .venue import normalize_venue, venue_value_lines

        venue_id = normalize_venue(venue)
        venue_lines = venue_value_lines(venue_id)
    except ValueError as exc:
        yield {
            "stage": "blocked",
            "missing_capability": "venue_profile",
            "unlock_condition": str(exc),
        }
        return
    rubric = value_lines() + venue_lines
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
    acquired_event: dict[str, Any] | None = None
    if bound_world is None and str(world_request or "").strip().lower() == "auto":
        # An acquired freeze (derived / harvested / wishlist) of this
        # object family is the WORLD the last mission asked for. Shipped
        # fixtures are still never auto-bound.
        acquired_world = pick_acquired_world(topic, world_catalog)
        if acquired_world is not None:
            bound_world = acquired_world
            acquired_event = {
                "stage": "world_acquired_bound",
                "world_id": acquired_world.id,
                "schema": acquired_world.schema,
                "provenance": acquired_world.provenance,
                "digest": acquired_world.digest,
            }
    bound_world = _ensure_task_world(
        topic, bound_world, world_request, workspace
    )
    slots = jumps + max(0, reframes)
    if client is None:
        try:
            chosen = backend or default_backend(
                model=model, base_url=base_url, api_key=api_key
            )
            declared_calls = (
                float(api_calls)
                if api_calls is not None
                else float(
                    jumps * candidates
                    + max(0, reframes)
                    + max(0, idea_budget) * slots
                    + jumps * candidates * 3  # compile retries on remaining far
                    + slots * (3 + 2 * polish_budget)
                    + 8
                    + 2
                    + 1  # one shadow-judge proposal at mission end
                )
            )
            client = default_client(
                declared_calls,
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
        "venue": venue_id,
        "world": bound_world.to_dict() if bound_world is not None else None,
        "explore": explore_mode,
        "jump_cap": jumps,
        "parallel": workers,
    }
    if acquired_event is not None:
        yield acquired_event
    world_fact_lines: list[str] = []
    # Look at the attested world before any card exists: the schema's
    # builtin descriptives, recorded as exploratory on the catalog chain.
    # The mission can then notice what its own data says (self-acceptance
    # vs oracle, validator writes vs failure) instead of leaving that to a
    # human with a scratch script. Motivation only; nothing here climbs.
    if (
        bound_world is not None
        and str(getattr(bound_world, "role", "") or "") == "world"
        and str(getattr(bound_world, "provenance", "") or "")
    ):
        try:
            from .exploratory import (
                AnalysisError,
                builtin_descriptives,
                fact_lines,
                load_records,
                run_analysis,
            )

            catalog_dir = Path(catalog_root) / "worlds"
            builtin = builtin_descriptives(str(getattr(bound_world, "schema", "") or ""))
            if builtin is not None and catalog_dir.is_dir():
                label = f"descriptives:{bound_world.schema}"
                already = [
                    row
                    for row in load_records(catalog_dir, bound_world.id)
                    if row.get("label") == label and row.get("world_digest") == bound_world.digest
                ]
                record = already[-1] if already else run_analysis(
                    bound_world, builtin, catalog_root=catalog_dir, label=label
                )
                yield {
                    "stage": "exploratory",
                    "world_id": bound_world.id,
                    "label": record.get("label"),
                    "script_digest": record.get("script_digest"),
                    "replayed": bool(already),
                    "results": record.get("results"),
                    "cannot_corroborate": True,
                }
                world_fact_lines = fact_lines(record)
        except AnalysisError as exc:
            yield {"stage": "exploratory_skipped", "world_id": bound_world.id, "reason": str(exc)[:300]}
        except Exception as exc:  # descriptives must never lose the mission
            yield {"stage": "exploratory_skipped", "world_id": bound_world.id, "reason": f"{type(exc).__name__}: {exc}"[:300]}
    mission_world = _mission_world_dir(workspace)
    if bound_world is not None and mission_world is not None:
        bind_world(mission_world, bound_world)
        _write_world_readme(Path(workspace), bound_world)

    mission_menu = None
    mission_dynamics: dict[str, Any] | None = None
    spray_closed_no_room = False
    notebook = load_notebook(workspace)
    mission_opportunity = opportunity_from_menu(None, notebook=notebook)
    if bound_world is not None and has_dynamics(bound_world):
        try:
            mission_dynamics = response_card(bound_world)
        except Exception:
            mission_dynamics = None
        if mission_dynamics is not None:
            mission_menu = menu_from_card(mission_dynamics, bound_world)
            mission_opportunity = opportunity_from_menu(
                mission_menu, notebook=notebook
            )
            yield {
                "stage": "world_scout",
                "card_id": "",
                "world_id": mission_dynamics["world_id"],
                "origin": compose_simulation(bound_world).get("origin") or "",
                "levers": sorted(mission_dynamics["levers"]),
                "observables": list(mission_dynamics["observables"]),
                "responses": dict(mission_dynamics["levers"]),
                "inert": list(mission_dynamics["inert"]),
                "room_to_move": mission_menu.room_to_move,
                "probe_room": mission_opportunity.probe_room,
                "opportunity": mission_opportunity.to_dict(),
                "report_digest": mission_dynamics["report_digest"],
            }
            if not mission_opportunity.probe_room:
                yield {
                    "stage": "probe_room_closed",
                    "reason": "no_probe_room",
                    "world_id": str(getattr(bound_world, "id", "") or ""),
                    "eligible_kinds": sorted(
                        mission_opportunity.eligible_kinds(local_problem=True)
                    ),
                    "detail": (
                        "scout found no manipulable lever that moves a "
                        "declared observable; probe is ineligible, but "
                        "question, acquire, and theory remain open"
                    ),
                }
    if bound_world is not None and workspace is not None:
        append_world_version(
            workspace,
            version_from_fixture(
                bound_world, path=str(_mission_world_dir(workspace) or "")
            ),
        )
        notebook.world_versions = load_world_lineage(workspace)
        save_notebook(workspace, notebook)

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

    anchored = select_near_anchors(
        assets.space, topic, assets.label_of, k=NEAR_LABELS
    )
    if not anchored:
        topic_vector = assets.space.embed(topic)
        anchored = tuple(
            sorted(
                (cosine_distance(topic_vector, vector), nid)
                for nid, vector in assets.space.vectors.items()
            )[:NEAR_LABELS]
        )
    near_ids = tuple(nid for _, nid in anchored[:NEAR_LABELS])
    graph_near = tuple(assets.label_of[n] for n in near_ids)
    preferred = preferred_object_labels(graph_near, topic)
    object_labels = surface_object_labels(graph_near, topic)
    graph_mechanisms = graph_supplies_mechanisms(graph_near, topic)
    gen_label_to_node = dict(assets.label_to_node)
    # Cosine DS neighbours stay a scout readout (`anchors`) and an oracle
    # proxy (`near_ids`). They are not the generation near menu: mixing
    # them in was how a reasoning topic inherited `finite metric`.
    if preferred:
        near = preferred
        selection = "topic_object"
        aligned = tuple(
            gen_label_to_node[label]
            for label in preferred
            if label in gen_label_to_node
        )
        if aligned:
            near_ids = aligned
    elif object_labels:
        near = object_labels
        selection = "topic_phrases"
        if near_ids:
            proxy = near_ids[0]
            near_ids = (proxy,) * len(object_labels) + tuple(near_ids)
    else:
        near = tuple(
            gram for gram in topic_object_phrases(topic) if gram
        )[:NEAR_LABELS] or graph_near
        selection = "topic_phrases" if near != graph_near else "embedding_fallback"
        if near_ids and selection == "topic_phrases":
            proxy = near_ids[0]
            near_ids = (proxy,) * max(1, len(near)) + tuple(near_ids)
    if near_ids:
        proxy = near_ids[0]
        for phrase in object_labels:
            gen_label_to_node.setdefault(phrase, proxy)
    feed_concepts = feed_query_concepts(topic, object_labels, near)
    yield {
        "stage": "anchor",
        "memory_anchor": near[0],
        "anchors": [
            {"concept": assets.label_of[nid], "distance": round(dist, 4)}
            for dist, nid in anchored[:NEAR_LABELS]
        ],
        "object_labels": list(object_labels),
        "selection": selection,
        "graph_mechanisms": graph_mechanisms,
        "feed_query": list(feed_concepts),
    }
    research_state = initial_state(topic)
    yield {"stage": "research_state", **research_state.to_dict()}

    # One long-term memory: the Research State carries what is known, what
    # is open, what contradicted, and which directions already died.
    topics_now = load_state(state_path)
    research = state_summary(topics_now, corpus_id, near[0])
    lessons = rejections_for(topics_now, corpus_id, near[0])
    stored_elites = elites_for(topics_now, corpus_id, near[0])
    stored_program = program_for(topics_now, corpus_id, near[0])
    stored_ideas = idea_programs_for(topics_now, corpus_id, near[0])
    stored_wiki = wiki_for(topics_now, corpus_id, near[0])
    program_lines = program_prompt_lines(stored_program)
    prior_live = is_live_program(stored_program)
    prior_continue = is_continue_program(stored_program)
    prior_plan = prior_live or prior_continue or stored_plan_executable(
        stored_program, *stored_ideas
    )
    if fresh:
        # A fresh mission on a topic that already has a plan: explore again
        # instead of dropping to execute-only. Lineage locks on finished
        # pairs still hold (they live in the state, not in these flags).
        prior_live = False
        prior_plan = False
    idea_skills = skills_from_payloads((stored_program or {}).get("skills") or ())
    if workspace is not None and idea_skills:
        for skill in idea_skills:
            write_dsh_skill(Path(workspace), skill)
    wiki_lines = wiki_prompt_lines(stored_wiki, topic=topic)
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
        harvest = harvest_recent(feed, feed_concepts)
        event: dict[str, Any] = {"stage": "fresh", **harvest.to_dict()}
        if harvest.blocked is not None and not harvest.works:
            yield event
        else:
            if harvest.works:
                added = record_wiki(
                    state_path,
                    corpus_id,
                    near[0],
                    harvest.works,
                    seen_at=assets.fresh_until,
                )
                event["wiki_added"] = added
                stored_wiki = wiki_for(load_state(state_path), corpus_id, near[0])
                wiki_lines = wiki_prompt_lines(stored_wiki, topic=topic)
                wiki_pool[:] = wiki_works(
                    load_state(state_path), corpus_id, near[0]
                )
            context = tuple(work.line() for work in harvest.works)
            yield event

    recovered = recover_trajectories(wiki_pool, topic, research_state)
    yield {
        "stage": "trajectories",
        "lines": [row.to_dict() for row in recovered],
        "patterns": list(
            dict.fromkeys(move for row in recovered for move in row.patterns)
        ),
    }

    # The agenda: the state's open questions become this mission's targets.
    # An executable plan already in the neighbourhood is executed, not
    # generated as a deepen card — including SYNTHETIC plans whose
    # remaining work is freeze + farfield execute.
    targets = [f"conflict to resolve: {line}" for line in research["contradictions"]]
    targets += [f"open question: {line}" for line in research["unknowns"]]
    targets += [f"literature gap: {line}" for line in wiki_gap_lines(stored_wiki, topic=topic)]
    farfield_cycle = preferred_operators(policy, JUMP_MOVES)
    reframe_cycle = preferred_operators(policy, tuple(sorted(REFRAME_OPERATORS)))
    explore_jumps = 0 if prior_plan else jumps
    if prior_plan:
        reframes = 0
    slot_plan: list[dict[str, Any]] = [
        {
            "track": "farfield",
            "operator": farfield_cycle[i % len(farfield_cycle)],
            "target": None,
            "role": "explore",
        }
        for i in range(explore_jumps)
    ] + [
        {"track": "reframe", "operator": reframe_cycle[i % len(reframe_cycle)],
         "target": None}
        for i in range(max(0, reframes))
    ]
    receivers = [s for s in slot_plan if s["track"] == "reframe"]
    receivers += [s for s in slot_plan if s["track"] == "farfield"]
    for slot_entry, target in zip(receivers, targets):
        slot_entry["target"] = target
    yield {
        "stage": "agenda",
        "targets": targets,
        "slots": list(slot_plan),
        "deepen": False,
        "continue": prior_plan,
        "continue_reason": (
            continue_reason(stored_program) or ("plan_executable" if prior_plan else "")
        ),
        "explore": explore_mode,
        "explore_jumps": explore_jumps,
        "reframe_cap": max(0, reframes),
        "plan_executable": prior_plan,
    }
    if prior_plan:
        yield {
            "stage": "explore_decision",
            "track": "farfield",
            "opened": 0,
            "cap": jumps,
            **decide_next(
                opened=0,
                cap=max(jumps, 1),
                records=[],
                prior_live=prior_live,
                prior_plan=True,
            ),
        }

    def slot_knowledge(entry: dict[str, Any]) -> tuple[str, ...]:
        if entry.get("target"):
            return knowledge + (
                f"This mission targets one gap on the books — {entry['target']}."
                " Prefer a claim that would move it.",
            )
        return knowledge

    def _card_view(card: GeneratedCard, *, plausibility: float) -> dict[str, Any]:
        return {
            "claim": card.claim,
            "mechanism": card.mechanism,
            "prediction": card.prediction,
            "falsifier": card.falsifier,
            "concept_a": card.pair[0],
            "concept_b": card.pair[1],
            "alienness": card.alienness,
            "plausibility_signal": plausibility,
        }

    def _run_text_gates(
        card: GeneratedCard, view: dict[str, Any]
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        text_kills: list[str] = []
        judge_errors: list[dict[str, Any]] = []
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
        return text_kills, judge_errors, shadow

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
            "works": [],
            "dead_end": card.dead_end,
            "why_failed": card.why_failed,
            "reframe": card.reframe,
            "objection": card.objection,
            "derivation": getattr(card, "derivation", "") or "",
            "claim_spec": getattr(card, "claim_spec", None),
            "idea_id": card.card_id,
            "lineage_id": str(getattr(card, "lineage_id", "") or ""),
            "origin": str(getattr(card, "origin", "") or ""),
            "research_target": str(getattr(card, "research_target", "") or ""),
            "world_version": str(getattr(card, "world_version", "") or ""),
            "kind_fields": dict(getattr(card, "kind_fields", None) or {}),
            "has_brief": False,
            "title": "",
            "plain_title": "",
            "one_liner": "",
            "why_it_matters": "",
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
            "world_provenance": getattr(bound_world, "provenance", "") or "",
            "host_ok": None,
            "world_incompatible": False,
            "generated_world": str(getattr(bound_world, "role", "") or "")
            == "generated",
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
            "world_lever": str(card.world_lever or ""),
            "world_observable": str(getattr(card, "world_observable", "") or ""),
            "idea_kind": idea_kind_of(card),
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
        if stage in ("margin_refused", "world_scout"):
            explore_prior.learn(event)
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
            record["works"] = list(event.get("works") or [])
        elif stage == "brief" and int(event.get("round") or 0) == 0:
            record["has_brief"] = True
            record["papers"] = len(event.get("papers") or [])
            record["title"] = str(event.get("title") or "")
            record["plain_title"] = str(event.get("plain_title") or "")
            record["one_liner"] = str(event.get("one_liner") or "")
            record["why_it_matters"] = str(event.get("why_it_matters") or "")
            record["first_steps"] = list(event.get("first_steps") or [])
            record["gap"] = str(event.get("gap") or "")
            record["approach"] = str(event.get("approach") or "")
            record["idea"] = str(event.get("idea") or "")
            record["risks"] = str(event.get("risks") or "")
            record["baseline"] = str(event.get("baseline") or "")
            if event.get("papers"):
                record["papers_list"] = list(event.get("papers") or [])
        elif stage == "review":
            record["novelty"] = int(event.get("novelty") or 0)
            record["value_vector"] = event.get("value_vector")
            notes = [
                str(item).strip()
                for item in (event.get("must_change") or [])
                if str(item or "").strip()
            ]
            critique = str(event.get("critique") or "").strip()
            if critique:
                notes.append(critique)
            if notes:
                record["design_notes"] = notes[:6]
        elif stage == "skill_distilled":
            if event.get("name") and event.get("body"):
                record["distilled_skill"] = {
                    "name": str(event.get("name") or ""),
                    "description": str(event.get("description") or ""),
                    "body": str(event.get("body") or ""),
                    "digest": str(event.get("digest") or ""),
                    "stages": list(event.get("stages") or ["generate", "diagnose", "probe"]),
                }
        elif stage == "idea_world":
            record["idea_world"] = {
                key: event.get(key)
                for key in (
                    "objects",
                    "missing",
                    "iterate",
                    "do_not_repeat",
                    "named_instance",
                    "source",
                )
                if event.get(key) not in (None, "", [], ())
            }
            record["idea_usefulness"] = idea_usefulness(record)
        elif stage == "world_held":
            record["world_binding"] = "held"
        elif stage == "world-sim":
            if event.get("issues"):
                record["world_sim_issues"] = list(event.get("issues") or [])
            if isinstance(event.get("h"), dict):
                record["world_sim_h"] = dict(event["h"])
            notes = [
                str(item.get("summary") or "").strip()
                for item in (event.get("issues") or [])
                if isinstance(item, dict)
                and item.get("type") == "structural"
                and str(item.get("summary") or "").strip()
            ]
            if notes:
                existing = [
                    str(item).strip()
                    for item in (record.get("design_notes") or [])
                    if str(item or "").strip()
                ]
                record["design_notes"] = list(dict.fromkeys(existing + notes))[:6]
        elif stage in ("idea_kind", "idea_parked"):
            if event.get("idea_kind"):
                record["idea_kind"] = str(event["idea_kind"])
            if event.get("runs_probe") is False:
                record["skip_probe"] = True
        elif stage == "world_expansion":
            record["world_expansion"] = True
            record["missing_observables"] = list(event.get("missing_observables") or [])
            record["in_world_testable"] = bool(event.get("in_world_testable"))
            if event.get("status"):
                record["compile_disposition"] = str(event.get("status") or "")
            if event.get("idea_kind"):
                record["idea_kind"] = str(event["idea_kind"])
        elif stage == "compile_rejected":
            record["compile_rejected"] = True
            record["compile_disposition"] = str(event.get("disposition") or "")
            record["skip_writeup"] = True
        elif stage == "idea_compile":
            if event.get("claim"):
                record["claim"] = str(event["claim"])
            if event.get("mechanism"):
                record["mechanism"] = str(event["mechanism"])
            if event.get("prediction"):
                record["prediction"] = str(event["prediction"])
            if event.get("world_lever"):
                record["world_lever"] = str(event["world_lever"])
            if event.get("world_observable"):
                record["world_observable"] = str(event["world_observable"])
            if isinstance(event.get("claim_spec"), dict):
                record["claim_spec"] = event["claim_spec"]
            record["compile_refined"] = True
        elif stage == "diagnosis":
            record["alternative"] = str(event.get("alternative") or "")
            record["experiment"] = str(event.get("experiment") or "")
            record["treatment_arm"] = str(event.get("treatment_arm") or "")
            record["control_arm"] = str(event.get("control_arm") or "")
            record["expected_direction"] = str(event.get("expected_direction") or "")
            if event.get("world_lever"):
                record["world_lever"] = str(event["world_lever"])
            if event.get("world_observable"):
                record["world_observable"] = str(event["world_observable"])
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
            if event.get("scoped_verdict"):
                record["scoped_verdict"] = event.get("scoped_verdict")
            if event.get("evidence_projection"):
                record["evidence_projection"] = event.get("evidence_projection")
            if event.get("supports_scope"):
                record["supports_scope"] = event.get("supports_scope")
            if event.get("prohibited_inferences"):
                record["prohibited_inferences"] = event.get("prohibited_inferences")
            record["object_absent"] = bool(event.get("object_absent"))
            record["dv_blind"] = bool(event.get("dv_blind"))
            if event.get("world_consumed") is False:
                record["world_consumed"] = False
            explore_prior.learn(
                {
                    **event,
                    "world_lever": event.get("world_lever")
                    or record.get("world_lever"),
                    "world_observable": event.get("world_observable")
                    or record.get("world_observable"),
                }
            )
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
            # The host reruns the same bytes. It can confirm a probe verdict;
            # it cannot lift one the in-loop gates lowered: a measure blind to
            # the oracle, or a separation that survived structure destruction,
            # reproduces identically on the host and is still not a
            # measurement. (v3 booked two such reruns as `supports`.)
            gated = bool(record.get("dv_blind")) or record.get("world_consumed") is False or bool(
                record.get("object_absent")
            )
            if event.get("verdict") and not gated:
                record["verdict"] = event["verdict"]
            elif event.get("verdict") and gated:
                record["host_gate_blocked"] = (
                    "dv_blind" if record.get("dv_blind") else
                    "placebo_fail" if record.get("world_consumed") is False else "object_absent"
                )
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
        elif stage == "world_schema_held":
            current = dict(record.get("world_requirement") or {})
            if event.get("schema"):
                current["schema"] = str(event["schema"])
                spec = SCHEMAS.get(str(event["schema"]))
                if spec is not None:
                    current["object_type"] = spec.object_type
            if event.get("requested"):
                current["freeze_schema"] = str(event["requested"])
            record["world_requirement"] = current
            record["world_schema_held"] = str(event.get("requested") or "")
        elif stage == "world_forecast":
            record["world_forecast_agrees"] = bool(event.get("agrees"))
            if isinstance(event.get("analysis"), dict):
                record["idea_analysis"] = dict(event["analysis"])
        elif stage == "probe_skipped":
            record["experiment_bottleneck"] = str(event.get("bottleneck") or "no_handle")
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
            # No default family: a world with no schema is recorded as
            # such, not booked as a protocol automaton.
            record["world_schema"] = str(world_row.get("schema") or "")
            req = dict(record.get("world_requirement") or {})
            if world_row.get("schema") and not req.get("schema"):
                req["schema"] = str(world_row["schema"])
            if not req.get("object_type") and not req.get("schema"):
                stored_pair = list(record.get("pair") or [])
                stored_far = str(stored_pair[1]) if len(stored_pair) > 1 else ""
                inferred = infer_requirement(
                    topic,
                    without_far_terms(
                        str(record.get("claim") or ""), stored_far, topic
                    ),
                )
                if inferred is not None:
                    req = {**inferred.to_dict(), **req}
            if req.get("object_type") or req.get("schema"):
                record["world_requirement"] = req

    # Trajectory-guided landings. `jumps` / `reframes` are a cap.
    # Auto: Explore → Evaluate → Learn → Jump again. Fixed: same
    # recovered trajectories, increasing distance, no landing feedback
    # (replay / control stake).
    explore_prior = ExplorationPrior(
        object_phrases=topic_object_phrases(topic)[:8]
    )
    explore_prior.world_facts.extend(world_fact_lines)
    if mission_dynamics is not None:
        explore_prior.learn({"stage": "world_scout", **mission_dynamics})
    last_chosen: dict[str, Any] = {}

    def _screen_candidates(
        jump: Any,
        jump_index: int,
        far_candidates: list[tuple[str, str]],
        *,
        lens: str,
        slot: dict[str, Any] | None = None,
        proposal: Any = None,
    ) -> dict[str, Any]:
        viable, partners, prescreened, extra_kills = _screen_far(
            assets,
            near,
            near_ids,
            far_candidates,
            object_labels=object_labels,
        )
        all_kills.extend(extra_kills)
        payload = {
            "index": jump_index,
            "jump": jump,
            "far": tuple(viable),
            "partners": partners,
            "prescreened": prescreened,
            "lens": lens,
        }
        if slot is not None:
            payload["slot"] = slot
        if proposal is not None:
            payload["proposal"] = proposal
            payload["reasoning"] = proposal.reasoning
            payload["band"] = proposal.band
        return payload

    def _trajectory_jump(proposal: Any) -> Jump:
        seed = near_ids[0] if near_ids else ""
        landing = (
            tuple(assets.space.vectors[seed])
            if seed and seed in assets.space.vectors
            else ()
        )
        alien = {"early": 0.25, "middle": 0.5, "far": 0.8}.get(
            getattr(proposal, "band", ""), 0.25
        )
        return Jump(
            operator=str(proposal.move),
            seed_id=seed,
            params={
                "source": "research_trajectory",
                "band": proposal.band,
                "shaped_by": proposal.shaped_by,
                "support": proposal.support,
                "novelty": proposal.novelty,
            },
            landing=landing,
            retrieved=(),
            alienness=float(alien),
        )

    def _lock_exploit(meta: dict[str, Any]) -> dict[str, Any]:
        if not (
            stored_program
            and slot_plan
            and meta["index"] == 0
            and slot_plan[0].get("role") == "exploit"
        ):
            return meta
        locked = list(stored_program.get("pair") or [])
        if len(locked) < 2:
            return {
                **meta,
                "far": (),
                "partners": {},
                "locked_pair": True,
                "lock_failed": "continue_pair_incomplete",
            }
        near_label, far_label = locked[0], locked[1]
        stored_nodes = list(stored_program.get("pair_nodes") or [])
        far_node = stored_nodes[1] if len(stored_nodes) > 1 else ""
        in_graph = far_label in assets.label_to_node
        literature = str(far_node).startswith(LITERATURE_NODE_PREFIX) or (
            not in_graph and not far_node
        )
        if not in_graph and not literature:
            return {
                **meta,
                "far": (),
                "partners": {},
                "locked_pair": True,
                "lock_failed": "continue_far_not_in_graph",
                "locked_pair_labels": locked[:2],
            }
        return {
            **meta,
            "far": (far_label,),
            "partners": {far_label: (near_label,)},
            "locked_pair": True,
            "lens": "trajectory" if literature else "graph",
        }

    def _sample_meta(jump_index: int, *, sequential: bool) -> dict[str, Any]:
        del sequential
        trajectories = recover_trajectories(wiki_pool, topic, research_state)
        proposal = propose_jump(
            index=jump_index,
            state=research_state,
            trajectories=trajectories,
            ledger=explore_prior.ledger,
            used_far=explore_prior.used_far,
        )
        far_candidates = [
            (phrase, f"{LITERATURE_NODE_PREFIX}{phrase}")
            for phrase in proposal.far_labels
        ]
        slot = None
        if 0 <= jump_index < len(slot_plan):
            slot = {
                **slot_plan[jump_index],
                "operator": proposal.move,
                "track": "farfield",
                "band": proposal.band,
            }
        return _lock_exploit(
            _screen_candidates(
                _trajectory_jump(proposal),
                jump_index,
                far_candidates,
                lens="trajectory",
                slot=slot,
                proposal=proposal,
            )
        )

    def _absorb_landing(meta: dict[str, Any]) -> None:
        card_id = last_chosen.get("card_id")
        record = None
        if card_id:
            for row in found:
                if row.get("card_id") == card_id:
                    record = row
                    break
        category = landing_category(
            record,
            killed_by=list(last_chosen.get("killed_by") or []),
            generated=bool(last_chosen),
        )
        why = ""
        iterate = ""
        if record:
            why = str(record.get("why_failed") or record.get("verdict") or "").strip()
            world_note = record.get("idea_world") if isinstance(record.get("idea_world"), dict) else {}
            iterate = str(world_note.get("iterate") or world_note.get("do_not_repeat") or "").strip()
            holes = [
                str(item.get("summary") or "").strip()
                for item in (record.get("world_sim_issues") or [])
                if isinstance(item, dict)
                and item.get("type") == "structural"
                and str(item.get("summary") or "").strip()
            ]
            if holes and not iterate:
                iterate = holes[0]
        explore_prior.absorb_landing(
            pair=last_chosen.get("pair"),
            far_menu=meta.get("far") or (),
            category=category,
            operator=str(
                getattr(meta.get("jump"), "operator", "")
                or last_chosen.get("operator")
                or ""
            ),
            world_lever=str(last_chosen.get("world_lever") or ""),
            world_observable=str(last_chosen.get("world_observable") or ""),
            why=why,
            iterate=iterate,
        )
        if (
            record is not None
            and explore_prior.how_found
            and not record.get("how_found")
        ):
            record["how_found"] = dict(explore_prior.how_found)
        proposal = meta.get("proposal")
        if proposal is not None:
            far = ""
            pair = last_chosen.get("pair") or []
            if len(pair) >= 2:
                far = str(pair[1])
            explore_prior.absorb_feedback(
                evaluate_landing(
                    proposal,
                    far=far,
                    category=category,
                    why=why,
                    trajectories=recover_trajectories(
                        wiki_pool, topic, research_state
                    ),
                    state=research_state,
                ),
                far=far,
            )

    jump_meta: list[dict[str, Any]] = []
    if explore_mode == "fixed":
        for jump_index in range(explore_jumps):
            jump_meta.append(_sample_meta(jump_index, sequential=False))
    else:
        jump_meta = []

    def _continue_lock_refused(meta: dict[str, Any]) -> dict[str, Any] | None:
        reason = str(meta.get("lock_failed") or "")
        if not reason:
            return None
        return {
            "stage": "refused",
            "missing_capability": reason,
            "unlock_condition": (
                "the neighbourhood continue pair must still be the same two "
                "labels; a graph far must remain a node of this corpus, a "
                "literature far stays a literature far. Sampling a different "
                "distant concept would abandon the locked line"
            ),
            "track": "farfield",
            "role": "exploit",
            "pair": list((stored_program or {}).get("pair") or []),
        }

    def _far_jobs(meta: dict[str, Any]) -> list[dict[str, Any]]:
        if not meta["far"]:
            allowed = tuple(item for item in (object_labels or near) if item)
            meta = {
                **meta,
                "far": (LOCAL_ORIGIN_LABEL,),
                "partners": {LOCAL_ORIGIN_LABEL: allowed},
                "origin": ORIGIN_LOCAL,
                "lens": "local",
            }
        slot = meta.get("slot") or slot_plan[meta["index"]]
        return [
            {
                "kind": "farfield",
                "jump_index": meta["index"],
                "k": k,
                "jump": meta["jump"],
                "far": meta["far"],
                "partners": meta["partners"],
                "slot": slot,
                "lens": meta.get("lens") or "trajectory",
                "proposal": meta.get("proposal"),
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
        slot = meta.get("slot")
        if slot is None and 0 <= int(meta["index"]) < len(slot_plan):
            slot = slot_plan[meta["index"]]
        event = {
            "stage": "jump",
            "index": meta["index"],
            "operator": jump.operator,
            "track": (slot or {}).get("track") or "farfield",
            "lens": meta.get("lens") or "trajectory",
            "target": (slot or {}).get("target"),
            "alienness": round(jump.alienness, 4),
            "far": list(meta["far"]),
            "prescreened_out": meta["prescreened"],
            "locked_pair": bool(meta.get("locked_pair")),
        }
        if meta.get("band"):
            event["band"] = meta["band"]
        if meta.get("reasoning"):
            event["reasoning"] = list(meta["reasoning"])
        proposal = meta.get("proposal")
        if proposal is not None:
            event["support"] = proposal.support
            event["novelty"] = proposal.novelty
            event["shaped_by"] = proposal.shaped_by
        if meta.get("lock_failed"):
            event["lock_failed"] = meta["lock_failed"]
            if meta.get("locked_pair_labels"):
                event["locked_pair_labels"] = list(meta["locked_pair_labels"])
        notebook = explore_prior.snapshot()
        event["used_far"] = list(notebook["used_far"])
        event["blocked_pairs"] = list(notebook["used_pairs"])
        event["blocked_routes"] = list(notebook["routes"])
        event["unused_menu"] = list(notebook["unused_menu"])
        event["object_phrases"] = list(notebook["object_phrases"])
        if notebook.get("how_found"):
            event["how_found"] = notebook["how_found"]
        if notebook.get("search"):
            event["search"] = notebook["search"]
        return event

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

    def _refine_until_enter(
        current: GeneratedCard,
        kills: list[str],
        jump_index: int,
        track: str,
    ) -> Iterator[dict[str, Any]]:
        """Text-gate deaths only. Graph-dead pairs do not burn idea_rounds."""
        nonlocal total_cards, survivors, last_chosen
        if idea_budget <= 0:
            return
        for attempt in range(idea_budget):
            decision = decide_idea_refine(
                attempt=attempt,
                cap=idea_budget,
                killed_by=kills,
                alive=False,
                idea_kind=idea_kind_of(current),
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
            field_seed, field_near = _field_seed_and_near()
            refine_near = field_near
            if current.pair[0] != field_seed and current.pair[0] not in refine_near:
                refine_near = (current.pair[0],) + tuple(refine_near)
            refine_skills = attach_idea_skills(
                skill_blocks,
                skills_from_payloads(
                    (
                        program_for(
                            topics_now, corpus_id, near[0], pair=current.pair
                        )
                        or {}
                    ).get("skills")
                    or ()
                ),
            )["generate"]
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
                        seed_label=field_seed,
                        near_labels=refine_near,
                        killed_by=kills,
                        context=context,
                        knowledge=slot_knowledge({"target": None, "track": track}),
                        criteria=standards,
                        topic=topic,
                        skills=refine_skills,
                        program=_program_for_refine(current),
                        wiki=wiki_lines,
                    )
                else:
                    current = refine_card(
                        client,
                        current,
                        seed_label=field_seed,
                        near_labels=refine_near,
                        label_to_node=(
                            literature_labels(gen_label_to_node, (current.pair[1],))
                            if str(current.pair_nodes[1]).startswith(
                                LITERATURE_NODE_PREFIX
                            )
                            else gen_label_to_node
                        ),
                        killed_by=kills,
                        capsules=lessons,
                        logprobs=True,
                        criteria=standards,
                        context=context,
                        knowledge=slot_knowledge({"target": None, "track": track}),
                        value_criteria=rubric,
                        topic=topic,
                        skills=refine_skills,
                        program=_program_for_refine(current),
                        wiki=wiki_lines,
                        world_menu=mission_menu,
                        used_levers=(
                            tuple(
                                dict.fromkeys(
                                    list(explore_prior.used_levers)
                                    + (
                                        [current.world_lever]
                                        if current.world_lever
                                        and "no_handle" in kills
                                        else []
                                    )
                                )
                            )
                        ),
                        blocked_handles=tuple(explore_prior.blocked_handles),
                        scientific=scientific_for(
                            card=current, world=bound_world, workspace=workspace
                        ),
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
                "world_lever": str(getattr(current, "world_lever", "") or ""),
                "world_observable": str(getattr(current, "world_observable", "") or ""),
                "refined": True,
            }
            graph_killed_by: list[str] = []
            graph_checks: list[dict[str, Any]] = []
            if track != "reframe":
                graph_verdict = probe_generated(assets.oracle, [current])[0]
                graph_killed_by = list(graph_verdict.killed_by)
                graph_checks = [check.to_dict() for check in graph_verdict.checks]
            view = _card_view(
                current,
                plausibility=(
                    0.0
                    if track == "reframe"
                    else assets.oracle.jaccard(*current.pair_nodes)
                ),
            )
            text_kills, _, shadow = _run_text_gates(current, view)
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
                last_chosen = {
                    "card_id": current.card_id,
                    "pair": list(current.pair),
                    "killed_by": [],
                    "alive": True,
                    "operator": current.operator,
                    "world_lever": current.world_lever,
                    "world_observable": getattr(current, "world_observable", ""),
                }
                entered_jobs.append((current, track))
                return

    def _field_seed_and_near() -> tuple[str, tuple[str, ...]]:
        """Near-side labels shown to the generator.

        The starting position is the user's object, not a cosine neighbour
        from a concept catalog. Covering graph labels may still appear
        when they name that object.
        """
        if object_labels:
            return object_labels[0], tuple(object_labels[1:])
        return near[0], tuple(near[1:])

    def _jump_notes(job: dict[str, Any]) -> tuple[str, ...]:
        proposal = job.get("proposal")
        notes: list[str] = []
        if proposal is not None:
            notes.extend(proposal.prompt_lines())
        notes.extend(
            trajectory_prompt_lines(
                research_state,
                recover_trajectories(wiki_pool, topic, research_state),
            )
        )
        extra = explore_prior.ledger.prompt_lines()
        if extra:
            notes.extend(extra)
        return tuple(notes)

    def _do_gen(job: dict[str, Any]) -> dict[str, Any]:
        slot = job.get("slot") or job.get("entry") or {}
        generate_program = program_lines if slot.get("role") == "exploit" else ()
        seed_label, near_labels = _field_seed_and_near()
        knowledge = slot_knowledge(job.get("slot") or job.get("entry") or slot)
        knowledge = knowledge + notebook.prompt_lines()
        generate_skills = skill_blocks["generate"]
        if slot.get("role") == "exploit" and stored_program:
            pair = list(stored_program.get("pair") or [])
            if len(pair) >= 2:
                locked_near = pair[0]
                if locked_near != seed_label and locked_near not in near_labels:
                    near_labels = (locked_near,) + tuple(near_labels)
                duty = continue_duty_lines(
                    {
                        **stored_program,
                        "world_id": stored_program.get("world_id")
                        or getattr(bound_world, "id", "")
                        or "",
                    }
                )
                if duty:
                    knowledge = knowledge + duty
                generate_skills = attach_idea_skills(
                    skill_blocks, idea_skills
                )["generate"]
        try:
            if job["kind"] == "farfield":
                exploit = slot.get("role") == "exploit" and stored_program
                locked_near = ""
                if exploit:
                    pair = list(stored_program.get("pair") or [])
                    if pair:
                        locked_near = str(pair[0])
                card_labels = gen_label_to_node
                if (job.get("lens") or "trajectory") == "trajectory":
                    card_labels = literature_labels(
                        gen_label_to_node, job.get("far") or ()
                    )
                card = generate_card(
                    client,
                    seed_label=seed_label,
                    near_labels=near_labels,
                    far_labels=_rotate(job["far"], job["k"]),
                    operator=job["jump"].operator,
                    alienness=job["jump"].alienness,
                    label_to_node=card_labels,
                    logprobs=True,
                    criteria=standards,
                    context=context,
                    knowledge=knowledge,
                    capsules=lessons,
                    viable_pairings=job["partners"],
                    value_criteria=rubric,
                    topic=topic,
                    skills=generate_skills,
                    program=generate_program,
                    wiki=wiki_lines,
                    object_labels=(locked_near,) if locked_near else object_labels,
                    prior_struggle=(
                        () if exploit else explore_prior.prompt_lines()
                    ),
                    world_menu=mission_menu,
                    used_levers=tuple(explore_prior.used_levers),
                    blocked_handles=tuple(explore_prior.blocked_handles),
                    jump_notes=_jump_notes(job),
                    scientific=scientific_for(
                        world=bound_world, workspace=workspace
                    ),
                )
            else:
                field_seed, field_near = _field_seed_and_near()
                card = generate_reframe(
                    client,
                    seed_label=field_seed,
                    near_labels=field_near,
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
        nonlocal total_cards, survivors, last_chosen
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
                "world_lever": str(getattr(card, "world_lever", "") or ""),
                "world_observable": str(getattr(card, "world_observable", "") or ""),
            }
        if not jump_cards:
            return
        rulings = []
        graph_verdicts = probe_generated(
            assets.oracle, [card for _, card in jump_cards]
        )
        for (k, card), verdict in zip(jump_cards, graph_verdicts):
            view = _card_view(
                card, plausibility=assets.oracle.jaccard(*card.pair_nodes)
            )
            text_kills, judge_errors, shadow = _run_text_gates(card, view)
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
                last_chosen = {
                    "card_id": card.card_id,
                    "pair": list(card.pair),
                    "killed_by": list(verdict.killed_by) + ruling["text_kills"],
                    "alive": alive,
                    "operator": card.operator,
                    "world_lever": card.world_lever,
                    "world_observable": getattr(card, "world_observable", ""),
                }
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
            "world_lever": str(getattr(card, "world_lever", "") or ""),
            "world_observable": str(getattr(card, "world_observable", "") or ""),
        }
        view = _card_view(card, plausibility=0.0)
        text_kills, judge_errors, shadow = _run_text_gates(card, view)
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

    def _do_entry(item: tuple[GeneratedCard, str]) -> dict[str, Any]:
        card, _track = item
        idea = program_for(topics_now, corpus_id, near[0], pair=card.pair)
        entry_blocks = attach_idea_skills(
            skill_blocks, skills_from_payloads((idea or {}).get("skills") or ())
        )
        lock_world = bound_world is not None
        hold_world = bool(
            stored_program
            and is_continue_program(stored_program)
            and list(card.pair)[:2] == list(stored_program.get("pair") or [])[:2]
        )
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
            skill_blocks=entry_blocks,
            experiment_rounds=experiment_budget,
            world=bound_world,
            wiki_works_pool=wiki_pool,
            program=tuple(program_prompt_lines(idea)) + explore_prior.constraint_lines(),
            catalog=world_catalog,
            world_request=world_request,
            bound_world=bound_world,
            lock_world=lock_world,
            hold_world=hold_world,
            dynamics=mission_dynamics,
            world_sim=world_sim,
        )

    def _emit_evidence(
        batch: list[tuple[GeneratedCard, str]],
    ) -> Iterator[dict[str, Any]]:
        nonlocal bound_world, mission_dynamics, mission_menu
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
            outcome["world_provenance"] = record.get("world_provenance") or ""
            outcome["world_id"] = record.get("world_id") or ""
            card_world = payload.get("world")
            if card_world is not None:
                record["world_id"] = getattr(card_world, "id", "") or ""
                record["world_schema"] = getattr(card_world, "schema", "") or ""
                record["world_provenance"] = getattr(card_world, "provenance", "") or ""
                if bound_world is None:
                    bound_world = card_world
                    dest = _mission_world_dir(workspace)
                    if dest is not None:
                        bind_world(dest, bound_world)
                        _write_world_readme(Path(workspace), bound_world)
                    if has_dynamics(bound_world):
                        try:
                            mission_dynamics = response_card(bound_world)
                        except Exception:
                            mission_dynamics = None
                        if mission_dynamics is not None:
                            mission_menu = menu_from_card(
                                mission_dynamics, bound_world
                            )
            _fold_record_into_outcome(outcome, record)
            live_card = payload.get("card") or card
            writeup_stash.append(
                {
                    "card": live_card,
                    "track": track,
                    "bundle": payload["bundle"],
                    "record": record,
                    "world": card_world,
                }
            )
            entered_cards[live_card.card_id] = live_card

    def _retry_no_handle(
        batch: list[tuple[GeneratedCard, str]],
        jump_index: int,
    ) -> Iterator[dict[str, Any]]:
        """Retry no_handle cards, then evidence only after review."""
        follow: list[tuple[GeneratedCard, str]] = []
        for card, track in batch:
            record = next(
                (row for row in found if row.get("card_id") == card.card_id),
                None,
            )
            if str((record or {}).get("experiment_bottleneck") or "") != "no_handle":
                continue
            before = len(entered_jobs)
            yield from _refine_until_enter(card, ["no_handle"], jump_index, track)
            follow.extend(entered_jobs[before:])
        if follow:
            yield from _emit_evidence_reviewed(follow)

    def _review_entered(
        batch: list[tuple[GeneratedCard, str]],
    ) -> Iterator[dict[str, Any]]:
        """Adversarial review of entered cards, before registration.

        Passing the gates is admission, not quality. Argus and ARIS both
        put a hostile reader between the first draft and the first
        experiment; FarField's version is bounded (one rewrite of the
        same pair), pre-registration (no result exists to fit), and
        gate-checked (a refined card re-runs the graph oracle and every
        text gate; a dead rewrite keeps the original). The review vector
        is recorded and never kills, never climbs the ladder.
        """
        nonlocal total_cards, survivors
        if idea_budget <= 0 or not batch:
            return
        papers = list(wiki_pool)[:10]
        for index, (card, track) in enumerate(list(batch)):
            try:
                review = critique_card(
                    client,
                    card,
                    topic,
                    papers,
                    scientific=scientific_for(
                        card=card, world=bound_world, workspace=workspace
                    ),
                )
            except GenerationRefused as exc:
                yield {
                    "stage": "idea_review_refused",
                    "card_id": card.card_id,
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
                }
                continue
            except LLMUnavailable as exc:
                yield {
                    "stage": "blocked",
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
                }
                return
            event = review.to_dict()
            event["stage"] = "idea_review"
            event["pair"] = list(card.pair)
            event["track"] = track
            yield event
            yield {"stage": "review_admitted", "card_id": card.card_id}
            changes = tuple(review.must_change) if review.keep_going else ()
            decision = decide_idea_refine(
                attempt=0,
                cap=1,
                alive=True,
                review_changes=changes,
                failure_types=tuple(getattr(review, "failure_types", ()) or ()),
                idea_kind=idea_kind_of(card),
            )
            yield {
                "stage": "idea_decision",
                "card_id": card.card_id,
                "attempt": 0,
                "pair": list(card.pair),
                "track": track,
                **decision,
            }
            if not decision["continue"]:
                continue
            kills = [
                f"reviewer_must_change: {item}" for item in review.must_change[:3]
            ]
            if review.critique:
                kills.append(f"reviewer_critique: {review.critique}")
            try:
                field_seed, field_near = _field_seed_and_near()
                refined = refine_card(
                    client,
                    card,
                    seed_label=field_seed,
                    near_labels=(
                        (card.pair[0],) + tuple(field_near)
                        if card.pair[0] != field_seed
                        else tuple(field_near)
                    ),
                    label_to_node=(
                        literature_labels(gen_label_to_node, (card.pair[1],))
                        if str(card.pair_nodes[1]).startswith(LITERATURE_NODE_PREFIX)
                        else gen_label_to_node
                    ),
                    killed_by=kills,
                    capsules=lessons,
                    logprobs=True,
                    criteria=standards,
                    context=context,
                    knowledge=slot_knowledge({"target": None, "track": track}),
                    value_criteria=rubric,
                    topic=topic,
                    skills=skill_blocks["generate"],
                    program=(),
                    wiki=wiki_lines,
                    world_menu=mission_menu,
                    used_levers=tuple(explore_prior.used_levers),
                    blocked_handles=tuple(explore_prior.blocked_handles),
                    scientific=scientific_for(
                        card=card, world=bound_world, workspace=workspace
                    ),
                )
            except GenerationRefused as exc:
                yield {
                    "stage": "refused",
                    "card_id": card.card_id,
                    "missing_capability": exc.record.missing_capability,
                    "unlock_condition": exc.record.unlock_condition,
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
            graph_killed_by: list[str] = []
            if track != "reframe" and tuple(refined.pair) == tuple(card.pair):
                graph_verdict = probe_generated(assets.oracle, [refined])[0]
                graph_killed_by = list(graph_verdict.killed_by)
            view = _card_view(
                refined,
                plausibility=(
                    0.0
                    if track == "reframe"
                    else assets.oracle.jaccard(*refined.pair_nodes)
                ),
            )
            text_kills, _, shadow = _run_text_gates(refined, view)
            same_pair = tuple(refined.pair) == tuple(card.pair)
            alive = same_pair and not graph_killed_by and not text_kills
            yield {
                "stage": "verdict",
                "candidate": "review_refine",
                "chosen": alive,
                "card_id": refined.card_id,
                "operator": refined.operator,
                "alienness": round(refined.alienness, 4),
                "pair": list(refined.pair),
                "claim": refined.claim,
                "mechanism": refined.mechanism,
                "prediction": refined.prediction,
                "preregistered_falsifier": refined.falsifier,
                "graph_killed_by": graph_killed_by,
                "text_kills": text_kills,
                "shadow_kills": shadow,
                "judge_errors": [],
                "alive": alive,
                "outcome": (
                    "entered"
                    if alive
                    else "pair_changed"
                    if not same_pair
                    else "ineligible"
                ),
                "refined": True,
                "refine_reason": "review",
            }
            if not alive:
                # The rewrite failed the gates or drifted off the pair:
                # the original entered card stands.
                continue
            survivors += 1
            batch[index] = (refined, track)
            yield {"stage": "review_admitted", "card_id": refined.card_id}
            for job_index, (existing, existing_track) in enumerate(entered_jobs):
                if (
                    existing.card_id == card.card_id
                    and existing_track == track
                ):
                    entered_jobs[job_index] = (refined, track)
                    break

    def _emit_evidence_reviewed(
        batch: list[tuple[GeneratedCard, str]],
    ) -> Iterator[dict[str, Any]]:
        reviews_seen: dict[str, dict[str, Any]] = {}
        review_events: list[dict[str, Any]] = []
        for event in _review_entered(batch):
            yield event
            review_events.append(event)
            if event["stage"] == "idea_review" and event.get("card_id"):
                reviews_seen[str(event["card_id"])] = event
            if event["stage"] == "blocked":
                return
        # The human gate sits between the adversarial review and the first
        # paid evidence step. With `human_gate="after_review"`, cards the
        # operator has not approved stop the mission here with a decision
        # file; rerunning the same command with `--approve <card_id>`
        # replays everything above from the cache and continues.
        if human_gate == "after_review":
            approved_ids = {str(item) for item in approved}
            pending = [
                (card, track)
                for card, track in batch
                if "all" not in approved_ids and card.card_id not in approved_ids
            ]
            if pending:
                rows = [
                    {
                        "card_id": card.card_id,
                        "track": track,
                        "pair": list(card.pair),
                        "claim": card.claim,
                        "mechanism": card.mechanism,
                        "prediction": getattr(card, "prediction", ""),
                        "review": {
                            k: reviews_seen.get(card.card_id, {}).get(k)
                            for k in ("novelty", "clarity", "feasibility", "importance", "critique", "must_change")
                        },
                    }
                    for card, track in pending
                ]
                if workspace is not None:
                    _write_human_gate(Path(workspace), rows, approved_ids)
                yield {
                    "stage": "awaiting_human",
                    "gate": "after_review",
                    "pending": [row["card_id"] for row in rows],
                    "approved": sorted(approved_ids),
                    "file": str(Path(workspace) / "HUMAN_GATE.md") if workspace is not None else "",
                }
                yield {
                    "stage": "blocked",
                    "missing_capability": "human_approval",
                    "unlock_condition": (
                        "read HUMAN_GATE.md, then rerun the same command with "
                        "--approve <card_id> (repeatable) or --approve all; recorded "
                        "generation and review replay from the cache"
                    ),
                }
                return
        eligible = cards_eligible_for_evidence(
            batch,
            review_events,
            review_required=idea_budget > 0,
        )
        skipped = [
            card.card_id
            for card, _track in batch
            if card.card_id not in {item[0].card_id for item in eligible}
        ]
        if skipped:
            yield {
                "stage": "evidence_skipped",
                "reason": "review_fail_closed",
                "card_ids": skipped,
            }
        yield from _emit_evidence(eligible)

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
            refused = _continue_lock_refused(meta)
            if refused is not None:
                yield refused
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
        first_entered = list(entered_jobs)
        for event in _emit_evidence_reviewed(first_entered):
            yield event
            if event["stage"] == "blocked":
                return
        for event in _retry_no_handle(first_entered, 0):
            yield event
            if event["stage"] == "blocked":
                return
    else:
        evidenced = 0
        for jump_index in range(explore_jumps):
            last_chosen = {}
            meta = _sample_meta(jump_index, sequential=True)
            jump_meta.append(meta)
            yield _jump_event_far(meta)
            refused = _continue_lock_refused(meta)
            if refused is not None:
                yield refused
            far_jobs = _far_jobs(meta)
            jobs = far_jobs
            if jobs:
                gen_results = []
                for event in _generate_wave(jobs):
                    if event["stage"] == "_gen_results":
                        gen_results = event["results"]
                        continue
                    yield event
                    if event["stage"] == "blocked":
                        return
                yield from _ingest_far(meta, list(zip(far_jobs, gen_results)))
                _absorb_landing(meta)
                batch = entered_jobs[evidenced:]
                evidenced = len(entered_jobs)
                for event in _emit_evidence_reviewed(batch):
                    yield event
                    if event["stage"] == "blocked":
                        return
                for event in _retry_no_handle(batch, jump_index):
                    yield event
                    if event["stage"] == "blocked":
                        return
                evidenced = len(entered_jobs)
            else:
                _absorb_landing(meta)
            far_opened += 1
            decision = decide_next(
                opened=far_opened,
                cap=explore_jumps,
                records=found,
                prior_live=prior_live,
                prior_plan=prior_plan,
                min_landings=min_ideas,
            )
            yield {
                "stage": "explore_decision",
                "track": "farfield",
                "opened": far_opened,
                "cap": explore_jumps,
                **decision,
            }
            if not decision["continue"]:
                break
        stay = decide_next(
            opened=max(1, far_opened),
            cap=max(explore_jumps, 1),
            records=found,
            prior_live=prior_live,
            prior_plan=prior_plan,
            min_landings=min_ideas,
        )
        if (
            not has_research_plan(found)
            and not prior_live
            and not prior_plan
            and stay["reason"] != "plan_executable"
        ):
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
                for event in _emit_evidence_reviewed(batch):
                    yield event
                    if event["stage"] == "blocked":
                        return
                for event in _retry_no_handle(batch, index):
                    yield event
                    if event["stage"] == "blocked":
                        return
                evidenced = len(entered_jobs)
                reframe_opened += 1
                decision = decide_next(
                    opened=reframe_opened,
                    cap=max(0, reframes),
                    records=found,
                    prior_live=prior_live,
                    prior_plan=prior_plan,
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
        gaps = wiki_gap_lines(wiki_for(load_state(state_path), corpus_id, near[0]), topic=topic)
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
    if stake is None:
        stake = missing_stake(
            anchor=near_ids[0] if near_ids else "",
            arm_n=len(arm_far_nodes),
            reason="literature_landing" if arm_far_nodes else "no_graph_arm",
        )
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
    lesson_kills = list(all_kills)
    for row in found:
        if row.get("verdict") not in {"uninformative", "weakens"}:
            continue
        pair = list(row.get("pair") or [])
        if len(pair) < 2:
            continue
        world_note = (
            row.get("idea_world") if isinstance(row.get("idea_world"), dict) else {}
        )
        lesson_kills.append(
            {
                "pair": pair,
                "killed_by": [str(row.get("verdict") or "approach")],
                "why": str(row.get("why_failed") or row.get("verdict") or "")[:240],
                "lesson": str(
                    world_note.get("do_not_repeat") or world_note.get("iterate") or ""
                )[:240],
            }
        )
    banked = record_rejections(state_path, corpus_id, near[0], lesson_kills)

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

    # Freeze recipes for the host operator. WORLD_INCOMPATIBLE unmatched
    # rows and constructed GENERATED worlds both qualify. A later freeze
    # binds new claims, never these. A scout with no room to move is the
    # same kind of unmatched object — do not spray four far nouns instead.
    recipes = wishlist_requirements(found, topic=topic)
    if recipes:
        try:
            wish_path = record_world_wishlist(
                world_root or root, recipes, topic=topic
            )
        except OSError:
            wish_path = None
        if wish_path is not None:
            yield {
                "stage": "world_wishlist",
                "unmatched": len(recipes),
                "path": str(wish_path),
            }

    briefs = sum(1 for record in found if record.get("has_brief"))
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
        lang="zh",
    )
    packet_md_en = render_mission_packet(
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
        lang="en",
    )
    idea_report_md = render_idea_report(
        topic,
        found=found,
        ranked=ranked,
        kills=all_kills,
        wiki=wiki_for(load_state(state_path), corpus_id, near[0]),
    )
    idea_packets: list[str] = []
    paper_targets: list[tuple[Path, str]] = []
    if workspace is not None:
        live_rows, _buckets = live_packet_records(ranked=ranked, found=found)
        sibling_ids = [
            str(row.get("card_id") or "")
            for row in live_rows
            if str(row.get("card_id") or "")
        ]
        stash_works = {
            str(stash["card"].card_id): list(
                (stash.get("bundle") or {}).get("works") or []
            )
            for stash in writeup_stash
        }
        taken: set[str] = set()
        for rec in live_rows:
            card_id = str(rec.get("card_id") or "")
            if not card_id:
                continue
            if not rec.get("works") and stash_works.get(card_id):
                rec = {**rec, "works": stash_works[card_id]}
            name = idea_folder_name(rec, taken=taken)
            rec = {**rec, "idea_name": name}
            folder = compile_live_idea(
                workspace, topic, rec, sibling_ids=sibling_ids
            )
            idea_packets.append(str(folder))
            paper_targets.append((Path(folder), card_id))
            # The evidence grid: cells this claim needs, which the probes
            # filled (typed), and which need a world that is not here yet.
            try:
                grid_event = _write_evidence_grid(
                    Path(workspace),
                    Path(folder),
                    card_id,
                    rec,
                    bound_world=bound_world,
                    dynamics_card=mission_dynamics,
                    catalog_root=Path(catalog_root) / "worlds",
                    topic=topic,
                )
                if grid_event is not None:
                    yield grid_event
            except Exception as exc:  # the grid must never lose the packets
                yield {"stage": "evidence_grid_skipped", "card_id": card_id, "reason": f"{type(exc).__name__}: {exc}"[:200]}
        prune_idea_packets(workspace, keep={Path(path).name for path in idea_packets})
        write_idea_report(workspace, idea_report_md)
        used = tuple(explore_prior.used_levers)
        menu = mission_menu
        write_question_board(
            workspace,
            render_question_board(
                scorable=tuple(menu.scorable_questions(used) if menu is not None else ()),
                records=found,
                interventional=tuple(
                    menu.unused_interventional(used) if menu is not None else ()
                ),
            ),
        )
        write_archive(workspace, render_archive(found))
    # The paper-level pass: background, related work, motivation, theory
    # sketch, hypotheses, limitations, and a claim-driven experiment
    # design, written by the model from the attested fact sheet and read
    # once by the same model as a reviewer. DRAFT; the ladder does not
    # move. Skipped, not faked, when the budget or the endpoint is gone.
    if workspace is not None and paper_rounds >= 0:
        from .paperplan import PaperError, compile_paper

        for idea_folder, paper_card in paper_targets:
            if not (Path(workspace) / "candidates" / paper_card / "protocol.json").is_file():
                continue
            try:
                paper_record = compile_paper(
                    Path(workspace),
                    idea_folder,
                    paper_card,
                    client=client,
                    rounds=paper_rounds,
                    catalog_root=Path(catalog_root) / "worlds"
                    if (Path(catalog_root) / "worlds").is_dir()
                    else None,
                )
            except (PaperError, LLMUnavailable) as exc:
                yield {
                    "stage": "paper_skipped",
                    "card_id": paper_card,
                    "idea": idea_folder.name,
                    "reason": str(exc)[:300],
                }
                continue
            except Exception as exc:  # a writing failure must not lose the mission
                yield {
                    "stage": "paper_skipped",
                    "card_id": paper_card,
                    "idea": idea_folder.name,
                    "reason": f"{type(exc).__name__}: {exc}"[:300],
                }
                continue
            yield {
                "stage": "paper",
                "card_id": paper_card,
                "idea": idea_folder.name,
                "title": paper_record.get("title"),
                "paper_dir": paper_record.get("paper_dir"),
                "score": paper_record.get("score"),
                "rounds": paper_record.get("rounds"),
                "hypotheses": paper_record.get("hypotheses"),
                "audit": paper_record.get("audit"),
                "status": "draft",
            }
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
        "packet_en": (
            str(Path(workspace) / "RESEARCH_PACKET_EN.md")
            if workspace is not None
            else None
        ),
        "idea_packets": idea_packets,
        "idea_report": (
            str(Path(workspace) / "idea-stage" / "IDEA_REPORT.md")
            if workspace is not None
            else None
        ),
        "packet_markdown": packet_md,
        "packet_markdown_en": packet_md_en,
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
            {k: v for k, v in done.items() if k not in ("stage", "packet_markdown", "packet_markdown_en")},
            packet_md=packet_md,
            packet_md_en=packet_md_en,
        )
    yield done


def _brief_event(brief: ResearchBrief, card: GeneratedCard, round_no: int) -> dict[str, Any]:
    event = brief.to_dict()
    event["stage"] = "brief"
    event["round"] = round_no
    event["pair"] = list(card.pair)
    event["claim"] = card.claim
    return event


def _write_evidence_grid(
    workspace: Path,
    idea_dir: Path,
    card_id: str,
    rec: dict[str, Any],
    *,
    bound_world: Any,
    dynamics_card: dict[str, Any] | None,
    catalog_root: Path,
    topic: str,
) -> dict[str, Any] | None:
    from .evidencegrid import compile_grid, write_grid
    from .objectprops import required_properties

    cand = candidate_dir(workspace, card_id)
    try:
        protocol = json.loads((cand / "protocol.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        protocol = {}
    try:
        probe = json.loads((cand / "probe.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        probe = {}
    claim = str(protocol.get("claim") or rec.get("claim") or "")
    if not claim:
        return None
    world_row: dict[str, Any] | None = None
    if bound_world is not None:
        world_row = bound_world.to_dict() if hasattr(bound_world, "to_dict") else {}
    catalog_rows: list[dict[str, Any]] = []
    if catalog_root.is_dir():
        for directory in sorted(catalog_root.iterdir()):
            if not (directory / "manifest.json").is_file():
                continue
            try:
                fixture = load_fixture(directory)
            except Exception:
                continue
            if fixture.role in {"test", "fixture", "synthetic"}:
                continue
            catalog_rows.append(fixture.to_dict())
    attempts = list(probe.get("attempts") or protocol.get("experiment_attempts") or [])
    # The latest verdict/reason belongs to the last attempt.
    if attempts and isinstance(attempts[-1], dict):
        attempts[-1] = {**attempts[-1], "reason": probe.get("reason") or attempts[-1].get("reason"), "dv_blind": probe.get("dv_blind"), "object_absent": probe.get("object_absent"), "evidence_id": probe.get("evidence_id")}
    grid = compile_grid(
        claim=claim,
        world=world_row,
        dynamics_card=dynamics_card,
        diagnosis=protocol.get("diagnosis") or {},
        attempts=attempts,
        catalog_worlds=catalog_rows,
        required_properties=required_properties(topic, claim),
    )
    path = write_grid(idea_dir, grid)
    return {
        "stage": "evidence_grid",
        "card_id": card_id,
        "idea": idea_dir.name,
        "path": str(path),
        "coverage": grid.get("coverage"),
        "reading": grid.get("reading"),
        "needs_world": [c["id"] for c in grid.get("cells") or [] if c.get("status") == "needs_world"],
    }


def _response_ceiling(dynamics_card: Any, diagnosis: Any) -> float | None:
    """The scout's |relative response| of the diagnosis's lever on its observable.

    None when the world has no scout card, the lever is not on it, or the
    observable was not measured: no ceiling means no refusal.
    """
    if not isinstance(dynamics_card, dict):
        return None
    lever = str(getattr(diagnosis, "world_lever", "") or "")
    observable = str(getattr(diagnosis, "world_observable", "") or "")
    responses = (dynamics_card.get("levers") or {}).get(lever)
    if not isinstance(responses, dict) or observable not in responses:
        return None
    try:
        return abs(float(responses[observable]))
    except (TypeError, ValueError):
        return None


def _write_human_gate(workspace: Path, rows: list[dict[str, Any]], approved: set[str]) -> None:
    """The decision file for the after-review human gate."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "HUMAN_GATE.json").write_text(
        json.dumps({"gate": "after_review", "pending": rows, "approved": sorted(approved)}, ensure_ascii=False, indent=1)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Human gate: after review, before evidence",
        "",
        "The mission stopped before spending on diagnosis and probes. Each card below passed the "
        "text gates and one adversarial review. Decide which claims deserve the evidence budget, then "
        "rerun the same command with `--approve <card_id>` (repeatable) or `--approve all`. Recorded "
        "generation and review replay from the cache; spend restarts at the first unrecorded call.",
        "",
    ]
    for row in rows:
        review = row.get("review") or {}
        lines.append(f"## `{row['card_id']}` · {row['pair'][0]} × {row['pair'][1]} ({row['track']})")
        lines.append("")
        lines.append(f"- claim: {row['claim']}")
        lines.append(f"- mechanism: {row['mechanism']}")
        if row.get("prediction"):
            lines.append(f"- prediction: {row['prediction']}")
        scores = ", ".join(
            f"{k} {review.get(k)}" for k in ("novelty", "clarity", "feasibility", "importance") if review.get(k) is not None
        )
        if scores:
            lines.append(f"- adversarial review: {scores}")
        if review.get("critique"):
            lines.append(f"- critique: {review['critique']}")
        for item in review.get("must_change") or []:
            lines.append(f"  - must change: {item}")
        lines.append("")
    (workspace / "HUMAN_GATE.md").write_text("\n".join(lines), encoding="utf-8")


def _run_probe_attempt(
    spec: Any,
    workspace: Path | None,
    card_id: str,
    attempt: int,
    world: Any = None,
    tier_name: str = "",
) -> Any:
    """The in-loop probe follows the pre-registered tier (capped) instead of
    a flat 20s: a history schema's script is allowed to replay the history."""
    tier = resolve_tier(tier_name) if tier_name else None
    seconds = probe_timeout(tier) if tier is not None else 20.0
    if workspace is not None:
        dest = candidate_dir(workspace, card_id) / "probe" / f"attempt-{attempt}"
        dest.mkdir(parents=True, exist_ok=True)
        if world is not None:
            bind_world(dest, world)
        return run_probe(spec, dest, timeout_seconds=seconds, tier=tier)
    with tempfile.TemporaryDirectory(prefix="ffprobe-") as tmp:
        dest = Path(tmp)
        if world is not None:
            bind_world(dest, world)
        return run_probe(spec, dest, timeout_seconds=seconds, tier=tier)


def _compile_entered_pair(
    client: Any,
    card: GeneratedCard,
    *,
    topic: str,
    diagnosis: Any,
    notes: list[str],
    program: str | tuple[str, ...],
    works: list,
    wiki_works_pool: list | None,
    skill_blocks: dict[str, str] | None,
    dynamics_card: dict[str, Any] | None,
    world: Any,
    prior_for_diag: dict[str, Any] | None,
    diagnose_skills: str,
    world_requirement: Any,
    world_path: Any,
    sink: dict[str, Any],
    workspace: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Same pair after diagnosis + rehearsal. Object and freeze stay locked.

    A refused compile keeps the entered card. Imagined numbers never enter.
    """
    labels = {
        card.pair[0]: card.pair_nodes[0],
        card.pair[1]: card.pair_nodes[1],
    }
    lines = program if isinstance(program, tuple) else ((program,) if program else ())
    titles: list[str] = []
    for work in list(works or []) + list(wiki_works_pool or []):
        if isinstance(work, dict):
            title = str(work.get("title") or "").strip()
        else:
            title = str(getattr(work, "title", "") or "").strip()
        if title and title not in titles:
            titles.append(title)
    wiki = tuple(titles[:8])
    menu = menu_from_card(dynamics_card, world)
    generate_skills = (skill_blocks or {}).get("generate", "")
    yield {
        "stage": "idea_refining",
        "card_id": card.card_id,
        "pair": list(card.pair),
        "duty": "compile",
        "killed_by": ["structural"],
    }
    try:
        refined = refine_card(
            client,
            card,
            seed_label=card.pair[0],
            near_labels=(),
            label_to_node=labels,
            bottleneck="compile",
            topic=topic,
            skills=generate_skills,
            program=lines,
            wiki=wiki,
            world_menu=menu if menu.levers else None,
            duty="compile",
            keep_id=True,
            compile_notes=notes,
            scientific=scientific_for(
                card=card, world=world, workspace=workspace
            ),
        )
    except GenerationRefused as exc:
        yield {
            "stage": "refused",
            "card_id": card.card_id,
            "duty": "compile",
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
    sink["card"] = refined
    yield {
        "stage": "idea_compile",
        "card_id": refined.card_id,
        "pair": list(refined.pair),
        "claim": refined.claim,
        "mechanism": refined.mechanism,
        "prediction": refined.prediction,
        "world_lever": refined.world_lever,
        "claim_spec": getattr(refined, "claim_spec", None),
        "reason": "compile",
    }
    prior = dict(prior_for_diag or {})
    if notes:
        prior["design_notes"] = list(notes)[:6]
    prior["compile_refine"] = True
    try:
        rewritten = write_diagnosis(
            client,
            refined,
            topic,
            prior_probe=prior,
            skills=diagnose_skills,
            world=world,
            works=works,
            program=lines,
            requirement=world_requirement,
            dynamics=dynamics_card,
            world_path=world_path,
            scientific=scientific_for(
                card=refined, world=world, workspace=workspace
            ),
        )
    except GenerationRefused as exc:
        yield {
            "stage": "diagnosis_refused",
            "card_id": refined.card_id,
            "duty": "compile",
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
    sink["diagnosis"] = rewritten
    event = rewritten.to_dict()
    event["stage"] = "diagnosis"
    event["attempt"] = 0
    event["compile_refined"] = True
    yield event


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
    lock_world: bool = False,
    dynamics: dict[str, Any] | None = None,
    world_sim: bool = True,
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
    def _progress(stage: str, status: str, **extra: Any) -> None:
        append_progress(
            Path(workspace) if workspace is not None else None,
            card.card_id,
            {"stage": stage, "status": status, **extra},
        )

    compile_row = getattr(card, "claim_spec", None)
    compile_row = compile_row if isinstance(compile_row, dict) else None
    kind = idea_kind_of(card)
    yield {
        "stage": "idea_kind",
        "card_id": card.card_id,
        "idea_kind": kind,
        "runs_probe": runs_probe(kind),
    }
    if should_queue_world_expansion(compile_row) or kind == ACQUIRE:
        expansion = world_expansion_record(card, compile_row)
        expansion["idea_kind"] = kind
        enqueue_world_expansion(
            Path(workspace) if workspace is not None else None, expansion
        )
        _progress(
            "world_expansion",
            "queued",
            mechanism=expansion.get("mechanism"),
            missing_observables=expansion.get("missing_observables"),
            idea_kind=kind,
        )
        yield {"stage": "world_expansion", **expansion}
        if bundle is not None:
            bundle["world_expansion"] = expansion
        if not in_world_testable(compile_row) and skip_writeup_on_needs_world(kind):
            if bundle is not None:
                bundle["skip_writeup"] = True
            return
    elif compile_row and not in_world_testable(compile_row):
        yield {
            "stage": "compile_rejected",
            "card_id": card.card_id,
            "disposition": compile_disposition_of(compile_row),
            "reasons": list(compile_row.get("compile_reasons") or []),
        }
        if bundle is not None:
            bundle["skip_writeup"] = True
        return

    works: list = []
    if feed is not None and hasattr(feed, "survey_around"):
        yield {"stage": "surveying", "card_id": card.card_id, "pair": list(card.pair)}
        harvest = harvest_survey(
            feed,
            card.pair[0],
            card.pair[1],
            claim=card.claim,
            topic=topic,
        )
        event = {"stage": "survey", "card_id": card.card_id, **harvest.to_dict()}
        if harvest.blocked is not None and not harvest.works:
            yield event
        else:
            works = list(harvest.works)
            yield event
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

    if not runs_probe(kind):
        if bundle is not None:
            bundle["works"] = works
            bundle["prior"] = prior
            bundle["skip_writeup"] = False
            bundle["skip_probe"] = True
        if kind == ACQUIRE:
            record = validate_acquire_recipe(propose_acquire(card))
            persist_acquire(Path(workspace) if workspace is not None else None, record)
            harvested = None
            if (
                world is not None
                and workspace is not None
                and record.state == "RECIPE_VALIDATED"
            ):
                child, harvested = execute_acquire_harvest(
                    Path(workspace), card, world
                )
                if child is not None:
                    yield {
                        "stage": "world_transition",
                        "card_id": card.card_id,
                        "parent_world": str(getattr(world, "id", "") or ""),
                        "child_world": str(getattr(child, "id", "") or ""),
                        "child_version": str(getattr(child, "version", "") or ""),
                        "child_evidence_id": str(
                            getattr(child, "world_evidence_id", "") or ""
                        ),
                    }
            if harvested is None:
                book = load_notebook(Path(workspace) if workspace is not None else None)
                book.record_idea(
                    card,
                    status="proposed" if record.state != "RECIPE_VALIDATED" else "entered",
                    epistemic="GENERATED",
                    extra={
                        "capability_gap": record.capability_gap,
                        "acquire_state": record.state,
                    },
                )
                save_notebook(Path(workspace) if workspace is not None else None, book)
            yield {
                "stage": "acquire_lifecycle",
                "card_id": card.card_id,
                "idea_kind": kind,
                "acquire": (harvested or record).to_dict(),
                "evidence_role": typed_evidence_role(kind),
            }
            return
        if kind == QUESTION:
            book = load_notebook(Path(workspace) if workspace is not None else None)
            book.record_idea(card, status="registered", epistemic="GENERATED")
            save_notebook(Path(workspace) if workspace is not None else None, book)
            yield {
                "stage": "question_registered",
                "card_id": card.card_id,
                "idea_kind": kind,
                "diagnose_goal": diagnose_goal(kind),
                "evidence_role": typed_evidence_role(kind),
            }
            return
        if kind == THEORY:
            book = load_notebook(Path(workspace) if workspace is not None else None)
            book.record_idea(card, status="registered", epistemic="GENERATED")
            save_notebook(Path(workspace) if workspace is not None else None, book)
            yield {
                "stage": "theory_registered",
                "card_id": card.card_id,
                "idea_kind": kind,
                "diagnose_goal": diagnose_goal(kind),
                "evidence_role": typed_evidence_role(kind),
            }
            return
        yield {
            "stage": "idea_parked",
            "card_id": card.card_id,
            "idea_kind": kind,
            "runs_probe": False,
            "reason": (
                "this kind is a live idea; remaining work is not a two-arm "
                "on this freeze"
            ),
        }
        return

    probe_payload: dict[str, Any] | None = None
    verdict: str | None = None
    diagnosis = None
    extra_rounds = max(0, int(experiment_rounds))
    attempts_log: list[dict[str, Any]] = []
    discarded: list[str] = []
    sim_notes: list[str] = []
    compiled_once = False
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
            if bundle is not None:
                bundle["world_path"] = world_path.to_dict()
            if not isinstance(world_requirement, dict):
                world_requirement = {}
            world_requirement = {
                **world_requirement,
                **world_path.wishlist_fields(),
            }
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
                    if lock_world and world is not None:
                        yield {
                            "stage": "world_held",
                            "card_id": card.card_id,
                            "world_id": str(getattr(world, "id", "") or ""),
                            "named_instance": world_path.named_instance,
                            "reason": reason,
                            "honesty": (
                                "this line stays on the bound freeze; a "
                                "different published instance is a wishlist, "
                                "not a substitute"
                            ),
                        }
                    else:
                        yield {
                            "stage": "world_surrogate",
                            "card_id": card.card_id,
                            "world_id": str(getattr(world, "id", "") or ""),
                            "reason": reason,
                        }
                        world = None

    idea_world = None
    try:
        idea_world = analyze_idea_world(
            client,
            card,
            topic,
            paper_pool,
            lineage=world_path,
        )
    except Exception:
        idea_world = None
    if idea_world is not None:
        yield {
            "stage": "idea_world",
            "card_id": card.card_id,
            **idea_world.to_dict(),
        }
        if bundle is not None:
            bundle["idea_world"] = idea_world.to_dict()

    # Construct from this mission's topic + retrieved papers. A catalog
    # cousin is never the default experimental world. Construction
    # cannot corroborate.
    dynamics_card: dict[str, Any] | None = None
    if not isinstance(world_requirement, dict):
        world_requirement = {}
    if not (world_requirement.get("object_type") or world_requirement.get("schema")):
        inferred = infer_requirement(topic)
        if inferred is None:
            # The far concept supplies a mechanism, not a field: its own
            # words are stripped before the claim may vote on a schema,
            # or `feedback edge set` welds a code-world claim into an
            # undirected graph.
            far_label = card.pair[1] if len(card.pair) > 1 else ""
            inferred = infer_requirement(
                topic,
                without_far_terms(getattr(card, "claim", ""), far_label, topic),
                without_far_terms(getattr(card, "prediction", ""), far_label, topic),
            )
        if inferred is not None:
            world_requirement = {**inferred.to_dict(), **world_requirement}
    # One mission, one schema. The task world built from the topic at
    # mission start is the family every card's construction must keep;
    # a lineage's freeze recipe or a card's prose may wish for another
    # family, and that wish goes to the wishlist, not into world_dest.
    held = _hold_task_schema(world, world_requirement, world_path)
    if held is not None:
        world_requirement = held["requirement"]
        yield {
            "stage": "world_schema_held",
            "card_id": card.card_id,
            "schema": held["schema"],
            "requested": held["requested"],
            "source": held["source"],
            "honesty": (
                "the mission world keeps the topic's schema; a different "
                "family is a freeze wish for the host, not a construction"
            ),
        }
    freeze_row = child_probe_freeze(
        card, load_world_lineage(Path(workspace) if workspace is not None else None)
    )
    if freeze_row and workspace is not None:
        from .lifecycle import freeze_dir_for

        world_dest = freeze_dir_for(
            Path(workspace),
            str(freeze_row.get("world_id") or ""),
            version=str(freeze_row.get("version") or ""),
        )
        yield {
            "stage": "world_version_freeze",
            "card_id": card.card_id,
            "world_id": freeze_row.get("world_id"),
            "parent_world": freeze_row.get("parent_world"),
            "evidence_id": freeze_row.get("evidence_id"),
            "version": freeze_row.get("version"),
        }
    if world_dest is None:
        world_dest = _mission_world_dir(workspace) or _world_work_dir(
            workspace, card.card_id
        )
    sim_world = world
    world_req_text = str(world_request or "").strip().lower()
    want_construct = world_req_text not in {"", "synthetic"}
    paper_ground = (
        world_path is not None
        and want_construct
        and not attested_freeze(sim_world)
    )
    need_construct = sim_world is None or paper_ground
    if (
        run_probes
        and need_construct
        and want_construct
        and in_mission_constructable(world_requirement)
        and world_dest is not None
    ):
        previous = None
        if paper_ground and sim_world is not None:
            try:
                previous = load_world_payload(sim_world.root)
            except Exception:
                previous = None
        constructed, built = _construct_iteration_world(
            client,
            card,
            world_requirement,
            Path(world_dest),
            diagnosis=None,
            previous=previous,
            lineage=world_path,
            topic=topic,
            idea_world=idea_world,
        )
        for event in built:
            yield event
        if constructed is not None and str(
            getattr(constructed, "role", "") or ""
        ) == "generated":
            payload = load_world_payload(constructed.root)
            if payload.get("acquisition") != "incomplete":
                sim_world = constructed
                if bundle is not None:
                    bundle["world"] = constructed
                if workspace is not None:
                    _write_world_readme(
                        Path(workspace), constructed, honesty=HONESTY
                    )
    if run_probes and sim_world is not None and compose_simulation(sim_world).get("available"):
        same_bound = (
            dynamics is not None
            and str(getattr(sim_world, "id", "") or "")
            == str((dynamics or {}).get("world_id") or "")
        )
        if same_bound:
            dynamics_card = dict(dynamics)
        else:
            try:
                dynamics_card = response_card(sim_world)
            except Exception:
                dynamics_card = None
        if dynamics_card is not None:
            plan = compose_simulation(sim_world)
            if not same_bound:
                yield {
                    "stage": "world_scout",
                    "card_id": card.card_id,
                    "world_id": dynamics_card["world_id"],
                    "origin": plan.get("origin") or "",
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

    def _adapt_bound_world(
        reasons: list[str], diagnosis_now: Any
    ) -> Iterator[dict[str, Any]]:
        """Adapt the constructed world in place for the same registration.

        `decide_retry` has said `adapt_world` since the loop existed; this
        is where the verb finally acts. Only a GENERATED world may be
        rewritten — an attested freeze never takes this path.
        """
        nonlocal sim_world, dynamics_card
        if sim_world is None or str(getattr(sim_world, "role", "") or "") != "generated":
            return
        try:
            previous_payload = load_world_payload(sim_world.root)
        except Exception:
            previous_payload = None
        constructed, built = _construct_iteration_world(
            client,
            card,
            world_requirement,
            Path(sim_world.root),
            diagnosis=diagnosis_now,
            previous=previous_payload,
            errors=reasons,
            adapted=True,
            lineage=world_path,
            topic=topic,
            idea_world=idea_world,
        )
        for evt in built:
            yield evt
        if constructed is not None and str(
            getattr(constructed, "role", "") or ""
        ) == "generated":
            fresh = load_world_payload(constructed.root)
            if fresh.get("acquisition") != "incomplete":
                sim_world = constructed
                if bundle is not None:
                    bundle["world"] = constructed
                try:
                    dynamics_card = (
                        response_card(constructed)
                        if compose_simulation(constructed).get("available")
                        else None
                    )
                except Exception:
                    dynamics_card = None

    if run_probes and runs_probe(kind):
        for attempt in range(extra_rounds + 1):
            inert_forecast = False
            skip_probe = False
            active_dyn = world if world is not None else sim_world
            generated_dyn = (
                str(getattr(active_dyn, "role", "") or "") == "generated"
            )
            need_new_diagnosis = diagnosis is None or last_bottleneck == "method"
            if need_new_diagnosis:
                _progress("diagnosing", "started", attempt=attempt)
                yield {
                    "stage": "diagnosing",
                    "card_id": card.card_id,
                    "attempt": attempt,
                }
                try:
                    for margin_try in range(3):
                        diagnosis = write_diagnosis(
                            client,
                            card,
                            topic,
                            prior_probe=prior_for_diag,
                            failed_experiments=discarded,
                            skills=diagnose_skills,
                            world=world if world is not None else sim_world,
                            works=works,
                            program=program,
                            requirement=world_requirement if world is None else None,
                            dynamics=dynamics_card,
                            world_path=world_path,
                            scientific=scientific_for(
                                card=card,
                                world=world if world is not None else sim_world,
                                workspace=workspace,
                            ),
                        )
                        # The schema's compute floor is applied here, before
                        # registration and before any probe: a history schema
                        # is measured at history scale, not at toy scale.
                        floor_world = world if world is not None else sim_world
                        floored = floor_tier(
                            diagnosis.compute_tier,
                            str(getattr(floor_world, "schema", "") or ""),
                        )
                        if floored != diagnosis.compute_tier:
                            diagnosis = replace(diagnosis, compute_tier=floored)
                            yield {
                                "stage": "tier_floor",
                                "card_id": card.card_id,
                                "attempt": attempt,
                                "schema": str(getattr(floor_world, "schema", "") or ""),
                                "compute_tier": floored,
                                "detail": (
                                    "pre-registered before any probe: the schema's "
                                    "object is a replayed history, so its tier floor "
                                    f"is {floored}"
                                ),
                            }
                        # Margin plausibility: the scout already measured how far
                        # this lever can move this observable on this world. A
                        # margin above that ceiling pre-announces "uninformative"
                        # and would cost a host-heavy probe to confirm it; refuse
                        # the registration and ask for a margin the world can pay.
                        # The retry is immediate and does not spend an attempt.
                        ceiling = _response_ceiling(dynamics_card, diagnosis)
                        if (
                            ceiling is not None
                            and float(diagnosis.margin or 0.0) > ceiling + 1e-9
                            and margin_try < 2
                        ):
                            yield {
                                "stage": "margin_refused",
                                "card_id": card.card_id,
                                "attempt": attempt,
                                "world_lever": diagnosis.world_lever,
                                "world_observable": diagnosis.world_observable,
                                "margin": diagnosis.margin,
                                "ceiling": round(ceiling, 6),
                                "detail": (
                                    "implausible_margin: the scout moved "
                                    f"{diagnosis.world_observable} by at most "
                                    f"{ceiling:+.1%} under {diagnosis.world_lever} on this "
                                    f"world; a pre-registered margin of {diagnosis.margin} "
                                    "cannot be met and would only buy a certain uninformative"
                                ),
                            }
                            if diagnosis.experiment and diagnosis.experiment not in discarded:
                                discarded.append(diagnosis.experiment)
                            prior_for_diag = _redesign_prior(
                                diagnosis,
                                verdict="uninformative",
                                dynamics_card=dynamics_card,
                                design_notes=[
                                    "implausible_margin: pre-register a margin no larger than "
                                    f"{ceiling:.3f} (the scout's response of {diagnosis.world_lever} "
                                    f"on {diagnosis.world_observable}), or compile onto a "
                                    "lever/observable the scout shows moving further; the effect "
                                    "the claim promises must be one this world can show"
                                ],
                            )
                            continue
                        break
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
                _progress("diagnosis", "finished", attempt=attempt)
                yield event

                if dynamics_card is not None and active_dyn is not None:
                    compiled = str(getattr(card, "world_lever", "") or "").strip()
                    lever = diagnosis.world_lever or compiled or "none"
                    # `none` is the honest skip on a freeze AND on a
                    # constructed world: running three probes against a
                    # fixture the mechanism cannot touch produces 0/0
                    # theatre, not evidence.
                    if lever == "none" and (
                        attested_freeze(world) if world is not None else generated_dyn
                    ):
                        analysis = {
                            "stop_reason": "no_handle",
                            "room_to_move": False,
                            "n_steps": 0,
                            "summary": (
                                f"On {getattr(active_dyn, 'id', '') or 'this object'}, "
                                "the registered prediction has no runtime handle; "
                                "the idea cannot be executed here."
                            ),
                        }
                        yield {
                            "stage": "probe_skipped",
                            "card_id": card.card_id,
                            "world_id": str(getattr(active_dyn, "id", "") or ""),
                            "reason": (
                                "the registered mechanism has no lever in this "
                                "world's runtime dynamics; skip the probe rather "
                                "than invent data"
                            ),
                            "bottleneck": "no_handle",
                            "analysis": analysis,
                        }
                        last_bottleneck = classify_bottleneck(refused="no_handle")
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
                        skip_probe = True
                    else:
                        # The prediction check: roll the world forward under
                        # the registered lever before the experiment exists.
                        # An analysis aid on the record, never a verdict.
                        # A GENERATED world rehearses the same way — its
                        # forecast still cannot corroborate anything.
                        try:
                            forecast = forward_simulate(active_dyn, lever)
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

            if (
                diagnosis is not None
                and attempt == 0
                and world_sim
                and workspace is not None
            ):
                extra = {
                    "world_id": str(getattr(world if world is not None else sim_world, "id", "") or ""),
                    "world_schema": str(
                        getattr(world if world is not None else sim_world, "schema", "") or ""
                    ),
                    "probe_kind": (
                        KIND_GENERATED
                        if str(getattr(sim_world, "role", "") or "") == "generated"
                        else KIND_WORLD
                        if world is not None
                        else KIND_SYNTHETIC
                    ),
                }
                for sim_event in rehearse_entered_idea(
                    workspace=Path(workspace),
                    topic=topic,
                    card=card,
                    diagnosis=diagnosis,
                    enabled=True,
                    client=client,
                    wiki=list(works) + list(wiki_works_pool or []),
                    world_lever=str(getattr(diagnosis, "world_lever", "") or ""),
                    schema=str(extra.get("world_schema") or ""),
                    scout=dynamics_card if isinstance(dynamics_card, dict) else None,
                    extra=extra,
                ):
                    yield sim_event
                    for item in sim_event.get("issues") or []:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") != "structural":
                            continue
                        note = str(item.get("summary") or "").strip()
                        if note and note not in sim_notes:
                            sim_notes.append(note)

            if (
                diagnosis is not None
                and attempt == 0
                and workspace is not None
                and sim_notes
                and not compiled_once
            ):
                compiled_once = True
                decision = decide_compile_refine(
                    attempt=0, reasons=["structural"]
                )
                yield {
                    "stage": "idea_decision",
                    "card_id": card.card_id,
                    "attempt": 0,
                    "pair": list(card.pair),
                    **decision,
                }
                if decision["continue"]:
                    compiled: dict[str, Any] = {}
                    for compile_event in _compile_entered_pair(
                        client,
                        card,
                        topic=topic,
                        diagnosis=diagnosis,
                        notes=sim_notes,
                        program=program,
                        works=works,
                        wiki_works_pool=wiki_works_pool,
                        skill_blocks=skill_blocks,
                        dynamics_card=dynamics_card,
                        world=world if world is not None else sim_world,
                        prior_for_diag=prior_for_diag,
                        diagnose_skills=diagnose_skills,
                        world_requirement=world_requirement if world is None else None,
                        world_path=world_path,
                        sink=compiled,
                        workspace=workspace,
                    ):
                        yield compile_event
                        if compile_event.get("stage") == "blocked":
                            return
                    if compiled.get("card") is not None:
                        card = compiled["card"]
                        if bundle is not None:
                            bundle["card"] = card
                    if compiled.get("diagnosis") is not None:
                        diagnosis = compiled["diagnosis"]

            if diagnosis is None:
                break
            if skip_probe:
                break
            if inert_forecast:
                last_bottleneck = "method"
                if diagnosis.experiment and diagnosis.experiment not in discarded:
                    discarded.append(diagnosis.experiment)
                prior_for_diag = _redesign_prior(
                    diagnosis,
                    verdict="uninformative",
                    dynamics_card=dynamics_card,
                    must_switch_mechanism=switch_mechanism_now(
                        uninformative_runs=sum(
                            1
                            for item in attempts_log
                            if item.get("verdict") == "uninformative"
                        )
                        + 1,
                        probe_kind=(
                            KIND_WORLD
                            if world is not None
                            else KIND_GENERATED
                            if generated_dyn
                            else KIND_SYNTHETIC
                        ),
                    ),
                    design_notes=sim_notes,
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
            probe_world = _active_probe_world(world, sim_world)
            if probe_world is None and not allow_synthetic:
                constructed = (
                    sim_world is not None
                    and str(getattr(sim_world, "role", "") or "") == "generated"
                )
                analysis = {
                    "stop_reason": "no_object",
                    "room_to_move": bool(
                        constructed
                        and dynamics_card
                        and any(
                            abs(float(delta)) >= RESPONSE_FLOOR
                            for resp in (dynamics_card.get("levers") or {}).values()
                            for delta in resp.values()
                        )
                    ),
                    "n_steps": 0,
                    "summary": (
                        (
                            f"No attested freeze; a same-family GENERATED "
                            f"world {getattr(sim_world, 'id', '')} was "
                            "scouted for idea analysis and cannot corroborate."
                        )
                        if constructed
                        else (
                            "No attested freeze matches this claim; inventing "
                            "data is not verification. A novel of another "
                            "field is not a playground."
                        )
                    ),
                }
                yield {
                    "stage": "probe_skipped",
                    "card_id": card.card_id,
                    "reason": (
                        "no attested freeze is bound and no same-family "
                        "world could be constructed; skip rather than "
                        "invent a substitute"
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
            _progress("probe_execute", "started", attempt=attempt)
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
                    world=probe_world,
                    scientific=scientific_for(
                        card=card, world=probe_world, workspace=workspace
                    ),
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
                        "world_digest": str(getattr(probe_world, "digest", "") or ""),
                    },
                )

            result = _run_probe_attempt(
                spec,
                workspace,
                card.card_id,
                attempt,
                world=probe_world,
                tier_name=str(getattr(diagnosis, "compute_tier", "") or ""),
            )
            probe_payload = result.to_dict()
            probe_payload["attempt"] = attempt
            probe_payload["experiment_digest"] = sha256_text(spec.source)
            if probe_world is not None:
                probe_payload["world_digest"] = probe_world.digest
                probe_payload["data_digest"] = probe_world.digest
                probe_payload["evidence_id"] = evidence_id(
                    claim_id=card.card_id,
                    experiment_digest=probe_payload["experiment_digest"],
                    world_digest=probe_world.digest,
                    data_digest=probe_world.digest,
                )
            if diagnosis is not None and ablation_required(
                diagnosis.alternative,
                diagnosis.experiment,
                levers=levers_of(probe_world) if probe_world is not None else (),
                world_lever=diagnosis.world_lever,
            ):
                probe_payload["ablation_required"] = True
                probe_payload["mechanism_identified"] = False
            probe_payload["kind"] = _probe_kind_for(probe_world, spec.source)
            probe_payload["world_has_dynamics"] = bool(
                probe_world is not None and has_dynamics(probe_world)
            )
            event = dict(probe_payload)
            event["stage"] = "probe"
            event["card_id"] = card.card_id
            yield event
            if result.status != "ran":
                last_bottleneck = classify_bottleneck(
                    status=result.status,
                    error=result.error,
                    world_role=str(getattr(active_dyn, "role", "") or ""),
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
                if last_action == "adapt_world":
                    for adapt_event in _adapt_bound_world(
                        [f"probe crashed on this world: {result.error}"[:400]],
                        diagnosis,
                    ):
                        yield adapt_event
                continue

            evidence = judge_probe(diagnosis, result.treatment, result.control)
            spec_payload = getattr(card, "claim_spec", None)
            if isinstance(spec_payload, dict):
                try:
                    projection = project_evidence_to_claim(
                        claim_spec_from_payload(spec_payload),
                        arithmetic_verdict=str(evidence.get("verdict") or ""),
                        observed_effect={
                            "treatment": evidence.get("treatment"),
                            "control": evidence.get("control"),
                            "separation": evidence.get("separation"),
                        },
                        metric=str(
                            getattr(diagnosis, "world_observable", "")
                            or getattr(card, "world_observable", "")
                            or ""
                        ),
                    )
                    evidence["scoped_verdict"] = projection.record.verdict.value
                    evidence["supports_scope"] = list(projection.record.supports_scope)
                    evidence["prohibited_inferences"] = list(
                        projection.record.prohibited_inferences
                    )
                    evidence["evidence_projection"] = projection.scientific_text()
                    evidence["claim_spec"] = spec_payload
                except ContractViolation:
                    pass
            probe_payload.update(evidence)
            _progress(
                "probe_execute",
                "finished",
                attempt=attempt,
                scoped_verdict=str(evidence.get("scoped_verdict") or ""),
            )
            verdict = str(evidence.get("verdict") or "") or None
            # DV sanity: a claim about an oracle-relative quantity (false
            # acceptance, task success, label) must have a measure that
            # responds to the oracle. Permute the oracle field alone and
            # rerun; if neither arm moves, the measure is blind to its
            # object. That is object_absent — it does not spend the
            # experiment budget as "uninformative", it cannot make the plan
            # executable, and it cannot be a supports.
            dv_probe_world = probe_world if probe_world is not None else world
            if (
                dv_probe_world is not None
                and oracle_field(str(getattr(dv_probe_world, "schema", "") or "")) is not None
                and oracle_dependent(
                    card.claim,
                    getattr(diagnosis, "experiment", ""),
                    getattr(diagnosis, "treatment_arm", ""),
                    getattr(diagnosis, "control_arm", ""),
                )
            ):
                dv_dir = (
                    candidate_dir(Path(workspace), card.card_id)
                    / "probe"
                    / f"attempt-{attempt}-oracle-placebo"
                    if workspace is not None
                    else Path(tempfile.mkdtemp(prefix="ff-oracle-placebo-"))
                )
                dv_record: dict[str, Any] = {
                    "stage": "dv_sanity",
                    "card_id": card.card_id,
                    "attempt": attempt,
                    "treatment": result.treatment,
                    "control": result.control,
                }
                try:
                    placebo_fixture = materialize_oracle_placebo(
                        dv_probe_world, dv_dir / "raw"
                    )
                    if placebo_fixture is None:
                        dv_record["skipped"] = "oracle field has fewer than two values"
                    else:
                        dv_dest = dv_dir / "run"
                        dv_dest.mkdir(parents=True, exist_ok=True)
                        bind_world(dv_dest, placebo_fixture)
                        # Same budget as the real run: a host-heavy script
                        # given 20s here only ever reports "timeout".
                        dv_tier = resolve_tier(str(getattr(diagnosis, "compute_tier", "") or ""))
                        dv_run = run_probe(
                            spec,
                            dv_dest,
                            timeout_seconds=probe_timeout(dv_tier),
                            tier=dv_tier,
                        )
                        dv_record["placebo_status"] = dv_run.status
                        if dv_run.status == "ran":
                            dv_record["placebo_treatment"] = dv_run.treatment
                            dv_record["placebo_control"] = dv_run.control
                            dv_record["blind"] = dv_blind(
                                (float(result.treatment), float(result.control)),
                                (float(dv_run.treatment), float(dv_run.control)),
                            )
                except Exception as exc:  # the check must not lose the run
                    dv_record["error"] = str(exc)[:300]
                yield dv_record
                if dv_record.get("blind"):
                    verdict = "uninformative"
                    evidence["verdict"] = verdict
                    evidence["dv_blind"] = True
                    evidence["reason"] = blind_reason(
                        str(getattr(dv_probe_world, "schema", "") or "")
                    )
                    probe_payload.update(evidence)
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
                # One permutation is a noisy draw: on the derived Live-SWE
                # world a single shuffle left a -0.17 "separation" next to a
                # real -0.20 and killed a genuine contrast. The placebo is a
                # small distribution (PLACEBO_SEEDS draws) and the real
                # separation is compared with its mean; each draw runs under
                # the same tier budget as the probe.
                placebo_seps: list[float] = []
                placebo_run = None
                placebo = None
                placebo_tier = resolve_tier(str(getattr(diagnosis, "compute_tier", "") or ""))
                try:
                    for placebo_seed in range(PLACEBO_SEEDS):
                        raw = placebo_dir / f"raw-{placebo_seed}"
                        dest = placebo_dir / f"run-{placebo_seed}"
                        dest.mkdir(parents=True, exist_ok=True)
                        placebo = materialize_placebo(world, raw, seed=placebo_seed)
                        bind_world(dest, placebo)
                        placebo_run = run_probe(
                            spec, dest, timeout_seconds=probe_timeout(placebo_tier), tier=placebo_tier
                        )
                        if placebo_run.status != "ran":
                            break
                        placebo_seps.append(
                            arm_separation(float(placebo_run.treatment), float(placebo_run.control))
                        )
                except Exception as exc:
                    placebo_run = None
                    probe_payload["world_consumed"] = False
                    probe_payload["placebo_error"] = str(exc)
                if placebo_run is not None and placebo_run.status == "ran" and placebo_seps:
                    sep_placebo = sum(placebo_seps) / len(placebo_seps)
                    consumed = world_consumed(
                        sep_real, sep_placebo, diagnosis.margin
                    )
                    probe_payload["world_consumed"] = consumed
                    probe_payload["placebo_separation"] = round(sep_placebo, 6)
                    probe_payload["placebo_separations"] = [round(v, 6) for v in placebo_seps]
                    yield {
                        "stage": "world_placebo",
                        "card_id": card.card_id,
                        "consumed": consumed,
                        "separation": round(sep_real, 6),
                        "placebo_separation": round(sep_placebo, 6),
                        "placebo_separations": [round(v, 6) for v in placebo_seps],
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
                        evidence["world_consumed"] = False
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
                    evidence["world_consumed"] = False
                    evidence["reason"] = (
                        "placebo control failed; consumption cannot be attested"
                    )
                    probe_payload.update(evidence)
            # Two arms that both measured exactly zero never met the
            # claim's object: the fixture is missing the instance, the
            # design is not sharp or dull. On a GENERATED world that is a
            # world_model bottleneck (adapt the same world in place); on a
            # freeze it stays a method problem.
            object_absent = bool(evidence.get("dv_blind")) or (
                verdict == "uninformative"
                and float(evidence.get("treatment") or 0.0) == 0.0
                and float(evidence.get("control") or 0.0) == 0.0
            )
            if object_absent:
                evidence["object_absent"] = True
                probe_payload["object_absent"] = True
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
                world_role=str(getattr(active_dyn, "role", "") or ""),
                last_action=last_action,
                object_absent=object_absent,
            )
            row = {
                "attempt": attempt,
                "bottleneck": last_bottleneck,
                "status": result.status,
                "verdict": verdict,
                "experiment": diagnosis.experiment,
                "treatment": evidence.get("treatment"),
                "control": evidence.get("control"),
                # The next diagnosis reads this: a blind measure is a
                # design fault to fix, not a null result to sharpen.
                "error": str(evidence.get("reason") or "") if evidence.get("dv_blind") else "",
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
                # A blind measure is not an uninformative design: it does
                # not vote for switching the mechanism, and the redesign is
                # told exactly what to fix.
                redesign_notes = list(sim_notes or [])
                if evidence.get("dv_blind"):
                    redesign_notes = [str(evidence.get("reason") or "")] + redesign_notes
                prior_for_diag = _redesign_prior(
                    diagnosis,
                    verdict="uninformative",
                    dynamics_card=dynamics_card,
                    must_switch_mechanism=switch_mechanism_now(
                        uninformative_runs=sum(
                            1
                            for item in attempts_log
                            if item.get("verdict") == "uninformative"
                            and "dv_blind" not in str(item.get("error") or "")
                        ),
                        probe_kind=str(
                            (probe_payload or {}).get("kind") or KIND_SYNTHETIC
                        ),
                    ),
                    design_notes=redesign_notes,
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
            if last_action == "adapt_world":
                claim_head = str(getattr(card, "claim", "") or "")[:200]
                for adapt_event in _adapt_bound_world(
                    [
                        "both arms measured exactly 0: the claimed object "
                        "is absent from this constructed world. Add real "
                        "instances of the quantities the claim names "
                        f"({claim_head}) so the registered two-arm can "
                        "count something; do not add a verdict."
                    ],
                    diagnosis,
                ):
                    yield adapt_event

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
        world=world if world is not None else sim_world,
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
            scientific=scientific_for(
                card=card, world=world, workspace=workspace
            ),
        )
    except GenerationRefused as exc:
        yield {
            "stage": "brief_refused",
            "card_id": card.card_id,
            "missing_capability": exc.record.missing_capability,
            "unlock_condition": exc.record.unlock_condition,
            "compiled_fallback": True,
        }
        brief = compile_brief(card, works, diagnosis=diagnosis, topic=topic)
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
            gate_note = None
            if probe_payload.get("dv_blind"):
                gate_note = "dv_blind"
            elif probe_payload.get("world_consumed") is False:
                gate_note = "placebo_fail"
            elif probe_payload.get("object_absent"):
                gate_note = "object_absent"
            if gate_note and host_record.get("verdict"):
                host_record = {
                    **host_record,
                    "host_verdict_raw": host_record.get("verdict"),
                    "verdict": str(probe_payload.get("verdict") or "uninformative"),
                    "gate_blocked": gate_note,
                    "honesty": (
                        f"the in-loop {gate_note} gate lowered this probe; the host rerun "
                        "reproduces the same bytes and cannot lift it"
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
                "description": minted.description,
                "body": minted.body,
                "digest": minted.digest,
                "stages": list(minted.stages),
                "paths": written,
                "executable": False,
            }


HEAVY_REPLICATES = 5
# Structure-placebo draws per supports; the real separation is compared with their mean.
PLACEBO_SEEDS = 5


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
            review = review_idea(
                client,
                card,
                current,
                topic,
                works,
                scientific=scientific_for(card=card),
            )
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


def execute_acquire_harvest(
    workspace: Path,
    card: GeneratedCard,
    parent_world: Any,
    *,
    dest: Path | None = None,
    new_id: str = "",
) -> tuple[Any, Any]:
    """In-mission or next-mission harvest. Creates W1; does not rewrite W0."""
    record = validate_acquire_recipe(propose_acquire(card))
    if record.state != "RECIPE_VALIDATED":
        persist_acquire(workspace, record)
        return None, record
    child_dir = Path(dest or (Path(workspace) / "worlds" / (new_id or f"{parent_world.id}-acq")))
    child, attested = harvest_acquired_world(
        parent_world,
        child_dir,
        record,
        new_id=new_id or f"{parent_world.id}-acq-{card.card_id[:8]}",
        parent_version=str(getattr(parent_world, "version", "") or "W0"),
    )
    bound = bind_acquired_world(workspace, child, attested)
    book = load_notebook(workspace)
    book.record_idea(
        card,
        status="attested",
        epistemic="WORLD",
        extra={
            "world_id": bound.world_id,
            "evidence_id": bound.evidence_id,
            "capability_gap": bound.capability_gap,
            "causal_ready": True,
        },
    )
    save_notebook(workspace, book)
    return child, bound


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
        claim_spec=row.get("claim_spec") if isinstance(row.get("claim_spec"), dict) else None,
        idea_kind=idea_kind_of(row),
        lineage_id=str(row.get("lineage_id") or ""),
        origin=str(row.get("origin") or ""),
        research_target=str(row.get("research_target") or ""),
        parent_idea_id=str(row.get("parent_idea_id") or ""),
        transition_reason=str(row.get("transition_reason") or ""),
        world_version=str(row.get("world_version") or ""),
        kind_fields=row.get("kind_fields") if isinstance(row.get("kind_fields"), dict) else None,
    )


def brief_from_dict(row: dict[str, Any], works: list[FreshWork]) -> ResearchBrief:
    return ResearchBrief(
        card_id=str(row.get("card_id") or ""),
        title=str(row.get("title") or ""),
        plain_title=str(row.get("plain_title") or ""),
        one_liner=str(row.get("one_liner") or ""),
        why_it_matters=str(row.get("why_it_matters") or ""),
        gap=str(row.get("gap") or ""),
        idea=str(row.get("idea") or ""),
        approach=str(row.get("approach") or ""),
        first_steps=tuple(row.get("first_steps") or ()),
        baseline=str(row.get("baseline") or ""),
        risks=str(row.get("risks") or ""),
        read_first=tuple(row.get("read_first") or ()),
        papers=tuple(work.to_dict() for work in works),
    )
