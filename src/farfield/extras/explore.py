"""Two bounded loops: rewrite the same idea, or rewrite the same test.

Far-field exploration is trajectory-guided search in research space
(`researchspace.py`): recover how related work evolved, jump with that
prior, evaluate, then jump again. The jump cap and the spray stop still
bound how many landings open. Crossover of two parents is not on the
product path.

The spray stop is *plan executable*: this mission already produced a
brief + two-arm plan that an intern can run. A WORLD support, a
registered uninformative / SYNTHETIC support, or a neighbourhood line
that already has that plan, all stop further far jumps. Remaining work
is `farfield execute`, not another research generate.

A later landing does not inherit another idea's program. It reads the
exploration notebook: tried routes, recovered trajectories, and the
support/novelty/value/feasibility/distance feedback of earlier jumps.
Depth is in-mission idea refine and experiment iteration — not a
handoff that asks the next `farfield research` to become the scientist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .lifecycle import (
    attractor_key,
    design_signature,
    failure_mode_of,
    refine_action_for_failures,
)
from .researchspace import JumpFeedback, SearchLedger
from .world import is_world_kind


DEFAULT_EXPLORE = "auto"
DEFAULT_JUMP_CAP = 4
POLISH_AUTO_CAP = 2
IDEA_AUTO = -1
IDEA_AUTO_CAP = 2
COMPILE_REFINE_CAP = 1
ATTRACTOR_MISS_FLOOR = 2

# Graph pair deaths cannot be repaired by rewriting the claim. A new
# landing is the fallback. Text-judge deaths can: the pair is still legal.
GRAPH_PAIR_MARKS = (
    "already_combined",
    "implied_by",
    "hub",
    "shortcut",
    "graph",
    "pair_not_already_combined",
    "endpoint_is_not_a_concept_hub",
    "pair_not_implied_by_neighborhood",
)


@dataclass
class ExplorationPrior:
    """Search memory for this mission. Not another idea's program.

    Later jumps read tried pairs, leftover menu items, recovered
    trajectories' feedback, and the policy the last evaluation named
    (increase distance, tighten the trajectory, retrieve support, or
    reformulate). Unused menu items are information, not a commanded
    destination. A concept catalog is not this notebook.
    """

    used_pairs: list[tuple[str, str]] = field(default_factory=list)
    used_far: list[str] = field(default_factory=list)
    used_levers: list[str] = field(default_factory=list)
    unused_menu: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    object_phrases: tuple[str, ...] = ()
    how_found: dict[str, Any] | None = None
    lessons: list[dict[str, Any]] = field(default_factory=list)
    ledger: SearchLedger = field(default_factory=SearchLedger)
    # Typed design constraints this mission learned from its own probes
    # (blind measure, placebo-surviving separation, effect ceiling, inert
    # lever). Binding for every later card and diagnosis; a negative
    # result is information only if it changes what gets written next.
    constraints: list[str] = field(default_factory=list)
    # What the bound world showed before any card existed (exploratory
    # descriptives; cannot corroborate). Priors for what this world can
    # show, so claims are written about phenomena the data contains.
    world_facts: list[str] = field(default_factory=list)
    # Handles that failed twice on this freeze. Lethal at compile, not a
    # prompt hint: the next landing cannot weld onto an attractor.
    blocked_handles: list[str] = field(default_factory=list)
    handle_misses: dict[str, int] = field(default_factory=dict)

    def add_constraint(self, text: str) -> None:
        line = " ".join(str(text or "").split())
        if line and line not in self.constraints:
            self.constraints.append(line)

    def _handle_of(self, event: dict[str, Any], diagnosis: Any = None) -> str:
        lever = str(
            event.get("world_lever") or getattr(diagnosis, "world_lever", "") or ""
        ).strip()
        observable = str(
            event.get("world_observable")
            or getattr(diagnosis, "world_observable", "")
            or ""
        ).strip()
        if lever and observable:
            return f"{lever}/{observable}"
        return lever

    def _note_miss(
        self,
        handle: str,
        why: str,
        *,
        world_id: str = "",
        failure_mode: str = "",
        design_sig: str = "",
    ) -> None:
        name = str(handle or "").strip()
        if not name:
            return
        key = attractor_key(
            world_id=world_id,
            handle=name,
            failure_mode=failure_mode,
            design_sig=design_sig,
        )
        self.handle_misses[key] = self.handle_misses.get(key, 0) + 1
        if (
            self.handle_misses[key] >= ATTRACTOR_MISS_FLOOR
            and key not in self.blocked_handles
        ):
            self.blocked_handles.append(key)
            self.add_constraint(
                f"attractor: {key} {why}; equivalent designs on this world "
                "version are refused; a different design signature may retry"
            )

    def learn(self, event: dict[str, Any], diagnosis: Any = None) -> None:
        """Turn one evidence-side event into a constraint, when it carries one."""
        stage = str(event.get("stage") or "")
        if stage == "evidence":
            reason = str(event.get("reason") or "")
            handle = self._handle_of(event, diagnosis)
            mode = failure_mode_of(event)
            sig = design_signature(
                lever=str(event.get("world_lever") or ""),
                observable=str(event.get("world_observable") or ""),
                treatment=str(event.get("treatment") or getattr(diagnosis, "treatment_arm", "") or ""),
                control=str(event.get("control") or getattr(diagnosis, "control_arm", "") or ""),
                measure=str(event.get("measure") or ""),
                mechanism=str(event.get("mechanism") or ""),
                prediction=str(event.get("prediction") or ""),
            )
            world_id = str(event.get("world_id") or event.get("world_version") or "")
            if event.get("dv_blind") or event.get("object_absent"):
                self._note_miss(
                    handle,
                    "failed as object_absent/dv_blind twice",
                    world_id=world_id,
                    failure_mode=mode,
                    design_sig=sig,
                )
                if event.get("dv_blind"):
                    self.add_constraint(
                        "dv_blind happened: a measure of an oracle-relative quantity must READ the "
                        "oracle field (task_return) — define false acceptance as accepted-and-oracle-"
                        "failed; measures that ignore the oracle are refused as object_absent"
                    )
            elif "did not consume the bound world" in reason:
                self._note_miss(
                    handle,
                    "failed the placebo consume check twice",
                    world_id=world_id,
                    failure_mode=mode,
                    design_sig=sig,
                )
                self.add_constraint(
                    "placebo_fail happened: a script whose arm separation survives structure "
                    "destruction of the world is refused — the measure must depend on the "
                    "world's records, never on the arm flag alone"
                )
            elif str(event.get("verdict") or "") == "uninformative":
                self._note_miss(
                    handle,
                    "was uninformative twice",
                    world_id=world_id,
                    failure_mode=mode,
                    design_sig=sig,
                )
                sep = event.get("separation")
                margin = event.get("margin")
                if sep is not None and margin is not None:
                    self.add_constraint(
                        f"under_margin happened: arms separated by {float(sep):+.3f} against a "
                        f"pre-registered margin of {float(margin):.3f}; promise effects this "
                        "world can show (see the scout responses), not 10-25% by habit"
                    )
        elif stage == "margin_refused":
            self.add_constraint(
                f"effect ceiling: {event.get('world_lever')} moves {event.get('world_observable')} "
                f"by at most {float(event.get('ceiling') or 0.0):+.1%} on this world; margins above "
                "that are refused before any probe runs"
            )
        elif stage == "world_scout":
            inert = [str(item) for item in (event.get("inert") or [])]
            levers = [str(item) for item in (event.get("levers") or [])]
            dead = [
                lever
                for lever in levers
                if lever and all(
                    f"{lever}/{obs}" in inert for obs in (event.get("observables") or [])
                )
            ]
            if dead:
                self.add_constraint(
                    f"inert levers on this world: {', '.join(dead)} move no observable "
                    "(a replayed history cannot feel an intervention on recorded acceptances); "
                    "do not compile onto them"
                )
            contrasts = [lever for lever in levers if lever.startswith("contrast:")]
            if contrasts:
                self.add_constraint(
                    "this world is a replayed history: observational handles "
                    f"({', '.join(contrasts)}) are the ones that ask about its own dynamics; "
                    "prefer them over subsampling levers"
                )

    def constraint_lines(self) -> tuple[str, ...]:
        lines: list[str] = []
        if self.world_facts:
            lines.append(
                "Measured on the bound world before any card (exploratory descriptives; "
                "cannot corroborate; use as priors for what this world can show):"
            )
            lines.extend(f"- {fact}" for fact in self.world_facts[:8])
        if self.constraints:
            lines.append(
                "Design constraints learned this mission from its own probes (binding for "
                "every later card and diagnosis):"
            )
            lines.extend(f"- {item}" for item in self.constraints[:8])
        if self.blocked_handles:
            lines.append(
                "Blocked attractor handles (failed twice on this freeze; "
                "compile refuses them): " + ", ".join(self.blocked_handles[:8])
            )
        return tuple(lines)

    def absorb_landing(
        self,
        *,
        pair: tuple[str, str] | list[str] | None,
        far_menu: tuple[str, ...] | list[str] = (),
        category: str,
        operator: str = "",
        world_lever: str = "",
        world_observable: str = "",
        why: str = "",
        iterate: str = "",
    ) -> None:
        menu = [str(item) for item in far_menu if str(item).strip()]
        chosen: tuple[str, str] | None = None
        if pair and len(pair) >= 2:
            chosen = (str(pair[0]), str(pair[1]))
            if chosen not in self.used_pairs:
                self.used_pairs.append(chosen)
            if chosen[1] and chosen[1] not in self.used_far:
                self.used_far.append(chosen[1])
            lever = str(world_lever or "").strip()
            observable = str(world_observable or "").strip()
            # A handle is lever/observable. Two landings on the same lever
            # that measure different observables ask different questions;
            # a world with two levers must not cap breadth at two.
            handle = f"{lever}/{observable}" if lever and observable else lever
            if lever and lever != "none" and handle not in self.used_levers:
                self.used_levers.append(handle)
            line = f"{chosen[0]} × {chosen[1]} [{category}]"
            if line not in self.routes:
                self.routes.append(line)
            reason = str(why or "").strip()
            next_step = str(iterate or "").strip()
            if reason or next_step:
                lesson = {
                    "pair": list(chosen),
                    "category": category,
                    "why": reason[:200],
                    "iterate": next_step[:200],
                }
                if lesson not in self.lessons:
                    self.lessons.append(lesson)
        elif category:
            line = f"landing [{category}]"
            if line not in self.routes:
                self.routes.append(line)
        for label in menu:
            if chosen and label == chosen[1]:
                continue
            if label not in self.unused_menu and label not in self.used_far:
                self.unused_menu.append(label)
        if category == "world_supports" and chosen and self.how_found is None:
            leftover = [item for item in menu if item != chosen[1]]
            self.how_found = {
                "operator": operator,
                "pair": list(chosen),
                "unused_menu": leftover[:8],
            }

    def absorb_feedback(
        self,
        feedback: JumpFeedback | dict[str, Any],
        *,
        far: str = "",
    ) -> None:
        """Bank one landing's scores so the next jump is not independent."""
        row = (
            feedback
            if isinstance(feedback, JumpFeedback)
            else JumpFeedback(
                support=str(feedback.get("support") or "none"),
                novelty=str(feedback.get("novelty") or "none"),
                value=str(feedback.get("value") or "unclear"),
                feasibility=str(feedback.get("feasibility") or "unknown"),
                distance=str(feedback.get("distance") or "on_band"),
                policy=str(feedback.get("policy") or "hold"),
                detail=str(feedback.get("detail") or ""),
            )
        )
        self.ledger.absorb(row, far=far)

    def prompt_lines(self) -> tuple[str, ...]:
        """Endpoints, leftover menu, verdict class, topic phrases. Nothing else.

        Empty keeps cached prompts byte-identical. No program, fixture,
        experiment numbers, or landing vectors.
        """
        if not self.routes and not self.world_facts and not self.constraints:
            return ()
        lines: list[str] = []
        if not self.routes:
            # Before any landing: only what the world showed and what the
            # scout ruled out. The first card is written against the data.
            return self.constraint_lines()
        if self.object_phrases:
            lines.append(
                "Topic object phrases (the scientific object stays these; "
                "a distant concept is only a mechanism): "
                + ", ".join(self.object_phrases[:8])
            )
        lines.append(
            "Routes already tried this mission (do not repeat the same "
            "kind on the same research target as a new spray; a different "
            "kind on the same pair is a new idea in the same lineage):"
        )
        lines.extend(f"- {row}" for row in self.routes[:8])
        leftover = [
            item for item in self.unused_menu if item not in self.used_far
        ][:8]
        if leftover:
            lines.append(
                "Unused distant concepts from earlier landings (information "
                "only; pair a concept from THIS jump's distant list, not as "
                "a commanded destination): " + ", ".join(leftover)
            )
        if self.used_levers:
            lines.append(
                "Spray breadth: compiled handles already used as a new "
                "landing this mission (do not mechanically repeat them on "
                "a new far jump). The same scientific lineage may reuse a "
                "handle when matching, control, validator, or measure "
                "changes: " + ", ".join(self.used_levers[:8])
            )
        if self.used_pairs:
            lines.append(
                "Search provenance already opened (pair is provenance, not "
                "identity; a different idea_kind on the same pair is a new "
                "idea): "
                + ", ".join(f"{a} × {b}" for a, b in self.used_pairs[:8])
            )
        if self.lessons:
            lines.append(
                "Approaches already tried (do not regenerate these "
                "mechanisms; a new far jump must weld a different lever):"
            )
            for row in self.lessons[:6]:
                pair = row.get("pair") or []
                combo = (
                    f"{pair[0]} × {pair[1]}"
                    if len(pair) >= 2
                    else str(row.get("category") or "landing")
                )
                bits = [f"{combo} [{row.get('category') or ''}]"]
                if row.get("why"):
                    bits.append(str(row["why"]))
                if row.get("iterate"):
                    bits.append("next: " + str(row["iterate"]))
                lines.append("- " + " — ".join(bits))
        lines.extend(self.constraint_lines())
        extra = self.ledger.prompt_lines()
        if extra:
            lines.extend(extra)
        return tuple(lines)

    def snapshot(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "used_pairs": [list(pair) for pair in self.used_pairs],
            "used_far": list(self.used_far),
            "used_levers": list(self.used_levers),
            "constraints": list(self.constraints),
            "world_facts": list(self.world_facts),
            "blocked_handles": list(self.blocked_handles),
            "unused_menu": list(self.unused_menu)[:12],
            "routes": list(self.routes)[:8],
            "object_phrases": list(self.object_phrases[:8]),
        }
        if self.how_found:
            payload["how_found"] = dict(self.how_found)
        if self.lessons:
            payload["lessons"] = list(self.lessons)[:8]
        search = self.ledger.snapshot()
        if search.get("feedback") or search.get("dead_directions"):
            payload["search"] = search
        return payload


def landing_category(
    record: dict[str, Any] | None = None,
    *,
    killed_by: list[str] | None = None,
    generated: bool = True,
) -> str:
    """Attested outcome of one landing, for the exploration notebook."""
    if not generated:
        return "generation_refused"
    if record:
        if _world_support(record):
            return "world_supports"
        kind = str(record.get("probe_kind") or "").upper()
        if record.get("verdict") == "weakens":
            return "world_weakens" if is_world_kind(kind) else "weakens"
        if record.get("world_incompatible"):
            return "world_incompatible"
        bottleneck = str(record.get("experiment_bottleneck") or "")
        if bottleneck == "method":
            return "probe_method"
        if bottleneck == "implementation":
            return "probe_implementation"
        if bottleneck == "world":
            return "world_incompatible"
        if bottleneck == "no_handle":
            return "no_handle"
        if bottleneck:
            return bottleneck
        if record.get("prior_kills"):
            return "prior"
        if record.get("verdict") == "uninformative":
            if record.get("object_absent"):
                return "object_absent"
            return "uninformative"
        if record.get("verdict") == "supports":
            return "synthetic_supports"
    if pair_structurally_dead(killed_by):
        return "graph_pair"
    if killed_by:
        return "text_gate"
    return "entered"


def normalize_explore(mode: str | None) -> str:
    text = str(mode or DEFAULT_EXPLORE).strip().lower()
    if text in ("fixed", "preset", "all"):
        return "fixed"
    return "auto"


def resolve_idea_rounds(requested: int) -> tuple[int, str]:
    """Map the idea-evolution knob to (extra refines of the same card, mode).

    `-1` means auto: up to `IDEA_AUTO_CAP` rewrites of the *same pair*
    after the first draft. `0` is the one-shot used by tests. A positive
    integer is an explicit extra-attempt cap. This is not a new far jump
    and not a polish of the briefing.
    """
    n = int(requested)
    if n < 0:
        return IDEA_AUTO_CAP, "auto"
    return max(0, n), "fixed"


def pair_structurally_dead(killed_by: list[str] | None) -> bool:
    """True when the *pair* failed the graph oracle.

    Rewriting claim/prediction cannot un-hub a node. Those deaths restore
    a new far landing instead of burning idea_rounds.
    """
    for name in killed_by or []:
        lowered = str(name).lower()
        if any(mark in lowered for mark in GRAPH_PAIR_MARKS):
            return True
    return False


def decide_idea_refine(
    *,
    attempt: int,
    cap: int,
    killed_by: list[str] | None = None,
    alive: bool = False,
    review_changes: tuple[str, ...] | list[str] = (),
    failure_types: tuple[str, ...] | list[str] = (),
    idea_kind: str = "",
) -> dict[str, Any]:
    """Whether to rewrite the same idea in this mission.

    Text-gate deaths are worth a refine: the model needs the kill names
    in H so the next draft is a better idea, not a new pair. A no_handle
    death is the same pair against the scouted lever table — the object
    was right; the mechanism did not compile. World gaps construct the
    experiment's world; they do not rewrite the claim to fit a freeze
    catalog or rebind Pride.

    `review_changes` is the adversarial reviewer's must_change list on a
    card that already *entered*: passing the gates is admission, not
    quality. One pre-registration rewrite against a named objection is
    idea evolution; it happens before any experiment is registered, so
    nothing is being rewritten to fit a result.
    """
    cap = max(0, int(cap))
    attempt = max(0, int(attempt))
    killers = [str(item) for item in (killed_by or []) if str(item).strip()]
    failures = [str(item) for item in (failure_types or []) if str(item).strip()]
    if failures:
        action = refine_action_for_failures(failures, idea_kind)
        if action == "transition_kind" and attempt < cap:
            named = ", ".join(failures[:3])
            return {
                "continue": True,
                "action": "transition_kind",
                "reason": named,
                "failure_types": failures,
                "detail": (
                    f"structured failure {named} admits a kind transition "
                    "on this lineage; do not rewrite the object or invent "
                    "a new far concept"
                ),
            }
        if action == "enter_acquire" and attempt < cap:
            return {
                "continue": True,
                "action": "transition_kind",
                "reason": "missing_world",
                "failure_types": failures,
                "detail": "the freeze cannot attest the needed capability; enter acquire",
            }
    if "no_handle" in killers:
        if attempt >= cap:
            return {
                "continue": False,
                "action": "stop",
                "reason": "hit_cap",
                "detail": f"attempt {attempt + 1} of {cap + 1} still ineligible",
            }
        return {
            "continue": True,
            "action": "refine_idea",
            "reason": "no_handle",
            "detail": (
                "the bound object is right but the mechanism did not compile "
                "onto a declared lever; rewrite the same pair against the "
                "lever table — do not spray a new far noun"
            ),
        }
    if alive:
        changes = [str(item) for item in review_changes if str(item).strip()]
        if changes and attempt < cap:
            named = "; ".join(changes[:3])
            return {
                "continue": True,
                "action": "refine_idea",
                "reason": "review",
                "detail": (
                    f"the adversarial review demands: {named}. Rewrite the "
                    "same pair before registration — do not change the "
                    "object or the far concept"
                ),
            }
        return {
            "continue": False,
            "action": "stop",
            "reason": "entered",
            "detail": "this idea entered research; further rewrites would be polish",
        }
    if pair_structurally_dead(killed_by):
        return {
            "continue": False,
            "action": "stop",
            "reason": "graph_pair",
            "detail": (
                "the pair failed the graph oracle; rewriting the claim "
                "cannot repair it — open a new landing"
            ),
        }
    if not killed_by:
        return {
            "continue": False,
            "action": "stop",
            "reason": "no_text_kill",
            "detail": "nothing textual to repair on this draft",
        }
    if attempt >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "detail": f"attempt {attempt + 1} of {cap + 1} still ineligible",
        }
    killers = ", ".join(str(item) for item in (killed_by or [])[:4])
    return {
        "continue": True,
        "action": "refine_idea",
        "reason": "gate",
        "detail": (
            f"rewrite the same pair so {killers} cannot recur; "
            "do not open a new distant landing"
        ),
    }


def compile_debt(
    *,
    structural_notes: list[str] | tuple[str, ...] | None = None,
    world_lever: str = "",
    alternative: str = "",
    levers: tuple[str, ...] | list[str] = (),
    competing_missing: bool = False,
) -> list[str]:
    """Why this entered pair still needs an in-mission compile rewrite.

    Structural holes come from world-sim. An unnamed competing
    explanation or an honest `none` lever is experiment / no_handle
    work — recorded here so H can see them, not so spray reopens.
    """
    reasons: list[str] = []
    notes = [str(item).strip() for item in (structural_notes or []) if str(item).strip()]
    if notes:
        reasons.append("structural")
    lever = str(world_lever or "").strip().lower()
    menu = [str(item).strip() for item in levers if str(item).strip() and item != "none"]
    if lever in {"", "none"} and menu:
        reasons.append("no_handle")
    if competing_missing or (menu and not str(alternative or "").strip()):
        reasons.append("competing_explanation")
    return reasons


def decide_compile_refine(
    *,
    attempt: int,
    cap: int = COMPILE_REFINE_CAP,
    reasons: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Rewrite the entered pair once against rehearsal / weld debt.

    This is not briefing polish and not a new far jump. The claim
    object, pair, and freeze stay locked. Empty reasons mean the
    registered diagnosis is already the compile.
    """
    cap = max(0, int(cap))
    attempt = max(0, int(attempt))
    debt = [str(item) for item in (reasons or []) if str(item).strip()]
    if not debt:
        return {
            "continue": False,
            "action": "stop",
            "reason": "no_compile_debt",
            "detail": "diagnosis and rehearsal left no structural hole on this pair",
        }
    if attempt >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "detail": f"attempt {attempt + 1} of {cap} already spent on this pair",
        }
    named = ", ".join(debt[:4])
    return {
        "continue": True,
        "action": "refine_idea",
        "reason": "compile",
        "detail": (
            f"rewrite the same pair so {named} cannot recur; "
            "do not rename the object, switch the far concept, or rebind the freeze"
        ),
    }


def resolve_polish_rounds(requested: int) -> tuple[int, str]:
    """Map the user-facing polish knob to (cap, mode).

    `-1` means auto: try up to `POLISH_AUTO_CAP` reviews, then
    `keep_going` stops. `0` still means skip polish entirely.
    """
    n = int(requested)
    if n < 0:
        return POLISH_AUTO_CAP, "auto"
    return max(0, n), "fixed"


def _world_support(row: dict[str, Any]) -> bool:
    """True when supports was measured on an attested freeze the script read."""
    if row.get("prior_kills") or row.get("killed"):
        return False
    if row.get("verdict") != "supports":
        return False
    return is_world_kind(row.get("probe_kind"))


def has_live_line(records: list[dict[str, Any]] | None) -> bool:
    """True when a WORLD `supports` is on the books.

    SYNTHETIC or GENERATED supports are coherence, not a live scientific
    line. Spray and self-evolution read this, not bare `verdict`.
    """
    return any(_world_support(row) for row in records or [])


def _world_kind(program: dict[str, Any] | None) -> bool:
    return is_world_kind((program or {}).get("probe_kind"))


def is_live_program(program: dict[str, Any] | None) -> bool:
    """True when the neighbourhood has a WORLD support.

    That committed line already has an executable plan. Further far
    landings would be spray. SYNTHETIC / GENERATED supports are not live.
    """
    if not program or program.get("verdict") != "supports":
        return False
    return _world_kind(program)


def is_continue_program(program: dict[str, Any] | None) -> bool:
    """True when this pair already holds an executable WORLD plan.

    Occupants are WORLD-only and lock the pair and freeze so a later
    mission cannot spray a new pair over them. Ablation, a sharper test,
    or a lever switch are must-run rows in `EXPERIMENT_PLAN.md` — they
    are executed, not generated as a new card. SYNTHETIC H cannot occupy
    this seat. Weakened lines do not continue.
    """
    if is_live_program(program):
        return True
    if not program or program.get("verdict") != "uninformative":
        return False
    if not _world_kind(program):
        return False
    if program.get("prior_kills") or program.get("killed"):
        return False
    return True


def continue_reason(program: dict[str, Any] | None) -> str:
    """Why spray stays closed, or empty when the neighbourhood may jump."""
    if is_live_program(program) or is_continue_program(program):
        return "plan_executable"
    return ""


def neighbourhood_priority(program: dict[str, Any] | None) -> int | None:
    """Lower wins the single neighbourhood seat. None is not an occupant.

    live support > mechanism-switch debt > first WORLD-uninformative.
    """
    if is_live_program(program):
        return 0
    if not is_continue_program(program):
        return None
    if program.get("must_switch_mechanism"):
        return 1
    return 2


def has_research_plan(records: list[dict[str, Any]] | None) -> bool:
    """Promotion stop: a WORLD support is already on the books.

    Follow-up experiments (uninformative, SYNTHETIC supports) are not
    promotion. They still make the *plan* executable, so ``decide_next``
    stops spray. They do not count as a live WORLD line.
    """
    return has_live_line(records)


def plan_executable(row: dict[str, Any] | None) -> bool:
    """True when remaining work is execute, not another far jump.

    A registered two-arm that already spoke (`supports`, or a WORLD
    `uninformative` whose sharper redesigns are must-run rows) has a
    brief and an `EXPERIMENT_PLAN.md`. Weakened or killed lines are
    closed, not executable. A GENERATED / SYNTHETIC `uninformative` is
    neither: the constructed rehearsal failed to speak, nothing about
    the plan became runnable, and treating it as an occupied seat is
    how a mission dies on three 0/0 probes. The same goes for a probe
    that never met its object (`object_absent`).
    """
    if not row:
        return False
    from .ideakind import ACQUIRE, THEORY, idea_kind_of

    if idea_kind_of(row) in {ACQUIRE, THEORY}:
        return False
    if row.get("prior_kills") or row.get("killed"):
        return False
    verdict = str(row.get("verdict") or "")
    if verdict == "weakens":
        return False
    if verdict == "supports":
        return True
    if verdict == "uninformative":
        if row.get("object_absent"):
            return False
        return is_world_kind(row.get("probe_kind"))
    if str(row.get("experiment_bottleneck") or "") == "no_handle":
        return False
    experiment = str(row.get("experiment") or "").strip()
    if not experiment:
        return False
    lowered = experiment.lower()
    if lowered.startswith("skip the probe") or "no runtime handle" in lowered:
        return False
    return True


def stored_plan_executable(*programs: dict[str, Any] | None) -> bool:
    """True when any stored pair already has an executable plan.

    WORLD lines occupy the neighbourhood seat. SYNTHETIC H does not,
    but its brief+plan is still execute work, not another far jump.
    """
    return any(plan_executable(row) for row in programs if row)


DEFAULT_MIN_LANDINGS = 3


def decide_next(
    *,
    opened: int,
    cap: int,
    records: list[dict[str, Any]] | None = None,
    prior_live: bool = False,
    prior_plan: bool = False,
    min_landings: int = 1,
) -> dict[str, Any]:
    """Whether to open another far landing.

    Stop when a plan is executable — but not before `min_landings`
    distinct landings have opened this mission (bounded by the cap). One
    idea with an executable plan is not a choice: the cwm-iclr2027 run
    opened one landing of a cap of four because its first WORLD probe
    was "uninformative", and the mission ended with nothing to rank. The
    floor is breadth, not spray: every landing under it is still
    generated, gated, and evidenced before the next opens, and the
    executable plan keeps its seat.

    Promotion (WORLD supports) stays narrow and is not a separate spray
    license. A neighbourhood line that already has a plan does not get a
    deepen generate — ablation and lever switches are must-run rows for
    `farfield execute`. A weakened or graph-dead line may open a new
    landing until the cap.
    """
    cap = max(0, int(cap))
    opened = max(0, int(opened))
    floor = min(max(1, int(min_landings)), cap) if cap else 0
    if has_research_plan(records) or prior_live or prior_plan:
        # The floor is about this mission's own first plan. A live line or
        # plan inherited from the neighbourhood is a continue: execute it,
        # do not open a generate slot against it.
        if opened < floor and not (prior_live or prior_plan):
            return {
                "continue": True,
                "action": "continue",
                "reason": "breadth_floor",
                "detail": (
                    f"an executable plan exists, but only {opened} of the "
                    f"{floor} landings this mission must open have opened; "
                    "one idea is not a choice. The plan keeps its seat"
                ),
            }
        return {
            "continue": False,
            "action": "stop",
            "reason": "plan_executable",
            "detail": (
                "this pair already has an executable plan; "
                "more far jumps would be spray. Remaining runs are "
                "farfield execute, not a later research generate"
            ),
        }
    if opened >= cap:
        return {
            "continue": False,
            "action": "stop",
            "reason": "hit_cap",
            "detail": f"opened {opened} of cap {cap} with no executable plan",
        }
    current: dict[str, Any] | None = None
    for row in reversed(list(records or [])):
        if not isinstance(row, dict):
            continue
        if row.get("verdict") or row.get("experiment") or row.get("probe_kind"):
            current = row
            break
    if current is None:
        return {
            "continue": True,
            "action": "continue",
            "reason": "no_plan",
            "detail": (
                "no idea has entered; keep opening distant landings until "
                "one does or the cap is hit"
            ),
        }
    if current.get("prior_kills") or current.get("killed"):
        return {
            "continue": True,
            "action": "continue",
            "reason": "idea_closed",
            "detail": "this pair is closed; a new landing is the fallback",
        }
    if str(current.get("verdict") or "") == "weakens":
        return {
            "continue": True,
            "action": "continue",
            "reason": "idea_closed",
            "detail": (
                "the registered two-arm weakened the claim; "
                "redesigning it to hunt a support is forbidden, "
                "so a new landing may open"
            ),
        }
    # Any record of this mission with an executable plan counts, not only
    # the most recent one: after a later landing died blind, the earlier
    # plan is still the one to execute.
    planned = [row for row in (records or []) if isinstance(row, dict) and plan_executable(row)]
    if plan_executable(current) or planned:
        if opened < floor:
            return {
                "continue": True,
                "action": "continue",
                "reason": "breadth_floor",
                "detail": (
                    f"the current idea's plan is executable, but only {opened} of "
                    f"the {floor} landings this mission must open have opened; "
                    "the next landing must be a different pair on a different lever"
                ),
            }
        return {
            "continue": False,
            "action": "stop",
            "reason": "plan_executable",
            "detail": (
                (
                    "the current idea's experiment plan is executable "
                    if plan_executable(current)
                    else f"an earlier landing this mission ({str((planned or [{}])[0].get('card_id') or 'earlier card')}) has an executable plan "
                )
                + "(see refine-logs/EXPERIMENT_PLAN.md). "
                "Must-run follow-ups are farfield execute, not a new pair."
            ),
        }
    return {
        "continue": True,
        "action": "continue",
        "reason": "no_plan",
        "detail": (
            "no executable plan yet; keep opening distant landings until "
            "one idea can be run or the cap is hit"
        ),
    }
